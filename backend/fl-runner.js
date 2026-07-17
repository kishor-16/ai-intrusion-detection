/**
 * Live Federated Learning runner — spawns run_federated.py --stream and
 * relays its per-round events (REAL local training + FedAvg aggregation,
 * happening right now on this machine) to the dashboard over Socket.IO.
 * One run at a time; history.json is rewritten by the script at the end,
 * so the replay view stays consistent with the last live run.
 */

const path = require('path');
const { spawn } = require('child_process');

class FlRunner {
  constructor({ onEvent, stmts, python = process.env.PYTHON || 'python' }) {
    this.onEvent = onEvent;
    this.stmts = stmts;
    this.python = python;
    this.proc = null;
    this.state = { running: false, engine: null, rounds: [], startedAt: null };
  }

  status() { return { ...this.state, rounds: this.state.rounds.slice(-30) }; }

  start({ rounds = 5, localEpochs = 1, engine = 'builtin' } = {}) {
    if (this.proc) return { ok: false, error: 'a federated run is already in progress' };
    const script = path.join(__dirname, '..', 'federated-learning', 'run_federated.py');
    const args = [script, '--stream', '--rounds', String(rounds),
      '--local-epochs', String(localEpochs), '--engine', engine];
    const runStarted = Date.now() / 1000;
    this.state = { running: true, engine, rounds: [], startedAt: runStarted };
    this.proc = spawn(this.python, args, { cwd: path.join(__dirname, '..') });
    this.onEvent('federated-status', this.status());

    let buffer = '';
    this.proc.stdout.on('data', (chunk) => {
      buffer += chunk.toString();
      let nl;
      while ((nl = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, nl).trim();
        buffer = buffer.slice(nl + 1);
        if (!line.startsWith('@@EVENT ')) continue;
        try {
          const ev = JSON.parse(line.slice(8));
          if (ev.type === 'round') {
            this.state.rounds.push(ev);
            this.stmts.insertFlRound.run({
              run_started: runStarted, round: ev.round,
              accuracy: ev.accuracy, f1_macro: ev.f1_macro ?? null,
              engine: this.state.engine,
            });
          }
          this.onEvent('federated-round', ev);
        } catch { /* non-JSON line — ignore */ }
      }
    });
    this.proc.stderr.on('data', (d) => console.error(`[fl-live] ${d.toString().trim()}`));
    this.proc.on('close', (code) => {
      this.proc = null;
      this.state.running = false;
      this.onEvent('federated-status', { ...this.status(), exitCode: code });
    });
    return { ok: true, ...this.status() };
  }

  stop() {
    if (!this.proc) return { ok: false, error: 'no run in progress' };
    this.proc.kill();
    return { ok: true };
  }
}

module.exports = { FlRunner };
