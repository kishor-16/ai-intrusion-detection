/**
 * Deterministic demo scenario engine.
 *
 * A FIXED timeline of ticks (no randomness in this file; the simulator's RNG is
 * reset to the global seed on every start/reset, so the whole run replays
 * identically). Each tick either emits normal traffic (devices round-robin) or
 * injects an attack wave on a specific device. Detections come from REAL
 * inference on each generated flow — nothing here is scripted output.
 *
 * Exposes: start(), pause(), resume(), reset(), status().
 */

const TICK_MS = Number(process.env.TICK_MS || 1100);

// Attack waves: [startTick, endTick] inclusive, device under attack.
const WAVES = [
  { from: 10, to: 13, deviceId: 'cam-01' },    // DDoS flooding
  { from: 18, to: 21, deviceId: 'med-04' },    // brute force
  { from: 26, to: 29, deviceId: 'router-03' }, // port scanning
  { from: 34, to: 37, deviceId: 'sensor-02' }, // spoofing
  { from: 42, to: 45, deviceId: 'home-05' },   // malware beaconing
];
const END_TICK = 52;

class Scenario {
  constructor({ simulator, inference, onEvent }) {
    this.sim = simulator;       // axios instance -> simulator service
    this.inf = inference;       // axios instance -> inference service
    this.onEvent = onEvent;     // (eventName, payload) => broadcast
    this.deviceIds = [];
    this._clear();
  }

  _clear() {
    this.tick = -1;
    this.running = false;
    this.finished = false;
    this.timer = null;
    this.waveState = {};        // deviceId -> true while under attack
  }

  status() {
    return {
      running: this.running,
      finished: this.finished,
      tick: this.tick,
      endTick: END_TICK,
      tickMs: TICK_MS,
      waves: WAVES,
    };
  }

  async start(deviceIds) {
    if (this.running) return this.status();
    if (this.finished) await this.reset(deviceIds);
    this.deviceIds = deviceIds;
    if (this.tick < 0) {
      // fresh run: reset the simulator RNG => deterministic replay
      await this.sim.post('/reset', {});
    }
    this.running = true;
    this.onEvent('scenario-status', this.status());
    this.timer = setInterval(() => this._step().catch((e) => {
      this.onEvent('service-status', { level: 'error', message: e.message });
    }), TICK_MS);
    return this.status();
  }

  pause() {
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
    this.running = false;
    this.onEvent('scenario-status', this.status());
    return this.status();
  }

  async reset() {
    if (this.timer) clearInterval(this.timer);
    await this.sim.post('/reset', {});
    this._clear();
    this.onEvent('scenario-reset', {});
    this.onEvent('scenario-status', this.status());
    return this.status();
  }

  _activeWave(t) {
    return WAVES.find((w) => t >= w.from && t <= w.to) || null;
  }

  async _step() {
    this.tick += 1;
    const t = this.tick;

    if (t > END_TICK) {
      this.pause();
      this.finished = true;
      this.onEvent('scenario-status', this.status());
      return;
    }

    // per-tick status so the dashboard's wave timeline tracks the run live
    this.onEvent('scenario-status', this.status());

    const wave = this._activeWave(t);

    // device status transitions (attack begins / ends)
    for (const w of WAVES) {
      if (t === w.from) {
        this.waveState[w.deviceId] = true;
        this.onEvent('device-status', { deviceId: w.deviceId, status: 'under-attack' });
      }
      if (t === w.to + 1) {
        this.waveState[w.deviceId] = false;
        this.onEvent('device-status', { deviceId: w.deviceId, status: 'protected' });
      }
    }

    if (wave) {
      // attack tick: 2 attack flows on the target + 1 normal flow elsewhere
      await this._emitFlows(wave.deviceId, 'attack', 2);
      const other = this.deviceIds[t % this.deviceIds.length];
      if (other !== wave.deviceId) await this._emitFlows(other, 'normal', 1);
    } else {
      // normal tick: round-robin one device
      const dev = this.deviceIds[t % this.deviceIds.length];
      await this._emitFlows(dev, 'normal', 1);
    }
  }

  /** Generate flows on the simulator, run REAL inference on each, broadcast. */
  async _emitFlows(deviceId, mode, count) {
    const gen = await this.sim.post('/generate', { device_id: deviceId, mode, count });
    for (const ev of gen.data.events) {
      const pred = await this.inf.post('/predict', { features: ev.features });
      const p = pred.data;
      const payload = {
        meta: ev.meta,
        prediction: p,             // label/confidence/probs/latency_ms/shap_top — real
        features: ev.features,     // kept so Explanation Center can request live SHAP
        tick: this.tick,
      };
      this.onEvent('prediction', payload);
      if (p.label !== 'Normal') this.onEvent('threat', payload);
    }
  }
}

module.exports = { Scenario, WAVES, END_TICK, TICK_MS };
