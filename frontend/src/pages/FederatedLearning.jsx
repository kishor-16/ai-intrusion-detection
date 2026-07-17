import React, { useEffect, useRef, useState } from 'react';
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ReferenceLine,
} from 'recharts';
import { useStore } from '../socket.jsx';

export default function FederatedLearning() {
  const { state, api } = useStore();
  const { flLive } = state;
  const [fl, setFl] = useState(null);
  const [shown, setShown] = useState(0);   // replay cursor
  const [playing, setPlaying] = useState(false);
  const timer = useRef(null);

  useEffect(() => {
    api.getFederated().then((d) => {
      setFl(d);
      if (d?.rounds?.length) setShown(d.rounds.length); // full curve on load; Replay animates
    }).catch(() => setFl({ available: false }));
  }, [api]);
  useEffect(() => () => clearInterval(timer.current), []);

  // when a live run finishes, history.json has been rewritten — reload it
  const wasRunning = useRef(false);
  useEffect(() => {
    if (wasRunning.current && !flLive.running) {
      api.getFederated().then((d) => {
        setFl(d);
        if (d?.rounds?.length) setShown(d.rounds.length);
      }).catch(() => {});
    }
    wasRunning.current = flLive.running;
  }, [flLive.running, api]);

  const liveMode = flLive.running && flLive.rounds.length > 0;

  if (!fl) return <div className="panel"><div className="empty">Loading…</div></div>;
  if (!fl.available) {
    return (
      <div className="panel"><h3>Federated Learning</h3>
        <div className="empty">
          No FL run recorded yet.<br />
          Run <code>python federated-learning/run_federated.py</code> (offline, before the demo).
          This page replays the real per-round history — it never animates invented numbers.
        </div>
      </div>
    );
  }

  const rounds = fl.rounds || [];
  if (!rounds.length) return <div className="panel"><div className="empty">history.json has no rounds.</div></div>;
  const data = liveMode
    ? flLive.rounds.map((r) => ({ ...r, pct: +(r.accuracy * 100).toFixed(2) }))
    : rounds.slice(0, shown).map((r) => ({ ...r, pct: +(r.accuracy * 100).toFixed(2) }));
  const done = shown >= rounds.length;

  function play() {
    clearInterval(timer.current);
    setShown(1); setPlaying(true);
    timer.current = setInterval(() => {
      setShown((s) => {
        if (s + 1 >= rounds.length + 1) { clearInterval(timer.current); setPlaying(false); return rounds.length; }
        return s + 1;
      });
    }, 700);
  }

  const final = rounds[rounds.length - 1];

  return (
    <div className="grid" style={{ gap: 16 }}>
      <div className="grid cols-4">
        <div className="panel"><div className="stat-value accent-blue">{fl.num_clients}</div>
          <div className="stat-label">edge clients (non-IID)</div><div className="stat-sub">{fl.partition}</div></div>
        <div className="panel"><div className="stat-value">{rounds.length - 1}</div>
          <div className="stat-label">FedAvg rounds</div><div className="stat-sub">{fl.algorithm}</div></div>
        <div className="panel"><div className="stat-value accent-green">{(final.accuracy * 100).toFixed(2)}%</div>
          <div className="stat-label">final global accuracy</div>
          <div className="stat-sub">centralized reference: {fl.centralized_reference_accuracy != null ? `${(fl.centralized_reference_accuracy * 100).toFixed(2)}%` : '—'}</div></div>
        <div className="panel"><div className="stat-value accent-green">0</div>
          <div className="stat-label">raw flows shared</div><div className="stat-sub">{fl.privacy_note}</div></div>
      </div>

      <div className="panel">
        <h3>
          <span>
            {liveMode
              ? <>Global model accuracy per round — <span style={{ color: 'var(--green)' }}>● TRAINING LIVE NOW</span></>
              : 'Global model accuracy per round — replay of the real offline run'}
          </span>
          <span style={{ display: 'flex', gap: 8 }}>
            <button className="btn" style={{ padding: '5px 14px', fontSize: 12 }}
              onClick={() => api.runFederated({ rounds: 5, localEpochs: 1 })}
              disabled={flLive.running}
              title="Spawns run_federated.py on this machine: 5 clients train locally and FedAvg aggregates, streamed here round by round.">
              {flLive.running ? `⏳ Training… round ${flLive.rounds.length ? flLive.rounds[flLive.rounds.length - 1].round : 0}` : '⚡ Run live FL now'}
            </button>
            <button className="btn primary" style={{ padding: '5px 14px', fontSize: 12 }}
              onClick={play} disabled={playing || flLive.running}>
              {playing ? 'Replaying…' : done && shown ? '↻ Replay' : '▶ Replay training'}
            </button>
          </span>
        </h3>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={data}>
            <CartesianGrid stroke="rgba(255,255,255,0.05)" />
            <XAxis dataKey="round" stroke="#8b96b8" fontSize={11}
              label={{ value: 'federated round', fill: '#8b96b8', fontSize: 11, dy: 14 }} />
            <YAxis domain={[0, 100]} stroke="#8b96b8" fontSize={11} unit="%" width={44} />
            <Tooltip contentStyle={{ background: '#0a0f24', border: '1px solid rgba(120,160,255,0.3)', borderRadius: 8 }} />
            {fl.centralized_reference_accuracy != null && (
              <ReferenceLine y={fl.centralized_reference_accuracy * 100} stroke="#8b96b8"
                strokeDasharray="5 5" label={{ value: 'centralized reference', fill: '#8b96b8', fontSize: 10 }} />
            )}
            <Line dataKey="pct" stroke="#22ff88" strokeWidth={2.5}
              dot={{ fill: '#22ff88', r: 3.5 }} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
        <div className="src">
          {liveMode
            ? 'source: live federated-round events — run_federated.py --stream is training on this machine right now'
            : `source: federated-learning/history.json · engine ${fl.engine} · run ${fl.generated_at} in ${fl.duration_sec}s · replay of a real run (⚡ Run live FL retrains for real)`}
        </div>
      </div>

      <div className="panel">
        <h3>Participating edge clients</h3>
        <div className="grid cols-5">
          {(fl.clients || []).map((c) => (
            <div key={c.id} className="device">
              <div><span className="dot online" /><span className="name">{c.name}</span></div>
              <div className="ip">{c.id}</div>
              <div className="stat-sub">{c.samples?.toLocaleString()} local flows</div>
              <div className="stat-sub">specialist: <b style={{ color: 'var(--cyan)' }}>{c.attack}</b></div>
            </div>
          ))}
        </div>
        <div className="src">non-IID shards: each client predominantly its own device profile's traffic — raw data never pooled</div>
      </div>
    </div>
  );
}
