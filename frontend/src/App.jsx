import React, { useEffect, useMemo, useState } from 'react';
import { NavLink, Route, Routes } from 'react-router-dom';
import { useStore } from './socket.jsx';
import { metaFor, classFromScenarioLabel } from './attackMeta.js';
import CommandCenter from './pages/CommandCenter.jsx';
import ThreatMonitor from './pages/ThreatMonitor.jsx';
import ExplanationCenter from './pages/ExplanationCenter.jsx';
import EdgePerformance from './pages/EdgePerformance.jsx';
import FederatedLearning from './pages/FederatedLearning.jsx';
import NetworkMap from './pages/NetworkMap.jsx';

const NAV = [
  { to: '/', label: 'Command Center', icon: 'M4 13h6V4H4v9zm0 7h6v-5H4v5zm10 0h6v-9h-6v9zm0-16v5h6V4h-6z' },
  { to: '/threats', label: 'Threat Monitor', icon: 'M12 2 3 6v6c0 5 3.8 9.7 9 10 5.2-.3 9-5 9-10V6l-9-4z' },
  { to: '/explain', label: 'AI Explanations', icon: 'M12 2a7 7 0 0 0-4 12.7V17h8v-2.3A7 7 0 0 0 12 2zM9 21h6v-2H9v2z' },
  { to: '/edge', label: 'Edge Performance', icon: 'M13 2 3 14h7v8l10-12h-7l0-8z' },
  { to: '/federated', label: 'Federated Learning', icon: 'M12 3a3 3 0 1 1-3 3 3 3 0 0 1 3-3zM5 15a3 3 0 1 1-3 3 3 3 0 0 1 3-3zm14 0a3 3 0 1 1-3 3 3 3 0 0 1 3-3zM11 9l-4 5m6-5 4 5' },
  { to: '/map', label: 'Network Map', icon: 'M12 2a4 4 0 1 1-4 4 4 4 0 0 1 4-4zM4 16a3 3 0 1 1-2 3m18-3a3 3 0 1 0 2 3M8.5 8.5 5 17m10.5-8.5L19 17' },
];

function Icon({ d }) {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="1.7" strokeLinejoin="round" strokeLinecap="round">
      <path d={d} />
    </svg>
  );
}

function Clock() {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const t = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(t);
  }, []);
  return (
    <span className="clock" title="local time">
      {now.toLocaleDateString(undefined, { day: '2-digit', month: 'short' })}{' '}
      {now.toLocaleTimeString(undefined, { hour12: false })}
    </span>
  );
}

/** Attack-wave timeline: driven entirely by the scenario engine's real status
    (tick, endTick, waves) — waves colored by the device's scenario attack. */
function ScenarioStrip() {
  const { state } = useStore();
  const { scenario, devices, metrics } = state;
  const { tick = -1, endTick, waves, running, finished } = scenario;

  const waveInfo = useMemo(() => (waves || []).map((w, i) => {
    const dev = devices.find((d) => d.id === w.deviceId);
    const cls = classFromScenarioLabel(dev?.attack_label) || 'DDoS';
    return { ...w, i, device: dev, cls, color: metaFor(cls).color };
  }), [waves, devices]);

  if (tick < 0 || !endTick || !waveInfo.length) return null;

  const active = waveInfo.find((w) => tick >= w.from && tick <= w.to);
  const nDone = waveInfo.filter((w) => tick > w.to).length;

  let phase;
  if (finished) {
    phase = <>Scenario complete — <b>{metrics.threatsDetected ?? 0}</b> detections, all waves contained</>;
  } else if (active) {
    phase = (
      <>Wave {active.i + 1}/{waveInfo.length} — <span className="atk-name">
        {active.device?.attack_label || active.cls}</span> on <b>{active.device?.name || active.deviceId}</b></>
    );
  } else if (!running) {
    phase = <>Paused at tick {tick}</>;
  } else if (nDone === 0) {
    phase = <>Warm-up — baseline normal traffic</>;
  } else if (nDone === waveInfo.length) {
    phase = <>Cool-down — post-attack traffic</>;
  } else {
    phase = <>Interval — next wave incoming</>;
  }

  return (
    <div className="scenario-strip" role="status" aria-live="polite">
      <span className="phase">{phase}</span>
      <div className="timeline" title="scenario timeline — colored bands are attack waves">
        {waveInfo.map((w) => (
          <div key={w.i} className={`seg ${tick >= w.from ? 'hit' : ''}`} style={{
            left: `${(w.from / endTick) * 100}%`,
            width: `${((w.to - w.from + 1) / endTick) * 100}%`,
            background: w.color,
          }} />
        ))}
        <div className="fill" style={{ width: `${Math.min((tick / endTick) * 100, 100)}%` }} />
      </div>
      <div className="wave-chips">
        {waveInfo.map((w) => (
          <span key={w.i} style={{ color: tick >= w.from && tick <= w.to ? w.color : undefined }}
            className={`wave-chip ${tick > w.to ? 'done' : tick >= w.from ? 'active' : ''}`}>
            {w.cls}
          </span>
        ))}
      </div>
      <span className="tickno">TICK {Math.min(tick, endTick)}/{endTick}</span>
    </div>
  );
}

export default function App() {
  const { state, api } = useStore();
  const { scenario, connected, dataSource } = state;

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="logo">
          <div className="logo-mark">⬡</div>
          <div>
            <div className="logo-name">EdgeShield AI</div>
            <div className="logo-sub">IOT SECURITY SOC</div>
          </div>
        </div>
        <nav className="nav">
          {NAV.map((n) => (
            <NavLink key={n.to} to={n.to} end={n.to === '/'} title={n.label}
              className={({ isActive }) => (isActive ? 'active' : '')}>
              <Icon d={n.icon} /> <span className="nav-label">{n.label}</span>
            </NavLink>
          ))}
        </nav>
        <div className="foot">
          Privacy-preserving, explainable Edge-AI IDS.<br />
          Traffic: <b>simulated</b> · Detections: <b>real model inference</b>.
        </div>
      </aside>

      <div className="main">
        <header className="topbar">
          <h1>Security Operations Center</h1>
          <Clock />
          <span className={`badge ${connected ? 'live' : 'red'}`}>
            {connected ? '● LINK ACTIVE' : '● BACKEND OFFLINE'}
          </span>
          <span className="badge sim" title="All IoT traffic is simulated (dataset rows + jitter). Detections are real model inferences.">
            SIMULATED TRAFFIC · {dataSource === 'synthetic-sample' ? 'SAMPLE DATA' : 'CICIoT2023'}
          </span>
          {scenario.running ? (
            <button className="btn" onClick={api.pause}>❚❚ Pause</button>
          ) : (
            <button className="btn primary" onClick={api.start} disabled={!connected}>
              ⚡ Launch Cyber Attack Simulation
            </button>
          )}
          <button className="btn ghost" onClick={api.reset} disabled={!connected}>⟲ Reset</button>
        </header>

        <ScenarioStrip />

        <main className="content">
          <Routes>
            <Route path="/" element={<CommandCenter />} />
            <Route path="/threats" element={<ThreatMonitor />} />
            <Route path="/explain" element={<ExplanationCenter />} />
            <Route path="/edge" element={<EdgePerformance />} />
            <Route path="/federated" element={<FederatedLearning />} />
            <Route path="/map" element={<NetworkMap />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}
