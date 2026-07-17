import React from 'react';
import { useStore } from '../socket.jsx';
import { metaFor } from '../attackMeta.js';

const DEVICE_GLYPHS = {
  'cam-01': '📷', 'sensor-02': '🏭', 'router-03': '📡', 'med-04': '🏥', 'home-05': '🏠',
};

// device security states — driven by the backend's autonomous response engine
const STATUS_COLOR = {
  online: '#00d4ff', suspicious: '#ffb020', 'under-attack': '#ff3b5c',
  quarantined: '#b06bff', protected: '#22ff88', offline: '#5a6486',
};
const STATUS_TEXT = {
  suspicious: '? suspicious', quarantined: '⛔ quarantined — traffic blocked',
  protected: '✓ contained', offline: 'offline',
};

/**
 * IoT Network Map (2D SVG: low-risk, offline, smooth).
 * Node/edge states come from live device-status events; attacked links animate
 * with packet dots flowing toward the gateway. Topology is illustrative.
 */
export default function NetworkMap() {
  const { state } = useStore();
  const { devices, threats, scenario } = state;

  const W = 900; const H = 520; const cx = W / 2; const cy = H / 2;
  const R = 190;
  const pos = devices.map((d, i) => {
    const a = (i / Math.max(devices.length, 1)) * Math.PI * 2 - Math.PI / 2;
    return { ...d, x: cx + R * Math.cos(a), y: cy + R * Math.sin(a) };
  });
  const lastByDevice = {};
  threats.forEach((t) => { lastByDevice[t.meta.device_id] = t; });
  const trafficLive = scenario.running;

  return (
    <div className="panel map-wrap">
      <h3>
        <span>IoT Network Topology — simulated fleet · live model verdicts</span>
        <span className="h3-note">{trafficLive ? '⦿ traffic flowing' : 'idle'}</span>
      </h3>
      {devices.length === 0 ? <div className="empty">Waiting for backend…</div> : (
        <svg className="map-svg" viewBox={`0 0 ${W} ${H}`} width="100%" role="img"
          aria-label="Star topology: five IoT devices connected to the EdgeShield gateway">
          <defs>
            <pattern id="mapgrid" width="36" height="36" patternUnits="userSpaceOnUse">
              <path d="M36 0H0V36" fill="none" stroke="rgba(120,160,255,0.05)" strokeWidth="1" />
            </pattern>
            <radialGradient id="gwGlow">
              <stop offset="0%" stopColor="rgba(47,129,247,0.35)" />
              <stop offset="100%" stopColor="rgba(47,129,247,0)" />
            </radialGradient>
          </defs>
          <rect x="0" y="0" width={W} height={H} fill="url(#mapgrid)" rx="10" />

          {/* edges + packet flows */}
          {pos.map((d) => {
            const attacked = d.status === 'under-attack';
            const blocked = d.status === 'quarantined';
            const t = lastByDevice[d.id];
            const pktColor = attacked ? (t ? metaFor(t.prediction.label).color : '#ff3b5c') : '#00d4ff';
            return (
              <g key={`e-${d.id}`}>
                <line x1={cx} y1={cy} x2={d.x} y2={d.y}
                  className={attacked ? 'edge-attack' : 'edge-ok'}
                  opacity={blocked ? 0.25 : 1}
                  strokeDasharray={blocked ? '2 6' : undefined} />
                {!blocked && (attacked || trafficLive) && (
                  <circle r={attacked ? 3.5 : 2.2} fill={pktColor} opacity="0.9">
                    <animateMotion
                      dur={attacked ? '0.8s' : '2.6s'}
                      repeatCount="indefinite"
                      path={`M ${d.x} ${d.y} L ${cx} ${cy}`} />
                  </circle>
                )}
                {attacked && (
                  <circle r="3.5" fill={pktColor} opacity="0.55">
                    <animateMotion dur="0.8s" begin="0.4s" repeatCount="indefinite"
                      path={`M ${d.x} ${d.y} L ${cx} ${cy}`} />
                  </circle>
                )}
              </g>
            );
          })}

          {/* gateway */}
          <g>
            <circle cx={cx} cy={cy} r={54} fill="url(#gwGlow)" />
            <circle cx={cx} cy={cy} r={26} fill="rgba(47,129,247,0.15)"
              stroke="#2f81f7" strokeWidth="2" />
            <circle cx={cx} cy={cy} r={38} fill="none" stroke="rgba(47,129,247,0.25)">
              <animate attributeName="r" values="30;46;30" dur="3s" repeatCount="indefinite" />
              <animate attributeName="opacity" values="0.6;0;0.6" dur="3s" repeatCount="indefinite" />
            </circle>
            <text x={cx} y={cy + 4} textAnchor="middle" fill="#e6ecff" fontWeight="700">GW</text>
            <text x={cx} y={cy + 58} textAnchor="middle">EdgeShield Gateway · 10.0.0.1</text>
            <text x={cx} y={cy + 72} textAnchor="middle" fontSize="9" fill="#5a6486">
              int8 model inference at the edge
            </text>
          </g>

          {/* devices */}
          {pos.map((d) => {
            const attacked = d.status === 'under-attack';
            const color = STATUS_COLOR[d.status] || '#00d4ff';
            const t = lastByDevice[d.id];
            return (
              <g key={d.id}>
                {(attacked || d.status === 'quarantined') && (
                  <circle cx={d.x} cy={d.y} r={22} fill="none" stroke={color}>
                    <animate attributeName="r" values="18;34;18" dur="1.1s" repeatCount="indefinite" />
                    <animate attributeName="opacity" values="0.8;0;0.8" dur="1.1s" repeatCount="indefinite" />
                  </circle>
                )}
                <circle cx={d.x} cy={d.y} r={17} fill={`${color}22`} stroke={color} strokeWidth="2" />
                <text x={d.x} y={d.y + 5} textAnchor="middle" fontSize="14" aria-hidden="true">
                  {DEVICE_GLYPHS[d.id] || '🔌'}
                </text>
                <text x={d.x} y={d.y - 26} textAnchor="middle" fill="#e6ecff" fontWeight="600">{d.name}</text>
                <text x={d.x} y={d.y + 34} textAnchor="middle">{d.ip}</text>
                {attacked && t && (
                  <text x={d.x} y={d.y + 48} textAnchor="middle"
                    fill={metaFor(t.prediction.label).color} fontWeight="700">
                    {t.prediction.label} · {(t.prediction.confidence * 100).toFixed(0)}%
                  </text>
                )}
                {STATUS_TEXT[d.status] && (
                  <text x={d.x} y={d.y + (attacked && t ? 62 : 48)} textAnchor="middle" fill={color}>
                    {STATUS_TEXT[d.status]}
                  </text>
                )}
              </g>
            );
          })}
        </svg>
      )}
      <div className="map-legend">
        <span><span className="dot online" />online</span>
        <span><span className="dot suspicious" />suspicious</span>
        <span><span className="dot under-attack" />under attack</span>
        <span><span className="dot quarantined" />quarantined</span>
        <span><span className="dot protected" />contained</span>
        <span style={{ marginLeft: 'auto' }}>packet dots = flow direction (illustrative)</span>
      </div>
      <div className="src">
        node status: device-status events from the scenario engine · attack labels: real model detections ·
        topology is illustrative (simulated network)
      </div>
    </div>
  );
}
