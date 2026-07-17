import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useStore } from '../socket.jsx';
import { metaFor, confTier } from '../attackMeta.js';

const tstr = (ts) => new Date(ts * 1000).toLocaleTimeString();

export default function ThreatMonitor() {
  const { state } = useStore();
  const { threats, predictions } = state;
  const [filter, setFilter] = useState('All');
  const nav = useNavigate();

  const classes = ['All', ...new Set(threats.map((t) => t.prediction.label))];
  const shown = [...threats].reverse()
    .filter((t) => filter === 'All' || t.prediction.label === filter);

  const openExplain = (flowId) => nav('/explain', { state: { flowId } });

  return (
    <div className="grid" style={{ gap: 16 }}>
      <div className="panel">
        <h3>
          <span>Live Threat Feed — real-time model detections</span>
          <span className="chip-row" role="group" aria-label="filter by attack class">
            {classes.map((c) => (
              <button key={c} className="chip" aria-pressed={filter === c}
                onClick={() => setFilter(c)}>{c}</button>
            ))}
          </span>
        </h3>
        <div className="feed scroll-slim" style={{ maxHeight: '46vh', overflowY: 'auto' }}>
          {shown.length === 0 && (
            <div className="empty">
              <span className="empty-icon">🛡</span>
              No threats detected. Press <b>⚡ Launch Cyber Attack Simulation</b> —
              attacks will be injected into the simulated traffic and every
              detection you see here is a live inference of the int8 edge model.
            </div>
          )}
          {shown.map((t) => {
            const meta = metaFor(t.prediction.label);
            const tier = confTier(t.prediction.confidence);
            return (
              <div key={t.meta.flow_id}
                className={`alert ${tier === 'low' ? 'low-conf' : ''}`}
                style={{ '--sev-color': meta.color }}
                role="button" tabIndex={0}
                title="Open in AI Explanations"
                onClick={() => openExplain(t.meta.flow_id)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openExplain(t.meta.flow_id); }
                }}>
                <div style={{ fontSize: 20 }} aria-hidden="true">{meta.icon}</div>
                <div>
                  <div>
                    <span className="atk">{t.prediction.label}</span>
                    {' '}on <b>{t.meta.device_name}</b>
                    {meta.sev !== 'none' && <span className={`tag sev-${meta.sev}`}>{meta.sev.toUpperCase()}</span>}
                    {meta.mitre && (
                      <span className="tag mitre" title={`MITRE ATT&CK — ${meta.mitreName}`}>
                        ATT&CK {meta.mitre}
                      </span>
                    )}
                    {tier === 'low' && (
                      <span className="tag lowconf"
                        title="The model is uncertain about this verdict — treat as a lead, not a conviction.">
                        LOW CONFIDENCE
                      </span>
                    )}
                  </div>
                  <div className="det">
                    {t.meta.src_ip} → {t.meta.dst_ip} · {t.meta.protocol} ·
                    {' '}{t.meta.packet_rate} pkt/s · {t.meta.total_bytes} B · {tstr(t.meta.timestamp)}
                    {t.meta.attack_label && <> · scenario: {t.meta.attack_label} (simulated)</>}
                  </div>
                </div>
                <div className="alert-right" style={{ textAlign: 'right' }}>
                  <div>
                    <div className={`conf-num ${tier}`}>{(t.prediction.confidence * 100).toFixed(1)}%</div>
                    <div className="det">{t.prediction.latency_ms} ms inference</div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
        <div className="src">source: Socket.IO `threat` events — inference-service /predict responses where label ≠ Normal</div>
      </div>

      <div className="panel">
        <h3>All Traffic — classification stream (most recent 25)</h3>
        <div className="rowline" style={{ color: 'var(--text)', fontWeight: 600 }}>
          <div>time</div><div>flow</div><div>model label</div>
          <div className="hide-sm">confidence</div><div className="hide-sm">latency</div>
        </div>
        <div className="scroll-slim" style={{ maxHeight: '28vh', overflowY: 'auto' }}>
          {[...predictions].reverse().slice(0, 25).map((p) => {
            const meta = metaFor(p.prediction.label);
            return (
              <div key={p.meta.flow_id} className={`rowline ${p.prediction.label === 'Normal' ? 'normal' : ''}`}>
                <div>{tstr(p.meta.timestamp)}</div>
                <div>{p.meta.device_name} · {p.meta.src_ip}→{p.meta.dst_ip}</div>
                <div className="label" style={{ color: meta.color }}>{p.prediction.label}</div>
                <div className="hide-sm">{(p.prediction.confidence * 100).toFixed(1)}%</div>
                <div className="hide-sm">{p.prediction.latency_ms} ms</div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
