import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useStore } from '../socket.jsx';
import { metaFor, confTier } from '../attackMeta.js';

const tstr = (ts) => new Date(ts * 1000).toLocaleTimeString();

const NEXT_STATUS = { Open: 'Investigating', Investigating: 'Contained', Contained: 'Resolved' };

export default function ThreatMonitor() {
  const { state, api } = useStore();
  const { threats, predictions, responses, incidents } = state;
  const [filter, setFilter] = useState('All');
  const nav = useNavigate();

  const classes = ['All', ...new Set(threats.map((t) => t.prediction.label))];
  // merged feed: model detections + autonomous response actions, newest first
  const feed = [
    ...threats.map((t) => ({ kind: 'threat', ts: t.meta.timestamp, item: t })),
    ...responses.map((r) => ({ kind: 'response', ts: r.ts, item: r })),
  ]
    .sort((a, b) => b.ts - a.ts)
    .filter((e) => filter === 'All'
      || (e.kind === 'threat' ? e.item.prediction.label === filter : e.item.label === filter));

  const openExplain = (flowId) => nav('/explain', { state: { flowId } });

  return (
    <div className="grid" style={{ gap: 16 }}>
      <div className="panel">
        <h3>
          <span>Live Threat Feed — real-time model detections &amp; auto-responses</span>
          <span className="chip-row" role="group" aria-label="filter by attack class">
            {classes.map((c) => (
              <button key={c} className="chip" aria-pressed={filter === c}
                onClick={() => setFilter(c)}>{c}</button>
            ))}
          </span>
        </h3>
        <div className="feed scroll-slim" style={{ maxHeight: '42vh', overflowY: 'auto' }}>
          {feed.length === 0 && (
            <div className="empty">
              <span className="empty-icon">🛡</span>
              No threats detected. Press <b>⚡ Launch Cyber Attack Simulation</b> —
              attacks will be injected into the simulated traffic and every
              detection you see here is a live inference of the int8 edge model.
            </div>
          )}
          {feed.map((e) => (e.kind === 'response' ? (
            <div key={e.item.id}
              className={`alert response-entry ${e.item.action === 'quarantine' ? 'quarantine' : ''}`}>
              <div style={{ fontSize: 18 }} aria-hidden="true">
                {e.item.action === 'quarantine' ? '⛔' : e.item.action === 'release' ? '🔓' : '🤖'}
              </div>
              <div className="resp-text">
                {e.item.text}
                {e.item.incident_id != null && (
                  <span className="tag mitre">INCIDENT #{e.item.incident_id}</span>
                )}
              </div>
              <div className="det">{tstr(e.item.ts)}</div>
            </div>
          ) : (() => {
            const t = e.item;
            const meta = metaFor(t.prediction.label);
            const tier = confTier(t.prediction.confidence);
            return (
              <div key={t.meta.flow_id}
                className={`alert ${tier === 'low' ? 'low-conf' : ''}`}
                style={{ '--sev-color': meta.color }}
                role="button" tabIndex={0}
                title="Open in AI Explanations"
                onClick={() => openExplain(t.meta.flow_id)}
                onKeyDown={(ev) => {
                  if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); openExplain(t.meta.flow_id); }
                }}>
                <div style={{ fontSize: 20 }} aria-hidden="true">{meta.icon}</div>
                <div>
                  <div>
                    <span className="atk">{t.prediction.label}</span>
                    {' '}on <b>{t.meta.device_name}</b>
                    {t.severity && t.severity !== 'None' && (
                      <span className={`tag sev-${t.severity.toLowerCase()}`}
                        title={`threat score ${t.threat_score}/100 — computed by the backend scoring engine`}>
                        {t.severity.toUpperCase()} {t.threat_score}
                      </span>
                    )}
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
          })()))}
        </div>
        <div className="src">
          source: Socket.IO `threat` events (inference-service /predict, label ≠ Normal) +
          `response` events (autonomous response engine) · scores: backend scoring engine
        </div>
      </div>

      <div className="panel">
        <h3>
          <span>Incidents — auto-created for high-risk detections</span>
          <a className="btn ghost inc-action" href={api.analyticsCsvUrl} download>⬇ Export CSV</a>
        </h3>
        {incidents.length === 0 ? (
          <div className="empty">No incidents yet — created automatically when a detection scores Critical
            or a device is quarantined.</div>
        ) : (
          <div className="table-scroll">
            <table className="cmp">
              <thead><tr>
                <th>ID</th><th>Device</th><th>Attack</th><th>Severity</th>
                <th>Score</th><th>Opened</th><th>Status</th><th>Action</th>
              </tr></thead>
              <tbody>
                {incidents.slice(0, 8).map((inc) => (
                  <tr key={inc.id}>
                    <td className="num">#{inc.id}</td>
                    <td>{inc.device_name}</td>
                    <td style={{ color: metaFor(inc.attack).color, fontWeight: 650 }}>{inc.attack}</td>
                    <td>{inc.severity}</td>
                    <td className="num">{inc.threat_score ?? '—'}</td>
                    <td className="num">{tstr(inc.created_at)}</td>
                    <td><span className={`inc-status ${inc.status}`}>{inc.status}</span></td>
                    <td>
                      {NEXT_STATUS[inc.status] ? (
                        <button className="btn ghost inc-action"
                          onClick={() => api.updateIncident(inc.id, NEXT_STATUS[inc.status])}>
                          → {NEXT_STATUS[inc.status]}
                        </button>
                      ) : <span className="det">closed</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <div className="src">source: incidents table (persisted) · lifecycle: Open → Investigating → Contained → Resolved</div>
      </div>

      <div className="panel">
        <h3>All Traffic — classification stream (most recent 25)</h3>
        <div className="rowline" style={{ color: 'var(--text)', fontWeight: 600 }}>
          <div>time</div><div>flow</div><div>model label</div>
          <div className="hide-sm">confidence</div><div className="hide-sm">latency</div>
        </div>
        <div className="scroll-slim" style={{ maxHeight: '26vh', overflowY: 'auto' }}>
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
