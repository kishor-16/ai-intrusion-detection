import React, { useMemo, useState } from 'react';
import { useLocation } from 'react-router-dom';
import { useStore } from '../socket.jsx';
import { metaFor, confTier } from '../attackMeta.js';

function ShapBars({ items, valueKey = 'impact' }) {
  const max = Math.max(...items.map((f) => Math.abs(f[valueKey])), 1e-9);
  return (
    <div>
      {items.map((f) => (
        <div className="shap-row" key={f.feature}>
          <div className="shap-feat" title={f.feature}>{f.feature}</div>
          <div className="shap-track">
            <div className={`shap-fill ${(f.direction === '-' || f[valueKey] < 0) ? 'neg' : ''}`}
              style={{ width: `${(Math.abs(f[valueKey]) / max) * 100}%` }} />
          </div>
          <div className="shap-val">{Number(f[valueKey]).toFixed(4)}</div>
        </div>
      ))}
    </div>
  );
}

export default function ExplanationCenter() {
  const { state, api } = useStore();
  const loc = useLocation();
  const { threats } = state;
  const [selId, setSelId] = useState(loc.state?.flowId || null);
  const [live, setLive] = useState(null);
  const [busy, setBusy] = useState(false);

  const recent = useMemo(() => [...threats].reverse().slice(0, 12), [threats]);
  const sel = recent.find((t) => t.meta.flow_id === selId) || recent[0] || null;

  async function runLive() {
    if (!sel) return;
    setBusy(true); setLive(null);
    try { setLive(await api.explain(sel.features)); }
    catch (e) { setLive({ error: String(e) }); }
    setBusy(false);
  }

  if (!sel) {
    return (
      <div className="panel"><h3>AI Explanation Center</h3>
        <div className="empty">
          <span className="empty-icon">🔬</span>
          No detections to explain yet.<br />
          Press <b>⚡ Launch Cyber Attack Simulation</b> in the top bar, then come back —
          every explanation here is computed by SHAP on the actual trained model.
        </div>
      </div>
    );
  }

  return (
    <div className="grid" style={{ gap: 16 }}>
      <div className="panel">
        <h3>Select Detection</h3>
        <div className="feed">
          {recent.map((t) => {
            const meta = metaFor(t.prediction.label);
            const tier = confTier(t.prediction.confidence);
            const select = () => { setSelId(t.meta.flow_id); setLive(null); };
            return (
              <div key={t.meta.flow_id}
                className={`alert ${t.meta.flow_id === sel.meta.flow_id ? 'selected' : ''} ${tier === 'low' ? 'low-conf' : ''}`}
                style={{ '--sev-color': meta.color }}
                role="button" tabIndex={0} onClick={select}
                onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); select(); } }}>
                <div aria-hidden="true">{meta.icon}</div>
                <div>
                  <span className="atk">{t.prediction.label}</span> on <b>{t.meta.device_name}</b>
                  {tier === 'low' && <span className="tag lowconf">LOW CONFIDENCE</span>}
                  <span className="det"> · conf {(t.prediction.confidence * 100).toFixed(1)}% · {t.meta.flow_id}</span>
                </div>
                <div className="det">{t.prediction.latency_ms} ms</div>
              </div>
            );
          })}
        </div>
      </div>

      <div className="grid cols-2">
        <div className="panel">
          <h3>Why the model says “{sel.prediction.label}” — class attribution profile</h3>
          {sel.prediction.shap_top?.length
            ? <ShapBars items={sel.prediction.shap_top} />
            : <div className="empty">Run <code>python ml-inference-service/explain/precompute_shap.py</code> to enable.</div>}
          <div className="src">
            source: SHAP values (Expected Gradients estimator) on the trained model, precomputed
            offline over correctly-classified test flows of this class (shap_summary.json)
          </div>
        </div>

        <div className="panel">
          <h3>Per-instance SHAP — this exact flow</h3>
          {!live && (
            <div className="empty">
              <button className="btn primary" onClick={runLive} disabled={busy}>
                {busy ? 'Computing SHAP…' : '🔬 Explain this flow (live)'}
              </button>
              <div style={{ marginTop: 10 }}>
                Runs SHAP on the trained network for this specific feature vector.
              </div>
            </div>
          )}
          {live?.error && <div className="empty">Failed: {live.error}</div>}
          {live?.features && (
            <>
              <ShapBars items={live.features} />
              <div className="src">
                computed live in {live.explain_time_ms} ms · model verdict {live.label}
                {' '}({(live.confidence * 100).toFixed(1)}%) · per-instance Expected Gradients (SHAP estimator)
              </div>
            </>
          )}
        </div>
      </div>

      <div className="panel">
        <h3>Model probability distribution for this flow</h3>
        <ShapBars items={Object.entries(sel.prediction.probs)
          .map(([feature, v]) => ({ feature, impact: v }))
          .sort((a, b) => b.impact - a.impact)} />
        <div className="src">source: softmax output of the int8 edge model for this flow (real inference)</div>
      </div>
    </div>
  );
}
