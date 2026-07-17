import React from 'react';
import { motion } from 'framer-motion';
import {
  ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip,
  BarChart, Bar, CartesianGrid, Cell,
} from 'recharts';
import { useStore } from '../socket.jsx';
import { metaFor } from '../attackMeta.js';

const fmt = (v, d = 0) => (v === null || v === undefined ? '—' : Number(v).toFixed(d));

const DEVICE_ICONS = {
  'cam-01': '📷', 'sensor-02': '🏭', 'router-03': '📡', 'med-04': '🏥', 'home-05': '🏠',
};

function Stat({ label, value, sub, accent, i }) {
  return (
    <motion.div className="panel" initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.06 }}>
      <div className={`stat-value ${accent || ''}`}>{value}</div>
      <div className="stat-label">{label}</div>
      {sub && <div className="stat-sub">{sub}</div>}
    </motion.div>
  );
}

export default function CommandCenter() {
  const { state } = useStore();
  const { metrics: m, devices, predictions, threats, modelInfo } = state;

  const latencySeries = predictions.slice(-40).map((p, i) => ({
    i, ms: p.prediction.latency_ms,
  }));
  const byClass = {};
  threats.forEach((t) => { byClass[t.prediction.label] = (byClass[t.prediction.label] || 0) + 1; });
  const classSeries = Object.entries(byClass).map(([name, count]) => ({ name, count }));

  return (
    <div className="grid" style={{ gap: 16 }}>
      <div className="grid cols-5">
        <Stat i={0} label="IoT Devices" value={devices.length || '—'} sub="virtual fleet (simulated)" />
        <Stat i={1} label="Flows Analyzed" value={m.totalFlows ?? '—'} accent="accent-blue"
          sub="each one a real model inference" />
        <Stat i={2} label="Threats Detected" value={m.threatsDetected ?? '—'} accent="accent-red"
          sub="live model detections" />
        <Stat i={3} label="Live AI Accuracy" accent="accent-green"
          value={m.liveAccuracy != null ? `${(m.liveAccuracy * 100).toFixed(1)}%` : '—'}
          sub={m.liveSamples ? `measured on ${m.liveSamples} flows this run` : 'runs during simulation'} />
        <Stat i={4} label="Avg Detection Latency" accent="accent-blue"
          value={m.avgLatencyMs != null ? `${fmt(m.avgLatencyMs, 2)} ms` : '—'}
          sub="mean of real per-flow inference times" />
      </div>

      <div className="grid cols-2">
        <div className="panel">
          <h3>Inference latency — live (ms)</h3>
          {latencySeries.length ? (
            <ResponsiveContainer width="100%" height={190}>
              <AreaChart data={latencySeries}>
                <defs>
                  <linearGradient id="lat" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#00d4ff" stopOpacity={0.45} />
                    <stop offset="100%" stopColor="#00d4ff" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="rgba(255,255,255,0.05)" vertical={false} />
                <XAxis dataKey="i" hide />
                <YAxis stroke="#8b96b8" fontSize={11} width={36} />
                <Tooltip contentStyle={{ background: '#0a0f24', border: '1px solid rgba(120,160,255,0.3)', borderRadius: 8 }} />
                <Area dataKey="ms" stroke="#00d4ff" fill="url(#lat)" strokeWidth={2} isAnimationActive={false} />
              </AreaChart>
            </ResponsiveContainer>
          ) : <div className="empty">Launch the simulation to stream live inferences.</div>}
          <div className="src">source: /predict → latency_ms (measured per request)</div>
        </div>

        <div className="panel">
          <h3>Detections by attack class — this run</h3>
          {classSeries.length ? (
            <ResponsiveContainer width="100%" height={190}>
              <BarChart data={classSeries}>
                <CartesianGrid stroke="rgba(255,255,255,0.05)" vertical={false} />
                <XAxis dataKey="name" stroke="#8b96b8" fontSize={11} />
                <YAxis stroke="#8b96b8" fontSize={11} allowDecimals={false} width={30} />
                <Tooltip contentStyle={{ background: '#0a0f24', border: '1px solid rgba(120,160,255,0.3)', borderRadius: 8 }} />
                <Bar dataKey="count" radius={[6, 6, 0, 0]} isAnimationActive={false}>
                  {classSeries.map((d) => <Cell key={d.name} fill={metaFor(d.name).color} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : <div className="empty">No threats yet — the model reports all traffic normal.</div>}
          <div className="src">source: live threat events (model label ≠ Normal)</div>
        </div>
      </div>

      <div className="panel">
        <h3><span>Device Fleet</span> <span className="h3-note">simulated devices, real per-flow classification</span></h3>
        <div className="grid cols-5">
          {devices.map((d) => (
            <div key={d.id} className={`device ${d.status}`}>
              <div>
                <span className={`dot ${d.status}`} />
                <span className="dev-icon" aria-hidden="true">{DEVICE_ICONS[d.id] || '🔌'}</span>
                <span className="name">{d.name}</span>
              </div>
              <div className="ip">{d.ip} · {d.id}</div>
              <div className="stat-sub">{d.normal_desc}</div>
              <div className="stat-sub" style={{ color: d.status === 'under-attack' ? 'var(--red)' : undefined }}>
                {d.status === 'under-attack' ? `⚠ ${d.attack_label}` : d.status === 'protected' ? '✓ threat contained' : 'nominal'}
              </div>
            </div>
          ))}
        </div>
      </div>

      {modelInfo && (
        <div className="panel">
          <h3>Deployed Edge Model</h3>
          <div className="grid cols-4">
            <div><div className="stat-value" style={{ fontSize: 20 }}>{modelInfo.model_file}</div><div className="stat-label">serving file</div></div>
            <div><div className="stat-value accent-blue" style={{ fontSize: 20 }}>{modelInfo.size_kb} KB</div><div className="stat-label">size on disk (int8)</div></div>
            <div><div className="stat-value" style={{ fontSize: 20 }}>{modelInfo.params?.toLocaleString?.() ?? '—'}</div><div className="stat-label">parameters</div></div>
            <div>
              <div className="stat-value accent-green" style={{ fontSize: 20 }}>
                {modelInfo.test_metrics ? `${(modelInfo.test_metrics.accuracy * 100).toFixed(2)}%` : '—'}
              </div>
              <div className="stat-label">held-out test accuracy</div>
            </div>
          </div>
          <div className="src">
            source: /model-info · trained on {modelInfo.data_source} · {modelInfo.training?.gpu ? `GPU: ${modelInfo.training.gpu}` : ''} · F1(macro) {modelInfo.test_metrics?.f1_macro ?? '—'}
          </div>
        </div>
      )}
    </div>
  );
}
