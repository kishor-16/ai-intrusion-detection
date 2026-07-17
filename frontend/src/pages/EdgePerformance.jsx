import React, { useEffect, useState } from 'react';
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, LabelList, Cell,
} from 'recharts';
import { useStore } from '../socket.jsx';

const tip = { background: '#0a0f24', border: '1px solid rgba(120,160,255,0.3)', borderRadius: 8 };

export default function EdgePerformance() {
  const { api } = useStore();
  const [b, setB] = useState(null);

  useEffect(() => { api.getBenchmarks().then(setB).catch(() => setB({ available: false })); }, [api]);

  if (!b) return <div className="panel"><div className="empty">Loading benchmarks…</div></div>;
  if (!b.available) {
    return (
      <div className="panel"><h3>Edge Performance</h3>
        <div className="empty">
          No benchmark results yet.<br />
          Run <code>python benchmarks/compare_baseline.py</code> — this page shows only
          measured numbers, so it stays empty until they exist.
        </div>
      </div>
    );
  }

  const base = b.baseline; const edge = b.edgeshield; const fp32 = b.edgeshield_fp32;
  const sizeData = [
    { name: 'Baseline (fp32)', kb: base.size_kb, fill: '#8b96b8' },
    { name: 'EdgeShield fp32', kb: fp32.size_kb, fill: '#2f81f7' },
    { name: 'EdgeShield int8', kb: edge.size_kb, fill: '#22ff88' },
  ];
  const latData = [
    { name: 'Baseline', mean: base.latency.mean_ms, p95: base.latency.p95_ms },
    { name: 'EdgeShield fp32', mean: fp32.latency.mean_ms, p95: fp32.latency.p95_ms },
    { name: 'EdgeShield int8', mean: edge.latency.mean_ms, p95: edge.latency.p95_ms },
  ];

  return (
    <div className="grid" style={{ gap: 16 }}>
      <div className="grid cols-4">
        <div className="panel">
          <div className="stat-value accent-green">{(base.size_kb / edge.size_kb).toFixed(1)}×</div>
          <div className="stat-label">smaller than baseline</div>
          <div className="stat-sub">{base.size_kb} KB → {edge.size_kb} KB (bytes on disk)</div>
        </div>
        <div className="panel">
          <div className="stat-value accent-blue">{(base.latency.mean_ms / edge.latency.mean_ms).toFixed(1)}×</div>
          <div className="stat-label">faster inference</div>
          <div className="stat-sub">{base.latency.mean_ms} ms → {edge.latency.mean_ms} ms mean, batch 1</div>
        </div>
        <div className="panel">
          <div className="stat-value accent-green">{(edge.accuracy * 100).toFixed(2)}%</div>
          <div className="stat-label">int8 accuracy</div>
          <div className="stat-sub">baseline {(base.accuracy * 100).toFixed(2)}% on same test set</div>
        </div>
        <div className="panel">
          <div className="stat-value">{edge.latency.p95_ms} ms</div>
          <div className="stat-label">p95 latency (edge)</div>
          <div className="stat-sub">{b.edgeshield.latency.runs} timed runs, 1 CPU thread</div>
        </div>
      </div>

      <div className="grid cols-2">
        <div className="panel">
          <h3>Model size — KB on disk</h3>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={sizeData}>
              <CartesianGrid stroke="rgba(255,255,255,0.05)" vertical={false} />
              <XAxis dataKey="name" stroke="#8b96b8" fontSize={11} />
              <YAxis stroke="#8b96b8" fontSize={11} />
              <Tooltip contentStyle={tip} />
              <Bar dataKey="kb" radius={[6, 6, 0, 0]}>
                {sizeData.map((d) => <Cell key={d.name} fill={d.fill} />)}
                <LabelList dataKey="kb" position="top" fill="#e6ecff" fontSize={11} />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="panel">
          <h3>Inference latency — ms (batch 1, single thread)</h3>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={latData}>
              <CartesianGrid stroke="rgba(255,255,255,0.05)" vertical={false} />
              <XAxis dataKey="name" stroke="#8b96b8" fontSize={11} />
              <YAxis stroke="#8b96b8" fontSize={11} />
              <Tooltip contentStyle={tip} />
              <Bar dataKey="mean" fill="#00d4ff" radius={[6, 6, 0, 0]} name="mean" />
              <Bar dataKey="p95" fill="#2f81f7" radius={[6, 6, 0, 0]} name="p95" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="panel">
        <h3>Baseline vs EdgeShield — measured comparison</h3>
        <div className="table-scroll">
        <table className="cmp">
          <thead><tr><th>Dimension</th><th>Baseline (centralized IEEE-style)</th><th>EdgeShield AI</th></tr></thead>
          <tbody>
            <tr><td>Accuracy</td><td className="num">{base.accuracy}</td><td className="num win">{edge.accuracy}</td></tr>
            <tr><td>F1 (macro)</td><td className="num">{base.f1_macro}</td><td className="num win">{edge.f1_macro}</td></tr>
            <tr><td>Model size</td><td className="num">{base.size_kb} KB (fp32)</td><td className="num win">{edge.size_kb} KB (int8)</td></tr>
            <tr><td>Parameters</td><td className="num">{base.params?.toLocaleString()}</td><td className="num win">smaller edge net</td></tr>
            <tr><td>Latency mean / p95</td><td className="num">{base.latency.mean_ms} / {base.latency.p95_ms} ms</td><td className="num win">{edge.latency.mean_ms} / {edge.latency.p95_ms} ms</td></tr>
            <tr><td>Raw data leaves device?</td><td>Yes</td><td className="win">No — weights only (FL)</td></tr>
            <tr><td>Explainable?</td><td>No</td><td className="win">Yes — SHAP</td></tr>
            <tr><td>Runs on edge?</td><td>No</td><td className="win">Yes</td></tr>
          </tbody>
        </table>
        </div>
        <div className="src">
          source: benchmarks/results.json, generated {b.generated_at} on this machine ·
          test set {b.test_rows?.toLocaleString()} flows · data: {b.data_source}
        </div>
      </div>
    </div>
  );
}
