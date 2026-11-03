# EdgeShield AI — Measured Results

*Generated 2026-11-03T21:24:56 by `benchmarks/compare_baseline.py` on this machine.
Data source: `synthetic-sample` · test set: 1,800 flows ·
latency = single flow, batch 1, 1 CPU thread (edge realism), 300 timed runs after warmup.*

| Dimension | Baseline (centralized IEEE-style) | EdgeShield AI | Source |
|---|---|---|---|
| Detection accuracy | 0.9978 | 0.9994 | this script, shared test set |
| F1 (macro) | 0.9978 | 0.9994 | this script, shared test set |
| Model size (KB) | 1,661.2 (float32) | **75.1 (int8)** | bytes on disk |
| Parameters | 422,150 | 54,470 | model definition |
| Inference latency, mean (ms) | 0.104 | **0.051** | this script |
| Inference latency, p95 (ms) | 0.123 | **0.056** | this script |
| Raw data leaves device? | Yes | **No (weights only)** | design (federated-learning/) |
| Explainable? | No | **Yes (SHAP)** | ml-inference-service/explain |
| Runs on edge? | No | **Yes** | size + latency above |

EdgeShield float32 reference: acc 0.9994, F1 0.9994, 218.4 KB,
0.036 ms mean — quantization cost is measured, not assumed.
