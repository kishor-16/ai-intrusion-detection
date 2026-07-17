# EdgeShield AI — Measured Results

*Generated 2026-07-17T22:46:43 by `benchmarks/compare_baseline.py` on this machine.
Data source: `ciciot2023-subset` · test set: 39,460 flows ·
latency = single flow, batch 1, 1 CPU thread (edge realism), 300 timed runs after warmup.*

| Dimension | Baseline (centralized IEEE-style) | EdgeShield AI | Source |
|---|---|---|---|
| Detection accuracy | 0.8600 | 0.8636 | this script, shared test set |
| F1 (macro) | 0.8298 | 0.8347 | this script, shared test set |
| Model size (KB) | 1,661.2 (float32) | **75.3 (int8)** | bytes on disk |
| Parameters | 422,150 | 54,470 | model definition |
| Inference latency, mean (ms) | 0.09 | **0.02** | this script |
| Inference latency, p95 (ms) | 0.098 | **0.022** | this script |
| Raw data leaves device? | Yes | **No (weights only)** | design (federated-learning/) |
| Explainable? | No | **Yes (SHAP)** | ml-inference-service/explain |
| Runs on edge? | No | **Yes** | size + latency above |

EdgeShield float32 reference: acc 0.8645, F1 0.8355, 218.4 KB,
0.115 ms mean — quantization cost is measured, not assumed.
