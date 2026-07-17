# Demo Script — the 4-minute judge walkthrough (rehearse this)

**Before judging:** `python scripts/run_all.py` already running · browser on
http://localhost:5173 · run `python scripts/smoke_test.py --live` once → all green.

## 0:00 — Frame it (Command Center, idle)
"Existing IEEE IoT-IDS models are centralized, heavy, private-data-hungry and
black-box. EdgeShield fixes all four in software. Everything you'll see is honest:
the *traffic* is simulated — see the amber badge — but every *detection* is a live
inference of a real int8 model we trained, and every number on screen cites its source."

## 0:30 — ⚡ Launch Cyber Attack Simulation
Warm-up normal traffic streams in (all classified Normal, green). Point at
**Avg Detection Latency** — "that's the measured per-flow inference time of the
quantized model, updating live" — and **Live AI Accuracy** — "predictions checked
against the simulator's ground truth, computed this run, not a training screenshot."

## 1:00 — First wave hits (Smart Camera, DDoS)
Threat Monitor: glowing alert — device, predicted class, confidence, latency.
"The model has never seen these exact flows; they're dataset-distribution samples
with jitter. Same seed, same run — press Reset later and it replays identically."

## 1:45 — AI Explanations
Click the DDoS alert → Explanation Center. Class attribution profile (precomputed
SHAP from the trained net), then press **Explain this flow (live)** — per-instance
SHAP computed on the spot, with its own measured compute time. "Not a canned chart."

## 2:30 — Edge Performance
"Here's the enhancement, measured: our reproduction of the centralized baseline vs
EdgeShield — size, latency mean and p95, accuracy on the same test set, benchmarked
on this laptop by `benchmarks/compare_baseline.py`. The table in `docs/results.md`
is generated from the same JSON — one source of truth."

## 3:15 — Federated Learning
"Training is private: five non-IID edge clients, FedAvg, only weights move.
This curve is a replay of the real offline run — round-by-round global accuracy."
Press **▶ Replay training**.

## 3:45 — Network Map + close
Map shows the last waves containing. "Scalable by design — microservice boundary,
schema contract, deterministic demo engine. Everything reproducible from one
command on a clean machine." **⟲ Reset** to show determinism if asked.

## Judge Q&A ammo
- *"Is any number fake?"* — No. Unmeasured values render as "—". Grep the frontend:
  it computes no metrics; each panel footer names its source file/endpoint.
- *"Why is accuracy so high?"* — Sample-data mode is labeled; on real CICIoT2023
  subsets expect high-90s too (flow features are highly separable) — the honest
  caveat is in provenance and we say it before they ask.
- *"Why not live FL?"* — Determinism during judging; the run is real and replayed.
- *"Why ONNX not TFLite?"* — Windows CUDA training + fragile conversion chain;
  quantized-edge story identical (docs/implementation.md).
