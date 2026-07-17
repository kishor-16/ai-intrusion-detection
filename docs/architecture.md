# EdgeShield AI — Architecture

## Runtime topology

Three backend processes + one frontend, one startup script (`scripts/run_all.py`):

```
[iot-simulator (Python/FastAPI) :8200]
        |  POST /generate  (schema-matched raw feature vectors + display metadata)
        v
[backend (Node + Express + Socket.IO) :4000]
        |  POST /predict  (one raw vector)          ^
        v                                           | REST snapshots
[ml-inference-service (Python/FastAPI) :8100]       |
   int8 ONNX + scaler + SHAP                        |
                                                    v
                       [frontend (React + Vite) :5173]  <-- Socket.IO events
```

Rationale: Python owns all ML (no PyTorch-in-Node pain); the documented
microservice boundary is a deliberate architecture decision — each service is
independently restartable and testable.

## Module contracts

### ml-inference-service (owns the model)
- `POST /predict` — body `{features: [46 raw floats, schema order]}` →
  `{label, confidence, probs, latency_ms, shap_top, shap_scope, quantized}`.
  `latency_ms` is the measured model-run time for this request.
  The service applies the persisted scaler internally — callers send RAW values.
- `POST /explain` — same body → per-instance SHAP (GradientExplainer, live).
- `GET /model-info` — real file size, params, test metrics, data provenance, benchmarks.

### iot-simulator
- `GET /devices` — the 5 device profiles.
- `POST /generate` `{device_id, mode: normal|attack, count}` → events, each:
  - `features`: raw vector in EXACT schema order (sampled real dataset row of the
    device's class, continuous features jittered ±3%, indicators untouched),
  - `meta`: display-only fields (IPs, protocol, rates, timestamp, `simulated: true`,
    plus ground truth `true_class` used to measure live accuracy honestly).
- `POST /reset` — reseeds the RNG → deterministic replay.

### backend (relay/orchestrator — NO ML)
- Socket.IO events: `init`, `prediction`, `threat`, `device-status`, `metrics`,
  `scenario-status`, `scenario-reset`, `service-status`.
- REST: `/devices /threats /predictions /metrics /model-performance
  /federated-status /benchmarks /explain /scenario/{start,pause,reset,status}`.
- `demo/scenario.js`: fixed tick timeline (no randomness in Node; the simulator
  RNG is seeded), 5 attack waves, `start/pause/reset`.

### frontend
Renders ONLY data received over Socket.IO/REST. Computes no metrics. Each panel
footer cites its data source.

## The feature-schema rule (demo-killer prevention)

`common/schema.py` lists the 46 CICIoT2023 feature columns in canonical order and
is imported by BOTH `ml-engine` (training) and `iot-simulator` (generation);
`common/artifacts/scaler.pkl` is fitted once in `preprocess.py` and applied only
inside the inference service. Human-friendly fields never enter the vector.

## Determinism

Global seed 42 everywhere (numpy, torch, simulator RNG, scenario timeline).
Reset → simulator `/reset` + state wipe → byte-identical rerun. No network
access needed at demo time; no live training during judging (FL replayed from
`history.json`).
