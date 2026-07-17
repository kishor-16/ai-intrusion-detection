# Implementation Decisions

## ML route: B — PyTorch → ONNX → ONNX Runtime (int8)

Chosen over TF/Keras→TFLite because:
- The team trains on an NVIDIA RTX GPU on **native Windows**; TensorFlow dropped
  native-Windows GPU support after 2.10, while PyTorch CUDA works out of the box.
- ONNX Runtime dynamic quantization (int8 weights) gives the same edge story —
  quantized model, measured size/latency — without the version-fragile
  PyTorch→ONNX→TF→TFLite bridge the build brief warns against.
- One exporter (`torch.onnx`), one runtime (`onnxruntime`) for training-side
  eval, benchmarks, and serving. float32-vs-int8 cost is *measured*, not assumed
  (`benchmarks/compare_baseline.py` reports both).

Edge realism: benchmark + serving sessions run with `intra_op_num_threads = 1`
(single core), batch size 1 — the honest edge-gateway scenario.

**Winsorized quantization (measured necessity).** Z-scored flow features are
extremely heavy-tailed, and we *measured* what that does to naive int8:
dynamic −14 accuracy points, static MinMax −12, static Percentile −7 —
256 quantization levels stretched over a range hundreds of σ wide leave ~2
bins for normal traffic. Fix: clip scaled inputs to ±c (c searched over a
grid on the validation set), calibrate, evaluate, and serve with the *same*
clip (`models/int8_meta.json`; applied identically by the inference service
and benchmarks, so deployed path == evaluated path). `quantize.py` refuses to
ship any candidate losing >3 points vs fp32.

## Model

`EdgeShieldNet`: 46 → 256 → 128 → 64 → 6 MLP with BatchNorm + Dropout (~55k
params) — still deliberately edge-sized (~8× fewer params than the baseline).
`BaselineNet`: 46 → 512 → 512 → 256 → 6 (~420k params) float32, our
reproduction of the centralized IEEE-style approach for the measured
comparison. Both defined once in `ml-engine/model.py`. Both consume identical
preprocessing: signed-log1p on heavy-tailed continuous features → standard
scaling → ±8 clip (spec persisted in `common/artifacts/transform.json` and
applied identically by the inference service — training path == serving path).
EdgeShield training additionally uses class-weighted cross-entropy (Brute
Force is ~4× rarer), label smoothing 0.05, and a cosine LR schedule.

## Dataset

CICIoT2023 (46 features, 33 raw attack labels mapped to 6 classes in
`common/schema.py::RAW_LABEL_MAP`). Because it is ~13 GB, `download_dataset.py`
builds a stratified subset; a bundled **synthetic sample** (statistically
realistic, class-conditional generators in `make_sample.py`) guarantees a
no-download bring-up. Provenance is recorded and displayed — the UI badge says
"SAMPLE DATA" until the real subset is used. Metrics are always real
measurements on whatever source trained the model.

## Simulator honesty model

Flows = real dataset rows of the target class + ±3% jitter on continuous
features (never on indicator/protocol columns) → guaranteed schema match and
in-distribution inputs, so detections are genuinely correct model behaviour.
The simulator also ships ground truth (`true_class`) with each flow, which the
backend uses to compute **live accuracy** (predictions vs truth this run) —
displayed instead of a static training number.

## Federated learning

FedAvg (McMahan et al. 2017), 5 non-IID clients partitioned by device profile
(60% of each attack class to its owner client). Two interchangeable engines:

- `--engine builtin` (default): self-contained PyTorch implementation of the
  same algorithm — chosen as default for one-laptop reliability (no Ray).
- `--engine flower`: Flower `start_simulation` (install
  `requirements-optional.txt`); identical partition and history format.

Run offline; the dashboard **replays** `federated-learning/history.json`
round-by-round. No live FL during judging (determinism rule).

## SHAP

Two levels, both from the real trained network. SHAP values are estimated with
**Expected Gradients** — the exact estimator `shap.GradientExplainer`
implements — in pure PyTorch (`ml-engine/explain_utils.py`). We dropped the
shap package from the runtime path because its numba/llvmlite native binaries
crash silently on some Windows machines; pure torch means: if training runs,
explanations run. Same math, dependency-proof.

- **Class profiles** (fast path shown on every detection): mean |attribution|
  over correctly-classified test flows per class, precomputed offline into
  `shap_summary.json`.
- **Per-instance** (`/explain`, Explanation Center button): live Expected
  Gradients (k=128) on the selected flow's exact vector; the response includes
  its own measured compute time.

## Determinism inventory

seed 42 in: sample generator, preprocess split, torch training, FL partition +
local training, benchmark sampling, simulator RNG (reset on scenario start),
scenario timeline (fixed tick table, zero randomness in Node).
