"""
EdgeShield ML Inference Service (FastAPI) — owns the model. Port 8100.

Endpoints
---------
POST /predict     one RAW feature vector in schema order -> real inference on the
                  int8 ONNX edge model. latency_ms is the actually-measured model
                  run time for THIS request. shap_top is the predicted class's
                  precomputed SHAP profile (see /explain for per-instance).
POST /explain     real per-instance SHAP (GradientExplainer on the trained
                  PyTorch net) for the Explanation Center.
GET  /model-info  real size/params/provenance from disk.
GET  /health

No fabricated numbers anywhere: every value is measured here or read from
artifacts produced by ml-engine/ and benchmarks/.
"""

import json
import os
import sys
import time
from pathlib import Path

# Guard against duplicate-OpenMP hard aborts on Windows (torch MKL + numba TBB)
# when /explain lazily imports shap. Must be set before native libs load.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import joblib
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "ml-engine"))
from common.schema import (BENCH_RESULTS, BEST_MODEL_PT, CLASSES,  # noqa: E402
                           FEATURE_COLUMNS, FP32_ONNX, INT8_META, INT8_ONNX,
                           METRICS_JSON, N_FEATURES, PROCESSED_DIR,
                           PROVENANCE_PATH, SCALER_PATH, SHAP_SUMMARY,
                           TRANSFORM_PATH)

app = FastAPI(title="EdgeShield ML Inference Service")

# ---------- startup: load artifacts ----------
STATE: dict = {}


@app.on_event("startup")
def load_artifacts():
    if not SCALER_PATH.exists():
        raise RuntimeError("scaler missing — run ml-engine/preprocess.py first")
    STATE["scaler"] = joblib.load(SCALER_PATH)

    # feature transform spec written by preprocess.py — MUST match training:
    # raw flow -> signed log1p (heavy-tailed cols) -> scaler -> clip
    STATE["log_idx"] = []
    STATE["post_clip"] = None
    if TRANSFORM_PATH.exists():
        t = json.loads(TRANSFORM_PATH.read_text())
        STATE["log_idx"] = [FEATURE_COLUMNS.index(c)
                            for c in t.get("signed_log1p_columns", [])]
        STATE["post_clip"] = t.get("post_scale_clip")

    model_path = INT8_ONNX if INT8_ONNX.exists() else FP32_ONNX
    if not model_path.exists():
        raise RuntimeError("no ONNX model — run ml-engine/train.py first")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1  # edge realism
    STATE["sess"] = ort.InferenceSession(str(model_path), sess_options=opts,
                                         providers=["CPUExecutionProvider"])
    STATE["model_path"] = model_path
    STATE["quantized"] = model_path == INT8_ONNX

    # winsorized quantization: the int8 model was calibrated AND evaluated with
    # scaled inputs clipped to ±input_clip — apply the identical clip here so
    # the deployed path matches the evaluated path (models/int8_meta.json).
    STATE["input_clip"] = None
    if STATE["quantized"] and INT8_META.exists():
        STATE["input_clip"] = json.loads(INT8_META.read_text()).get("input_clip")

    STATE["metrics"] = json.loads(METRICS_JSON.read_text()) if METRICS_JSON.exists() else {}
    STATE["provenance"] = (json.loads(PROVENANCE_PATH.read_text())
                           if PROVENANCE_PATH.exists() else {})
    STATE["shap_summary"] = (json.loads(SHAP_SUMMARY.read_text())
                             if SHAP_SUMMARY.exists() else None)
    STATE["torch_model"] = None  # lazy-loaded for /explain
    print(f"[inference] serving {model_path.name} (quantized={STATE['quantized']})")


def _lazy_torch():
    """Load the PyTorch net + background sample once, on first /explain call.
    Attribution uses Expected Gradients (the SHAP GradientExplainer estimator)
    in pure PyTorch — no shap-package native deps (see ml-engine/explain_utils)."""
    if STATE["torch_model"] is None:
        import torch
        from model import EdgeShieldNet
        net = EdgeShieldNet()
        ckpt = torch.load(BEST_MODEL_PT, map_location="cpu")
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        bg = np.load(PROCESSED_DIR / "train.npz")["X"]
        rng = np.random.default_rng(42)
        STATE["torch_model"] = net
        STATE["background"] = torch.from_numpy(bg[rng.choice(len(bg), 200, replace=False)])
    return STATE["torch_model"], STATE["background"]


class PredictIn(BaseModel):
    features: list[float]  # RAW values, exact schema order (scaler applied here)


class ExplainIn(BaseModel):
    features: list[float]


def _softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def _transform(features: list[float]) -> np.ndarray:
    """Raw schema-order vector -> model-ready scaled vector (same path as
    training: signed log1p -> scaler -> post-scale clip)."""
    x = np.asarray(features, dtype=np.float64).reshape(1, -1)
    if STATE["log_idx"]:
        li = STATE["log_idx"]
        x[:, li] = np.sign(x[:, li]) * np.log1p(np.abs(x[:, li]))
    xs = STATE["scaler"].transform(x).astype(np.float32)
    if STATE["post_clip"] is not None:
        xs = np.clip(xs, -STATE["post_clip"], STATE["post_clip"])
    return xs


@app.post("/predict")
def predict(body: PredictIn):
    if len(body.features) != N_FEATURES:
        raise HTTPException(422, f"expected {N_FEATURES} features, got {len(body.features)}")
    xs = _transform(body.features)
    if STATE["input_clip"] is not None:
        xs = np.clip(xs, -STATE["input_clip"], STATE["input_clip"])

    t0 = time.perf_counter()
    logits = STATE["sess"].run(None, {"features": xs})[0][0]
    latency_ms = (time.perf_counter() - t0) * 1000  # REAL measured inference time

    probs = _softmax(logits)
    idx = int(probs.argmax())
    label = CLASSES[idx]

    shap_top = []
    summary = STATE["shap_summary"]
    if summary and label in summary["per_class"]:
        shap_top = summary["per_class"][label]["top_features"][:6]

    return {
        "label": label,
        "confidence": round(float(probs[idx]), 4),
        "probs": {CLASSES[i]: round(float(p), 4) for i, p in enumerate(probs)},
        "latency_ms": round(latency_ms, 3),
        "shap_top": shap_top,          # class-level profile (precomputed offline)
        "shap_scope": "class-level",   # per-instance available via /explain
        "quantized": STATE["quantized"],
    }


@app.post("/explain")
def explain(body: ExplainIn):
    """Real per-instance SHAP-value estimate (Expected Gradients) for the
    selected detection."""
    if len(body.features) != N_FEATURES:
        raise HTTPException(422, f"expected {N_FEATURES} features")
    import torch
    from explain_utils import expected_gradients
    net, background = _lazy_torch()
    x = np.asarray(body.features, dtype=np.float64).reshape(1, -1)  # raw, for display
    xs = _transform(body.features)
    xt = torch.from_numpy(xs)

    with torch.no_grad():
        probs = torch.softmax(net(xt), dim=1)[0].numpy()
    idx = int(probs.argmax())

    t0 = time.perf_counter()
    vals = expected_gradients(net, xt[0], background, idx, k=128).numpy()
    elapsed = (time.perf_counter() - t0) * 1000
    order = np.argsort(-np.abs(vals))[:10]
    return {
        "label": CLASSES[idx],
        "confidence": round(float(probs[idx]), 4),
        "shap_scope": "per-instance",
        "explain_time_ms": round(elapsed, 1),
        "features": [
            {"feature": FEATURE_COLUMNS[i], "impact": round(float(vals[i]), 5),
             "value": round(float(x[0, i]), 4)}
            for i in order
        ],
    }


@app.get("/model-info")
def model_info():
    m = STATE["metrics"]
    p = STATE["provenance"]
    info = {
        "model_file": STATE["model_path"].name,
        "size_kb": round(STATE["model_path"].stat().st_size / 1024, 1),
        "params": m.get("params"),
        "quantized": STATE["quantized"],
        "input_clip": STATE["input_clip"],
        "classes": CLASSES,
        "n_features": N_FEATURES,
        "test_metrics": m.get("test"),
        "training": m.get("training"),
        "data_source": p.get("data_source", "unknown"),
        "data_note": p.get("note", ""),
    }
    if BENCH_RESULTS.exists():
        info["benchmarks"] = json.loads(BENCH_RESULTS.read_text())
    return info


@app.get("/health")
def health():
    return {"status": "ok", "model": STATE["model_path"].name}


if __name__ == "__main__":
    import os
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("INFERENCE_PORT", 8100)),
                log_level="warning")
