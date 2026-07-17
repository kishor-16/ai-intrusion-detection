"""
THE money artifact: measured baseline vs EdgeShield comparison.

Baseline  = BaselineNet (large centralized float32 MLP, no privacy, no
            explainability) — trained here if models/baseline_model.pt missing.
EdgeShield = EdgeShieldNet int8 ONNX (train.py + quantize.py) + federated
            training (federated-learning/) + SHAP (ml-inference-service/explain).

Every numeric cell is MEASURED by this script on this machine:
  - accuracy/F1: evaluated on the same held-out test set
  - model size: actual bytes on disk
  - latency: mean single-flow inference over N runs (after warmup)

Outputs: benchmarks/results.json + docs/results.md
Usage:   python benchmarks/compare_baseline.py [--runs 300]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml-engine"))
from common.schema import (BASELINE_ONNX, BASELINE_PT, BENCH_RESULTS, CLASSES,
                           FL_HISTORY, FP32_ONNX, GLOBAL_SEED, INT8_META,
                           INT8_ONNX, METRICS_JSON, PROCESSED_DIR, REPO_ROOT)
from model import BaselineNet, count_params, export_onnx  # noqa: E402

RESULTS_MD = REPO_ROOT / "docs" / "results.md"


def load(name):
    d = np.load(PROCESSED_DIR / f"{name}.npz")
    return d["X"], d["y"]


def train_baseline(epochs=25):
    """Train the centralized float32 baseline (once) on the same data."""
    torch.manual_seed(GLOBAL_SEED); np.random.seed(GLOBAL_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tr, y_tr = load("train"); X_val, y_val = load("val")
    model = BaselineNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = torch.nn.CrossEntropyLoss()
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    dl = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=True,
                                     generator=torch.Generator().manual_seed(GLOBAL_SEED))
    from sklearn.metrics import f1_score
    best, best_state = -1, None
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); lossf(model(xb), yb).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(torch.from_numpy(X_val).to(device)).argmax(1).cpu().numpy()
        f1 = f1_score(y_val, pred, average="macro")
        print(f"[baseline] epoch {ep+1:02d} val_f1={f1:.4f}")
        if f1 > best:
            best, best_state = f1, {k: v.cpu().clone() for k, v in model.state_dict().items()}
    print(f"[baseline] trained in {time.time()-t0:.1f}s (params={count_params(model):,})")
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "arch": "BaselineNet"}, BASELINE_PT)
    export_onnx(model.cpu(), BASELINE_ONNX)
    return model.cpu()


def eval_onnx(onnx_path, X, y, clip=None):
    import onnxruntime as ort
    from sklearn.metrics import accuracy_score, f1_score
    if clip is not None:  # winsorized int8 path (models/int8_meta.json)
        X = np.clip(X, -clip, clip)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    preds = []
    for i in range(0, len(X), 4096):
        out = sess.run(None, {"features": X[i:i+4096].astype(np.float32)})[0]
        preds.append(out.argmax(1))
    yp = np.concatenate(preds)
    return (round(float(accuracy_score(y, yp)), 4),
            round(float(f1_score(y, yp, average="macro")), 4))


def latency_onnx(onnx_path, X, runs, warmup=50, clip=None):
    """Mean single-flow (batch=1) CPU latency in ms — the edge scenario."""
    import onnxruntime as ort
    if clip is not None:
        X = np.clip(X, -clip, clip)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1  # edge-device realism: single core
    sess = ort.InferenceSession(str(onnx_path), sess_options=opts,
                                providers=["CPUExecutionProvider"])
    idx = np.random.default_rng(GLOBAL_SEED).integers(0, len(X), runs + warmup)
    times = []
    for k, i in enumerate(idx):
        x = X[i:i+1].astype(np.float32)
        t0 = time.perf_counter()
        sess.run(None, {"features": x})
        dt = (time.perf_counter() - t0) * 1000
        if k >= warmup:
            times.append(dt)
    arr = np.array(times)
    return {"mean_ms": round(float(arr.mean()), 3),
            "p50_ms": round(float(np.percentile(arr, 50)), 3),
            "p95_ms": round(float(np.percentile(arr, 95)), 3),
            "runs": runs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=300)
    ap.add_argument("--baseline-epochs", type=int, default=25)
    ap.add_argument("--retrain-baseline", action="store_true",
                    help="retrain even if a baseline exists (REQUIRED after any "
                         "preprocessing change, or the comparison is stale)")
    args = ap.parse_args()

    for p in (FP32_ONNX, INT8_ONNX):
        if not p.exists():
            sys.exit(f"[compare] {p.name} missing — run train.py and quantize.py first")

    if args.retrain_baseline or not BASELINE_ONNX.exists():
        print("[compare] training baseline on current preprocessed data")
        train_baseline(args.baseline_epochs)

    X_te, y_te = load("test")

    # winsorized quantization: int8 is calibrated/served with clipped inputs
    int8_clip = None
    if INT8_META.exists():
        int8_clip = json.loads(INT8_META.read_text()).get("input_clip")
        print(f"[compare] int8 evaluated with input clip ±{int8_clip} (as deployed)")

    print("[compare] evaluating accuracy/F1 on shared test set...")
    b_acc, b_f1 = eval_onnx(BASELINE_ONNX, X_te, y_te)
    e_acc, e_f1 = eval_onnx(INT8_ONNX, X_te, y_te, clip=int8_clip)
    f_acc, f_f1 = eval_onnx(FP32_ONNX, X_te, y_te)

    print("[compare] measuring latency (batch=1, 1 thread)...")
    b_lat = latency_onnx(BASELINE_ONNX, X_te, args.runs)
    e_lat = latency_onnx(INT8_ONNX, X_te, args.runs, clip=int8_clip)
    f_lat = latency_onnx(FP32_ONNX, X_te, args.runs)

    kb = lambda p: round(p.stat().st_size / 1024, 1)
    edge_metrics = json.loads(METRICS_JSON.read_text()) if METRICS_JSON.exists() else {}
    edge_params = edge_metrics.get("params")
    edge_params_str = f"{edge_params:,}" if isinstance(edge_params, int) else "n/a"
    fl = json.loads(FL_HISTORY.read_text()) if FL_HISTORY.exists() else None

    results = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "test_rows": int(len(y_te)),
        "classes": CLASSES,
        "data_source": edge_metrics.get("data_source", "unknown"),
        "baseline": {
            "name": "Centralized IEEE-style MLP (float32)",
            "accuracy": b_acc, "f1_macro": b_f1,
            "size_kb": kb(BASELINE_ONNX), "latency": b_lat,
            "params": count_params(BaselineNet()),
            "raw_data_leaves_device": True, "explainable": False, "edge_ready": False,
        },
        "edgeshield_fp32": {
            "accuracy": f_acc, "f1_macro": f_f1,
            "size_kb": kb(FP32_ONNX), "latency": f_lat,
        },
        "edgeshield": {
            "name": "EdgeShield AI (int8 edge model + FL + SHAP)",
            "accuracy": e_acc, "f1_macro": e_f1,
            "size_kb": kb(INT8_ONNX), "latency": e_lat,
            "raw_data_leaves_device": False, "explainable": True, "edge_ready": True,
            "fl_final_accuracy": fl["rounds"][-1]["accuracy"] if fl else None,
        },
    }
    BENCH_RESULTS.write_text(json.dumps(results, indent=2))

    md = f"""# EdgeShield AI — Measured Results

*Generated {results['generated_at']} by `benchmarks/compare_baseline.py` on this machine.
Data source: `{results['data_source']}` · test set: {results['test_rows']:,} flows ·
latency = single flow, batch 1, 1 CPU thread (edge realism), {args.runs} timed runs after warmup.*

| Dimension | Baseline (centralized IEEE-style) | EdgeShield AI | Source |
|---|---|---|---|
| Detection accuracy | {b_acc:.4f} | {e_acc:.4f} | this script, shared test set |
| F1 (macro) | {b_f1:.4f} | {e_f1:.4f} | this script, shared test set |
| Model size (KB) | {results['baseline']['size_kb']:,} (float32) | **{results['edgeshield']['size_kb']:,} (int8)** | bytes on disk |
| Parameters | {results['baseline']['params']:,} | {edge_params_str} | model definition |
| Inference latency, mean (ms) | {b_lat['mean_ms']} | **{e_lat['mean_ms']}** | this script |
| Inference latency, p95 (ms) | {b_lat['p95_ms']} | **{e_lat['p95_ms']}** | this script |
| Raw data leaves device? | Yes | **No (weights only)** | design (federated-learning/) |
| Explainable? | No | **Yes (SHAP)** | ml-inference-service/explain |
| Runs on edge? | No | **Yes** | size + latency above |

EdgeShield float32 reference: acc {f_acc:.4f}, F1 {f_f1:.4f}, {results['edgeshield_fp32']['size_kb']:,} KB,
{f_lat['mean_ms']} ms mean — quantization cost is measured, not assumed.
"""
    RESULTS_MD.parent.mkdir(exist_ok=True)
    RESULTS_MD.write_text(md)
    print(f"[compare] wrote {BENCH_RESULTS} and {RESULTS_MD}")
    print(md)


if __name__ == "__main__":
    main()
