"""
Quantize the trained EdgeShield model to int8 — and PROVE it kept its accuracy.

Measured problem on CICIoT2023: z-scored flow features are extremely heavy-
tailed (single features can sit hundreds of σ out), so per-tensor int8 input
quantization stretches 256 levels over a huge range and the in-distribution
region collapses into a few bins. We measured: dynamic quant −14 pts,
static MinMax −12 pts, static Percentile −7 pts. Unshippable.

Fix implemented here: WINSORIZED static quantization — clip the scaled inputs
to [−c, c] (searched over several c), calibrate and evaluate with the same
clip. The clip value ships in models/int8_meta.json and the inference service
+ benchmarks apply the identical clip, so the deployed path is exactly the
evaluated path. Selection on the VALIDATION set; test numbers reported by
benchmarks/compare_baseline.py. Refuses to ship if best candidate loses more
than 3 accuracy points vs fp32.

Usage: python ml-engine/quantize.py
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.schema import (FP32_ONNX, GLOBAL_SEED, INT8_META, INT8_ONNX,
                           PROCESSED_DIR)

MAX_ACC_DROP = 0.03
CLIP_GRID = [4.0, 6.0, 10.0, 16.0]


def load(name):
    d = np.load(PROCESSED_DIR / f"{name}.npz")
    return d["X"], d["y"]


def accuracy(onnx_path, X, y, clip=None):
    import onnxruntime as ort
    if clip is not None:
        X = np.clip(X, -clip, clip)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    preds = []
    for i in range(0, len(X), 4096):
        out = sess.run(None, {"features": X[i:i + 4096].astype(np.float32)})[0]
        preds.append(out.argmax(1))
    return float((np.concatenate(preds) == y).mean())


def make_reader(X, clip=None, batch=64, max_batches=200):
    from onnxruntime.quantization import CalibrationDataReader
    if clip is not None:
        X = np.clip(X, -clip, clip)

    class Reader(CalibrationDataReader):
        def __init__(self):
            n = min(len(X), batch * max_batches)
            self.batches = [{"features": X[i:i + batch].astype(np.float32)}
                            for i in range(0, n, batch)]
            self.it = iter(self.batches)

        def get_next(self):
            return next(self.it, None)

    return Reader()


def main():
    if not FP32_ONNX.exists():
        sys.exit("[quantize] models/model_fp32.onnx missing — run train.py first")

    from onnxruntime.quantization import (CalibrationMethod, QuantFormat,
                                          QuantType, quantize_static)

    rng = np.random.default_rng(GLOBAL_SEED)
    X_tr, _ = load("train")
    X_val, y_val = load("val")
    calib = X_tr[rng.choice(len(X_tr), min(12800, len(X_tr)), replace=False)]

    model_in = FP32_ONNX
    try:
        from onnxruntime.quantization.shape_inference import quant_pre_process
        pre = Path(tempfile.gettempdir()) / "edgeshield_preproc.onnx"
        quant_pre_process(str(FP32_ONNX), str(pre))
        model_in = pre
    except Exception as e:
        print(f"[quantize] note: quant_pre_process skipped ({e})")

    fp32_acc = accuracy(FP32_ONNX, X_val, y_val)
    print(f"[quantize] fp32 val_acc={fp32_acc:.4f}")
    for c in CLIP_GRID:
        a = accuracy(FP32_ONNX, X_val, y_val, clip=c)
        print(f"[quantize] fp32 with input clip ±{c:<4}: val_acc={a:.4f} "
              f"(clipping sanity — should barely move)")

    tmpdir = Path(tempfile.gettempdir())
    candidates = []  # (name, path, clip, val_acc)

    def try_candidate(name, clip, fn):
        out = tmpdir / f"edgeshield_{name}.onnx"
        try:
            fn(out)
            acc = accuracy(out, X_val, y_val, clip=clip)
            candidates.append((name, out, clip, acc))
            print(f"[quantize] candidate {name:24s} val_acc={acc:.4f}")
        except Exception as e:
            print(f"[quantize] candidate {name:24s} FAILED: {e}")

    for c in CLIP_GRID:
        try_candidate(f"clip{c:g}-static-minmax", c, lambda out, c=c: quantize_static(
            str(model_in), str(out), make_reader(calib, clip=c),
            quant_format=QuantFormat.QDQ, per_channel=True,
            weight_type=QuantType.QInt8, activation_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.MinMax))

    # unclipped percentile kept as a reference candidate
    try_candidate("static-percentile", None, lambda out: quantize_static(
        str(model_in), str(out), make_reader(calib),
        quant_format=QuantFormat.QDQ, per_channel=True,
        weight_type=QuantType.QInt8, activation_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.Percentile))

    if not candidates:
        sys.exit("[quantize] all quantization candidates failed")

    name, path, clip, acc = max(candidates, key=lambda c: c[3])
    drop = fp32_acc - acc
    print(f"[quantize] BEST: {name} val_acc={acc:.4f} (fp32 {fp32_acc:.4f}, "
          f"drop {drop:.4f})")

    if drop > MAX_ACC_DROP:
        sys.exit(f"[quantize] REFUSING to ship: best int8 loses {drop:.3f} "
                 f"(> {MAX_ACC_DROP}). Investigate before demoing.")

    shutil.copyfile(path, INT8_ONNX)
    INT8_META.write_text(json.dumps({
        "method": name,
        "input_clip": clip,
        "val_acc_int8": round(acc, 4),
        "val_acc_fp32": round(fp32_acc, 4),
        "note": ("Scaled inputs are clipped to ±input_clip before the int8 model "
                 "(winsorized quantization). The inference service and benchmarks "
                 "apply the identical clip — deployed path == evaluated path."),
    }, indent=2))

    kb = lambda p: p.stat().st_size / 1024
    print(f"[quantize] float32: {kb(FP32_ONNX):8.1f} KB  ({FP32_ONNX.name})")
    print(f"[quantize] int8:    {kb(INT8_ONNX):8.1f} KB  ({INT8_ONNX.name}, {name})")
    print(f"[quantize] compression: {kb(FP32_ONNX)/kb(INT8_ONNX):.2f}x")
    print(f"[quantize] meta -> {INT8_META}")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(INT8_ONNX), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"features": np.zeros((1, 46), dtype=np.float32)})
    assert out[0].shape == (1, 6), "quantized model output shape wrong"
    print("[quantize] int8 model sanity check OK")


if __name__ == "__main__":
    main()
