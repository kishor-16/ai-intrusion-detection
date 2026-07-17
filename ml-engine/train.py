"""
Train the EdgeShield edge model on the preprocessed data.

- GPU-ready: auto-detects CUDA, prints GPU name and training time.
- Deterministic under GLOBAL_SEED.
- Reports Accuracy / Precision / Recall / F1 / confusion matrix on the TEST set
  and saves plots to docs/. Every number in models/metrics.json is a real
  measurement from this script — the dashboard reads from there, never retypes.

Usage: python ml-engine/train.py [--epochs 60] [--batch 256]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.schema import (BEST_MODEL_PT, CLASSES, FP32_ONNX, GLOBAL_SEED,
                           METRICS_JSON, MODEL_DIR, PROCESSED_DIR,
                           PROVENANCE_PATH, REPO_ROOT)
from model import EdgeShieldNet, count_params, export_onnx  # noqa: E402  (ml-engine dir)

DOCS = REPO_ROOT / "docs"


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load(name):
    d = np.load(PROCESSED_DIR / f"{name}.npz")
    return torch.from_numpy(d["X"]), torch.from_numpy(d["y"])


def evaluate(model, X, y, device, batch=2048):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            logits = model(X[i:i + batch].to(device))
            preds.append(logits.argmax(1).cpu())
    return torch.cat(preds).numpy()


def metrics_report(y_true, y_pred):
    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                 precision_recall_fscore_support)
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    pc, rc, f1c, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(CLASSES)), zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    return {
        "accuracy": round(float(acc), 4),
        "precision_macro": round(float(p), 4),
        "recall_macro": round(float(r), 4),
        "f1_macro": round(float(f1), 4),
        "per_class": {
            CLASSES[i]: {"precision": round(float(pc[i]), 4),
                         "recall": round(float(rc[i]), 4),
                         "f1": round(float(f1c[i]), 4),
                         "support": int(sup[i])}
            for i in range(len(CLASSES))
        },
        "confusion_matrix": cm.tolist(),
    }


def save_plots(cm, history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    DOCS.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("EdgeShield — Test Confusion Matrix")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i][j], ha="center", va="center",
                    color="white" if cm[i][j] > cm.max() / 2 else "black", fontsize=8)
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(DOCS / "confusion_matrix.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history["train_loss"], label="train loss")
    ax.plot(history["val_f1"], label="val F1 (macro)")
    ax.set_xlabel("epoch"); ax.legend(); ax.set_title("EdgeShield — Training")
    fig.tight_layout(); fig.savefig(DOCS / "training_curves.png", dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--patience", type=int, default=15)
    args = ap.parse_args()

    set_seed(GLOBAL_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "none (CPU)"
    print(f"[train] device={device}  GPU={gpu}  CUDA available={torch.cuda.is_available()}")

    X_tr, y_tr = load("train"); X_val, y_val = load("val"); X_te, y_te = load("test")
    print(f"[train] train={len(y_tr):,} val={len(y_val):,} test={len(y_te):,}")

    model = EdgeShieldNet().to(device)
    print(f"[train] EdgeShieldNet params={count_params(model):,}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    # class-weighted loss (Brute Force has ~4x fewer samples) + label smoothing
    counts = np.bincount(y_tr.numpy(), minlength=len(CLASSES)).astype(np.float64)
    weights = counts.sum() / (len(CLASSES) * counts)
    print(f"[train] class weights: "
          f"{ {CLASSES[i]: round(float(w),2) for i, w in enumerate(weights)} }")
    lossf = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32).to(device),
        label_smoothing=0.05)
    ds = torch.utils.data.TensorDataset(X_tr, y_tr)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=True,
        generator=torch.Generator().manual_seed(GLOBAL_SEED))

    from sklearn.metrics import f1_score
    best_f1, best_state, bad, history = -1.0, None, 0, {"train_loss": [], "val_f1": []}
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        tot = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward(); opt.step()
            tot += loss.item() * len(yb)
        sched.step()
        train_loss = tot / len(y_tr)
        val_f1 = f1_score(y_val.numpy(), evaluate(model, X_val, y_val, device),
                          average="macro")
        history["train_loss"].append(round(train_loss, 4))
        history["val_f1"].append(round(float(val_f1), 4))
        print(f"[train] epoch {epoch+1:02d}  loss={train_loss:.4f}  val_f1={val_f1:.4f}")
        if val_f1 > best_f1 + 1e-4:
            best_f1, bad = val_f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[train] early stop at epoch {epoch+1}")
                break
    train_time = time.time() - t0

    model.load_state_dict(best_state)
    y_pred = evaluate(model, X_te, y_te, device)
    rep = metrics_report(y_te.numpy(), y_pred)
    save_plots(np.array(rep["confusion_matrix"]), history)

    MODEL_DIR.mkdir(exist_ok=True)
    torch.save({"state_dict": best_state, "arch": "EdgeShieldNet"}, BEST_MODEL_PT)
    export_onnx(model.cpu(), FP32_ONNX)

    provenance = json.loads(PROVENANCE_PATH.read_text()) if PROVENANCE_PATH.exists() else {}
    out = {
        "model": "EdgeShieldNet",
        "params": count_params(model),
        "test": rep,
        "training": {
            "device": str(device), "gpu": gpu,
            "train_time_sec": round(train_time, 1),
            "epochs_run": len(history["train_loss"]),
            "seed": GLOBAL_SEED,
            "best_val_f1": round(float(best_f1), 4),
        },
        "data_source": provenance.get("data_source", "unknown"),
        "history": history,
    }
    METRICS_JSON.write_text(json.dumps(out, indent=2))
    print(f"[train] TEST acc={rep['accuracy']}  f1_macro={rep['f1_macro']}  "
          f"time={train_time:.1f}s")
    print(f"[train] saved {BEST_MODEL_PT.name}, {FP32_ONNX.name}, metrics.json, plots -> docs/")


if __name__ == "__main__":
    main()
