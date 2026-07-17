"""
Precompute per-class SHAP attribution profiles from the REAL trained model.

For each class: take up to 150 correctly-classified test flows of that class,
estimate SHAP values via Expected Gradients (the shap.GradientExplainer
estimator, implemented in pure PyTorch — see ml-engine/explain_utils.py for
why), average |attribution| per feature, keep the top 8. The /predict endpoint
attaches the result to every detection (fast path); per-instance attribution
stays available live via /explain.

Output: ml-inference-service/explain/shap_summary.json
Usage:  python ml-inference-service/explain/precompute_shap.py
"""

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "ml-engine"))
from common.schema import (BEST_MODEL_PT, CLASSES, FEATURE_COLUMNS,  # noqa: E402
                           GLOBAL_SEED, PROCESSED_DIR, SHAP_SUMMARY)


def main():
    import torch
    from explain_utils import expected_gradients
    from model import EdgeShieldNet

    rng = np.random.default_rng(GLOBAL_SEED)
    net = EdgeShieldNet()
    net.load_state_dict(torch.load(BEST_MODEL_PT, map_location="cpu")["state_dict"])
    net.eval()

    tr = np.load(PROCESSED_DIR / "train.npz")["X"]
    te = np.load(PROCESSED_DIR / "test.npz")
    X_te, y_te = te["X"], te["y"]

    background = torch.from_numpy(tr[rng.choice(len(tr), 200, replace=False)])

    with torch.no_grad():
        preds = net(torch.from_numpy(X_te)).argmax(1).numpy()

    out = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "method": ("Expected Gradients (SHAP GradientExplainer estimator, "
                      "pure PyTorch), 200-flow background, "
                      "correctly-classified test flows"),
           "per_class": {}}

    t0 = time.time()
    for ci, cls in enumerate(CLASSES):
        idx = np.where((y_te == ci) & (preds == ci))[0]
        if len(idx) == 0:
            print(f"[shap] WARN: no correct predictions for {cls}")
            continue
        take = idx[rng.choice(len(idx), min(150, len(idx)), replace=False)]
        acc_abs = np.zeros(len(FEATURE_COLUMNS))
        acc_signed = np.zeros(len(FEATURE_COLUMNS))
        for j, i in enumerate(take):
            x = torch.from_numpy(X_te[i])
            attr = expected_gradients(net, x, background, ci, k=32,
                                      seed=GLOBAL_SEED + int(i)).numpy()
            acc_abs += np.abs(attr)
            acc_signed += attr
        mean_abs = acc_abs / len(take)
        mean_signed = acc_signed / len(take)
        order = np.argsort(-mean_abs)[:8]
        out["per_class"][cls] = {
            "n_flows": int(len(take)),
            "top_features": [
                {"feature": FEATURE_COLUMNS[i],
                 "impact": round(float(mean_abs[i]), 5),
                 "direction": "+" if mean_signed[i] >= 0 else "-"}
                for i in order
            ],
        }
        print(f"[shap] {cls}: top = "
              f"{[f['feature'] for f in out['per_class'][cls]['top_features'][:4]]}")

    SHAP_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SHAP_SUMMARY.write_text(json.dumps(out, indent=2))
    print(f"[shap] wrote {SHAP_SUMMARY} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
