"""
Preprocess the dataset into model-ready arrays + persist the shared scaler/schema.

Data source resolution (automatic):
  1. dataset/subset/subset.parquet   (real CICIoT2023 subset, via download_dataset.py)
  2. dataset/sample/sample.csv       (bundled synthetic sample; --sample forces this)

Outputs:
  dataset/processed/{train,val,test}.npz     (X float32 SCALED, y int64)
  common/artifacts/scaler.pkl                (fitted StandardScaler — the ONE scaler
                                              used by training, benchmarks, and the
                                              inference service)
  common/artifacts/label_encoder.json        (class name <-> index)
  common/artifacts/provenance.json           (which data source was used — surfaced
                                              honestly in the UI)

Usage: python ml-engine/preprocess.py [--sample]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.schema import (ARTIFACT_DIR, CLASSES, FEATURE_COLUMNS, GLOBAL_SEED,
                           INDICATOR_COLUMNS, LABEL_COLUMN, LABEL_ENCODER_PATH,
                           PROCESSED_DIR, PROVENANCE_PATH, SAMPLE_CSV,
                           SCALER_PATH, SUBSET_DIR, TRANSFORM_PATH)

# Post-scale winsorization: bound z-scores so training, quantization and
# serving all see the same bounded range (heavy-tailed traffic features).
POST_SCALE_CLIP = 8.0


def load_source(force_sample: bool):
    subset = SUBSET_DIR / "subset.parquet"
    if not force_sample and subset.exists():
        print(f"[preprocess] using REAL CICIoT2023 subset: {subset}")
        return pd.read_parquet(subset), "ciciot2023-subset"
    if not SAMPLE_CSV.exists():
        sys.exit("[preprocess] no data found. Run ml-engine/scripts/make_sample.py "
                 "(bundled sample) or download_dataset.py (real data) first.")
    print(f"[preprocess] using bundled SYNTHETIC sample: {SAMPLE_CSV}")
    print("[preprocess] NOTE: metrics will be real measurements on synthetic data; "
          "the UI will carry a 'sample data' provenance badge.")
    return pd.read_csv(SAMPLE_CSV), "synthetic-sample"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="force the bundled sample")
    args = ap.parse_args()

    t0 = time.time()
    df, provenance = load_source(args.sample)

    # --- schema check (fail loudly, this is the demo-killer) ---
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        sys.exit(f"[preprocess] FATAL: data is missing schema columns: {missing}")
    df = df[FEATURE_COLUMNS + [LABEL_COLUMN]]

    # --- cleaning ---
    n0 = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[df[LABEL_COLUMN].isin(CLASSES)]
    print(f"[preprocess] cleaned: {n0:,} -> {len(df):,} rows")

    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y = np.array([CLASSES.index(v) for v in df[LABEL_COLUMN]], dtype=np.int64)

    # --- signed log1p on heavy-tailed continuous features ---------------------
    # Traffic features span many orders of magnitude; log-compression is the
    # standard fix for MLPs AND makes int8 quantization far friendlier.
    # sign(x)*log1p(|x|) is safe for any value. The exact column list is
    # persisted to transform.json — the inference service applies the same
    # transform to raw flows before the scaler (single source of truth).
    log_cols = [c for c in FEATURE_COLUMNS if c not in INDICATOR_COLUMNS]
    log_idx = [FEATURE_COLUMNS.index(c) for c in log_cols]
    X[:, log_idx] = np.sign(X[:, log_idx]) * np.log1p(np.abs(X[:, log_idx]))

    # --- stratified 70/15/15 split ---
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=GLOBAL_SEED)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=GLOBAL_SEED)

    # --- fit scaler on TRAIN ONLY, persist it for every downstream consumer ---
    scaler = StandardScaler().fit(X_tr)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, (Xs, ys) in {
        "train": (X_tr, y_tr), "val": (X_val, y_val), "test": (X_te, y_te),
    }.items():
        Z = np.clip(scaler.transform(Xs), -POST_SCALE_CLIP, POST_SCALE_CLIP)
        np.savez_compressed(PROCESSED_DIR / f"{name}.npz",
                            X=Z.astype(np.float32), y=ys)
        print(f"[preprocess] {name}: {len(ys):,} rows")

    TRANSFORM_PATH.write_text(json.dumps({
        "signed_log1p_columns": log_cols,
        "post_scale_clip": POST_SCALE_CLIP,
        "note": "Applied by preprocess (training) AND the inference service "
                "(serving) — raw flow -> signed log1p -> scaler -> clip.",
    }, indent=2))

    LABEL_ENCODER_PATH.write_text(json.dumps(
        {"classes": CLASSES, "n_features": len(FEATURE_COLUMNS)}, indent=2))
    PROVENANCE_PATH.write_text(json.dumps({
        "data_source": provenance,
        "rows_after_cleaning": int(len(df)),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": GLOBAL_SEED,
        "note": ("SYNTHETIC sample used — pipeline bring-up mode. All metrics are real "
                 "measurements on this data. Swap in the real CICIoT2023 subset via "
                 "download_dataset.py for publication-grade numbers.")
                if provenance == "synthetic-sample" else
                "Real CICIoT2023 stratified subset.",
    }, indent=2))

    print(f"[preprocess] scaler -> {SCALER_PATH}")
    print(f"[preprocess] provenance = {provenance}")
    print(f"[preprocess] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
