"""
Download CICIoT2023 and build a stratified training subset.

The full dataset is ~13 GB across 169 CSV part-files, published by the
Canadian Institute for Cybersecurity (UNB):
    https://www.unb.ca/cic/datasets/iotdataset-2023.html
It is mirrored on Kaggle ("CICIoT2023"). Because UNB requires a form and
mirrors change, this script supports two modes:

  1) --from-dir <folder>   you already downloaded/unzipped the CSVs anywhere
                           on disk; the script scans, maps labels, and builds
                           the stratified subset (recommended).
  2) --url <direct-url>    a direct link to a zip of CSV part-files, if you
                           have one.

Output: dataset/subset/subset.parquet  (a few hundred k rows, stratified
across the 6 EdgeShield classes) which preprocess.py picks up automatically.

Usage:
  python ml-engine/scripts/download_dataset.py --from-dir "D:/CICIoT2023/csv"
  python ml-engine/scripts/download_dataset.py --from-dir ... --rows-per-class 60000
"""

import argparse
import sys
import zipfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.schema import (DATASET_DIR, FEATURE_COLUMNS, GLOBAL_SEED,
                           LABEL_COLUMN, RAW_LABEL_MAP, SUBSET_DIR)

RAW_DIR = DATASET_DIR / "raw"


def download(url: str) -> Path:
    import requests
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / url.split("/")[-1]
    if dest.exists():
        print(f"[download] {dest.name} already present, skipping")
        return dest
    print(f"[download] {url} -> {dest}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    if dest.suffix == ".zip":
        print("[download] extracting...")
        with zipfile.ZipFile(dest) as z:
            z.extractall(RAW_DIR)
    return RAW_DIR


def build_subset(csv_dir: Path, rows_per_class: int):
    csvs = sorted(csv_dir.rglob("*.csv"))
    if not csvs:
        sys.exit(f"[subset] no CSV files found under {csv_dir}")
    print(f"[subset] scanning {len(csvs)} CSV files")

    per_class_frames: dict[str, list[pd.DataFrame]] = {}
    per_class_rows: dict[str, int] = {}

    for path in tqdm(csvs, desc="CSV files"):
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as e:  # corrupt part-file: skip, don't die
            print(f"[subset] WARN: skipping {path.name}: {e}")
            continue
        if LABEL_COLUMN not in df.columns:
            # some dumps use 'Label'
            cand = [c for c in df.columns if c.lower() == "label"]
            if not cand:
                continue
            df = df.rename(columns={cand[0]: LABEL_COLUMN})
        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing:
            print(f"[subset] WARN: {path.name} missing {len(missing)} schema cols, skipping")
            continue
        df[LABEL_COLUMN] = df[LABEL_COLUMN].map(RAW_LABEL_MAP)
        df = df.dropna(subset=[LABEL_COLUMN])
        for cls, grp in df.groupby(LABEL_COLUMN):
            have = per_class_rows.get(cls, 0)
            need = rows_per_class - have
            if need <= 0:
                continue
            take = grp[FEATURE_COLUMNS + [LABEL_COLUMN]].head(need)
            per_class_frames.setdefault(cls, []).append(take)
            per_class_rows[cls] = have + len(take)
        if per_class_rows and all(v >= rows_per_class for v in per_class_rows.values()) \
                and len(per_class_rows) >= 6:
            print("[subset] all classes filled, stopping early")
            break

    frames = [pd.concat(v, ignore_index=True) for v in per_class_frames.values()]
    subset = pd.concat(frames, ignore_index=True)
    subset = subset.sample(frac=1.0, random_state=GLOBAL_SEED).reset_index(drop=True)

    SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    out = SUBSET_DIR / "subset.parquet"
    subset.to_parquet(out, index=False)
    print(f"[subset] wrote {len(subset):,} rows -> {out}")
    print(subset[LABEL_COLUMN].value_counts().to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-dir", type=Path, help="folder containing the CICIoT2023 CSVs")
    ap.add_argument("--url", help="direct URL to a zip of the CSVs (optional)")
    ap.add_argument("--rows-per-class", type=int, default=50000)
    args = ap.parse_args()

    if args.url:
        build_subset(download(args.url), args.rows_per_class)
    elif args.from_dir:
        build_subset(args.from_dir, args.rows_per_class)
    else:
        print(__doc__)
        sys.exit("Provide --from-dir or --url. (Get the dataset from "
                 "https://www.unb.ca/cic/datasets/iotdataset-2023.html or Kaggle.)")


if __name__ == "__main__":
    main()
