"""
Download CICIoT2023 from Kaggle using your Kaggle API key, then build the
stratified training subset — one command, no kaggle CLI needed (pure requests).

API key: Kaggle -> your avatar -> Settings -> API -> "Create New Token"
downloads kaggle.json ({"username": "...", "key": "..."}). Put it in ONE of:
  1. %USERPROFILE%\\.kaggle\\kaggle.json   (standard location, recommended)
  2. the repo root (git-ignored)
  3. env vars KAGGLE_USERNAME + KAGGLE_KEY
  4. flags: --username you --key xxxxxxxx

Usage:
  python ml-engine/scripts/download_kaggle.py                 # download + extract + subset
  python ml-engine/scripts/download_kaggle.py --no-subset     # download + extract only
  python ml-engine/scripts/download_kaggle.py --rows-per-class 60000

The zip is ~2-3 GB (13 GB extracted) — make sure the drive has ~20 GB free.
Interrupted downloads resume automatically where supported.
"""

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from common.schema import DATASET_DIR  # noqa: E402

DEFAULT_DATASET = "akashdogra/cic-iot-2023"   # CSV mirror of CICIoT2023
RAW_DIR = DATASET_DIR / "raw" / "kaggle"
API = "https://www.kaggle.com/api/v1/datasets/download/{}"


def find_credentials(args):
    if args.username and args.key:
        return args.username, args.key
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"]
    for p in (REPO / "kaggle.json", Path.home() / ".kaggle" / "kaggle.json"):
        if p.exists():
            try:
                d = json.loads(p.read_text())
                print(f"[kaggle] using credentials from {p}")
                return d["username"], d["key"]
            except (json.JSONDecodeError, KeyError):
                sys.exit(f"[kaggle] {p} exists but is not a valid kaggle.json")
    sys.exit("[kaggle] no API key found.\n"
             "  Get one: kaggle.com -> Settings -> API -> Create New Token,\n"
             "  then place kaggle.json in ~/.kaggle/ (or repo root), or pass\n"
             "  --username/--key, or set KAGGLE_USERNAME/KAGGLE_KEY.")


def download(dataset: str, username: str, key: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / (dataset.replace("/", "_") + ".zip")
    url = API.format(dataset)

    headers = {}
    mode = "wb"
    done = dest.stat().st_size if dest.exists() else 0
    if done:
        headers["Range"] = f"bytes={done}-"
        mode = "ab"
        print(f"[kaggle] resuming at {done/1e6:.0f} MB")

    with requests.get(url, auth=(username, key), headers=headers,
                      stream=True, timeout=120, allow_redirects=True) as r:
        if r.status_code == 401:
            sys.exit("[kaggle] 401 Unauthorized — check your username/key.")
        if r.status_code == 403:
            sys.exit("[kaggle] 403 Forbidden — open the dataset page in a browser "
                     "once and accept its terms, then retry:\n"
                     f"  https://www.kaggle.com/datasets/{dataset}")
        if r.status_code == 404:
            sys.exit(f"[kaggle] 404 — dataset '{dataset}' not found.")
        if done and r.status_code == 200:
            print("[kaggle] server ignored resume; restarting from zero")
            mode, done = "wb", 0
        elif r.status_code not in (200, 206):
            sys.exit(f"[kaggle] HTTP {r.status_code}: {r.text[:300]}")

        total = int(r.headers.get("content-length", 0)) + done
        with open(dest, mode) as f, tqdm(total=total, initial=done, unit="B",
                                         unit_scale=True, desc="download") as bar:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))

    print(f"[kaggle] saved {dest} ({dest.stat().st_size/1e9:.2f} GB)")
    return dest


def extract(zip_path: Path) -> Path:
    out = RAW_DIR / "extracted"
    out.mkdir(exist_ok=True)
    print(f"[kaggle] extracting -> {out} (this can take a while)")
    with zipfile.ZipFile(zip_path) as z:
        members = z.namelist()
        for m in tqdm(members, desc="extract"):
            target = out / m
            if not (target.exists() and target.stat().st_size == z.getinfo(m).file_size):
                z.extract(m, out)
    n_csv = len(list(out.rglob("*.csv")))
    print(f"[kaggle] extracted: {n_csv} CSV files")
    if n_csv == 0:
        sys.exit("[kaggle] no CSVs in the archive — wrong dataset slug?")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DATASET,
                    help=f"Kaggle dataset slug (default: {DEFAULT_DATASET})")
    ap.add_argument("--username")
    ap.add_argument("--key")
    ap.add_argument("--rows-per-class", type=int, default=50000)
    ap.add_argument("--no-subset", action="store_true",
                    help="download+extract only, skip subset building")
    ap.add_argument("--keep-zip", action="store_true",
                    help="keep the .zip after successful extraction")
    args = ap.parse_args()

    username, key = find_credentials(args)
    zip_path = download(args.dataset, username, key)
    csv_dir = extract(zip_path)

    if not args.keep_zip:
        zip_path.unlink()
        print("[kaggle] removed zip to save disk (use --keep-zip to keep it)")

    if args.no_subset:
        print(f"[kaggle] done. Build the subset later with:\n"
              f"  python ml-engine/scripts/download_dataset.py --from-dir \"{csv_dir}\"")
        return

    from download_dataset import build_subset  # same directory
    build_subset(csv_dir, args.rows_per_class)
    print("\n[kaggle] READY — now run:  python scripts/prepare_all.py")


if __name__ == "__main__":
    main()
