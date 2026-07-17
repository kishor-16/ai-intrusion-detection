"""
One-command offline preparation: builds every artifact the demo needs, in order.

  python scripts/prepare_all.py            # uses real subset if present, else bundled sample
  python scripts/prepare_all.py --sample   # force the bundled sample (no downloads ever)
  python scripts/prepare_all.py --fast     # smaller epochs/rounds (smoke-test in minutes)

Steps: sample data (if needed) -> preprocess -> train -> quantize -> SHAP
precompute -> federated run -> baseline benchmarks. Each step prints real
measurements; nothing later in the UI exists unless produced here.
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(desc, args):
    print(f"\n{'='*70}\n[prepare] {desc}\n{'='*70}")
    r = subprocess.run([PY, *args], cwd=REPO)
    if r.returncode != 0:
        sys.exit(f"[prepare] FAILED at: {desc}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()

    sample_csv = REPO / "dataset" / "sample" / "sample.csv"
    subset = REPO / "dataset" / "subset" / "subset.parquet"
    if not sample_csv.exists() and not subset.exists():
        run("generate bundled sample dataset", ["ml-engine/scripts/make_sample.py"])

    pre = ["ml-engine/preprocess.py"] + (["--sample"] if args.sample else [])
    run("preprocess (fit scaler, splits)", pre)

    train = ["ml-engine/train.py"] + (["--epochs", "12"] if args.fast else [])
    run("train EdgeShield model (real metrics)", train)

    run("quantize to int8 ONNX", ["ml-engine/quantize.py"])
    run("precompute SHAP class profiles", ["ml-inference-service/explain/precompute_shap.py"])

    fed = ["federated-learning/run_federated.py"] + (["--rounds", "5"] if args.fast else [])
    run("federated FedAvg simulation (5 non-IID clients)", fed)

    bench = ["benchmarks/compare_baseline.py", "--retrain-baseline"] + \
        (["--baseline-epochs", "8", "--runs", "150"] if args.fast else [])
    run("baseline vs EdgeShield benchmarks", bench)

    print("\n[prepare] ALL ARTIFACTS READY. Start the demo:  python scripts/run_all.py")


if __name__ == "__main__":
    main()
