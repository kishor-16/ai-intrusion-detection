"""
End-to-end smoke test. Run it twice:

  python scripts/smoke_test.py            # artifact check (services not needed)
  python scripts/smoke_test.py --live     # with services running: full round-trip
                                          # simulator -> /predict, schema + keys asserted

Exit code 0 = green. Use this as the phase checkpoint verifier.
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

PASS, FAIL = "  [OK]", "  [!!]"
failures = []


def check(label, cond, hint=""):
    print(f"{PASS if cond else FAIL} {label}" + ("" if cond else f"   -> {hint}"))
    if not cond:
        failures.append(label)


def artifacts():
    print("\n-- schema contract --")
    from common.schema import CLASSES, FEATURE_COLUMNS
    check("schema has 46 features", len(FEATURE_COLUMNS) == 46)
    check("schema has 6 classes", len(CLASSES) == 6)
    sj = json.loads((REPO / "common" / "schema.json").read_text())
    check("schema.json mirrors schema.py",
          sj["feature_columns"] == FEATURE_COLUMNS and sj["classes"] == CLASSES,
          "regenerate schema.json from schema.py")

    print("\n-- artifacts --")
    for rel, hint in [
        ("dataset/sample/sample.csv", "python ml-engine/scripts/make_sample.py"),
        ("common/artifacts/scaler.pkl", "python ml-engine/preprocess.py"),
        ("common/artifacts/provenance.json", "python ml-engine/preprocess.py"),
        ("models/best_model.pt", "python ml-engine/train.py"),
        ("models/model_fp32.onnx", "python ml-engine/train.py"),
        ("models/metrics.json", "python ml-engine/train.py"),
        ("models/model_int8.onnx", "python ml-engine/quantize.py"),
        ("ml-inference-service/explain/shap_summary.json",
         "python ml-inference-service/explain/precompute_shap.py"),
        ("federated-learning/history.json", "python federated-learning/run_federated.py"),
        ("benchmarks/results.json", "python benchmarks/compare_baseline.py"),
        ("docs/results.md", "python benchmarks/compare_baseline.py"),
    ]:
        check(rel, (REPO / rel).exists(), hint)

    mp = REPO / "models" / "metrics.json"
    if mp.exists():
        m = json.loads(mp.read_text())
        acc = m.get("test", {}).get("accuracy")
        check(f"metrics.json test accuracy present ({acc})", isinstance(acc, float))


def live():
    import urllib.request

    def get(url):
        with urllib.request.urlopen(url, timeout=8) as r:
            return json.loads(r.read())

    def post(url, body):
        req = urllib.request.Request(url, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())

    print("\n-- live services --")
    check("inference /health", get("http://127.0.0.1:8100/health")["status"] == "ok")
    check("simulator /health", get("http://127.0.0.1:8200/health")["status"] == "ok")
    check("backend /health", get("http://127.0.0.1:4000/health")["status"] == "ok")

    print("\n-- golden path round-trip --")
    ev = post("http://127.0.0.1:8200/generate",
              {"device_id": "cam-01", "mode": "attack", "count": 1})["events"][0]
    check("simulator emits 46-feature vector", len(ev["features"]) == 46)
    check("simulator marks traffic simulated", ev["meta"]["simulated"] is True)
    p = post("http://127.0.0.1:8100/predict", {"features": ev["features"]})
    for k in ("label", "confidence", "probs", "latency_ms"):
        check(f"/predict returns {k}", k in p)
    check("latency_ms is a real measurement > 0", p["latency_ms"] > 0)
    print(f"       model verdict for simulated DDoS flow: "
          f"{p['label']} ({p['confidence']:.2%}, {p['latency_ms']} ms)")

    mi = get("http://127.0.0.1:8100/model-info")
    check("model-info reports provenance", "data_source" in mi)
    check("model-info size from disk", isinstance(mi["size_kb"], (int, float)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args()
    artifacts()
    if args.live:
        live()
    print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} FAILURES'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
