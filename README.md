# AI-BASED-INTRUTION-DETECTION-SYSTEM
EdgeShield AI is an IEEE-inspired Edge AI Intrusion Detection System for IoT networks, enhanced with Federated Learning, Explainable AI (SHAP), TensorFlow Lite optimization, and a real-time cybersecurity dashboard. Features realistic IoT attack simulation, privacy-preserving learning, and edge deployment for intelligent threat detection.

# EdgeShield AI

**Privacy-Preserving, Explainable, Edge-AI Intrusion Detection for IoT Networks using Federated Learning.**

An enhancement of existing IEEE ML-based IoT-IDS approaches: instead of a heavy, centralized,
black-box classifier, EdgeShield runs a **quantized int8 edge model**, trains it **federatedly
(FedAvg, raw data never leaves a device)**, explains every detection with **SHAP**, and streams
everything into a real-time SOC dashboard. Fully software — all IoT devices are **clearly-labeled
simulations**; every detection is a **real inference** of the trained model.

## Honesty contract (read this first, judges)

- **No fabricated metrics.** Every number in the UI (accuracy, latency, sizes, FL curve,
  confidence) traces to `ml-engine/train.py`, `benchmarks/`, `federated-learning/history.json`,
  or the live `/predict` response. UI panels cite their source; unmeasured values show "—".
- **Simulated vs real.** IoT *traffic* is simulated (dataset rows + ±3% jitter — labeled
  "SIMULATED" in the top bar). Model *detections* are real inferences on those schema-matched
  vectors — not scripted popups. "Live AI accuracy" = predictions vs simulator ground truth,
  measured during the run.
- **Offline + deterministic.** One global seed (42), no network calls at demo time, one-click
  Reset replays the identical scenario.

## Architecture

```
[iot-simulator :8200] --flows--> [backend :4000 (Node+Express+Socket.IO)] --ws--> [frontend :5173]
                                       |         ^ scenario engine (deterministic)
                                       v
                       [ml-inference-service :8100 (FastAPI)]
                        int8 ONNX model + SHAP  (owns all ML)
```

Details: `docs/architecture.md` · ML route rationale: `docs/implementation.md` ·
baseline gap: `docs/research_gap.md` · measured results: `docs/results.md` (generated).

## Quick start

```bash
# 1) Python env (3.10–3.12). For NVIDIA GPU training install CUDA torch first:
#    pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 2) Node deps
cd backend  && npm install && cd ..
cd frontend && npm install && cd ..

# 3) Build every artifact offline (sample data, train, quantize, SHAP, FL, benchmarks)
python scripts/prepare_all.py            # add --fast for a quick smoke run

# 4) One-command bring-up (all four services)
python scripts/run_all.py                # or scripts\run.bat / scripts/run.sh
```

Open **http://localhost:5173** → press **⚡ Launch Cyber Attack Simulation**.

## Real dataset (CICIoT2023, optional but recommended before the event)

The pipeline runs out-of-the-box on a bundled synthetic sample (clearly labeled in the UI).
For real numbers, get CICIoT2023 (~13 GB). Easiest — straight from Kaggle with your API key
(kaggle.com → Settings → API → Create New Token → put `kaggle.json` in `~/.kaggle/`):

```bash
python ml-engine/scripts/download_kaggle.py   # download + extract + build subset
python scripts/prepare_all.py                 # now uses the real subset automatically
```

Or manually from [UNB CIC](https://www.unb.ca/cic/datasets/iotdataset-2023.html) /
[Kaggle](https://www.kaggle.com/datasets/akashdogra/cic-iot-2023), then:

```bash
python ml-engine/scripts/download_dataset.py --from-dir "D:/path/to/CICIoT2023/csv"
python scripts/prepare_all.py
```

`preprocess.py` records which source was used (`common/artifacts/provenance.json`)
and the dashboard badge switches from "SAMPLE DATA" to "CICIoT2023".

## GPU training

`train.py` auto-detects CUDA and prints the GPU name and measured training time.
Verify with: `python -c "import torch; print(torch.cuda.get_device_name(0))"`.

## The demo (golden path)

1. `python scripts/run_all.py` → dashboard loads with the fleet online.
2. **⚡ Launch Cyber Attack Simulation** — a seeded, scripted timeline: warm-up normal
   traffic, then 5 attack waves (DDoS → Brute Force → Scanning → Spoofing → Malware),
   one per device. The int8 model classifies **every flow live**; threats light up the
   feed and the map; SHAP explains any detection; the FL page replays the real
   federated run; Edge Performance shows measured baseline-vs-edge numbers.
3. **⟲ Reset** — identical replay, every time.

## Repository map

| Path | What it is |
|---|---|
| `common/schema.py` | THE feature-schema contract (46 CICIoT2023 features) both sides import |
| `ml-engine/` | sample/download, preprocess, train, quantize |
| `ml-inference-service/` | FastAPI: `/predict`, `/explain`, `/model-info` (owns model + SHAP) |
| `iot-simulator/` | 5 virtual devices emitting schema-matched flows |
| `backend/` | Node relay + deterministic scenario engine (no ML) |
| `frontend/` | React+Vite SOC dashboard (renders only received data) |
| `federated-learning/` | FedAvg simulation (5 non-IID clients) → `history.json` |
| `benchmarks/` | measured size/latency/accuracy, baseline comparison |
| `scripts/` | `prepare_all.py` (build artifacts) · `run_all.py` (bring-up) |
