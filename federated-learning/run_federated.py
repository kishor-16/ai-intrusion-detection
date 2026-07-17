"""
Federated training simulation — 5 non-IID clients (the device profiles),
FedAvg aggregation, run OFFLINE before the demo. The dashboard replays
history.json; no live FL during judging (build brief §4.4).

Two engines, same algorithm and same partition:
  --engine builtin  (default) self-contained FedAvg in pure PyTorch. Zero extra
                    deps, deterministic, one process — the reliable path.
  --engine flower   Flower's simulation engine (needs requirements-optional.txt).
                    Kept as an option because Flower is the standard framework;
                    the builtin engine implements exactly FedAvg (McMahan et al.
                    2017), which is also Flower's default strategy.

Non-IID partition: each client predominantly holds its own device profile's
traffic (60% of its attack class + an equal slice of Normal), remainder spread
across the other clients — realistic heterogeneous edge data.

Outputs: federated-learning/history.json + models/fl_global_model.pt
Usage:   python federated-learning/run_federated.py [--rounds 10] [--local-epochs 2]
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "ml-engine"))
from common.schema import (CLASSES, FL_HISTORY, FL_MODEL_PT, GLOBAL_SEED,  # noqa: E402
                           METRICS_JSON, PROCESSED_DIR)
from model import EdgeShieldNet  # noqa: E402

CLIENT_PROFILES = [
    {"id": "client-camera",  "name": "Smart Camera edge node",    "attack": "DDoS"},
    {"id": "client-sensor",  "name": "Industrial Sensor edge node", "attack": "Spoofing"},
    {"id": "client-router",  "name": "Smart Router edge node",    "attack": "Scanning"},
    {"id": "client-medical", "name": "Medical IoT edge node",     "attack": "Brute Force"},
    {"id": "client-home",    "name": "Smart Home edge node",      "attack": "Malware"},
]


def load(name):
    d = np.load(PROCESSED_DIR / f"{name}.npz")
    return d["X"], d["y"]


def emit(args, obj):
    """--stream mode: machine-readable per-round events on stdout so the
    backend can broadcast REAL training progress live to the dashboard."""
    if getattr(args, "stream", False):
        print("@@EVENT " + json.dumps(obj), flush=True)


def partition_non_iid(X, y, rng):
    """Deterministic non-IID split: 60% of each attack class to its profile
    client, 40% spread over the rest; Normal split evenly."""
    n_clients = len(CLIENT_PROFILES)
    shards = {i: [] for i in range(n_clients)}
    for ci, cls in enumerate(CLASSES):
        idx = np.where(y == ci)[0]
        rng.shuffle(idx)
        if cls == "Normal":
            for i, part in enumerate(np.array_split(idx, n_clients)):
                shards[i].append(part)
        else:
            owner = next(i for i, p in enumerate(CLIENT_PROFILES) if p["attack"] == cls)
            cut = int(0.6 * len(idx))
            shards[owner].append(idx[:cut])
            rest = [i for i in range(n_clients) if i != owner]
            for i, part in zip(rest, np.array_split(idx[cut:], len(rest))):
                shards[i].append(part)
    return {i: np.concatenate(parts) for i, parts in shards.items()}


def local_train(model, X, y, epochs, lr, batch, seed):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = torch.nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True, generator=g)
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad(); lossf(model(xb), yb).backward(); opt.step()
    return model.state_dict()


def fedavg(states, weights):
    total = sum(weights)
    out = copy.deepcopy(states[0])
    for k in out:
        if out[k].dtype.is_floating_point:
            out[k] = sum(s[k] * (w / total) for s, w in zip(states, weights))
        else:  # e.g. batchnorm num_batches_tracked (int64)
            out[k] = states[0][k]
    return out


def evaluate(model, X, y):
    from sklearn.metrics import accuracy_score, f1_score
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X)).argmax(1).numpy()
    return (round(float(accuracy_score(y, pred)), 4),
            round(float(f1_score(y, pred, average="macro")), 4))


def run_builtin(args):
    torch.manual_seed(GLOBAL_SEED)
    rng = np.random.default_rng(GLOBAL_SEED)
    X_tr, y_tr = load("train"); X_te, y_te = load("test")
    shards = partition_non_iid(X_tr, y_tr, rng)

    clients_meta = []
    for i, prof in enumerate(CLIENT_PROFILES):
        ys = y_tr[shards[i]]
        dist = {CLASSES[c]: int((ys == c).sum()) for c in range(len(CLASSES))}
        clients_meta.append({**prof, "samples": int(len(ys)), "class_distribution": dist})
        print(f"[fl] {prof['id']}: {len(ys):,} samples, dist={dist}")

    global_model = EdgeShieldNet()
    rounds = []
    acc0, f10 = evaluate(global_model, X_te, y_te)
    rounds.append({"round": 0, "accuracy": acc0, "f1_macro": f10,
                   "note": "untrained global model"})
    emit(args, {"type": "start", "engine": "builtin-fedavg", "rounds": args.rounds,
                "clients": [{"id": p["id"], "name": p["name"], "attack": p["attack"],
                             "samples": int(len(shards[i]))}
                            for i, p in enumerate(CLIENT_PROFILES)]})
    emit(args, {"type": "round", "round": 0, "accuracy": acc0, "f1_macro": f10,
                "total_rounds": args.rounds})
    t0 = time.time()
    for r in range(1, args.rounds + 1):
        states, weights = [], []
        for i in range(len(CLIENT_PROFILES)):
            emit(args, {"type": "client", "round": r, "client": CLIENT_PROFILES[i]["id"],
                        "status": "training"})
            local = copy.deepcopy(global_model)
            sd = local_train(local, X_tr[shards[i]], y_tr[shards[i]],
                             args.local_epochs, args.lr, args.batch,
                             seed=GLOBAL_SEED * 1000 + r * 10 + i)
            states.append(sd); weights.append(len(shards[i]))
        global_model.load_state_dict(fedavg(states, weights))
        acc, f1 = evaluate(global_model, X_te, y_te)
        rounds.append({"round": r, "accuracy": acc, "f1_macro": f1})
        emit(args, {"type": "round", "round": r, "accuracy": acc, "f1_macro": f1,
                    "total_rounds": args.rounds,
                    "elapsed_sec": round(time.time() - t0, 1)})
        print(f"[fl] round {r:02d}/{args.rounds}  test_acc={acc:.4f}  f1={f1:.4f}")

    torch.save({"state_dict": global_model.state_dict(), "arch": "EdgeShieldNet"},
               FL_MODEL_PT)
    return rounds, clients_meta, time.time() - t0, "builtin-fedavg"


def run_flower(args):
    """Flower simulation engine (optional; pip install -r requirements-optional.txt)."""
    import flwr as fl

    torch.manual_seed(GLOBAL_SEED)
    rng = np.random.default_rng(GLOBAL_SEED)
    X_tr, y_tr = load("train"); X_te, y_te = load("test")
    shards = partition_non_iid(X_tr, y_tr, rng)
    clients_meta = [{**p, "samples": int(len(shards[i]))}
                    for i, p in enumerate(CLIENT_PROFILES)]
    rounds = []

    def get_params(model):
        return [v.cpu().numpy() for v in model.state_dict().values()]

    def set_params(model, params):
        sd = model.state_dict()
        for k, v in zip(sd.keys(), params):
            sd[k] = torch.tensor(v)
        model.load_state_dict(sd)

    class Client(fl.client.NumPyClient):
        def __init__(self, cid):
            self.cid = int(cid)
            self.model = EdgeShieldNet()

        def fit(self, parameters, config):
            set_params(self.model, parameters)
            local_train(self.model, X_tr[shards[self.cid]], y_tr[shards[self.cid]],
                        args.local_epochs, args.lr, args.batch,
                        seed=GLOBAL_SEED + self.cid)
            return get_params(self.model), len(shards[self.cid]), {}

    def evaluate_fn(server_round, parameters, config):
        m = EdgeShieldNet(); set_params(m, parameters)
        acc, f1 = evaluate(m, X_te, y_te)
        rounds.append({"round": server_round, "accuracy": acc, "f1_macro": f1})
        if server_round == args.rounds:
            torch.save({"state_dict": m.state_dict(), "arch": "EdgeShieldNet"}, FL_MODEL_PT)
        return 0.0, {"accuracy": acc}

    t0 = time.time()
    fl.simulation.start_simulation(
        client_fn=lambda cid: Client(cid).to_client(),
        num_clients=len(CLIENT_PROFILES),
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=fl.server.strategy.FedAvg(
            min_fit_clients=5, min_available_clients=5,
            fraction_fit=1.0, evaluate_fn=evaluate_fn),
    )
    return rounds, clients_meta, time.time() - t0, "flower-simulation"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["builtin", "flower"], default="builtin")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--local-epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--stream", action="store_true",
                    help="emit @@EVENT JSON lines per round (live dashboard mode)")
    args = ap.parse_args()

    if args.engine == "flower":
        try:
            rounds, clients, dur, engine = run_flower(args)
        except ImportError:
            sys.exit("[fl] flwr not installed — pip install -r requirements-optional.txt "
                     "or use --engine builtin")
    else:
        rounds, clients, dur, engine = run_builtin(args)

    central = None
    if METRICS_JSON.exists():
        central = json.loads(METRICS_JSON.read_text())["test"]["accuracy"]

    FL_HISTORY.write_text(json.dumps({
        "engine": engine,
        "algorithm": "FedAvg (McMahan et al. 2017)",
        "seed": GLOBAL_SEED,
        "num_clients": len(clients),
        "partition": "non-IID by device profile (60% primary attack class)",
        "local_epochs": args.local_epochs,
        "duration_sec": round(dur, 1),
        "privacy_note": "Only model weights are exchanged; raw flows never leave a client shard.",
        "centralized_reference_accuracy": central,
        "clients": clients,
        "rounds": rounds,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))
    emit(args, {"type": "done", "engine": engine, "final_accuracy": rounds[-1]["accuracy"],
                "duration_sec": round(dur, 1), "centralized_reference": central})
    print(f"[fl] engine={engine} final_acc={rounds[-1]['accuracy']} "
          f"(centralized ref={central}) -> {FL_HISTORY}")


if __name__ == "__main__":
    main()
