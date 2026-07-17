"""
One-command bring-up of all EdgeShield services (cross-platform).

  python scripts/run_all.py            # inference + simulator + backend + frontend
  python scripts/run_all.py --no-frontend

Preflight-checks artifacts and node_modules, starts the four processes,
streams their logs with prefixes, and shuts everything down on Ctrl+C.
The demo then runs FULLY OFFLINE at http://localhost:5173
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
NPM = "npm.cmd" if os.name == "nt" else "npm"

REQUIRED = [
    ("common/artifacts/scaler.pkl", "python ml-engine/preprocess.py"),
    ("models/model_fp32.onnx", "python ml-engine/train.py"),
]
RECOMMENDED = [
    ("models/model_int8.onnx", "python ml-engine/quantize.py"),
    ("ml-inference-service/explain/shap_summary.json",
     "python ml-inference-service/explain/precompute_shap.py"),
    ("federated-learning/history.json", "python federated-learning/run_federated.py"),
    ("benchmarks/results.json", "python benchmarks/compare_baseline.py"),
]


def ports_free(with_frontend: bool):
    """Fail fast if any service port is already taken (stale processes serve
    STALE MODELS — always kill them and restart)."""
    import socket
    ports = [8100, 8200, 4000] + ([5173] if with_frontend else [])
    busy = []
    for port in ports:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            busy.append(port)
        finally:
            s.close()
    if busy:
        print(f"[run] ports already in use: {busy} — stale services from a "
              "previous run are still alive (and serving old artifacts).")
        if os.name == "nt":
            print("[run] free them with (PowerShell):\n"
                  "  Get-NetTCPConnection -LocalPort " + ",".join(map(str, busy)) +
                  " -State Listen | ForEach-Object "
                  "{ Stop-Process -Id $_.OwningProcess -Force }")
        else:
            print(f"[run] free them with:  kill $(lsof -ti :{','.join(map(str, busy))})")
        sys.exit(1)


def preflight(with_frontend: bool):
    ports_free(with_frontend)
    ok = True
    for rel, fix in REQUIRED:
        if not (REPO / rel).exists():
            print(f"[run] MISSING {rel}  ->  {fix}")
            ok = False
    if not ok:
        sys.exit("[run] required artifacts missing — run: python scripts/prepare_all.py")
    for rel, fix in RECOMMENDED:
        if not (REPO / rel).exists():
            print(f"[run] note: {rel} missing (page will show '—')  ->  {fix}")
    for d in (["backend"] + (["frontend"] if with_frontend else [])):
        if not (REPO / d / "node_modules").exists():
            sys.exit(f"[run] {d}/node_modules missing — run: cd {d} && npm install")


def spawn(name, cmd, cwd):
    print(f"[run] starting {name}: {' '.join(cmd)}")
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    return name, subprocess.Popen(
        cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, creationflags=flags)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-frontend", action="store_true")
    args = ap.parse_args()
    preflight(not args.no_frontend)

    env = os.environ.copy()
    procs = [
        spawn("inference", [PY, "main.py"], REPO / "ml-inference-service"),
        spawn("simulator", [PY, "simulator.py"], REPO / "iot-simulator"),
        spawn("backend  ", ["node", "server.js"], REPO / "backend"),
    ]
    if not args.no_frontend:
        procs.append(spawn("frontend ", [NPM, "run", "dev"], REPO / "frontend"))

    print("\n[run] EdgeShield AI is starting — dashboard: http://localhost:5173\n"
          "[run] Ctrl+C stops everything.\n")

    import threading

    def pump(name, p):
        for line in p.stdout:
            print(f"[{name}] {line.rstrip()}")

    threads = [threading.Thread(target=pump, args=np, daemon=True) for np in procs]
    for t in threads:
        t.start()

    reported = set()
    try:
        while True:
            time.sleep(1)
            for name, p in procs:
                if p.poll() is not None and name not in reported:
                    reported.add(name)
                    print(f"[run] {name.strip()} exited with code {p.returncode}"
                          + (" — port already in use? close the other dev server "
                             "and restart run_all.py" if p.returncode else ""))
    except KeyboardInterrupt:
        print("\n[run] shutting down...")
        for _, p in procs:
            if p.poll() is not None:
                continue
            try:
                if os.name == "nt":
                    # tree-kill: npm spawns node children that would otherwise
                    # survive and squat on the ports for the next run
                    subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"],
                                   capture_output=True)
                else:
                    p.send_signal(signal.SIGTERM)
            except Exception:
                p.terminate()
        for _, p in procs:
            try:
                p.wait(timeout=8)
            except Exception:
                p.kill()
        print("[run] all services stopped.")


if __name__ == "__main__":
    main()
