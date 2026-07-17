#!/usr/bin/env bash
# One-command bring-up (Linux/macOS/WSL/Git-Bash)
cd "$(dirname "$0")/.." && exec python scripts/run_all.py "$@"
