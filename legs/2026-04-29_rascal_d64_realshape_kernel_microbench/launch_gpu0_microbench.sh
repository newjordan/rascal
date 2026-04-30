#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
/venv/main/bin/python3 run.py "$@"
