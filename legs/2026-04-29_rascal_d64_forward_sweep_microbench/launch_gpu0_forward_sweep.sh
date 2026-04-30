#!/usr/bin/env bash
set -euo pipefail

cd /workspace/sota_rascal/legs/2026-04-29_rascal_d64_forward_sweep_microbench
mkdir -p logs
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/workspace/sota_rascal/legs/2026-04-29_rascal_d64_forward_sweep_microbench:${PYTHONPATH:-}"

/venv/main/bin/python3 run.py 2>&1 | tee "logs/forward_sweep_gpu0_$(date +%Y%m%d_%H%M%S).txt"
