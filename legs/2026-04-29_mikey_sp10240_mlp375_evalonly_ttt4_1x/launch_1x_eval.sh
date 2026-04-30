#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
TOKENIZER=/workspace/data/tokenizers/fineweb_10240_bpe.model
VAL=/workspace/data/datasets/fineweb10B_sp10240/fineweb_val_000000.bin
test -f "$TOKENIZER"
test -f "$VAL"
test -f final_model.int6.ptz
CUDA_VISIBLE_DEVICES=0 /venv/main/bin/python3 run.py
