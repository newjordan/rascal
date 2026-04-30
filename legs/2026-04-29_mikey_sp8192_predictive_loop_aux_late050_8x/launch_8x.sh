#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

TOKENIZER=/workspace/SOTA_FINAL/data/tokenizers/fineweb_8192_bpe.model
DATASET=/workspace/SOTA_FINAL/data/datasets/fineweb10B_sp8192
test -f "$TOKENIZER"
test -f "$DATASET/fineweb_val_000000.bin"
test -f "$DATASET/fineweb_train_000000.bin"
test -f "$DATASET/fineweb_train_000079.bin"
train_count=$(find "$DATASET" -maxdepth 1 -name 'fineweb_train_*.bin' | wc -l)
val_count=$(find "$DATASET" -maxdepth 1 -name 'fineweb_val_*.bin' | wc -l)
test "$train_count" -eq 80
test "$val_count" -eq 1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  /venv/main/bin/torchrun --standalone --nproc_per_node=8 run.py
