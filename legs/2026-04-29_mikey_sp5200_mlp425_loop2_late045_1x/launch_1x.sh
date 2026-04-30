#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
TOKENIZER=/workspace/data/tokenizers/fineweb_5200_bpe.model
DATASET=/workspace/data/datasets/fineweb10B_sp5200
test -f "$TOKENIZER"
test -f "$DATASET/fineweb_val_000000.bin"
test -f "$DATASET/fineweb_train_000000.bin"
test -f "$DATASET/fineweb_train_000136.bin"
train_count=$(find "$DATASET" -maxdepth 1 -name 'fineweb_train_*.bin' | wc -l)
val_count=$(find "$DATASET" -maxdepth 1 -name 'fineweb_val_*.bin' | wc -l)
test "$train_count" -eq 137
test "$val_count" -eq 1
CUDA_VISIBLE_DEVICES=0 \
  /venv/main/bin/torchrun --standalone --nproc_per_node=1 run.py
