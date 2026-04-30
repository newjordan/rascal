#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
set -a
source ../../conditions/rascal_sp4096_mlp35_2x.env
set +a

RUN_ID="${RUN_ID:-rascal_sp4096_mlp35_2x_seed${SEED}_$(date -u +%Y%m%d_%H%M%S)}"
export RUN_ID

test -f "$TOKENIZER_PATH"
test -d "$DATA_PATH"
test -n "$(find "$DATA_PATH" -maxdepth 1 -name 'fineweb_val_*.bin' -print -quit)"
test -n "$(find "$DATA_PATH" -maxdepth 1 -name 'fineweb_train_*.bin' -print -quit)"
sha256sum "$TOKENIZER_PATH"

mkdir -p logs
echo "run_id:${RUN_ID}"
echo "condition_id:${CONDITION_ID}"
echo "cuda_visible_devices:${CUDA_VISIBLE_DEVICES}"
echo "data_path:${DATA_PATH}"
echo "tokenizer_path:${TOKENIZER_PATH}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  /venv/main/bin/torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" run.py
