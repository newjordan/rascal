#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
set -a
source ../../conditions/mikey_III_sp5200_Loop_2x.env
set +a

RUN_ID="${RUN_ID:-mikey_III_sp5200_Loop_2x_seed${SEED}_$(date -u +%Y%m%d_%H%M%S)}"
export RUN_ID

test -f "$TOKENIZER_PATH"
printf '%s  %s\n' "$TOKENIZER_MODEL_SHA256" "$TOKENIZER_PATH" | sha256sum -c -
if [[ ! -d "$DATA_PATH" ]]; then
  echo "BLOCKED: missing SP5200 dataset at $DATA_PATH" >&2
  echo "Build it first: $DATASET_BUILD_SCRIPT" >&2
  exit 42
fi

python3 ../../scripts/validate_fineweb_shards.py "$DATA_PATH"

mkdir -p logs
echo "run_id:${RUN_ID}"
echo "condition_id:${CONDITION_ID}"
echo "cuda_visible_devices:${CUDA_VISIBLE_DEVICES}"
echo "data_path:${DATA_PATH}"
echo "tokenizer_path:${TOKENIZER_PATH}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  /venv/main/bin/torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" run.py
