#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

DATA_ROOT="${DATA_ROOT:-/workspace/SOTA_FINAL/data}"
CASEOPS_ROOT="${DATA_ROOT}/datasets/fineweb10B_sp8192_caseops/datasets"
DATASET_DIR="${CASEOPS_ROOT}/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved"
TOKENIZER_PATH="${CASEOPS_ROOT}/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model"

test -d "${DATASET_DIR}" || { echo "missing dataset: ${DATASET_DIR}" >&2; exit 2; }
test -f "${TOKENIZER_PATH}" || { echo "missing tokenizer: ${TOKENIZER_PATH}" >&2; exit 2; }
command -v lrzip >/dev/null || { echo "missing lrzip; install system package lrzip before running PR1855 pergroup compression" >&2; exit 2; }

train_count="$(find "${DATASET_DIR}" -maxdepth 1 -type f -name 'fineweb_train_*.bin' | wc -l)"
val_count="$(find "${DATASET_DIR}" -maxdepth 1 -type f -name 'fineweb_val_*.bin' ! -name 'fineweb_val_bytes_*.bin' | wc -l)"
val_bytes_count="$(find "${DATASET_DIR}" -maxdepth 1 -type f -name 'fineweb_val_bytes_*.bin' | wc -l)"
if [ "${train_count}" -lt 80 ] || [ "${val_count}" -lt 1 ] || [ "${val_bytes_count}" -ne "${val_count}" ]; then
  echo "incomplete PR1855 CaseOps data: train=${train_count} val=${val_count} val_bytes=${val_bytes_count}; need train>=80 and val_bytes==val" >&2
  exit 2
fi

export DATA_DIR="${DATA_ROOT}"
export VOCAB_SIZE=8192
export DATA_PATH="${DATASET_DIR}"
export TOKENIZER_PATH="${TOKENIZER_PATH}"
export CASEOPS_ENABLED=1
export ITERATIONS=20000
export MAX_WALLCLOCK_SECONDS=600
export PHASED_TTT_ENABLED=1
export PHASED_TTT_PREFIX_DOCS=2500
export PHASED_TTT_NUM_PHASES=3
export EMBED_BITS=7
export MATRIX_LR=0.026
export MIN_LR=0.1
export MLP_CLIP_SIGMAS=11.5
export ATTN_CLIP_SIGMAS=13.0
export EMBED_CLIP_SIGMAS=14.0
export GRAD_CLIP_NORM=0.3
export TTT_CHUNK_SIZE=48
export WARMUP_STEPS=20
export MUON_BACKEND_STEPS=5
export GLOBAL_TTT_MOMENTUM=0.9
export WARMDOWN_FRAC=0.85
export BETA2=0.99
export TTT_BETA2=0.99
export TTT_WEIGHT_DECAY=0.5
export TTT_LORA_RANK=80
export SPARSE_ATTN_GATE_SCALE=0.5
export GPTQ_RESERVE_SECONDS=0.5
export GPTQ_CALIBRATION_BATCHES=16
export VAL_LOSS_EVERY=0
export GATED_ATTN_QUANT_GATE=1
export SPARSE_ATTN_GATE_ENABLED=1
export GATE_WINDOW=12
export SMEAR_GATE_ENABLED=1
export LQER_ENABLED=1
export LQER_ASYM_ENABLED=1
export LQER_RANK=4
export LQER_FACTOR_BITS=4
export LQER_ASYM_GROUP=64
export LQER_TOP_K=3
export FUSED_CE_ENABLED=1
export COMPRESSOR=pergroup
export NCCL_NET=Socket
export SEED="${SEED:-42}"
export RUN_ID="pr1855_sp8192_lqer_smeargate_repro_8x_seed${SEED}"
export ARTIFACT_DIR="${ARTIFACT_DIR:-logs}"

/venv/main/bin/torchrun --standalone --nproc_per_node=8 run.py
