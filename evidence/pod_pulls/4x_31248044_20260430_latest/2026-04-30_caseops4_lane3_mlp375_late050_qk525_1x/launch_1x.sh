#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

DATA_ROOT="/workspace/SOTA_FINAL/data"
CASEOPS_ROOT="${DATA_ROOT}/datasets/fineweb10B_sp10240_caseops/datasets"
MANIFEST="${CASEOPS_ROOT}/caseops_manifest.json"
DATASET_DIR="${CASEOPS_ROOT}/datasets/fineweb10B_sp10240_lossless_caps_caseops_v1_reserved"
TOKENIZER_PATH="${CASEOPS_ROOT}/tokenizers/fineweb_10240_bpe_lossless_caps_caseops_v1_reserved.model"

test -f "${MANIFEST}" || { echo "missing manifest: ${MANIFEST}" >&2; exit 2; }
test -d "${DATASET_DIR}" || { echo "missing dataset: ${DATASET_DIR}" >&2; exit 2; }
test -f "${TOKENIZER_PATH}" || { echo "missing tokenizer: ${TOKENIZER_PATH}" >&2; exit 2; }
command -v lrzip >/dev/null || { echo "missing lrzip; install system package lrzip before running pergroup compression" >&2; exit 2; }

train_count="$(find "${DATASET_DIR}" -maxdepth 1 -type f -name 'fineweb_train_*.bin' | wc -l)"
val_count="$(find "${DATASET_DIR}" -maxdepth 1 -type f -name 'fineweb_val_*.bin' ! -name 'fineweb_val_bytes_*.bin' | wc -l)"
val_bytes_count="$(find "${DATASET_DIR}" -maxdepth 1 -type f -name 'fineweb_val_bytes_*.bin' | wc -l)"
if [ "${train_count}" -lt 80 ] || [ "${val_count}" -lt 1 ] || [ "${val_bytes_count}" -ne "${val_count}" ]; then
  echo "incomplete SP10240 CaseOps data: train=${train_count} val=${val_count} val_bytes=${val_bytes_count}; need train>=80 and val_bytes==val" >&2
  exit 2
fi

/venv/main/bin/python3 - <<'PY' "${TOKENIZER_PATH}"
import sentencepiece as spm
import sys
sp = spm.SentencePieceProcessor(model_file=sys.argv[1])
assert sp.vocab_size() == 10240, sp.vocab_size()
assert [sp.piece_to_id(chr(0xE001 + i)) for i in range(4)] == [4, 5, 6, 7]
PY

NCCL_NET=Socket CUDA_VISIBLE_DEVICES="${GPU_ID:-0}" \
  /venv/main/bin/torchrun --standalone --nproc_per_node=1 run.py
