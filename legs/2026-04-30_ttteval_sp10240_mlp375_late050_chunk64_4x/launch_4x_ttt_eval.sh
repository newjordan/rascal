#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

DATA_ROOT="/workspace/SOTA_FINAL/data"
CASEOPS_ROOT="${DATA_ROOT}/datasets/fineweb10B_sp10240_caseops/datasets"
MANIFEST="${CASEOPS_ROOT}/caseops_manifest.json"
DATASET_DIR="${CASEOPS_ROOT}/datasets/fineweb10B_sp10240_lossless_caps_caseops_v1_reserved"
TOKENIZER_PATH="${CASEOPS_ROOT}/tokenizers/fineweb_10240_bpe_lossless_caps_caseops_v1_reserved.model"
PARENT_ARTIFACT="/workspace/sota_rascal/legs/2026-04-30_pr1855_sp10240_caseops_mlp375_late050_8x/logs/final_model.int6.ptz"

test -f "${MANIFEST}" || { echo "missing manifest: ${MANIFEST}" >&2; exit 2; }
test -d "${DATASET_DIR}" || { echo "missing dataset: ${DATASET_DIR}" >&2; exit 2; }
test -f "${TOKENIZER_PATH}" || { echo "missing tokenizer: ${TOKENIZER_PATH}" >&2; exit 2; }
test -f "${PARENT_ARTIFACT}" || { echo "missing parent artifact: ${PARENT_ARTIFACT}" >&2; exit 2; }
ln -sfn "${PARENT_ARTIFACT}" logs/final_model.int6.ptz

NCCL_NET=Socket CUDA_VISIBLE_DEVICES=0,1,2,3 TTT_EVAL_ONLY=1 \
  torchrun --standalone --nproc_per_node=4 run.py
