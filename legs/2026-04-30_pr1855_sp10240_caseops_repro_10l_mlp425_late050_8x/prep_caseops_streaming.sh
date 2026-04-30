#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

DATA_ROOT="${DATA_ROOT:-/workspace/SOTA_FINAL/data}"
OUT_ROOT="${DATA_ROOT}/datasets/fineweb10B_sp8192_caseops/datasets"
TOK_DIR="${OUT_ROOT}/tokenizers"
TOK_PATH="${TOK_DIR}/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model"
DOCS_URL="${DOCS_URL:-https://huggingface.co/datasets/kilojoules/parameter-golf-sp8192/resolve/main/docs_selected.jsonl?download=true}"

mkdir -p "${TOK_DIR}"
cp -f tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model "${TOK_PATH}"

echo "streaming docs from ${DOCS_URL}"
echo "writing CaseOps shards under ${OUT_ROOT}"

curl -L --fail --retry 8 --retry-delay 5 "${DOCS_URL}" \
  | /venv/main/bin/python3 stream_prepare_caseops_data.py \
      --docs - \
      --out "${OUT_ROOT}" \
      --sp "${TOK_PATH}"
