#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

REPO="${REPO:-Frosty40/10k_caseops_golfer}"
OUT_ROOT="${OUT_ROOT:-/home/frosty40/SOTA_FINAL/data/datasets/fineweb10B_sp10240_caseops/datasets}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HF_BIN="${HF_BIN:-$(command -v hf || true)}"
ACTION="${1:-check}"

TOKENIZER_MODEL="${OUT_ROOT}/tokenizers/fineweb_10240_bpe_lossless_caps_caseops_v1_reserved.model"
TOKENIZER_VOCAB="${OUT_ROOT}/tokenizers/fineweb_10240_bpe_lossless_caps_caseops_v1_reserved.vocab"
DATASET_DIR="${OUT_ROOT}/datasets/fineweb10B_sp10240_lossless_caps_caseops_v1_reserved"
MANIFEST="${OUT_ROOT}/caseops_manifest.json"

echo "repo=${REPO}"
echo "out_root=${OUT_ROOT}"

missing=0
for path in "${TOKENIZER_MODEL}" "${TOKENIZER_VOCAB}" "${DATASET_DIR}" "${MANIFEST}"; do
  if [[ ! -e "${path}" ]]; then
    echo "MISSING: ${path}" >&2
    missing=1
  fi
done

train_count=0
val_count=0
val_bytes_count=0
if [[ -d "${DATASET_DIR}" ]]; then
  inventory="$("${PYTHON_BIN}" "${ROOT}/scripts/validate_fineweb_shards.py" --inventory "${DATASET_DIR}")"
  printf '%s\n' "${inventory}"
  train_count="$(printf '%s\n' "${inventory}" | awk -F= '/^train=/{print $2}')"
  val_count="$(printf '%s\n' "${inventory}" | awk -F= '/^val=/{print $2}')"
  val_bytes_count="$(printf '%s\n' "${inventory}" | awk -F= '/^val_bytes=/{print $2}')"
else
  printf '%s\n' "train=0" "val=0" "val_bytes=0"
fi

echo "train_shards=${train_count} val_shards=${val_count} val_byte_sidecars=${val_bytes_count}"

if [[ "${missing}" -ne 0 || "${train_count}" -eq 0 || "${val_count}" -eq 0 || "${val_bytes_count}" -eq 0 ]]; then
  echo "Dataset is not upload-ready yet." >&2
  exit 2
fi

if [[ "${ACTION}" != "check" && "${ACTION}" != "upload" ]]; then
  echo "Usage: $0 [check|upload]" >&2
  exit 64
fi

if [[ -z "${HF_BIN}" || ! -x "${HF_BIN}" ]]; then
  echo "FATAL: hf CLI not found. Set HF_BIN=/path/to/hf." >&2
  exit 1
fi

"${HF_BIN}" auth whoami >/dev/null

if [[ "${ACTION}" == "check" ]]; then
  echo "Upload-ready. To upload:"
  echo "  REPO=${REPO} OUT_ROOT=${OUT_ROOT} HF_BIN=${HF_BIN} $0 upload"
  exit 0
fi

"${HF_BIN}" repo create "${REPO}" --repo-type dataset --exist-ok
"${HF_BIN}" upload-large-folder "${REPO}" "${OUT_ROOT}" --repo-type dataset
echo "DONE: https://huggingface.co/datasets/${REPO}"
