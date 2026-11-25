#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

OUT="${OUT:-/workspace/data}"
SOURCE_SCRIPT="${SOURCE_SCRIPT:-/workspace/rascal/data/download_hf_docs_and_tokenize.py}"
SPEC="${SPEC:-$ROOT/scripts/tokenizer_specs_sp5200.json}"
MODEL="${MODEL:-/workspace/data/tokenizers/fineweb_5200_bpe.model}"
REPO_ID="${REPO_ID:-willdepueoai/parameter-golf}"
REMOTE_ROOT="${REMOTE_ROOT:-datasets}"
MIN_FREE_GB="${MIN_FREE_GB:-70}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x /venv/main/bin/python3 ]]; then
    PYTHON_BIN=/venv/main/bin/python3
  else
    PYTHON_BIN=python3
  fi
fi

free_gb="$(df -BG "$OUT" | awk 'NR==2 {gsub(/G/, "", $4); print $4}')"
if (( free_gb < MIN_FREE_GB )); then
  echo "BLOCKED: ${OUT} has ${free_gb}GB free; need at least ${MIN_FREE_GB}GB for docs plus SP5200 shards." >&2
  exit 42
fi

test -f "$SOURCE_SCRIPT"
test -f "$SPEC"
test -f "$MODEL"
mkdir -p "$OUT/datasets" "$OUT/tokenizers"

"$PYTHON_BIN" "$SOURCE_SCRIPT" \
  --repo-id "$REPO_ID" \
  --remote-root "$REMOTE_ROOT" \
  --output-root "$OUT" \
  --tokenizer-config "$SPEC" \
  --skip-byte \
  --reuse-sp-model "5200=${MODEL}" \
  2>&1 | tee "$OUT/sp5200_build.log"

"$PYTHON_BIN" "$ROOT/scripts/validate_fineweb_shards.py" \
  "$OUT/datasets/fineweb10B_sp5200"

sha256sum "$OUT/tokenizers/fineweb_5200_bpe.model" "$OUT/tokenizers/fineweb_5200_bpe.vocab"
