#!/usr/bin/env bash
# Canonical sp5200 build on pod. Train BPE on train-portion only (val docs = first 50k untouched).
# Output mirrors the sp4096 layout the train scripts expect.
set -euo pipefail

OUT=/workspace/data
SPEC=/workspace/data_prep/tokenizer_specs_sp5200.json

mkdir -p "$OUT/datasets" "$OUT/tokenizers"

cd /workspace/data_prep

/venv/main/bin/python download_hf_docs_and_tokenize.py \
  --output-root "$OUT" \
  --tokenizer-config "$SPEC" \
  --skip-byte 2>&1 | tee /workspace/sp5200_build.log

# After completion, rename layout to match canonical /workspace/data/datasets/fineweb10B_sp5200/
# (download_hf_docs_and_tokenize.py already writes to <out>/datasets/<dataset_suffix>/)

echo "DONE. Verify:"
ls -la /workspace/data/datasets/fineweb10B_sp5200/ | head
ls -la /workspace/data/tokenizers/fineweb_5200_bpe.model
