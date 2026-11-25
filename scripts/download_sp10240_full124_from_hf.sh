#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

BASE_URL="https://huggingface.co/datasets/Frosty40/10k_golfer/resolve/main"
DATA_ROOT="${DATA_ROOT:-/workspace/SOTA_FINAL/data}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASET_DIR="$DATA_ROOT/datasets/fineweb10B_sp10240"
TOKENIZER_DIR="$DATA_ROOT/tokenizers"
FIRST80_DIR="$DATA_ROOT/datasets/fineweb10B_sp10240_first80"

mkdir -p "$DATASET_DIR" "$TOKENIZER_DIR"

download_file() {
  local name="$1"
  local dest="$2"
  if [[ -s "$dest" ]]; then
    echo "exists $dest"
    return
  fi
  echo "download $name -> $dest"
  curl -fL --retry 8 --retry-all-errors --connect-timeout 20 -C - \
    -o "$dest.part" "$BASE_URL/$name?download=true"
  mv "$dest.part" "$dest"
}

download_file "fineweb_10240_bpe.model" "$TOKENIZER_DIR/fineweb_10240_bpe.model"
download_file "fineweb_10240_bpe.vocab" "$TOKENIZER_DIR/fineweb_10240_bpe.vocab"

for idx in $(seq 0 123); do
  shard="$(printf 'fineweb_train_%06d.bin' "$idx")"
  if [[ -s "$FIRST80_DIR/$shard" && ! -e "$DATASET_DIR/$shard" ]]; then
    mv "$FIRST80_DIR/$shard" "$DATASET_DIR/$shard"
  fi
  download_file "$shard" "$DATASET_DIR/$shard"
done

if [[ -s "$FIRST80_DIR/fineweb_val_000000.bin" && ! -e "$DATASET_DIR/fineweb_val_000000.bin" ]]; then
  mv "$FIRST80_DIR/fineweb_val_000000.bin" "$DATASET_DIR/fineweb_val_000000.bin"
fi
download_file "fineweb_val_000000.bin" "$DATASET_DIR/fineweb_val_000000.bin"

"$PYTHON_BIN" "$ROOT/scripts/validate_fineweb_shards.py" "$DATASET_DIR" \
  --expect-train 124 --expect-val 1
test -f "$TOKENIZER_DIR/fineweb_10240_bpe.model"
