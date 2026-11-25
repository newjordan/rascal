#!/usr/bin/env bash
set -euo pipefail

DOCS="${DOCS:-/home/frosty40/parameter-golf-lab/data/docs_selected.jsonl}"
KEY="${KEY:-/home/frosty40/.ssh/id_ed25519_apollo}"
HOST="${HOST:-root@206.125.32.60}"
PORT="${PORT:-56335}"
KNOWN="${KNOWN:-/tmp/codex_vast_known_hosts}"

if [[ ! -f "$DOCS" ]]; then
  echo "FATAL: docs not found: $DOCS" >&2
  echo "Set DOCS to the local docs_selected.jsonl. This script does not download it." >&2
  exit 1
fi
if [[ ! -f "$KEY" ]]; then
  echo "FATAL: ssh key not found: $KEY" >&2
  exit 1
fi

echo "[caseops-stream] docs=${DOCS}"
echo "[caseops-stream] host=${HOST} port=${PORT}"

ssh \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile="$KNOWN" \
  -o ServerAliveInterval=15 \
  -o ServerAliveCountMax=8 \
  -i "$KEY" \
  -p "$PORT" \
  "$HOST" \
  'cd /workspace/sota_rascal/legs/2026-04-30_pr1855_sp8192_lqer_smeargate_repro_8x &&
   mkdir -p /workspace/SOTA_FINAL/data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers &&
   cp -f tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model /workspace/SOTA_FINAL/data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model &&
   /venv/main/bin/python3 stream_prepare_caseops_data.py \
     --docs - \
     --out /workspace/SOTA_FINAL/data/datasets/fineweb10B_sp8192_caseops/datasets \
     --sp /workspace/SOTA_FINAL/data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
     --val-docs 50000 \
     --max-train-shards 80' \
  < "$DOCS"
