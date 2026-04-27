#!/usr/bin/env bash
# Whale donor reproduction.
# Body: PR 1493 train_gpt.py (untouched, 48583 bytes, md5 ca24f3448c0a806110a158c4533765c4).
# Env: hyperparameters from
#   /home/frosty40/SOTA_FINAL/legs/2026-04-15_whale_l12_dim576_matrixbits5/tracked_env.sh
# Original record:
#   raw 1.0802 step 3295 → post-EMA 1.07953 → quant 1.11723 → quant+sliding 1.10040
#   → quant+TTT 1.09561 → artifact 15,995,671 bytes (legal, under 16 MB cap)
set -euo pipefail

cd "$(dirname -- "${BASH_SOURCE[0]}")"

SEED="${SEED:-444}"
NPROC="${NPROC_PER_NODE:-8}"

# Whale donor hyperparameters (verbatim from tracked_env.sh)
export RUN_ID="whale_donor_l12_dim576_matrixbits5_s${SEED}_$(date +%Y%m%d_%H%M%S)"
export NUM_LAYERS=12
export XSA_LAST_N=11
export MATRIX_BITS=5
export MODEL_DIM=576
export EMBEDDING_DIM=576
export VOCAB_SIZE=8192
export ROPE_DIMS=16
export NUM_LOOPS=2
export LOOP_START=3
export LOOP_END=5
export ENABLE_LOOPING_AT=0.35
export PARALLEL_RESIDUAL_START=8
export QK_GAIN_INIT=5.25
export MLP_MULT=4.0
export MATRIX_LR=0.022
export MUON_MOMENTUM=0.99
export MUON_WD=0.095
export EMA_DECAY=0.9965
export WARMDOWN_FRAC=0.72
export TTT_ENABLED=1
export TTT_LR=0.005
export TTT_EPOCHS=3
export TTT_MOMENTUM=0.9
export DATA_DIR=./data
export MAX_WALLCLOCK_SECONDS=600
export SEED NPROC

/venv/main/bin/torchrun --standalone --nproc_per_node="${NPROC}" train_gpt.py
