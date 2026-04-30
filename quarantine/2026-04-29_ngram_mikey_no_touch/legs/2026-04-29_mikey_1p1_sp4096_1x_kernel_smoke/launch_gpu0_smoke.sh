#!/usr/bin/env bash
set -euo pipefail

LEG_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${LEG_DIR}"

mkdir -p logs
TS="$(date +%Y%m%d_%H%M%S)"
RUN_ID="mikey_1p1_sp4096_1x_kernel_winner_gpu0_${TS}"
LOG="logs/${RUN_ID}.launcher.log"

{
  echo "condition_source=/workspace/sota_rascal/conditions/mikey_1p1_sp4096_1x_kernel_smoke.env"
  echo "run_label=kernel_integration_scout"
  echo "kernel_anchor=winner_anchor primary qmr1984"
  echo "changed_fields=8x source -> 1x scout, GPU0 only, ITERATIONS=3, COMPILE_ENABLED=0, LATE_QAT_THRESHOLD=0, KERNEL_SMOKE_ONLY=1, FA3 -> whale_native primary winner"
  echo "run_id=${RUN_ID}"
  echo "gpu_before:"
  nvidia-smi --query-gpu=index,name,temperature.gpu,temperature.memory,utilization.gpu,memory.used,power.draw,clocks.sm,clocks_throttle_reasons.sw_thermal_slowdown --format=csv,noheader,nounits
  echo "file_hashes:"
  sha256sum run.py vault/whale_kernel_triton.py
} | tee "${LOG}"

CUDA_VISIBLE_DEVICES=0 \
DATA_PATH=/workspace/data/datasets/fineweb10B_sp4096 \
TOKENIZER_PATH=/workspace/data/tokenizers/fineweb_4096_bpe.model \
VOCAB_SIZE=4096 \
SEED=444 \
MAX_WALLCLOCK_SECONDS=180 \
ITERATIONS=3 \
WARMUP_STEPS=0 \
TRAIN_LOG_EVERY=1 \
VAL_LOSS_EVERY=0 \
TRAIN_BATCH_TOKENS=786432 \
TRAIN_SEQ_LEN=2048 \
LOADER_MODE=coprime \
COPRIME_MAX_LOADED_SHARDS=143 \
COPRIME_SHARDS_PER_BATCH=1 \
COPRIME_SHARD_HOLD_STEPS=64 \
SKIP_GPTQ=1 \
SKIP_FINAL_EVAL=1 \
KERNEL_SMOKE_ONLY=1 \
KERNEL_SMOKE_ALLOW_WORLD_SIZE=1 \
COMPILE_ENABLED=0 \
LATE_QAT_THRESHOLD=0 \
TRIGRAM=0 \
NGRAM_EVAL_ORDER=0 \
KERNEL_SMOKE_ATTENTION_MODE=whale_native \
WHALE_BWD_VARIANT=fused_delta_tma \
WHALE_BWD_Q_CONFIG=128,64,8,5 \
WHALE_BWD_Q_MAXNREG=1984 \
WHALE_BWD_KV_TMA_CONFIG=128,64,8,3 \
WHALE_BWD_KV_TMA_MAXNREG=256 \
RUN_ID="${RUN_ID}" \
/venv/main/bin/python3 run.py 2>&1 | tee -a "${LOG}"

{
  echo "gpu_after:"
  nvidia-smi --query-gpu=index,name,temperature.gpu,temperature.memory,utilization.gpu,memory.used,power.draw,clocks.sm,clocks_throttle_reasons.sw_thermal_slowdown --format=csv,noheader,nounits
  echo "launcher_log=${LOG}"
  echo "run_log=logs/${RUN_ID}.txt"
} | tee -a "${LOG}"
