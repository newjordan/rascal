#!/usr/bin/env bash
set -euo pipefail

LEG_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${LEG_DIR}"
mkdir -p logs

TS="$(date -u +%Y%m%d_%H%M%S)"
SUMMARY="logs/mikey_sp4096_gpu0_mlp_ab_${TS}.summary.tsv"
echo -e "mode\tstatus\tlast_step\tstep_avg_ms\ttok_s\tlog" > "${SUMMARY}"

run_one() {
  local mode="$1"
  local run_id="mikey_sp4096_gpu0_mlp_${mode:-eager}_${TS}"
  local log="logs/${run_id}.log"
  {
    echo "condition_source=/workspace/sota_rascal/conditions/mikey_1p1_sp4096_1x_kernel_smoke.env"
    echo "run_label=mlp_kernel_ab_scout"
    echo "changed_fields=GPU0 only, ITERATIONS=20, COMPILE_ENABLED=0, KERNEL_SMOKE_ONLY=1, attention=FA3, MLP_KERNEL_MODE=${mode:-eager}"
    echo "run_id=${run_id}"
    echo "gpu_before:"
    nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits
    echo "file_hashes:"
    sha256sum run.py
  } | tee "${log}"

  CUDA_VISIBLE_DEVICES=0 \
  DATA_PATH=/workspace/data/datasets/fineweb10B_sp4096 \
  TOKENIZER_PATH=/workspace/data/tokenizers/fineweb_4096_bpe.model \
  VOCAB_SIZE=4096 \
  SEED=444 \
  MAX_WALLCLOCK_SECONDS=150 \
  ITERATIONS=20 \
  WARMUP_STEPS=0 \
  TRAIN_LOG_EVERY=5 \
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
  KERNEL_SMOKE_ATTENTION_MODE=fa3 \
  MLP_KERNEL_MODE="${mode}" \
  RUN_ID="${run_id}" \
  /venv/main/bin/python3 run.py 2>&1 | tee -a "${log}"

  local last
  last="$(grep -E 'step:[0-9]+/20 train_loss:' "${log}" | tail -1 || true)"
  if [[ -z "${last}" ]]; then
    echo -e "${mode:-eager}\tno_train_line\t\t\t\t${log}" | tee -a "${SUMMARY}"
    return 1
  fi
  local step avg toks
  step="$(sed -nE 's/.*step:([0-9]+)\\/20.*/\\1/p' <<<"${last}")"
  avg="$(sed -nE 's/.*step_avg:([0-9.]+)ms.*/\\1/p' <<<"${last}")"
  toks="$(sed -nE 's/.*tok\\/s:([0-9]+).*/\\1/p' <<<"${last}")"
  echo -e "${mode:-eager}\tok\t${step}\t${avg}\t${toks}\t${log}" | tee -a "${SUMMARY}"
}

run_one ""
run_one "triton_act"

echo "summary=${SUMMARY}"
cat "${SUMMARY}"
