#!/usr/bin/env bash
set -euo pipefail

LEG_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${LEG_DIR}"

mkdir -p logs results

RUN_TS="$(date +%Y%m%d_%H%M%S)"
MANIFEST="logs/gpu0_kernel_smoke_${RUN_TS}.condition.txt"
SHAPE="${SMOKE_SHAPE:-2,2048,8,4,64}"
BACKENDS="${SMOKE_BACKENDS:-fa3,whale}"
ROUNDS="${SMOKE_ROUNDS:-2}"
ITERS="${SMOKE_ITERS:-10}"

{
  echo "condition_source=/workspace/sota_rascal/conditions/mikey_kernel_smoke_gpu0.env"
  echo "run_label=kernel_only_scout"
  echo "changed_fields=GPU_VISIBLE=0, WORLD_SIZE=1, NPROC_PER_NODE=1, no_dataset_or_tokenizer"
  echo "shape=${SHAPE}"
  echo "backends=${BACKENDS}"
  echo "rounds=${ROUNDS}"
  echo "iters=${ITERS}"
  echo "metric=fwd_mean_ms,fb_mean_ms"
  echo "comparator=FA3 same-process same-GPU0"
  echo "source_body=Mikey body integration pending; kernel microbench only"
  echo "gpu_before:"
  nvidia-smi --query-gpu=index,name,temperature.gpu,temperature.memory,utilization.gpu,memory.used,power.draw,clocks.sm,clocks_throttle_reasons.sw_thermal_slowdown --format=csv,noheader,nounits
  echo "file_hashes:"
  sha256sum bench_stable_scratch.py frozen_vault/whale_kernel_triton.py winner_anchor.json fallback_anchor.json benchmark_matrix.json
} | tee "${MANIFEST}"

run_profile() {
  local profile="$1"
  local dq="$2"
  local qmr="$3"
  local dkdv="$4"
  local dkdv_mr="$5"
  local out="results/${profile}_${RUN_TS}.json"
  local log="logs/${profile}_${RUN_TS}.log"

  {
    echo "profile=${profile}"
    echo "WHALE_BWD_VARIANT=fused_delta_tma"
    echo "WHALE_BWD_Q_CONFIG=${dq}"
    echo "WHALE_BWD_Q_MAXNREG=${qmr}"
    echo "WHALE_BWD_KV_TMA_CONFIG=${dkdv}"
    echo "WHALE_BWD_KV_TMA_MAXNREG=${dkdv_mr}"
    echo "WHALE_BWD_KV_CONFIG=${dkdv}"
    echo "WHALE_BWD_KV_MAXNREG=${dkdv_mr}"
  } | tee -a "${MANIFEST}"

  CUDA_VISIBLE_DEVICES=0 \
  WHALE_BWD_VARIANT=fused_delta_tma \
  WHALE_FUSED_DELTA_T_MAX=3072 \
  WHALE_BWD_Q_CONFIG="${dq}" \
  WHALE_BWD_Q_MAXNREG="${qmr}" \
  WHALE_BWD_KV_TMA_CONFIG="${dkdv}" \
  WHALE_BWD_KV_TMA_MAXNREG="${dkdv_mr}" \
  WHALE_BWD_KV_CONFIG="${dkdv}" \
  WHALE_BWD_KV_MAXNREG="${dkdv_mr}" \
  /venv/main/bin/python3 bench_stable_scratch.py \
    --label "gpu0_${profile}" \
    --shape "${SHAPE}" \
    --backends "${BACKENDS}" \
    --rounds "${ROUNDS}" \
    --iters "${ITERS}" \
    --out "${out}" \
    2>&1 | tee "${log}"

  echo "result_${profile}=${out}" | tee -a "${MANIFEST}"
}

run_profile "primary_qmr1984" "128,64,8,5" "1984" "128,64,8,3" "256"
run_profile "fallback_qmr928" "128,64,8,2" "928" "128,128,8,3" "256"

{
  echo "gpu_after:"
  nvidia-smi --query-gpu=index,name,temperature.gpu,temperature.memory,utilization.gpu,memory.used,power.draw,clocks.sm,clocks_throttle_reasons.sw_thermal_slowdown --format=csv,noheader,nounits
  echo "manifest=${MANIFEST}"
} | tee -a "${MANIFEST}"
