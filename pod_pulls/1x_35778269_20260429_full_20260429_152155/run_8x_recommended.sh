#!/bin/bash
# Recommended 8x H100 training config based on 1x kernel/schedule sweeps.
#
# Best findings from sweeps:
# - EMA hurts at short budgets (decay=0.9965 averages bad early weights when only ~800 steps)
# - WARMDOWN_FRAC=0.5 marginally beats 0.72 default
# - NUM_LOOPS=0 keeps full throughput; loops costs 1.55x compute for unclear gain at 10min
# - mode='reduce-overhead' (cuda graphs) gives +1.2% on 8x (grad_accum=1)
# - All v3 toggles (foreach EMA, prefetch, async save, gpu warmup snapshot) are free wins
#
# Hardware: 8x H100 with torchrun --nproc_per_node=8
# Output: model_path=final_model.pt, quantized=final_model.int6.ptz (~13.5MB compressed)

cd /workspace/Mikey_II_v5

# Schedule
export NUM_LOOPS=0
export SLIDING_WINDOW_ENABLED=0
export EMA_DECAY=0
export WARMDOWN_FRAC=0.5

# Kernel toggles (8x H100 ⇒ grad_accum=1 ⇒ cuda graphs viable)
export COMPILE_MODE=reduce-overhead
export ROTARY_PRECOMPUTE=1
export EMA_FOREACH=1
export DATA_PREFETCH=1
export ASYNC_SAVE=1
export GPU_WARMUP_SNAPSHOT=1

# Skip these (validated worse):
# export PACK_QKV=0          # default; -15.7% if set
# export INDUCTOR_EPILOGUE_FIRST=0   # default; -0.6% if set

# Wallclock
export MAX_WALLCLOCK_SECONDS=600
export DATA_DIR=/workspace/Fartmagic/data/

torchrun --nproc_per_node=8 --master_port=29500 train_gpt.py 2>&1 | tee logs/run_8x_$(date +%Y%m%d_%H%M%S).log
