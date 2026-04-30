# Kernel Optimization Findings (1x H100, 30M params)

Per-microbatch profile (B=48, T=2048, 98K tokens):

| Component | CUDA time | % of fwd+bwd |
|---|---|---|
| aten::mm (matmuls) | 26.1ms | 32% |
| flash_attn_3 fwd+bwd | 24.6ms | 30% |
| Command Buffer Full (launch overhead) | 11.5ms | 14% |
| Triton fused activations | 6.0ms | 7% |
| Other (norms, casts, reshapes) | 14ms | 17% |
| **Total fwd+bwd** | **82.5ms** | 100% |

## What Worked

| Change | Speedup | Notes |
|---|---|---|
| `mode='reduce-overhead'` (cuda graphs) | +1.2% | Only works at grad_accum=1 (8x H100 case). Breaks with grad_accum=8 due to autograd-keeps-tensors-alive across micro batches. |
| Full kernel stack on grad_accum=1 (cuda graphs + rotary precompute + EMA off + WD=0.5) | **+3.4%** | 905k vs 875k tok/s. Stacked toggles compound. |
| Foreach EMA + data prefetch + async save + GPU warmup snapshot (grad_accum=8) | +0.4% | From earlier ablation (1.097M baseline → 1.098M with toggles). Free wins. |

## What Didn't Work

| Change | Result | Why |
|---|---|---|
| `PACK_QKV=1` (fuse c_q/c_k/c_v into one matmul) | **−15.7%** | split() on output creates strided views; backward through cat hurts more than the saved matmul launches. |
| `ROTARY_PRECOMPUTE=1` alone | 0% | Cache only ran once anyway in steady state. |
| `mode='reduce-overhead'` at grad_accum=8 | broken | "Gradient addition node due to multiple use of tensor" — autograd retains forward activations across all 8 microbatches before backward, conflicting with cuda graph memory reuse. |
| `INDUCTOR_EPILOGUE_FIRST=1` | **−0.6%** | Compile already fuses elementwise after matmul as standalone Triton kernel; flipping order doesn't help and may produce worse plan. |
| FP8 (E4M3) on MLP linears (torchao) | **−4.4%** | At dim=512 the FP8 amax/scaling overhead per matmul exceeds tensor-core savings. CastedLinear may also not convert cleanly. |
| FP8 (E4M3) on all linears (torchao) | **−14.9%** | Attention has many small projections (kvdim=256) where FP8 fixed overhead dominates. FP8 needs compute-bound matmul to win, and our small dims aren't. |
| `INDUCTOR_COORD_DESCENT=1` | untested (slow) | Compile time exceeds 15+ min for our model — unworkable in 10min build budget. Would need offline-compiled artifact. |
| `mode='max-autotune-no-cudagraphs'` | untested (slow) | Same compile-time problem. |

## What's Bottleneck-Limited

- **aten::mm 32%**: cuBLAS already vendor-optimal at these tile sizes. Matmul shapes (B*T × dim) for dim=512 hit cuBLAS sweet spots.
- **flash_attn_3 30%**: FA3 is heavily optimized for H100. Hard to beat.
- **Triton activations 7%**: Already fused by compile (`leaky_relu(0.5).square()` collapses into one Triton kernel).

## Why Architecture Wins More Than Kernel

**Empirical ceiling on per-step kernel speedup at this model size: ~5%.** Architecture/schedule changes have ~10-30% leverage on bpb (per the v5 sweeps showing EMA-off = 0.054 bpb saved, looping = 0.018 bpb cost).

The model is small enough (30M params, 11 layers, dim=512) that GPU launch overhead can't be amortized further; each layer's compute is comparable to launch latency.

## Recommendations for 8x H100 build

1. **Use `COMPILE_MODE=reduce-overhead`** (grad_accum=1 there). +1.2% throughput, safe. Already wired with `cudagraph_mark_step_begin()` in `step_fn`.
2. **Keep `DATA_PREFETCH=1`, `EMA_FOREACH=1`, `GPU_WARMUP_SNAPSHOT=1`** (combined +0.4%, free).
3. **Set `EMA_DECAY=0`** for short runs (10min budget) — saves 0.054 bpb (raw model is better than EMA-averaged at 800-step horizon).
4. **Skip `PACK_QKV`** — hurts perf at this dim (-15.7%).
5. **Skip `INDUCTOR_EPILOGUE_FIRST`** — slight regression (-0.6%).
6. **Skip `mode='max-autotune'` and `INDUCTOR_COORD_DESCENT`** — compile time exceeds 15+ min, unworkable in 10min budget.

## Combined kernel toggle, 8x H100, ready to set:

```bash
COMPILE_MODE=reduce-overhead
ROTARY_PRECOMPUTE=1
EMA_FOREACH=1
DATA_PREFETCH=1
ASYNC_SAVE=1
GPU_WARMUP_SNAPSHOT=1
EMA_DECAY=0
NUM_LOOPS=0
SLIDING_WINDOW_ENABLED=0
WARMDOWN_FRAC=0.5
```

## Why the kernel ceiling is low here

- 30M-param model with dim=512 hits cuBLAS sweet spots already.
- Each layer's compute is ~7ms (forward+backward), comparable to launch overhead.
- The only "free" reductions in launch overhead require cuda graphs, which conflict with the 8-step gradient accumulation pattern on 1x.
- On 8x with grad_accum=1, cuda graphs work but only save 1-2%.
- Bigger absolute wins would need: a meaningfully larger model (more compute per launch), FP8 matmul (2x on H100 tensor cores), or a fundamentally different attention (sliding window, etc.).
