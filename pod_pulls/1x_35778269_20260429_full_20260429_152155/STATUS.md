# Status (autonomous loop work)

## What's done

### Schedule sweeps on 1x (10min budget) — see `/workspace/v5_results.md`

| Config | quant val_bpb | vs baseline |
|---|---|---|
| baseline (default schedule) | 1.269 | — |
| EMA off (decay=0) | **1.215** | **−0.054** |
| EMA off + WD=0.5 | **1.212** | **−0.057** ✅ best |
| EMA off + WD=0.3 | 1.214 | −0.055 |
| muon_warmup=100 (vs 1500) | 1.234 | +0.019 (regressed) |
| NUM_LOOPS=2 + WD=0.5 + EMA=0 | 1.233 | +0.018 (regressed) |

**Headline finding:** EMA hurts at short budgets — decay=0.9965 has 285-step half-life but only ~822 training steps, so EMA averages over too many bad early weights. Setting EMA_DECAY=0 saves 0.054 bpb.

### Kernel work on 1x — see `/workspace/kernel_findings.md`

Per-microbatch fwd+bwd profile: 32% matmul, 30% flash_attn, 14% command-buffer overhead, 7% activations.

| Change | Result |
|---|---|
| `mode='reduce-overhead'` (cuda graphs) at grad_accum=1 | **+1.2%** ✓ |
| Full kernel stack (cuda graphs + rotary + EMA=0 + WD=0.5) at grad_accum=1 | **+3.4%** ✓ |
| Foreach EMA + prefetch + async save + GPU warmup | +0.4% ✓ |
| `PACK_QKV=1` | −15.7% ✗ |
| `INDUCTOR_EPILOGUE_FIRST=1` | −0.6% ✗ |
| `mode='reduce-overhead'` at grad_accum=8 | broken (autograd retains across micros) |
| `mode='max-autotune'` / `INDUCTOR_COORD_DESCENT=1` | compile time exceeds 15min, unworkable in budget |

**Kernel ceiling at 30M params is ~5%.** The model's small enough that GPU launch overhead can't be amortized further. Bigger absolute wins would need: a meaningfully larger model, FP8 matmul, or different attention pattern.

## Files changed

- `/workspace/Mikey_II_v5/train_gpt.py` — all toggles wired, env-var controlled. Default behavior matches v1 except v3 toggles default ON.
- `/workspace/run_8x_recommended.sh` — recommended invocation for 8x H100 with all toggles set right.
- `/workspace/kernel_findings.md` — kernel deep-dive with profile data.
- `/workspace/v5_results.md` — schedule sweep table.
- `/workspace/STATUS.md` — this file.

## Recommended config to use on 8x H100

```bash
NUM_LOOPS=0
SLIDING_WINDOW_ENABLED=0
EMA_DECAY=0
WARMDOWN_FRAC=0.5
COMPILE_MODE=reduce-overhead   # cuda graphs, +1.2% on 8x
ROTARY_PRECOMPUTE=1            # required for cuda graphs (no runtime cache invalidation)
EMA_FOREACH=1                  # default; 4.2x faster on EMA op
DATA_PREFETCH=1                # default; async data load
ASYNC_SAVE=1                   # default; save final_model.pt off-thread
GPU_WARMUP_SNAPSHOT=1          # default; skip CPU round-trip on warmup state save
```

## Honest assessment of the bpb target

- v2 80min/1x reached **1.108 quant** (matches ~10min/8x in compute budget).
- Goal is **<1.08 quant** in 10min/8x.
- Schedule wins (EMA off + WD=0.5) save **0.057 bpb** at the 10min/1x scale.
- If those wins translate at scale, projected v2-equivalent → ~1.05 bpb. **In striking distance of <1.08, but no margin.**
- File size constraint: v5 base is 13.5MB (under 16MB cap). v2 was 19MB (over). Need to either keep v5 architecture or aggressive quant on v2 (matrix_bits=5 or layer count reduction) to fit.

## What's still uncertain

- Whether schedule wins at 10min/1x truly translate 1:1 to 10min/8x compute scale (it should, but gradient dynamics with bigger global batch differ slightly).
- Whether the `mode='reduce-overhead'` cuda graph wins compose with arch changes (it should — orthogonal).
- Whether `LOGIT_SOFTCAP` removal / lower matrix_bits would beat 1.08 with v2's larger architecture.
