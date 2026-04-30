# 1x Pod Full Pull Summary

Source pod: `35778269`  
Endpoint: `root@34.48.171.202 -p 4997`  
Pulled at: `2026-04-29T15:21:55Z`  
Local root: `pod_pulls/1x_35778269_20260429_full_20260429_152155/`

## Pull Contents

Pulled `311` files, `665M` total. This includes source trees, run logs, result notes, kernel/profile scripts, final model checkpoints, quantized artifacts, and the Claude/Jupyter research state.

Intentionally excluded: dataset shard directories, Hugging Face cache, `node_modules`, wheels, and `__pycache__`.

Manifest: `_sha256_manifest.txt`

Key hashes:

| File | SHA256 |
|---|---|
| `Mikey_II_v5/train_gpt.py` | `8bd88b159d44f5b3c59f7dd5e14447199b9c21c6f26920cdc5f05f9416540135` |
| `kernel_findings.md` | `b8d5e232ad6a0cf4cc9003a433fe0037d3d776c590de56029438b70deb4ebf1e` |
| `run_8x_recommended.sh` | `3f58d59909dbd198cea8f608fbf6252693a1409c672bdaa451867a23f4e33851` |

## Schedule Evidence

All rows are 1x H100 mechanics evidence on SP8192, not standard 8x leaderboard conditions.

| Run | Loops | EMA | Warmdown | Steps | Quant BPB | Total bytes | Last tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `v5_baseline_noloop` | 0 | 0.9965 | 0.72 | 821 | 1.26926549 | 13,595,691 | 1,097,845 |
| `v5_no_ema` | 0 | 0.0 | 0.72 | 818 | 1.21545317 | 13,593,522 | 1,092,736 |
| `v5_wd05` | 0 | 0.0 | 0.5 | 822 | 1.21220168 | 13,594,517 | 1,099,248 |
| `v5_wd03` | 0 | 0.0 | 0.3 | 824 | 1.21357581 | 13,594,651 | 1,101,018 |
| `v5_muon100` | 0 | 0.0 | 0.72 | 823 | 1.23359889 | 13,596,983 | 1,099,627 |
| `v5_loops2` | 2 | 0.0 | 0.5 | 561 | 1.23319879 | 13,597,605 | 749,694 |

## Kernel Evidence

From `kernel_findings.md`:

- `COMPILE_MODE=reduce-overhead` is a safe +1.2% at `grad_accum=1`, which is the 8x case.
- Full kernel stack on grad-accum 1 reached +3.4% throughput.
- `PACK_QKV=1` regressed 15.7%.
- FP8 MLP/all-linears regressed 4.4% / 14.9%.
- `INDUCTOR_EPILOGUE_FIRST=1` regressed 0.6%.
- `max-autotune` and coordinate descent compile too slowly for the 10 minute build budget.

## Direct 8x Implication

The strongest schedule prior from this pod is:

```text
NUM_LOOPS=0
EMA_DECAY=0
WARMDOWN_FRAC=0.5
MUON_MOMENTUM_WARMUP_STEPS=1500
COMPILE_MODE=reduce-overhead
ROTARY_PRECOMPUTE=1
EMA_FOREACH=1
DATA_PREFETCH=1
ASYNC_SAVE=1
GPU_WARMUP_SNAPSHOT=1
```

This does not prove the 8x result, but it is a real 1x mechanics signal. The largest measured lever is disabling EMA at short horizon; the kernel work is useful but capped at a few percent throughput.
