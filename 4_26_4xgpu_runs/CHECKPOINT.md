# Checkpoint — 2026-04-26 → 2026-04-27 (4K PAYDIRT)

## TL;DR — paydirt hit 2026-04-27 ~02:09 UTC

**Run `rascal_4k_8x_seed444_20260427_015932` produced sliding val_bpb 0.8672 — 0.243 bpb better than PR 1120's 1.10987 record.** Quant survived (EMA 0.8813 → int6 quant 0.8914 = only 0.01 collapse, unlike all prior tests today). Artifact is 17,766,043 bytes — **1.77 MB over the 16 MB legal cap**. Salvage paths in flight.

The 4k vocab axis was previously dismissed by the 11-day research collate as a null result — but that dismissal was based on stale-shard configs (only 17 sp4096 shards on prior pod) and short-horizon evaluation. With correct setup (143 shards on disk, MAX_LOADED=143, full 600s 8x canonical), 4k vocab is the strongest signal Jordan has produced.

## Pods
- **8x H100 SXM** (instance `35657525`, ssh `34.69.145.155:3159`) — running. Cost ~$12.36/hr.
  - `/workspace/rascal/data` → symlink to `/workspace/Fartmagic/data` ✓
  - Datasets present: sp1024 (80+1), sp8192 (80+1), **sp4096 (143+1, rsync'd from local master)** ✓
  - Tokenizers present: 1024, 4096, 8192 ✓
  - **Saved model from paydirt run**: `/workspace/rascal/final_model.pt` (105 MB raw post-EMA) and `/workspace/rascal/final_model.int6.ptz` (17 MB current oversized blob)
- **1x H100 SXM** (instance `35082273`) — smoke pod, sp1024+sp8192 only.

## Runs executed today
| run | result | verdict |
|---|---|---|
| `rascal_loop_8x_seed444_20260427_012605` | mid-train 1.13 → quant 1.46 (collapse), 16.29 MB (over cap) | bust × 2 |
| `rascal_mom95_mlp45_8x_seed444_20260427_014418` | post-EMA 1.12 → quant 1.17, 15.06 MB (legal but underpowered, +0.06 over PR 1120) | bust |
| `rascal_4k_8x_seed444_20260427_015932` | post-EMA 0.8813 → quant 0.8914 → **sliding 0.8672**, 17.77 MB | **PAYDIRT (oversized)** |

All logs saved at `4_26_4xgpu_runs/_run_logs/`.

## Salvage paths (in flight)

| path | what | status |
|---|---|---|
| `4k_vocab_rascal_10l/train_gpt_4K_10L_8xgpu.py` | NUM_LAYERS 11→10, retrain | pushed to pod, not yet fired |
| `4k_vocab_rascal_10l_brotli_mixed/` (subagent building) | 10L + brotli compression + mixed-int per-layer policy, retrain | subagent in progress |
| `scripts/compression_sweep_4k_8x.py` (subagent building) | compression-only sweep on existing `final_model.pt`, no retrain | subagent in progress |

The compression-only sweep follows the FINISH_LINE_HANDOFF rule: "use compression-only against a saved final_model.pt" instead of retraining to test compression policy.

## Files state (local)

**4xgpu silos** (4-GPU mechanics-proxy, wallclock 1200, world_size=4):
| silo | file | smoked | in bundle |
|---|---|---|---|
| `control_NOCHANGE/` | `train_gpt_4xgpu.py` | ✓ | ✓ |
| `midnight/` | `train_gpt_4xgpu.py` | ✓ | ✓ |
| `rascal_loop/` | `train_gpt_loop_4xgpu.py` | ✓ | ✓ |
| `4k_vocab_rascal/` | `train_gpt_4K_4xgpu.py` | ✓ | ✓ |
| `4k_vocab_rascal_loop/` | `train_gpt_4K_loop_4xgpu.py` | ✓ | ✓ |
| `4k_vocab_midnight/` | `train_gpt_4K_4xgpu.py` | ✓ | pending |

**8xgpu silos** (canonical 8xH100, wallclock 600, world_size=8):
| silo | file | run history |
|---|---|---|
| `control_NOCHANGE/` | `train_gpt_8xgpu.py` | not run |
| `midnight/` | `train_gpt_8xgpu.py` | not run |
| `rascal_loop/` | `train_gpt_loop_8xgpu.py` | RAN — bust (size + quant collapse) |
| `4k_vocab_rascal/` | `train_gpt_4K_8xgpu.py` | **RAN — PAYDIRT (sliding 0.8672, 17.77 MB)** |
| `4k_vocab_rascal_10l/` | `train_gpt_4K_10L_8xgpu.py` | pushed, ready to fire |
| `4k_vocab_rascal_10l_brotli_mixed/` | (subagent building) | — |
| `mom95_mlp45/` | `train_gpt_8xgpu.py` | RAN — bust (underpowered) |
| `mom95_mlp45_kernel/` | `train_gpt_8xgpu.py` | not run (would just be a faster bust) |
| `whale_donor/` | `train_gpt.py` (PR 1493 body untouched) + `run_whale.sh` | not yet fired |

**Tok/s logging added to all 12 rascal-lineage train_gpt files** as of 2026-04-27 — all future runs include `tok/s:<N>` in step logs.

## Hyperparameter facts baked across the silos
- All 4xgpu files: `MAX_WALLCLOCK_SECONDS=1200`, `world_size != 4` raise.
- All 8xgpu files: `MAX_WALLCLOCK_SECONDS=600`, `world_size != 8` raise.
- All rascal-lineage: `SEED=444`, `LOADER_MODE=coprime`, `COPRIME_SHARDS_PER_BATCH=1` (canonical, do NOT max).
- `COPRIME_MAX_LOADED_SHARDS`: 80 for 1k vocab files, **143 for all 4k files** (matches sp4096 shard count).
- All 4k files: VOCAB_SIZE=4096, DATA_PATH=fineweb10B_sp4096, TOKENIZER_PATH=fineweb_4096_bpe.model.
- 4K_8xgpu specifically: NUM_LAYERS=11. The 10L variant has NUM_LAYERS=10.
- whale_donor: PR 1493 body verbatim (md5 `ca24f344...`, 48,583 bytes) + `run_whale.sh` exporting tracked_env.sh hyperparameters (12L, MODEL_DIM=576, NUM_LOOPS=2, TTT enabled, etc.)

## Hard rules confirmed today
- **No CONDITION blocks** anywhere — defaults baked into Hyperparameters class only.
- **PER_BATCH stays at canonical 1** — only MAX_LOADED gets maxed.
- **MAX_LOADED = on-disk shard count of the actual dataset**.
- **The file IS the experiment** — silo per variant, no shared imports, no bad rewrites.
- **Score-first TTT is leaderboard-compliant** — merged-PR techniques are fair game.
- **NEW (FINISH_LINE_HANDOFF)**: "use compression-only against a saved final_model.pt" — don't retrain to test compression/quant policy.

## What changed everything
The 4k 8x paydirt overrode the 11-day research collate's pessimistic framing. Two things broke prior 4k attempts:
1. Stale dataset state — only 17 sp4096 shards on the dead 4xH100 pod (vs 143 in Jordan's local master at `/home/frosty40/parameter-golf-lab/data/datasets/fineweb10B_sp4096/`).
2. `MAX_LOADED_SHARDS=17` baked into the 4k files based on that stale state, causing I/O page-fault thrash.

Once corrected (rsynced full 143-shard master to pod + bumped MAX_LOADED to 143 + ran full 600s canonical horizon), the 4k vocab axis went from "null finding" to "biggest signal in the lab."
