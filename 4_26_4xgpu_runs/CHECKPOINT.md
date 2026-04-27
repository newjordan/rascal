# Checkpoint — 2026-04-26 → 2026-04-27 (NEW SOTA SHIPPED)

## TL;DR

Two leaderboard PRs live as of session end:

- **Mikey** (PR `openai/parameter-golf#1848`) — **NEW SOTA candidate**. 12L brotli+mixed at 4k vocab. **3-seed mean val_bpb 0.86548 (sliding), std 0.00109, 15.65 MB legal.** Beats PR 1120's 1.10987 by **0.245 bpb**.
- **Raphe** (PR `openai/parameter-golf#1846`) — predecessor. 10L brotli+mixed at 4k vocab. 3-seed mean val_bpb 0.87206, std 0.00052, 13.49 MB legal.

The blessed-whale donor (1.09561 quant-TTT, prior best legal artifact) is dethroned.

## Final depth scan (4k vocab + brotli + mixed-int recipe, seed 444 baseline)

| layers | val_bpb (sliding) | bytes | legal | submission |
|---|---|---|---|---|
| 10L | 0.87168 (seed 444); 0.87206 (3-seed mean) | 13,487,656 | ✓ | Raphe (PR #1846) |
| 11L | 0.86718 | 17,766,043 | ✗ | original paydirt (uniform int6+zstd) |
| 12L | 0.86441 (seed 444); **0.86548 (3-seed mean)** | 15,653,512 | ✓ | **Mikey (PR #1848) — SOTA** |
| 13L | 0.85957 | 16,765,032 | ✗ +765 KB | quality probe (file pushed, ran once seed 444) |
| 16L | (file pushed, not run) | — | likely ✗ | quality ceiling probe |

Quality monotonically improves with depth. Bytes also monotonically increase. Sweet spot = largest legal depth = 12L (Mikey).

## Pods used today
- **8x H100 SXM** (instance `35657525`, ssh `34.69.145.155:3159`) — running for the bulk of the work, **shut down** at session end. ~$12.36/hr. Total session compute spent: roughly 3-4 hours of 8x time. Worth every dollar.
- **1x H100 SXM** (instance `35082273`) — smoke pod, also shut down.
- **4x H100 SXM** (instance `34405931`) — exited before session, source of stale "17 sp4096 shards" data that initially poisoned the 4k axis.

## Local artifacts preserved
- `Raphe/artifacts/final_model.pt` (96 MB) + `final_model.int6.ptz` (13 MB) — paydirt run output
- `Mikey/artifacts/final_model.pt` (115 MB) + `final_model.int6.ptz` (15 MB) — most recent run on pod (seed 300)
- `4_26_4xgpu_runs/_run_logs/` — 11+ logs covering every run from the session (paydirt, busts, all 6 SOTA-candidate seed runs)

## Branches
- `newjordan/rascal:Rascal_lab` — base lab branch with all silos
- `newjordan/rascal:raphe` — clean single-commit Raphe submission (commit `8fdd3a7`)
- `newjordan/rascal:mikey` — clean single-commit Mikey submission (commit `fc52e36`, technique_summary blank)
- `newjordan/parameter-golf-1:submission/raphe` — fork branch for PR #1846
- `newjordan/parameter-golf-1:submission/mikey` — fork branch for PR #1848 (squashed to single commit `5626bbe`)

## Hard rules confirmed today
- **No CONDITION blocks** anywhere — defaults baked into Hyperparameters class only.
- **PER_BATCH stays at canonical 1** — only MAX_LOADED gets maxed.
- **MAX_LOADED = actual on-disk shard count** (sp4096 = 143, NOT the stale 17 from a prior pod state).
- **The file IS the experiment** — silo per variant, no shared imports.
- **Score-first TTT is leaderboard-compliant** (none of the SOTA submissions use it).
- **FINISH_LINE_HANDOFF rule**: use compression-only against saved final_model.pt — don't retrain to test compression policy.

## What's queued / ready but not yet run
- `4k_vocab_rascal_8l_brotli/`, `4k_vocab_rascal_9l_brotli/` — backup salvage variants if Mikey/Raphe needed shrinking
- `4k_vocab_rascal_16l_brotli_mixed/` — capacity ceiling probe (likely over cap)
- `scripts/compression_sweep_4k_8x.py` — offline compression sweep on Mikey's saved checkpoint (cheap, ~2min on 8x)
- `whale_donor/` — PR 1493 body verbatim (the prior donor's source); not relevant now that Mikey beat it

## Tournament focus going forward
See `~/.claude/projects/-home-frosty40-sota-rascal/memory/project_tournament_focus.md` for the ranked next-direction list. Top candidate: land **13L legal** by extending mixed-int to attention banks (free ~500 KB; would shave ~0.006 bpb off Mikey).

## What changed everything
The 4k vocab axis was misjudged as null in the 11-day research collate (best at 1.7448 short-horizon = "well above the 1.10 worth-discussing gate"). That dismissal was based on:
1. Stale dataset state on dead pod (only 17 sp4096 shards on disk, vs 143 in master)
2. `MAX_LOADED_SHARDS=17` baked from that stale state → I/O page-fault thrash
3. Short-horizon evaluation

With those three corrected (rsynced full 143-shard master to fresh pod, bumped MAX_LOADED to 143, ran full 600s canonical horizon), 4k vocab went from "null finding" to **biggest signal in the lab — record-shattering by 0.245 bpb.**
