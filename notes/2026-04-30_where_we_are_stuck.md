# Where We Are Stuck - 2026-04-30

Snapshot: `2026-04-30T22:08:47Z`

This note is for humans and agents picking up the Parameter Golf push after the
PR1855 pivot. It summarizes the current wall, the real data from today's runs,
and which ideas should or should not be promoted.

## Target

The active target is still the accepted PR1855 SP8192 CaseOps stack:

```text
SP8192 CaseOps + LQER + pergroup compression + phased LoRA TTT
accepted seed42 post-phased-TTT: 1.05989454 BPB
accepted 3-seed mean:          1.06107587 BPB
```

Any new candidate must either beat that stack or clearly produce a frozen
artifact that eval-time work can push below it.

## Current Stuck Point

We are not square one, but we are stuck between three failure modes:

1. The 10k/SP10240 CaseOps lane can make legal, good-looking models, but the
   best clean legal 10k final scores are still around `1.063-1.064`, not under
   the accepted PR1855 target.
2. Direct loop/body edits on the SP8192 PR1855 stack are currently damaging
   neural quality. The latest MLP shrink test was legal but too weak:
   `post-EMA 1.06993111`, `quant 1.07890818`.
3. Eval-time TTT chunk/phase tuning has not been enough by itself. It produces
   attractive local chunk numbers, but the final aggregate stays above the
   record.

The hard read: we need a stronger neural body or a loop-shape change that does
not damage the accepted PR1855 body before quant/TTT.

## Confirmed Data

| Run / lane | Type | Key change | Post-EMA BPB | Quant BPB | TTT BPB | Size bytes | Read |
|---|---:|---|---:|---:|---:|---:|---|
| PR1855 accepted SP8192 | record target | accepted stack | n/a | n/a | `1.05989454` | legal | Target to beat |
| `2026-04-30_pr1855_sp10240_caseops_repro_8x` | standard 8x | SP10240 CaseOps 11L MLP4 | `1.06351951` | `1.07211776` | n/a | `16450237` | Good neural, over cap |
| `2026-04-30_pr1855_sp10240_caseops_repro_lqertop2_8x` | standard 8x | LQER top3 -> top2 | `1.06384143` | n/a | n/a | `16441071` | Still over cap |
| `2026-04-30_pr1855_sp10240_caseops_mlp375_late050_8x` | standard 8x | 11L MLP3.75 late050 | `1.06788148` | `1.07681080` | `1.06429180` | `15818165` | Legal, not strong enough |
| `2026-04-30_pr1855_sp10240_caseops_repro_embed6_8x` | standard 8x | embed6 size fix | `1.06337337` | `1.07659480` | `1.06374960` | `15789033` | Best clean legal 10k body, still short |
| `caseops4_gpu1_mlp375_late045_dup_1x` + 4x TTT | mechanics proxy | side4 MLP3.75 late045 | `1.06417834` | `1.07288689` | `1.06022223` | `15812190` | High-water clue, not clean standard 8x |
| `2026-04-30_ttteval_sp10240_repro_embed6_chunk64_phase3_4x` | mechanics 4x eval | chunk64/phase3 on frozen embed6 | n/a | n/a | `1.06360109` | n/a | TTT knob alone not enough |
| `2026-04-30_ttteval_sp10240_repro_embed6_chunk48_phase5_2x` | mechanics 2x eval | phase5 on frozen embed6 | n/a | n/a | `1.06334774` | n/a | Better, still not enough |
| `2026-04-30_pr1855_sp10240_caseops_mlp4_late050_h1_hotloop_8x` | standard 8x | global int5 + hot-loop attention int6 | `1.06509411` | `1.09777040` | n/a | `12733990` | Global int5 is toxic |
| `2026-04-30_pr1855_sp10240_caseops_mlp4_late050_h2_allattn6_mlp5_4x` | mechanics 4x | all attention int6, MLP int5 | `1.06434618` | `1.09113614` | `1.07729729` | `13514468` | Still quant-fail |
| `2026-04-30_pr1855_sp8192_lqer_smeargate_loopoff40_8x` | standard 8x | hard loop-off final 40s | `1.09171737` | n/a | n/a | n/a | Dead; hard loop-off damages neural |
| `2026-04-30_pr1855_sp8192_caseops_mlp375_late045_8x` | standard 8x | MLP4 -> MLP3.75, loop 0.45 | `1.06993111` | `1.07890818` | n/a | `15288021` | Legal but body too weak |
| `2026-04-30_pr1855_sp8192_lqer_smeargate_loop_late050_4x_proxy` | mechanics 4x | accepted body, loop 0.50 | pending | pending | pending | pending | Running value proxy |

## What We Learned

### Transferable

- **SP8192 PR1855 remains the main scoring lane.** The accepted target is still
  the only proven sub-`1.061` path.
- **MLP shrink is not the SP8192 answer so far.** MLP3.75 late045 made size easy
  but lost too much neural quality on the accepted 8k stack.
- **Hard loop-off is dead.** Turning loops off near the end hurt the neural body
  before quant/TTT. Do not promote more hard-disable variants without a very
  specific new reason.
- **Loop shape is still plausible, but it must preserve body quality.** The
  useful loop axis is likely start/ramp/pressure, not hard shutoff.
- **Broad int5 policies are bad.** H1/H2 saved bytes but quant damage dominated.
  Mixed precision must be surgical if revisited.
- **TTT phase/chunk tuning is a finisher, not a rescue.** Phase5 and chunk64 did
  not move a frozen artifact far enough.

### Not Transferable Yet

- The `side4` `1.06022223` TTT result is a valuable clue, but it came through
  nonstandard mechanics/proxy training and did not reproduce cleanly as an 8x
  standard run. Do not treat it as the base score.
- SP10240/10k vocabulary is not dead, but it has not beaten accepted PR1855.
  Keep it as a side lane unless a new 10k body produces a legal frozen artifact
  under about `1.061`.

## Current Useful Pod Split

- **8x pod:** only standard record-lane tests. Current best next test is the
  accepted-body timing-only run:

  ```bash
  cd /workspace/sota_rascal/legs/2026-04-30_pr1855_sp8192_lqer_smeargate_loop_late050_8x
  ./launch_8x.sh
  tail -f logs/pr1855_sp8192_lqer_smeargate_loop_late050_8x_seed42.txt
  ```

- **4x pod:** value proxy on SP8192 loop timing/shape. Current running proxy:

  ```bash
  cd /workspace/sota_rascal/legs/2026-04-30_pr1855_sp8192_lqer_smeargate_loop_late050_4x_proxy
  tail -f logs/pr1855_sp8192_lqer_smeargate_loop_late050_4x_proxy_seed42.txt
  ```

- **2x/Claude pod:** length-aware or stratified TTT work. The 8k dataset is
  present on the Claude pod (`train=80`, `val=5`, `val_bytes=5`), but a good
  frozen 8k artifact is not available yet. Use the frozen 10k `repro_embed6`
  artifact for code development, then swap in a strong 8k artifact if one lands.

## Ranked Next Hypotheses

1. **SP8192 timing-only late050.** Keep accepted body/MLP/quant/TTT fixed and
   only delay loop activation from `0.35` to `0.50`. This isolates loop timing
   without repeating the MLP shrink miss.
2. **Loop pressure/ramp, not loop off.** If late050 is neutral or slightly good,
   test a gradual loop ramp or lower loop pressure. This should happen on 4x
   first unless the 8x result is very promising.
3. **Length-aware TTT.** The logs show local low-BPB regions, but the global
   average drifts upward on short-doc tail regions. The next eval-time idea is
   stratified or length-aware prefix/order selection, not more blind phase count.
4. **Exact PR1855 reproduction/seed validation.** If loop timing fails, stop
   body/loop edits on 8x and return to reproducing accepted PR1855 conditions.

## Do Not Spend 8x On

- More hard loop-off.
- More broad global-int5 mixed-precision variants.
- More MLP shrink on SP8192 unless the late050 accepted-body test changes the
  read.
- TTT-only sweeps as standard 8x training runs.
- 12L 10k depth unless a new quant policy exists; the depth lane is closed for
  now.

## Agent Instructions

Before preparing any new run:

1. Open the relevant `CONDITION.md` and `run.py`.
2. Label 4x/2x as mechanics proxies.
3. Record metric name exactly: post-EMA, diagnostic quantized, phased TTT, or
   another specific eval.
4. Do not compare proxy scores to accepted PR1855 as official results.
5. Do not launch from a condition that is only remembered in chat.
