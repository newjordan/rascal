# PR1493 Parity Check - 2026-04-29

## Source Contract

- Accepted artifact: `/home/frosty40/SOTA_FINAL/evidence/pod_pulls/instance_35002131_20260416_1113/legs/2026-04-15_whale_pr1493_faithful/tracked_env.sh`
- Submission JSON: `/home/frosty40/SOTA_FINAL/evidence/pod_pulls/35372056_2026-04-24_all_after_megahelix/SOTA_FINAL/pr1493_upstream_submission.json`
- Accepted score path: `quantized_ttt`, with sliding-window eval and legal score-first TTT enabled.
- Baseline anchor: `quantized_ttt val_bpb:1.08036236` in the faithful x-ray artifact.

## Current SP5200 Run Checked

- Runner: `legs/2026-04-29_mikey_II_v5_sp5200_loop2_11l_8x/run.py`
- Log: `pod_pulls/8x_35002131_20260429_sp5200/mikey_II_v5_sp5200_loop2_11l_8x_seed1337.txt`
- Result: `quantized val_bpb:1.18624269`, `Total submission size quantized+brotli:12633202`

## Parity Result

This was not PR1493 parity. The loop topology matched, but core PR1493 settings drifted.

| Field | PR1493 accepted | Current SP5200 run | Status |
| --- | --- | --- | --- |
| Vocab | `8192` | `5200` | intentional experiment |
| Layers | `11` | `11` | parity |
| Model dim | `512` | `512` | parity |
| Loops | `2`, L3-5, enabled at `0.35` | same | parity |
| Parallel residual | `L7+` | `L7+` | parity |
| MLP | `4.0` | `3.0` | broken |
| QK gain | `5.25` | `5.0` | drift |
| EMA | `0.9965` | `0.0` | drift |
| Warmdown | `0.72` | `0.3` | drift |
| WD | `MUON_WD=0.095` | `0.5` across optimizers | drift |
| Sliding eval | enabled | disabled | drift |
| TTT | enabled, SGD 3 epochs | disabled | drift |

## Next Valid 5200 Test

Built the next SP5200 leg as a parity-repair test, not as a new architecture:

- Local runner: `legs/2026-04-29_mikey_pr1493parity_sp5200_loop2_mlp4_8x/run.py`
- Remote runner: `/workspace/sota_rascal/legs/2026-04-29_mikey_pr1493parity_sp5200_loop2_mlp4_8x/run.py`
- `run.py` SHA256: `e154e571d824ca56f2ac361133cc5300f6486293235ce08b0e25d8a208fd1ea1`

- `NUM_LAYERS=11`
- `MODEL_DIM=512`
- `MLP_MULT=4.0`
- `NUM_LOOPS=2`
- `LOOP_START=3`
- `LOOP_END=5`
- `ENABLE_LOOPING_AT=0.35`
- `PARALLEL_RESIDUAL_START=7`
- `QK_GAIN_INIT=5.25`
- `WARMDOWN_FRAC=0.72`
- `EMA_DECAY=0.9965`
- `MUON_WD=0.095`
- `EMBED_WD=0.085`
- `ADAM_WD=0.02`
- `SLIDING_WINDOW_ENABLED=1`
- `TTT_ENABLED=1`
- `TTT_LR=0.005`
- `TTT_EPOCHS=3`
- `TTT_MOMENTUM=0.9`

Risk: MLP4 may approach the 16,000,000 byte cap, but SP5200 saves embedding bytes versus SP8192, so it is the right parity repair to test before judging 5200 or loops.
