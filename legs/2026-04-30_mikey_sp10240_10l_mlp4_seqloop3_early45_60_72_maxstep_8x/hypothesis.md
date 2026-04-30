# Hypothesis - 2026-04-30_mikey_sp10240_9l_mlp4_seqloop3_late55_68_80_maxstep_8x

## Purpose

This is the real 9-layer bigger-loop/max-step test.

## Evidence Check

We found local `seqloop3_late55_68_80_9l` runners, but no completed local logs and no copy on the active 8x pod. The active pod only had simpler 9L dim576 loop2 variants. This exact idea has not been cleanly run to a result.

## Parent

`2026-04-29_mikey_sp10240_full124_loop2_late050_9l_mlp4_clip1305_8x`

## Change

- `num_layers=9`
- `model_dim=512`
- `mlp_mult=4.0`
- `num_loops=3`
- staged sequential loop pressure at `0.55`, `0.68`, `0.80`
- `loop_start=3`, `loop_end=5`
- `matrix_bits=6`
- `matrix_clip_sigmas=13.05`
- compile repaired with `compile_fullgraph=False`, `compile_dynamic=True`

## Readout

This tests whether the smaller/faster 9L body can get more useful steps before applying heavier recurrence, then use late loop pressure as the final quality lever. It is a different idea from the simpler 9L dim576 loop2 body.
