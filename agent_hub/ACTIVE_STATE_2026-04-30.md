# Active State - 2026-04-30 CaseOps / PR1855 SP10240

Updated snapshot: `2026-04-30T15:54Z`.

## Strategic Read

The current main lane is PR1855 CaseOps/LQER/pergroup/phased-TTT adapted to
SP10240. The 10L MLP4 layer drop solved size too aggressively:

```text
2026-04-30_pr1855_sp10240_caseops_repro_10l_mlp4_8x
post_ema_bpb: 1.07121277
compressed_total_bytes: 15201001
free_bytes_vs_cap: 798999
```

Conclusion: do not shrink further. Spend bytes back into neural quality.

Best observed mechanics clue so far is MLP4 with later loop activation:

```text
caseops6_lane4_mlp4_late050_1x post_ema_bpb: 1.06275866
caseops6_gpu2_pr1855_repro_mlp4_late035_embed6_1x post_ema_bpb: 1.06298225
caseops6_lane0_repro_mlp4_late035_1x post_ema_bpb: 1.06318199
caseops6_lane1_mlp375_late050_1x post_ema_bpb: 1.06384344
caseops6_lane3_mlp375_late060_1x post_ema_bpb: 1.06474948
caseops6_lane5_mlp375_loop3_late060_1x post_ema_bpb: 1.06700154
```

Interpretation: loop3 is weak here. Late050 is worth testing on 8x. The next
byte-spend axis is MLP width, not dim. Avoid dim528/dim536: FA3 requires
attention head size multiple of 8, and those make head_dim 66/67 with 8 heads.
Dim576 is legal but a much larger step/size shock.

## Current 8x Pod

SSH:

```bash
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/tmp/codex_runpod_known_hosts -o IdentitiesOnly=yes -i /home/frosty40/.ssh/id_ed25519_apollo -p 15756 root@103.207.149.83
```

Current live run at snapshot:

```text
/workspace/sota_rascal/legs/2026-04-30_pr1855_sp10240_caseops_repro_10l_mlp4_late050_8x
```

Watch:

```bash
tail -f /workspace/sota_rascal/legs/2026-04-30_pr1855_sp10240_caseops_repro_10l_mlp4_late050_8x/logs/pr1855_sp10240_caseops_repro_10l_mlp4_late050_8x_seed42.txt
```

At snapshot this run had reached:

```text
3000/20000 train_loss: 2.6677 train_time: 4.4m tok/s: 8920225
```

## Next 8x Run If Current Misses

Prepared on the 8x pod and locally:

```bash
cd /workspace/sota_rascal/legs/2026-04-30_pr1855_sp10240_caseops_repro_10l_mlp425_late050_8x
./launch_8x.sh
tail -f logs/pr1855_sp10240_caseops_repro_10l_mlp425_late050_8x_seed42.txt
```

Hypothesis: 10L MLP4 left about 0.8MB unused, so raise MLP to 4.25 while
keeping 10L and late050 loop timing.

## TTT 4x Pod

SSH:

```bash
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/tmp/codex_runpod_known_hosts -o IdentitiesOnly=yes -i /home/frosty40/.ssh/id_ed25519_apollo -p 19902 root@103.207.149.104
```

Known under-cap parent artifact:

```text
/workspace/sota_rascal/legs/2026-04-30_pr1855_sp10240_caseops_mlp375_late050_8x/logs/final_model.int6.ptz
```

Completed eval-only baseline:

```text
quantized_ttt_phased val_bpb: 1.06416311
eval_time: 790.631s
condition: 4x mechanics, same artifact, chunk48 baseline
```

Current chunk test at snapshot:

```text
/workspace/sota_rascal/legs/2026-04-30_ttteval_sp10240_mlp375_late050_chunk32_4x
```

Watch:

```bash
tail -f /workspace/sota_rascal/legs/2026-04-30_ttteval_sp10240_mlp375_late050_chunk32_4x/logs/ttteval_sp10240_mlp375_late050_chunk32_4x_seed444.txt
```

Next staged TTT fallback:

```bash
cd /workspace/sota_rascal/legs/2026-04-30_ttteval_sp10240_mlp375_late050_chunk64_4x
./launch_4x_ttt_eval.sh
tail -f logs/ttteval_sp10240_mlp375_late050_chunk64_4x_seed444.txt
```

## Cut Rules

- If `Total submission size ... > 16000000`, cut immediately.
- If post-EMA is much worse than `1.079`, cut without paying TTT.
- On this CaseOps lane, a useful 8x candidate should ideally show post-EMA
  near `1.06x`, fit under cap, and not quantize above the low `1.07x`.
- Do not cut from early train loss alone.
- Do not compare 1x/4x/6x mechanics results as official 8x results.

## GitHub / Pod Sync

Remote:

```text
https://github.com/newjordan/rascal
```

Current local branch at this snapshot:

```text
matrix-lane-20260430
```

For fresh pods, clone or pull the branch, then use the exact `legs/<test>`
directory named above. Critical settings are in `run.py`, not shell overrides.

