# 2026-04-30 CaseOps 6x Mechanics Pack

Run label: `mechanics_1x` for each lane.

This pack is for a 6-GPU pod split into six independent 1x H100 mechanics
experiments. These are not official 8x standard leaderboard runs.

All lanes use the SP10240 CaseOps dataset:

- `/workspace/SOTA_FINAL/data/datasets/fineweb10B_sp10240_caseops/datasets`
- tokenizer vocab `10240`
- CaseOps ids `[4, 5, 6, 7]`

## Lanes

- GPU0: `2026-04-30_caseops6_lane0_repro_mlp4_late035_1x`
  - PR1855/SP10240 CaseOps repro mechanics proxy, MLP4, loop2, loop on `0.35`.
- GPU1: `2026-04-30_caseops6_lane1_mlp375_late050_1x`
  - Same stack, MLP3.75, loop2, loop on `0.50`.
- GPU2: `2026-04-30_caseops6_lane2_mlp385_late050_1x`
  - MLP3.85 midpoint check, loop2, loop on `0.50`.
- GPU3: `2026-04-30_caseops6_lane3_mlp375_late060_1x`
  - MLP3.75 with later loop pressure, loop2, loop on `0.60`.
- GPU4: `2026-04-30_caseops6_lane4_mlp4_late050_1x`
  - MLP4 with later loop pressure, loop2, loop on `0.50`.
- GPU5: `2026-04-30_caseops6_lane5_mlp375_loop3_late060_1x`
  - Risk lane: MLP3.75, loop3, loop on `0.60`.

## Command

```bash
cd /workspace/sota_rascal/legs/2026-04-30_caseops6_mechanics_pack
chmod +x launch_all_6x.sh
./launch_all_6x.sh
```

Watch all lane logs:

```bash
tail -n +1 -f /workspace/sota_rascal/legs/2026-04-30_caseops6_lane*/logs/*.txt
```
