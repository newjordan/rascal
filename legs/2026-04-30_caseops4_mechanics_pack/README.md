# 2026-04-30 CaseOps 4x Mechanics Pack

Run label: `mechanics_1x` for each lane.

This pack is for a 4-GPU pod split into four independent 1x H100 mechanics
experiments. These are not official 8x standard leaderboard runs.

All lanes use the SP10240 CaseOps dataset and the PR1855
LQER/pergroup/phased-TTT stack.

## Lanes

- GPU0: `2026-04-30_caseops4_lane0_mlp395_late050_1x`
  - MLP3.95, loop2, loop on `0.50`.
- GPU1: `2026-04-30_caseops4_lane1_mlp365_late050_1x`
  - Smaller MLP3.65, loop2, loop on `0.50`.
- GPU2: `2026-04-30_caseops4_lane2_mlp375_late045_1x`
  - MLP3.75, earlier loop pressure, loop on `0.45`.
- GPU3: `2026-04-30_caseops4_lane3_mlp375_late050_qk525_1x`
  - MLP3.75, loop on `0.50`, QK gain `5.25`.

## Command

```bash
cd /workspace/sota_rascal/legs/2026-04-30_caseops4_mechanics_pack
chmod +x launch_all_4x.sh
./launch_all_4x.sh
```

Watch all lane logs:

```bash
tail -n +1 -f /workspace/sota_rascal/legs/2026-04-30_caseops4_lane*/logs/*.txt
```
