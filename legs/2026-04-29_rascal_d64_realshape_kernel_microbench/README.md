# Rascal D64 Real-Shape Kernel Microbench

Run class: `mechanics_1x_kernel_microbench_scout`.

This is the first kernel step for the active 4k Rascal/Raphe body. It does not train and does not read the dataset. It compares `whale_fwd_fa3_bwd` against FA3 on random BF16 tensors at the real per-rank training shape:

`B=48,T=2048,H=8,KV=4,D=64`

Primary source evidence: `/home/frosty40/SOTA_FINAL/legs/2026-04-18_whale_cross_gpu_validation_prep_t2048/winner_anchor.json`.

Primary config:

- `DQ_TILE=128,64,8,5`
- `QMR=1984`
- `DKDV_TILE=128,64,8,3`
- `DKDV_MAXNREG=256`

Command:

```bash
cd /home/frosty40/sota_rascal/legs/2026-04-29_rascal_d64_realshape_kernel_microbench
./launch_gpu0_microbench.sh
```

Decision rule: only integrate this kernel into a Rascal runner if this real-shape microbench is green against FA3.
