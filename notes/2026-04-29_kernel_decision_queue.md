# Kernel Decision Queue - 2026-04-29

Purpose: keep kernel work evidence-driven and prevent D64, D72, loop topology, and training-score evidence from being mixed.

## Current Shape Map

- Active 4k Rascal/Raphe run: D64 attention shape (`MODEL_DIM=512`, `NUM_HEADS=8`, `NUM_KV_HEADS=4`, `T=2048`). With `TRAIN_BATCH_TOKENS=786432`, `WORLD_SIZE=2`, and `grad_accum=4`, the real per-rank training shape is `B=48,T=2048,H=8,KV=4,D=64`.
- Whale donor D72 work: `MODEL_DIM=576`, `NUM_HEADS=8`, shape `B=48,T=2048,H=8,KV=4,D=72`.
- PR1493/Mikey-style 8k loop body in this repo is D64 unless changed to a 576-wide body.

## Promote

### 1. D64 real-shape microbench for Rascal

Run class: kernel microbench only, not training.

Condition source to clone from: D64 multi-GPU winner anchor in `/home/frosty40/SOTA_FINAL/legs/2026-04-18_whale_cross_gpu_validation_prep_t2048/winner_anchor.json`.

Kernel config to test first:

- `family_id=primary`
- `DQ_TILE=128,64,8,5`
- `DKDV_TILE=128,64,8,3`
- `DKDV_MAXNREG=256`
- `QMR=1984`

Comparator: FA3 on the exact Rascal training shape `B=48,T=2048,H=8,KV=4,D=64`.

Decision gate: only integrate into a Rascal runner if the real-shape D64 microbench is green. The old D64 winner was proven on benchmark shapes, not this exact per-rank train shape.

### 2. D72 tail16dot for whale-donor loop work

Run class: already promoted from microbench to loop-transition diagnostics.

Condition source: `/home/frosty40/SOTA_FINAL/legs/2026-04-26_d72_split64_tail16dot_loop_transition_probe100/resolved_config.json`.

Kernel config:

- `fwd_variant=d72_split64_tail16dot`
- `fwd_d72_config=128,64,4,4`
- `hybrid_backend=fa3_bwd`
- `kernel_source_sha256=4d8340fc2ed9a656527d25bc8fbd7dfddc691e9ece287ff35017c9b0ee4d26f8`

Evidence: 100-step loop-transition probe reached `797720 tok/s` versus FA3 same-horizon control at `727722 tok/s`. This is valid for D72 whale donor work, not automatically for D64 Rascal.

## Hold Until Comparator Exists

### Rascal-loop kernel integration

Do not combine fresh loop topology and custom kernel in the same first run. First run the FA3/control Rascal-loop smoke in `legs/2026-04-29_rascal_sp4096_loop_1x_smoke`. If that passes, the next kernel step is a D64 real-shape microbench, then a short 1x kernel training smoke.

### Mikey III 8k loop kernel work

Do not launch until the SP8192 dataset is complete and the model shape is confirmed. If it remains `MODEL_DIM=512`, use the D64 queue. If it becomes `MODEL_DIM=576`, use the D72 queue.

## Reject / Do Not Repeat

- `hybrid_x180`: too slow and later produced a bad long result.
- `hybrid_warmstart180`: failed with `fail_rc_1` after CPU spin at first validation.
- `gradclip020`: worse at every breakpoint.
- D72 `tail8`: correct-ish but about `2.25x` FA3; reject.
- D72 persistent forward: slower than default; reject.
- Long-T D64 whale at T>=4096: prior evidence shows large losses; do not spend current tournament time here.
- CUDA persistent-L2 window: prior ranked draft marked it no-go for prep.

## Immediate Queue

1. Finish/watch active 4k MLP35 2x run; do not disturb GPUs 2/3.
2. Use GPU0 for Rascal-loop FA3/control smoke, because it is the missing comparator.
3. Prepare D64 real-shape kernel microbench for `B=48,T=2048,H=8,KV=4,D=64`.
4. If D64 real-shape microbench is green, prepare a 1x kernel training smoke.
5. Only after a 1x kernel smoke is clean, consider a 2x kernelized Rascal run.
