# Kernel and Loop Mining Brief - 2026-04-29

No further kernel work should launch from chat memory. Any kernel/loop runner must cite the exact condition artifact, source hash, tokenizer/data, GPU count, wallclock, compile policy, and comparator.

## Evidence Chain

### D64 / T2048 Kernel Work

- `2026-04-16_whale_bwd_ablations`: pure Triton whale reached near FA3 parity at `(B=4,T=2048,H=8,KV=4,D=64)` by using inline delta. It lost badly at T>=4096, so the short-context win does not transfer to long context.
- `2026-04-17_whale_bwd_variant_sweep`: `fused_delta_tma` was the best backward variant at D64/T2048. Whale won forward by 14-17%, lost backward by about 9.5%, and was only about 3.6% slower fwd+bwd before later autotune.
- `2026-04-17_whale_dkdv_wide_autotune_t2048` plus `2026-04-17_whale_dq_wide_autotune_t2048`: DKDV and DQ tuning crossed under FA3. Early heavy validation used `DKDV=128,128,8,3`, `DQ=128,64,8,5`, `DQ_MAXNREG=224`.
- `2026-04-18_whale_cross_gpu_validation_prep_t2048/winner_anchor.json`: the multi-GPU blessed D64 anchor is `family_id=primary`, `DQ=128,64,8,5`, `DKDV=128,64,8,3`, `DKDV_MAXNREG=256`, `QMR=1984`, with whale `0.28777ms` vs FA3 `0.30734ms`. This overrides raw single-pass table picking.

### D72 / Loop Transition Work

- `2026-04-21_whale_d72_repair_t2048`: root cause was explicit: the D64 winner was not donor-shape-blessed. Whale donor uses `(2,2048,8,4,72)`.
- `2026-04-26_d72_hybrid_kernel_loop_pair_summary.tsv`: primary hybrid loop-off beat loop-on on 1200s wallclock: loop-off step 1270 / bpb 1.1420, loop-on step 1006 / bpb 1.1475. Loop-on looked better per step but lost wallclock.
- `2026-04-26_d72_kernel_loop_transition_summary.md`: the old `x180` dynamic path was too slow. Static hybrid was better but still behind FA3. The first useful D72 custom-forward result was `tail16dot_128_64_w4_s4`.
- Same summary: matched 100-step resume from loop-onset checkpoint put tail16dot at `797720 tok/s` vs FA3 clean100 at `727722 tok/s`. This is the strongest loop-transition kernel signal, but only for that checkpoint-resume diagnostic condition.

## Failure Modes Not To Repeat

- Do not reuse `hybrid_x180` as a promotion path; it completed but was slow.
- Do not use `hybrid_warmstart180` as a fix; it failed with `fail_rc_1` after CPU spin at first validation.
- Do not tighten global grad clip to `0.20`; it was worse at every breakpoint.
- Do not chase tail8 or persistent-forward D72 variants; the summary rejects both.
- Do not carry D64 kernel conclusions into D72 or vice versa without a condition label.
- Do not use CUDA persistent-L2 window as a prep patch; the prior ranked draft marked it no-go.

## Condition-Locked Next Rule

For 4k Rascal/D64 work, the only defensible kernel source is the D64 multi-GPU anchor plus a fresh condition artifact. For 8k/Mikey/D72 loop work, the only defensible kernel source is the D72 tail16dot loop-transition diagnostic family. A proper Rascal-with-looping test should first run FA3/control loop topology cleanly; kernel integration comes after that comparator exists.
