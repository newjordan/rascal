# Mikey 1.1

Mikey with the eval fix from `AUDIT_Mikey_Raphe_2026-04-27.md` applied.

The original Mikey ran `eval_val_sliding` on the in-memory full-precision `base_model`
after deleting the decompressed quantized `eval_model`. That meant the headline
`final_sliding_window_exact` was a pre-quantized score, not an artifact-sliding score.

This 1.1 variant moves the `del eval_model` past the sliding/ngram block and passes
`eval_model` (the decompressed `final_model.int6.ptz` round-trip) into every post-quant
sliding eval call. Same training recipe, same artifact format, same legal-size budget.

## What changed vs Mikey

- `eval_val_sliding(... base_model ...)` → `eval_val_sliding(... eval_model ...)` (3 sites)
- `del eval_model, deq_state, quant_state, sd_cpu` moved to after the final eval block
- run / condition labels: `mikey_8x_seed444` → `mikey_1p1_8x_seed444`

## Pending

Headline `val_bpb` from the original Mikey runs is the base_model sliding score and is
**not** the corrected metric. Re-run on 8xH100 to obtain the artifact-sliding score;
expected direction is upward (worse) by ~0.04 bpb based on the audit's seed-300 spot
check.

| seed | original sliding BPB (base_model) | int6 round-trip BPB | bytes |
|------|-----------------------------------|---------------------|-------|
| 42   | 0.86503709                        | 0.90824024          | 15,639,737 |
| 300  | 0.86698133                        | 0.91128978          | 15,594,375 |
| 444  | 0.86441066                        | 0.90971723          | 15,653,512 |
| **mean** | **0.86547636**                | **0.90974908**      | legal |

```
torchrun --standalone --nproc_per_node=8 Mikey_1.1/train_gpt_8xgpu.py
```
