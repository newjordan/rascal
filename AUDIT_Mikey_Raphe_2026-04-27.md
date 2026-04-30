# Mikey / Raphe Audit - 2026-04-27

## Verdict

Mikey and Raphe are not clean as currently reported.

The submitted headline scores use `final_sliding_window_exact`, but the code computes that score on the in-memory full precision/EMA `base_model`, after deleting the decompressed quantized artifact model. Under the Parameter Golf artifact rule, the final reported score should be based on the model represented by the counted artifact bytes.

This does not look like validation leakage or a malicious cheat. It looks like score attribution drift: legal artifact size is reported correctly, but the reported BPB is the pre-quantized sliding score rather than an artifact round-trip score.

## Evidence

Official rule surface:

- The 16 MB artifact is code bytes plus compressed model bytes.
- Evaluation is compression on FineWeb validation.
- Validation data cannot be accessed during training, except legal score-first test-time training after tokens are scored.

Local code path:

- `Mikey/train_gpt_8xgpu.py` and `Raphe/train_gpt_8xgpu.py` build train data from `fineweb_train_*.bin` and validation from `fineweb_val_*.bin`.
- GPTQ calibration uses `args.train_files`, not validation.
- `NGRAM_EVAL_ORDER=0` in all six submitted logs.
- The final artifact is written to `final_model.int6.ptz`, loaded back into `eval_model`, and evaluated once by `eval_val`.
- Immediately after `final_int6_roundtrip_exact`, the code deletes `eval_model`.
- The later `final_sliding_window_exact` call evaluates `base_model`, not the decompressed artifact model.

Relevant code sequence:

```text
eval_model.load_state_dict(deq_state, strict=True)
q_val_loss, q_val_bpb = eval_val(... eval_model ...)
log final_int6_roundtrip_exact
del eval_model, deq_state, quant_state, sd_cpu
...
sw_val_loss, sw_val_bpb = eval_val_sliding(... base_model ...)
log final_sliding_window_exact
```

## Submitted vs Artifact Round-Trip Scores

Mikey:

| seed | submitted sliding BPB | artifact round-trip BPB | artifact bytes incl code |
|---|---:|---:|---:|
| 42 | 0.86503709 | 0.90824024 | 15,639,737 |
| 300 | 0.86698133 | 0.91128978 | 15,594,375 |
| 444 | 0.86441066 | 0.90971723 | 15,653,512 |
| mean | 0.86547636 | 0.90974908 | legal |

Raphe:

| seed | submitted sliding BPB | artifact round-trip BPB | artifact bytes incl code |
|---|---:|---:|---:|
| 42 | 0.87280270 | 0.91748677 | 13,492,522 |
| 300 | 0.87170361 | 0.91993641 | 13,475,412 |
| 444 | 0.87168317 | 0.91768479 | 13,487,656 |
| mean | 0.87206316 | 0.91836932 | legal |

Important: `final_int6_roundtrip_exact` is not the same evaluation metric as `final_sliding_window_exact`; it is a conservative artifact sanity score, not the exact corrected leaderboard score. The exact repair is artifact round-trip sliding eval.

## Local Artifact-Sliding Spot Check

I ran a local one-GPU spot check on Mikey seed 300 artifacts:

- full precision `final_model.pt`, `eval_val_sliding`, first 6,144 validation tokens: `1.17598585` BPB
- decompressed `final_model.int6.ptz`, `eval_val_sliding`, same tokens: `1.20270942` BPB
- delta: `+0.02672358` BPB against the quantized artifact

This proves the model object matters for sliding eval. It does not produce the full validation score; full artifact-sliding still needs to be run on 8xH100 or an equivalent acceptable environment.

## Other Checks

- No validation/train file path crossover found in the submitted code.
- No n-gram cache was active in submitted logs.
- The three submitted seeds match the logged `SEED` values.
- All six run logs stop at the wallclock cap around 600 seconds.
- The tokenizer byte-count helper was spot-checked against the local `fineweb10B_sp4096` validation shard and matched decoded UTF-8 bytes for a 200k-token sample.
- PR #1848 and PR #1846 were open with no review comments at audit time.

## Required Fix

Do not report `final_sliding_window_exact` as the record score unless it is computed on the decompressed artifact model.

Two acceptable repair paths:

1. Conservative immediate correction: update README/submission JSON to report the `final_int6_roundtrip_exact` means as a non-sliding artifact sanity score, not as the sliding leaderboard metric:
   - Mikey: `0.90974908`
   - Raphe: `0.91836932`

2. Correct leaderboard repair: run a new artifact round-trip sliding eval by calling `eval_val_sliding(... eval_model ...)` before deleting `eval_model`, then report those three seed means. This will likely be worse than the current full-precision sliding scores, but it must be measured.

## Bottom Line

Mikey/Raphe may still represent a real record-class result, but the public PR numbers `0.86548` and `0.87206` are full-precision `base_model` sliding scores, not artifact-sliding scores from the submitted logs.
