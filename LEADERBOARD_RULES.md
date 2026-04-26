# Parameter Golf Rules

Source: https://github.com/openai/parameter-golf/tree/main?tab=readme-ov-file#leaderboard

## Core Constraints

- The goal is the best language model that fits in a 16 MB artifact.
- Leaderboard submissions must train in under 10 minutes on 8xH100s.
- Scoring is based on compression over the FineWeb validation set, measured in bits per byte.

## Submission Policy

- Record submissions are considered in PR creation order.
- A submission needs a statistically meaningful improvement over the current SOTA to be accepted as a leaderboard record.
- If it is interesting but not record-breaking, it may still be accepted as a non-record submission.

## Evaluation Integrity

- Do not leak validation data into training.
- Do not cheat on test loss or use validation tokens before they have already been evaluated.
- Test-time training is only allowed on validation tokens that have already been scored.

## Tokenizer and Dataset Changes

- If you change the tokenizer or dataset, you must prove the reported `val_bpb` is still correct.
- Those submissions get closer scrutiny than ordinary model-only changes.

## External Code

- Extra libraries are allowed if they do not violate compute, training-time, evaluation, or code-size rules.
- Include a `requirements.txt` and setup notes in the submission if you rely on extra packages.

## Practical Read

- Treat the leaderboard as public and chronological.
- Treat score gains as insufficient unless the run is reproducible under the submission constraints.
- If the change is new but not a record, document why it is still worth keeping.

## Midnight III Notes

Pulled from `sota_nueral` so this directory also carries the current neural working base.

- Official leader: Midnight 12L at 1.10567949 BPB on seed 444, 15,631,603 bytes.
- Working base: Midnight III at 1.10616680 BPB on seed 444, with seed 300 confirmation intentionally pending.
- Vault source for new work: `vault/train_gpt_midnight_iii_base.py`.
- Current campaign: close the 0.45 BPB quant gap with loop-aware GPTQ.
- New experiments should start from the Midnight III base, not the older Midnight 12L leader file.
- Previous champion Rascal II is quarantined in `sota_nueral/records/quarantine/2026-04-07_Rascal_II_11L_demoted/2026-03-30_Rascal_8xH100`.

