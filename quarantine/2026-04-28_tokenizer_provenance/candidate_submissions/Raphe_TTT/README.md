# Raphe 1.2

Raphe 1.1 (eval-fix) plus score-first test-time training (TTT) on the
deserialized int6 artifact. PR #1493 precedent.

For each 32k-token chunk: score windows under `no_grad` first (commits bytes),
then SGD-train on the chunk (lr=0.005 cosine-decayed across chunks, momentum=0.9,
3 epochs, grad clip 1.0). Last chunk is eval-only. Every token is scored before
training, satisfying `LEADERBOARD_RULES.md`.

The artifact bytes do **not** change — TTT modifies only the in-memory
deserialized model. Legal-size box stays checked.

## What's reported

Each run logs three sliding scores on the int6 artifact:

- `final_sliding_window_exact` — artifact-sliding, no TTT (the 1.1 audit-fixed metric)
- `final_sliding_window_s64_exact` — same, stride 64
- `final_sliding_window_ttt_exact` — **new**: artifact-sliding with score-first TTT

Pick the headline from whichever wins (TTT typically claws back ~0.02–0.04 bpb on
quantization-induced loss; not guaranteed).

## Knobs

| env var | default | notes |
|---|---|---|
| `TTT_ENABLED` | 1 | gate; `TTT_ENABLED=0` skips TTT eval entirely |
| `TTT_LR` | 0.005 | initial SGD lr (cosine-decayed across chunks) |
| `TTT_EPOCHS` | 3 | inner-loop SGD epochs per chunk |
| `TTT_MOMENTUM` | 0.9 | SGD momentum |
| `TTT_CHUNK_TOKENS` | 32768 | chunk size for the score-then-train loop |

## Comparison points

| variant | what it measures | use |
|---|---|---|
| Raphe 1.1 | artifact-sliding only | re-establish the corrected baseline; fix PR #1846 |
| Raphe 1.2 (this) | artifact-sliding + TTT | upside test on top of 1.1 baseline |

## Pending

`val_bpb` is null until the 8xH100 run lands. Run Raphe 1.1 first to lock the
baseline, then 1.2 to measure the TTT delta.

```
torchrun --standalone --nproc_per_node=8 Raphe_1.2/train_gpt_8xgpu.py
```
