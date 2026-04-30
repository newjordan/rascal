# Donnie_TTT

Donnie (11L Raphe-recipe + layer-loops) plus score-first test-time training (TTT)
on the deserialized int6 artifact. PR #1493 precedent.

For each 32k-token chunk: score windows under `no_grad` first (commits bytes),
then SGD-train on the chunk (lr=0.005 cosine-decayed across chunks, momentum=0.9,
3 epochs, grad clip 1.0). Last chunk is eval-only.

The artifact bytes do **not** change — TTT modifies only the in-memory deserialized
model. `eval_model.looping_active` is set to True before TTT so the loop forward
path is used (matches the trained model). Legal-size box stays checked.

## What's reported

- `final_sliding_window_exact` — artifact-sliding, no TTT (loop-aware)
- `final_sliding_window_s64_exact` — same, stride 64
- `final_sliding_window_ttt_exact` — **new**: artifact-sliding with score-first TTT

## TTT knobs

| env var | default | notes |
|---|---|---|
| `TTT_ENABLED` | 1 | gate; `TTT_ENABLED=0` skips TTT |
| `TTT_LR` | 0.005 | initial SGD lr (cosine-decayed across chunks) |
| `TTT_EPOCHS` | 3 | inner-loop SGD epochs per chunk |
| `TTT_MOMENTUM` | 0.9 | SGD momentum |
| `TTT_CHUNK_TOKENS` | 32768 | chunk size for the score-then-train loop |

## Loop knobs (inherited from Donnie)

| env var | default |
|---|---|
| `NUM_LAYERS` | 11 |
| `NUM_LOOPS` | 2 |
| `LOOP_START` | 2 |
| `LOOP_END` | 4 |
| `ENABLE_LOOPING_AT` | 0.35 |

## Pending

`val_bpb` is null until the 8xH100 run lands.

```
torchrun --standalone --nproc_per_node=8 Donnie_TTT/train_gpt_8xgpu.py
```
