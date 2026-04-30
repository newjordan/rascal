# Hypothesis - 2026-04-30_h1_surgical_hot_loop_precision_8x

Parent: `2026-04-29_whale_l12_dim576_matrixbits5_sp10240_8x`

Run label: `standard_8x`

Experiment class: `new_experiment_hot_loop_precision`

## Change

Keep the 12L dim576 SP10240 whale and the same loop topology/budgets, but spend precision only where the recurrent segment reuses attention:

- `NUM_LAYERS=12`
- `MODEL_DIM=576`
- `EMBEDDING_DIM=576`
- `MLP_MULT=4.0`
- default matrix quantization: `int5`
- hot-loop attention matrices in blocks `3`, `4`, and `5`: `int6`
- other attention matrices: `int5`
- all MLP matrices: `int5`
- token embeddings: `int8`
- `NUM_LOOPS=2`, `LOOP_START=3`, `LOOP_END=5`, `ENABLE_LOOPING_AT=0.35`
- `HOT_LOOP_START=3`, `HOT_LOOP_END=5`, `HOT_LOOP_ATTN_BITS=6`
- `TTT_ENABLED=1`, `TTT_EPOCHS=3`

## Readout

H1 tests whether the looped attention segment is the quantization-sensitive core. A useful result preserves MLP4.0 and the SP10240 tokenizer while improving quantized/sliding/TTT behavior enough to justify the small targeted int6 byte spend.
