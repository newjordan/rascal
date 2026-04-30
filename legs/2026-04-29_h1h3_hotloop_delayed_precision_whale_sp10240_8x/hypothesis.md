# Hypothesis - 2026-04-29_h1h3_hotloop_delayed_precision_whale_sp10240_8x

Parent: `2026-04-29_whale_l12_dim576_matrixbits5_sp10240_8x`

Run label: `standard_8x`

Experiment class: `new_experiment_hotloop_precision_delay`

## Change

H1H3 hot-loop delayed precision car. Preserve the 12L dim576 SP10240 whale geometry and MLP size, but spend precision only on the active loop segment while delaying recurrent activation:

- `NUM_LAYERS=12`
- `MODEL_DIM=576`
- `EMBEDDING_DIM=576`
- `MLP_MULT=4.0`
- global `MATRIX_BITS=5`
- loop segment attention matrices in `blocks.3`, `blocks.4`, `blocks.5`: `int6`
- all other non-embedding matrices: `int5`
- token embeddings: `int8`
- `NUM_LOOPS=2`, `LOOP_START=3`, `LOOP_END=5`, `ENABLE_LOOPING_AT=0.55`
- loop warmup disabled explicitly with `loop_warmup_enabled=False`
- `TTT_ENABLED=1`, `TTT_EPOCHS=3`

## Readout

This tests whether the hot recurrent segment is the precision bottleneck. A useful signal is improved quantized and quantized sliding BPB without paying int6 bytes for all attention matrices, and without early-loop destabilization from loop warmup or 0.35 activation.
