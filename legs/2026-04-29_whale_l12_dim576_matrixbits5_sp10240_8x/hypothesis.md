# Hypothesis - 2026-04-29_whale_l12_dim576_matrixbits5_sp10240_8x

Parent: `2026-04-15_whale_l12_dim576_matrixbits5`

Run label: `standard_8x`

Experiment class: `new_experiment_size_risk`

## Change

Move the locked whale donor geometry from SP8192 to SP10240 while keeping the whale shape fixed:

- `NUM_LAYERS=12`
- `MODEL_DIM=576`
- `EMBEDDING_DIM=576`
- `MLP_MULT=4.0`
- `MATRIX_BITS=5`
- `NUM_LOOPS=2`, `LOOP_START=3`, `LOOP_END=5`, `ENABLE_LOOPING_AT=0.35`
- `TTT_ENABLED=1`, `TTT_EPOCHS=3`

## Readout

This is expected to be size-risk because the tied embedding table grows by roughly 1.18M raw int8 weights versus SP8192. The useful signal is whether post-EMA and quantized sliding/TTT improve enough to justify a follow-up byte-repair pass.
