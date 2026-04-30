# Hypothesis - 2026-04-29_whale_h3_delayed_loop_activation_l12_dim576_matrixbits5_sp10240_8x

Parent: `2026-04-29_whale_l12_dim576_matrixbits5_sp10240_8x`

Donor lineage: `2026-04-15_whale_l12_dim576_matrixbits5`

Run label: `standard_8x`

Experiment class: `new_experiment_size_risk`

## Change

H3 delayed-loop activation test. Preserve the SP10240 whale test car and delay the recurrent loop activation:

- `NUM_LAYERS=12`
- `MODEL_DIM=576`
- `EMBEDDING_DIM=576`
- `MLP_MULT=4.0`
- `MATRIX_BITS=5`
- `EMBED_BITS=8`
- `NUM_LOOPS=2`, `LOOP_START=3`, `LOOP_END=5`, `ENABLE_LOOPING_AT=0.55`
- `LOOP_WARMUP_STEPS=0`; the reset-only loop warmup stage is explicitly disabled
- `TTT_ENABLED=1`, `TTT_EPOCHS=3`

## Readout

The useful signal is whether delaying loop activation from `0.35` to `0.55` improves the post-EMA and quantized sliding/TTT readouts while preserving the 12L dim576 SP10240 whale shape and matrixbits5 compression policy.
