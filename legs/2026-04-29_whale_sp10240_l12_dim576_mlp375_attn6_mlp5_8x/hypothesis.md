# Hypothesis - 2026-04-29_whale_sp10240_l12_dim576_mlp375_attn6_mlp5_8x

Parent: `2026-04-29_whale_l12_dim576_matrixbits5_sp10240_8x`

Run label: `standard_8x`

Experiment class: `new_experiment_quant_repair`

## Change

Keep the whale donor shape and SP10240 data path, but use the MLP byte budget to repair quantization:

- `NUM_LAYERS=12`
- `MODEL_DIM=576`
- `EMBEDDING_DIM=576`
- `MLP_MULT=3.75`
- attention matrices: `int6`
- MLP matrices: `int5`
- token embeddings: `int8`
- `NUM_LOOPS=2`, `LOOP_START=3`, `LOOP_END=5`, `ENABLE_LOOPING_AT=0.35`
- `TTT_ENABLED=1`, `TTT_EPOCHS=3`

## Readout

This tests whether the whale spinout is mainly attention quant damage. A useful result is either legal size with lower plain quant BPB, or an over-cap result close enough to justify a tighter MLP/bit allocation pass.
