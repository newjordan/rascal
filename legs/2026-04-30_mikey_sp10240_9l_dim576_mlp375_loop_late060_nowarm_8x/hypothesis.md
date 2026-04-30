# Hypothesis - 2026-04-30_mikey_sp10240_9l_dim576_mlp375_loop_late060_nowarm_8x

## Purpose

This is the switch-body swing after cutting whale as primary. It keeps the SP10240 tokenizer/data lane, but changes the body away from 12L whale sensitivity.

## Parent

`2026-04-29_mikey_sp10240_full124_loop2_late050_9l_dim576_mlp325_clip1305_8x`

## Change

- `num_layers=9`
- `model_dim=576`
- `embedding_dim=576`
- `num_heads=8`
- `head_dim=72`
- `mlp_mult=3.75`
- `num_loops=2`
- `loop_start=3`
- `loop_end=5`
- `enable_looping_at=0.60`
- `loop_warmup_enabled=False`
- `matrix_bits=6`
- `matrix_clip_sigmas=13.05`

This spends capacity through FA3-valid width instead of whale depth. It also preserves the loop repair idea without adding submission bytes.

## Readout

This should be treated as high risk. It is interesting if the neural score moves toward the whale signal while staying under the byte cap. It should be cut if plain quant jumps above the competitive band or if the 9L width body learns too slowly before loop activation.
