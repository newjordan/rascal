# Hypothesis - 2026-04-30_mikey_sp10240_mlp375_loop_late050_emb7_mlpclip12_8x

## Parent

`2026-04-29_mikey_sp10240_full124_loop2_late050_mlp375_clip1305_8x`

Known parent signal:

- total bytes: `15,954,736`
- quantized BPB: `1.098550`
- sliding BPB: `1.08229317`
- TTT BPB: `1.08066751`

## Change

This is a clean quant-policy isolation:

- keep `num_layers=11`
- keep `mlp_mult=3.75`
- keep `num_loops=2`
- keep `enable_looping_at=0.50`
- keep loop warmup enabled
- keep non-MLP `matrix_clip_sigmas=13.05`
- change `embed_bits=7`
- add `mlp_clip_sigmas=12.0`

## Readout

This tests whether the newly discovered `embed_bits=7 + MLPClip12` axis improves byte margin or quant damage on the actual best body, without confounding loop schedule.
