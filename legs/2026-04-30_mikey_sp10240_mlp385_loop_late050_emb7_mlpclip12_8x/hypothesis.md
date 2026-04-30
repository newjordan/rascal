# Hypothesis - 2026-04-30_mikey_sp10240_mlp385_loop_late050_emb7_mlpclip12_8x

## Parent

`2026-04-30_mikey_sp10240_mlp375_loop_late050_emb7_mlpclip12_8x`

Original best-body signal before the quant-policy test:

- total bytes: `15,954,736`
- quantized BPB: `1.098550`
- sliding BPB: `1.08229317`
- TTT BPB: `1.08066751`

## Change

This is the follow-up if the clean quant-policy isolation buys usable byte margin:

- keep `num_layers=11`
- change `mlp_mult=3.75 -> 3.85`
- keep `num_loops=2`
- keep `enable_looping_at=0.50`
- keep loop warmup enabled
- keep non-MLP `matrix_clip_sigmas=13.05`
- keep `embed_bits=7`
- keep `mlp_clip_sigmas=12.0`

## Readout

This spends the quant-policy byte margin on the most sensitive capacity lever we have been seeing: MLP width on the proven late050 loop body.
