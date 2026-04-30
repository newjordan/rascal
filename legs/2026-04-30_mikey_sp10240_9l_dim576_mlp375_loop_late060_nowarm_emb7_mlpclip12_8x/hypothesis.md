# Hypothesis - 2026-04-30_mikey_sp10240_9l_dim576_mlp375_loop_late060_nowarm_emb7_mlpclip12_8x

## Parent

`2026-04-30_mikey_sp10240_9l_dim576_mlp375_loop_late060_nowarm_8x`

## Change

This keeps the 9L dim576 MLP3.75 delayed/no-warm loop body and changes quant policy:

- `embed_bits=7`
- `mlp_clip_sigmas=12.0`
- non-MLP matrix clip remains `matrix_clip_sigmas=13.05`
- matrix weights remain `matrix_bits=6`

## Why

This tests whether the byte savings from embedding int7 can make the wider 9L body more viable while MLPClip12 reduces MLP quant damage.
