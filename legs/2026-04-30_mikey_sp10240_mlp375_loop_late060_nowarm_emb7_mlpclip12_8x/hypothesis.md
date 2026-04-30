# Hypothesis - 2026-04-30_mikey_sp10240_mlp375_loop_late060_nowarm_emb7_mlpclip12_8x

## Parent

`2026-04-30_mikey_sp10240_mlp375_loop_late060_nowarm_8x`

## Change

This keeps the Mikey SP10240 11L MLP3.75 delayed/no-warm loop body and changes quant policy:

- `embed_bits=7`
- `mlp_clip_sigmas=12.0`
- non-MLP matrix clip remains `matrix_clip_sigmas=13.05`
- matrix weights remain `matrix_bits=6`

## Why

`embed_bits=7` buys byte budget from the tied embedding table. MLP-specific clip prevents the MLP tensors from inheriting the global matrix clip when the better comparator calls out `MLPClip12`.

## Readout

This is useful if byte size drops without a large embedding quant penalty and if MLP quant damage improves. Compare against parent quant, sliding, TTT, and total bytes.
