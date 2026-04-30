# Hypothesis - 2026-04-30_mikey_sp10240_mlp375_loop_late060_nowarm_8x

## Parent

`2026-04-29_mikey_sp10240_full124_loop2_late050_mlp375_clip1305_8x`

Known parent signal:

- `vocab_size=10240`
- `num_layers=11`
- `model_dim=512`
- `mlp_mult=3.75`
- `num_loops=2`
- `loop_start=3`
- `loop_end=5`
- `enable_looping_at=0.50`
- `matrix_bits=6`
- `matrix_clip_sigmas=13.05`
- `Total submission size quantized+brotli: 15954736`
- `quantized_sliding_window val_bpb: 1.08229317`
- `quantized_ttt val_bpb: 1.08066751`

## Change

This is a switch-body loop repair test. It keeps the byte-stable Mikey SP10240 11L MLP3.75 body and changes only the loop schedule:

- `enable_looping_at=0.60`
- `loop_warmup_enabled=False`

All tokenizer, dataset, 8x budget, GPTQ, matrix clip, and TTT settings stay aligned with the parent.

## Readout

This test is useful if delayed loop pressure improves the post-EMA neural score or keeps quant/sliding under the parent. It should be cut if plain quant jumps above the competitive band or if the delayed loop loses the parent neural curve.
