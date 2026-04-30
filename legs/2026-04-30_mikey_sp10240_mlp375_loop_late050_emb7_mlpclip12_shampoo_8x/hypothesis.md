# Hypothesis - 2026-04-30_mikey_sp10240_mlp375_loop_late050_emb7_mlpclip12_shampoo_8x

## Parent

`2026-04-30_mikey_sp10240_mlp375_loop_late050_emb7_mlpclip12_8x`

Underlying best-body signal before the quant-policy patch:

- total bytes: `15,954,736`
- quantized BPB: `1.098550`
- sliding BPB: `1.08229317`
- TTT BPB: `1.08066751`

## Change

This is an optimizer-axis isolation stacked on the clean quant-policy runner:

- keep SP10240 full124 data/tokenizer
- keep `num_layers=11`, `mlp_mult=3.75`, `num_loops=2`
- keep late loop enable at `0.50` with loop warmup enabled
- keep `embed_bits=7`, `matrix_clip_sigmas=13.05`, `mlp_clip_sigmas=12.0`
- replace the Muon zero-power update with a real running inverse-root preconditioner
- use `S_t = beta2*S_{t-1} + (1-beta2)*G^T*G` or `G*G^T` on the smaller matrix side
- use `shampoo_beta2=0.95`, `shampoo_eps=1e-4`, update every optimizer step

## Readout

This tests whether the optimizer hidden axis is real. The risk is lower throughput from exact eig inverse roots; the payoff would be better neural quality or quant stability without changing the model body.
