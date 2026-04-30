# 2026-04-30 Matrix Lane Two-Step Plan

Branch: `matrix-lane-20260430`

## Goal

Stop one-run improvisation. Every active test should have a next action and a second action based on the observed failure class.

Target band remains the personal-best Mikey 10k result:

- quantized sliding BPB: `1.08229317`
- quantized TTT BPB: `1.08066751`
- eval-only TTT5 BPB: `1.08051259`
- total bytes: `15,954,736`

## Matrix Lessons To Carry Forward

- Attention is less quant-tolerant than MLP. Old sweep code encodes this explicitly: `mlp_int5` is plausible, `attn_int5` is a falsification canary.
- Embeddings are not free. Treat `embed_bits=7` as an experiment, not a default.
- Global int5 on whale is toxic: whale post-EMA `1.07648677` became quant `1.11297402`.
- Broad attention repair helped directionally but not enough: `attn6/mlp5` whale got quant `1.10769976` and busted size at `17,587,249`.
- Loop blocks are higher-value matrices because loop execution reuses their quant error. If we spend int6 surgically, spend it first on looped attention in blocks `3,4,5`.
- Exact Shampoo inverse roots are real but too slow as implemented: early throughput around `3.8M tok/s` versus normal `7-8M`.

## Active Test

Clean quant-policy isolation:

```bash
cd /workspace/sota_rascal/legs/2026-04-30_mikey_sp10240_mlp375_loop_late050_emb7_mlpclip12_8x && ./launch_8x.sh
```

Changes only:

- `embed_bits: 8 -> 7`
- `mlp_clip_sigmas: unset -> 12.0`

Keep `late050` and loop warmup.

## Two-Step Decision Tree

### Case A: Clean Quant Wins Or Threatens

Signal:

- post-EMA near `1.0868-1.0875`
- plain quant not materially over `1.10`
- sliding/TTT at or under the `1.0805-1.0810` band

Next run:

- same body and loop schedule
- keep `embed_bits=7`
- keep `mlp_clip_sigmas=12.0`
- bump `mlp_mult` from `3.75` to `3.85`

Prepared runner:

```bash
cd /workspace/sota_rascal/legs/2026-04-30_mikey_sp10240_mlp385_loop_late050_emb7_mlpclip12_8x && ./launch_8x.sh
```

Second step:

- if bytes still have real margin and quant damage stays flat, try `mlp_mult=3.90`
- if bytes are tight or quant worsens, use eval-time TTT5/TTT budget on the winner instead of another body change

### Case B: Clean Quant Saves Bytes But Loses Quality

Signal:

- total bytes improve materially
- post-EMA or sliding/TTT misses the personal-best band

Next run:

- revert `embed_bits=8`
- keep `mlp_clip_sigmas=12.0` only if plain quant damage improved
- otherwise revert both and stop using this quant patch as a primary lane

Second step:

- spend bytes through MLP micro-capacity only if the reverted embed result stays near the parent curve
- otherwise switch to Case D

### Case C: Clean Quant Is Bad

Signal:

- post-EMA around `1.089+`
- plain quant deep over `1.10`
- final TTT misses `1.082+`

Next run:

- drop the `embed_bits=7 + MLPClip12` lane
- run a body switch with proven loop behavior, not the no-warm result:
  - `9L dim576`
  - `mlp_mult=3.75`
  - `late050`
  - loop warmup enabled
  - `embed_bits=8`
  - `matrix_clip_sigmas=13.05`

Prepared runner:

```bash
cd /workspace/sota_rascal/legs/2026-04-30_mikey_sp10240_9l_dim576_mlp375_loop_late050_warm_8x && ./launch_8x.sh
```

Second step:

- if 9L dim576 is too small but byte-safe, push MLP upward before changing loop timing
- if 9L dim576 is neural-bad, abandon 9L and go back to Mikey 11L matrix/eval-time fixes

### Case D: Neural Good, Quant Bad

Signal:

- post-EMA is good, but quant damage exceeds about `+0.020 bpb`
- or size forces int5

Next run:

- use matrix precision, not body size:
  - attention int6
  - MLP int5/6 depending on size
  - embeddings int8
  - prioritize looped attention blocks `3,4,5`

Prepared runner:

```bash
cd /workspace/sota_rascal/legs/2026-04-30_h1_surgical_hot_loop_precision_8x && ./launch_8x.sh
```

Second step:

- if H1 reduces quant damage but still busts size, shave MLP from `4.0` toward `3.75`
- if H1 does not reduce quant damage, stop whale/hot-loop precision and move to diagonal/smoothing only as a targeted eval/export repair

## Do Not Run Blind

- Do not rerun exact Shampoo on 8x until it has a cheaper cadence or approximation.
- Do not carry `late060/no-warm` forward unless the current evidence specifically calls for loop-delay again.
- Do not spend PPM on a body whose plain quant is above `1.10`; PPM is a finisher, not a rescue rope.

## Belief Test: 9L Bigger Loop / Max Step

Evidence check: local `seqloop3_late55_68_80_9l` runners existed, but there were no completed logs and no copy on the active 8x pod. This idea was not cleanly tried.

Prepared runner:

```bash
cd /workspace/sota_rascal/legs/2026-04-30_mikey_sp10240_9l_mlp4_seqloop3_late55_68_80_maxstep_8x && ./launch_8x.sh
```

Settings:

- `9L`, `model_dim=512`, `MLP=4.0`
- `num_loops=3`
- staged loop pressure at `0.55`, `0.68`, `0.80`
- `compile_fullgraph=False`, `compile_dynamic=True`
- `matrix_bits=6`, `matrix_clip_sigmas=13.05`

Why: this is the actual max-step version of the 9L loop idea, separate from the 9L dim576 loop2 body switch.
