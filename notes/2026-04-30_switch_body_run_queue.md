# 2026-04-30 Switch-Body Run Queue

## Decision

Cut whale as the primary lane. The 12L whale body showed the neural ceiling, but its artifact is too size-sensitive and the int5 quant jump is too large for the remaining time.

Keep loop repair alive, but port it to the byte-stable Mikey SP10240 body.

## Current Comparator

Parent:

```bash
cd /workspace/sota_rascal/legs/2026-04-29_mikey_sp10240_full124_loop2_late050_mlp375_clip1305_8x && ./launch_8x.sh
```

Known signal:

- total bytes: `15954736`
- quantized sliding BPB: `1.08229317`
- quantized TTT BPB: `1.08066751`
- eval-only TTT5 BPB: `1.08051259`

## Next Primary Test

Clean quant-policy isolation on the best parent:

```bash
cd /workspace/sota_rascal/legs/2026-04-30_mikey_sp10240_mlp375_loop_late050_emb7_mlpclip12_8x && ./launch_8x.sh
```

Changed fields:

- `embed_bits: 8 -> 7`
- `mlp_clip_sigmas: unset -> 12.0`

Unchanged:

- `enable_looping_at=0.50`
- loop warmup enabled
- `num_layers=11`
- `mlp_mult=3.75`
- `matrix_bits=6`
- non-MLP `matrix_clip_sigmas=13.05`

Why: this isolates the newly discovered quant-policy axis without mixing in the late060/no-warm loop change.

## Secondary Loop Test

Delayed-loop/no-warmup repair on the same Mikey 10k MLP3.75 body:

```bash
cd /workspace/sota_rascal/legs/2026-04-30_mikey_sp10240_mlp375_loop_late060_nowarm_8x && ./launch_8x.sh
```

Changed fields:

- `enable_looping_at: 0.50 -> 0.60`
- `loop_warmup_enabled: True -> False`

Decision rule: keep if neural/post-EMA or quant/sliding improves against the parent without losing byte safety. Cut if plain quant jumps above the competitive band or the neural curve trails badly after loop activation.

## Next Swing

Real optimizer-axis test on the clean quant-policy body:

```bash
cd /workspace/sota_rascal/legs/2026-04-30_mikey_sp10240_mlp375_loop_late050_emb7_mlpclip12_shampoo_8x && ./launch_8x.sh
```

Changed fields versus the clean quant-policy runner:

- `shampoo_enabled: False -> True`
- `shampoo_beta2=0.95`
- replaces Muon zero-power updates with a running Gram inverse-root preconditioner

Why: the logged `muon_beta2` axis was phantom in the old optimizer path. This is the first real test of `S_t = beta2*S_{t-1} + (1-beta2)*G^T*G` / inverse-root style conditioning. Expect step loss from exact eig roots; read it as quality-per-step, not speed-safe.

## Body Switch

High-risk 9-layer cake body switch:

```bash
cd /workspace/sota_rascal/legs/2026-04-30_mikey_sp10240_9l_dim576_mlp375_loop_late060_nowarm_8x && ./launch_8x.sh
```

Changed fields versus the older 9L dim576 scout:

- `mlp_mult: 3.25 -> 3.75`
- `enable_looping_at: 0.50 -> 0.60`
- `loop_warmup_enabled: True -> False`

Why: `dim576` is FA3-valid (`head_dim=72`) and spends capacity through width rather than whale depth. This is the real body-switch swing if the conservative repair does not move.

## Deprioritized

Whale hot-loop repair variants are preserved under `legs/`, but are no longer primary until a stable Mikey body fails to move.
