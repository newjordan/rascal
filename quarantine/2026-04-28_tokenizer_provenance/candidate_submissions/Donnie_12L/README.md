# Donnie_12L

12-layer Donnie (Raphe-recipe + layer-loop reuse, no TTT).

## Recipe vs Donnie

- `num_layers`: 11 → 12 (everything else identical)

Donnie 11L int6 round-trip: 0.90415915 / 14,878,185 bytes (~1.1 MB headroom under
the 16 MB cap). 12L adds one more layer of banks; expected size growth ~1.0–1.2 MB.

Default loop indices for 12L: `[0,1, 2,3,4, 2,3,4, 2,3,4, 5,6,7,8,9,10,11]` →
encoder = `[0,1, 2,3,4, 2,3,4, 2]` (9 passes), decoder = `[3,4, 5,6,7,8,9,10,11]`
(9 passes), `num_skip_weights = 9`.

## Knobs

| env var | default | notes |
|---|---|---|
| `NUM_LAYERS` | 12 | flat depth |
| `NUM_LOOPS` | 2 | extra repeats of `[loop_start, loop_end]` |
| `LOOP_START` | 2 |
| `LOOP_END` | 4 |
| `ENABLE_LOOPING_AT` | 0.35 |

```
torchrun --standalone --nproc_per_node=8 Donnie_12L/train_gpt_8xgpu.py
```
