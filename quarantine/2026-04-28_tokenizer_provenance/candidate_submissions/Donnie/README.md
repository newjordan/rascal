# Donnie

11-layer Raphe 1.1 with layer-loop reuse.

The hypothesis: Raphe's 13.4 MB artifact has ~2.5 MB of headroom under the 16 MB cap.
The earlier loop test bumped the final val score but busted size on a tighter recipe.
With Raphe's compact compression stack (sp4096 + brotli + bshf + mixed-int) the loop
size hit should fit. Donnie tests that.

## Recipe vs Raphe 1.1

- `num_layers`: 10 → 11
- `num_loops`: 0 → 2 (loop_start=2, loop_end=4 — block indices 2..4 reused 3× total)
- `enable_looping_at`: 0.35 of wallclock — flat-path warmup first, then toggle to loops
- Two-stage warmup: flat warmup → loop warmup (revert state) → main run
- Eval fix from Raphe 1.1 retained: post-quant sliding eval runs on `eval_model`
  (decompressed int6 artifact) with `looping_active=True` to match the trained model

Default loop indices for 11L: `[0,1, 2,3,4, 2,3,4, 2,3,4, 5,6,7,8,9,10]` →
encoder = `[0,1, 2,3,4, 2,3,4]` (8 effective passes), decoder = `[2,3,4, 5,6,7,8,9,10]`
(9 passes), `num_skip_weights = 8`.

## Knobs

| env var | default | notes |
|---|---|---|
| `NUM_LAYERS` | 11 | flat depth (with loops, effective forward depth = 17) |
| `NUM_LOOPS` | 2 | extra repeats of `[loop_start, loop_end]`; total passes = NUM_LOOPS + 1 |
| `LOOP_START` | 2 | first looped block index |
| `LOOP_END` | 4 | last looped block index (inclusive) |
| `ENABLE_LOOPING_AT` | 0.35 | wallclock fraction to flip `looping_active` on |

## Pending

`val_bpb` is null until the 8xH100 run lands. Compare against:

- Raphe 1.1 artifact-sliding (also pending)
- Raphe original base_model sliding: 0.87206316 mean
- Mikey 1.1 (12L flat) artifact-sliding (also pending)

Watch the artifact size — Donnie adds one full layer of banks vs Raphe; expected
size is ~14.5–15 MB (still under cap), but the int6 + brotli ratio is what we care
about.

```
torchrun --standalone --nproc_per_node=8 Donnie/train_gpt_8xgpu.py
```
