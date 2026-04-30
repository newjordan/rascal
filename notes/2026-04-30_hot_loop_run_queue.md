# Hot-Loop Run Queue - 2026-04-30

## Current Evidence

Baseline whale SP10240 showed the target is real but the artifact fails:

```text
post-EMA val_bpb: 1.07648677
quantized val_bpb: 1.11297402
Total submission size: 16,701,823
```

Interpretation: the 12L dim576 whale body is strong; global int5 quantization and/or early recurrence is the failure.

## Run Order

### 1. H2 Broad Attention Repair

Purpose: diagnostic for whether attention precision broadly repairs quant spin.

```bash
cd /workspace/sota_rascal/legs/2026-04-29_whale_sp10240_l12_dim576_mlp375_attn6_mlp5_8x && ./launch_8x.sh
```

Settings: `12L`, `dim576`, `SP10240`, `MLP3.75`, all attention int6, all MLP int5, embeddings int8, loop at `0.35`.

Readout: if plain quant improves materially from `1.11297`, attention precision is the right direction. If over cap, shave MLP to `3.6-3.65`.

### 2. H1 Surgical Hot-Loop Precision

Purpose: spend bytes only on recurrent attention blocks.

```bash
cd /workspace/sota_rascal/legs/2026-04-30_h1_surgical_hot_loop_precision_8x && ./launch_8x.sh
```

Settings: `MLP4.0`, global int5, blocks `3,4,5` attention int6, embeddings int8, loop at `0.35`.

Readout: if it repairs quant without huge size cost, hot-loop precision beats broad attention repair.

### 3. H3 Delayed Loop Activation

Purpose: test whether early recurrence causes float/quant instability.

```bash
cd /workspace/sota_rascal/legs/2026-04-29_whale_h3_delayed_loop_activation_l12_dim576_matrixbits5_sp10240_8x && ./launch_8x.sh
```

Settings: original whale SP10240 global int5, `MLP4.0`, loop activation delayed from `0.35` to `0.55`, loop warmup disabled.

Readout: compare post-EMA, step count, and plain quant to baseline. This may lose some float but reduce quant chaos.

### 4. H1H3 Combined Hot-Loop Precision + Delayed Loop

Purpose: combine the two loop hypotheses if either H1 or H3 looks promising, or run as a high-risk direct swing.

```bash
cd /workspace/sota_rascal/legs/2026-04-29_h1h3_hotloop_delayed_precision_whale_sp10240_8x && ./launch_8x.sh
```

Settings: `MLP4.0`, global int5, blocks `3,4,5` attention int6, loop activation `0.55`, loop warmup disabled.

Readout: best evidence if it keeps float near whale while reducing quant spin.

## Cut Rules

- For these neural/quant tests, do not wait for TTT if size is badly over cap or plain quant is clearly bad.
- Save: post-EMA BPB, total bytes, plain quant BPB, final step count, and `tok/s`.
- Plain quant target: must move below `1.10`; near `1.08-1.09` is worth sliding/TTT.

## Not Ready Yet

H4 loop-aware diagonal smoothing is not prepared. It needs an explicit SmoothQuant/AWQ/SpinQuant-style implementation scoped to loop blocks `3,4,5`, attention first.
