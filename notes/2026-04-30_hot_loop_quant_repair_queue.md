# Hot-Loop Quant Repair Queue - 2026-04-30

## Trigger Evidence

The 10k whale body proved the neural lane is alive but the artifact fails:

- `post-EMA val_bpb: 1.07648677`
- `quantized val_bpb: 1.11297402`
- `Total submission size: 16,701,823`
- over cap by `701,823` bytes
- quant damage: about `+0.03649 bpb`

Interpretation: the whale body is strong, but global int5 is poisoning recurrent/hot layers.

## Core Hypothesis

The win is likely in the loop path, not a generic bigger/smaller body. Blocks reused by the loop are executed more than once, so quant error in those blocks is amplified. Treat recursive layers as higher-value/higher-risk tensors.

Current 12L loop topology:

```text
loop segment: 3,4,5
encoder active: [0,1,2,3,4,5,3,4,5]
decoder active: [3,4,5,6,7,8,9,10,11]
```

## Test Queue

### H1 - Hot-Loop Attention Precision

Give int6 only to attention matrices in looped blocks `3,4,5`; keep other attention and all MLP matrices int5 if size allows.

Reason: attention is less quant-tolerant than MLP, and looped attention error is reused.

Readout: plain quant must drop materially from `1.11297`; size must be close enough to repair.

### H2 - Broad Attention Repair

Prepared runner:

```text
legs/2026-04-29_whale_sp10240_l12_dim576_mlp375_attn6_mlp5_8x
```

Settings: `12L dim576 SP10240`, `MLP_MULT=3.75`, all attention matrices int6, all MLP matrices int5, embeddings int8.

Reason: fast diagnostic for whether the spinout is attention-driven before writing the surgical hot-loop variant.

### H3 - Delayed Loop Activation

Move `ENABLE_LOOPING_AT` from `0.35` to `0.50-0.60`. Consider disabling loop warmup for this branch.

Reason: let the base 12L body learn cleaner representations before recurrence starts. The goal is less quant chaos even if float BPB gives back a little.

Readout: compare post-EMA loss, final step count, and plain quant delta. Do not judge only float BPB.

### H4 - Loop-Aware Diagonal Smoothing

Use SmoothQuant/AWQ/SpinQuant-style diagonal or orthogonal transforms only on looped blocks `3,4,5`, attention first.

User concept mapping: "diagonals on int5 indicators" = quantization-aware activation/weight rescaling for tensors forced to int5.

Reason: make int5 safer without paying full int6 byte cost.

### H5 - PPM After Artifact Is Close

Do not spend PPM on bodies with plain quant above `1.10`. Revisit only if the artifact is already near `1.08-1.09`.

Possible later form: uncertainty-gated eval residual expert, optionally conditioned on loop-vs-no-loop confidence. It must stay prefix-only.

## Priority

1. Run H2 if the current whale is cut or finished.
2. If H2 improves quant but busts size, shave MLP to `3.6-3.65`.
3. If H2 fits but loses too much float quality, make H1 surgical hot-loop precision.
4. If quant remains bad, test H3 delayed loop before abandoning whale.

## Short Name

Use `hotloop` for future leg names involving execution-count-weighted precision or delayed recurrence.
