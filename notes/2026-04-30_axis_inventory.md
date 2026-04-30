# 2026-04-30 Axis Inventory

Use this before picking the next paid run. Classify the failure first:

- neural quality miss
- quantization damage
- size overflow
- eval-time weakness
- runtime/step loss
- condition drift

## Axes

1. Tokenizer/data: vocab size, tokenizer hash, train shard count, val set, byte accounting, shard order, curriculum, leakage/integrity.
2. Body capacity: layers, width, MLP multiplier, heads, KV heads, FA3 head dim legality, embedding dim, residual/skip topology.
3. Loop/recurrence: loop count, loop span, activation time, warmup behavior, ramp/cold-start, shared weights, eval-time loop state.
4. Optimization: LR family, optimizer, true second-moment/preconditioner state, per-group WD, EMA, warmup/warmdown, batch tokens, grad accumulation, clipping, seed, steps.
5. Quant policy: bits per module, per-module clip sigmas, GPTQ calibration data, calibration batches, block size, scale dtype, pass-through tensors, tied embeddings, mixed int5/int6/int7/int8.
6. Artifact budget: serialized bytes, compressed bytes, code bytes, compressor, unused code, tensor names/layout.
7. Eval-time adaptation: sliding window, TTT, PPM/byte mixture, tag features, legal CPU helpers, eval budget usage.
8. Runtime/kernel: FA3/native path, compile mode, cudagraphs, dataloader mode, GPU count, thermals, tok/s, recompile warnings.
9. Integrity: exact parent path, source hash, condition artifact, metric name, wallclock class, seed, standard vs proxy.

## Rule

When neural BPB is good but quant/size is bad, audit quant policy and artifact budget before changing model body. For today that means `embed_bits`, per-module clips, mixed bit allocation, GPTQ calibration, and pass-through tensors before another architecture swing.

When an optimizer flag is logged but not used in the optimizer math, treat it as condition drift. The live test for this is the Shampoo runner, which adds a real running Gram inverse-root state instead of relying on phantom `muon_beta2`.

## Tool

After pulling pod logs, run:

```bash
scripts/mine_axis_deltas.py 'pod_pulls/**/*.txt' 'legs/**/logs/*.txt'
```

Use `--baseline path/to/baseline_log.txt` when comparing against a known parent. The output flags best score, quant damage, byte margin, failure class, and changed axes versus baseline.
