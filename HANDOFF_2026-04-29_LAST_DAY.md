# 2026-04-29 Last-Day Handoff

## Critical Context

We are in `sota_rascal` and the active research lane is Mikey/SP10240, not Rascal. The best verified personal-record family is the 10k vocab Mikey loop2 line. Do not drift back to 4k/5k or NGRAM unless explicitly asked.

The user correction is important: the next higher-dimension 10L loop2 test should use **MLP 3.75**, not MLP4. The higher dim is meant to spend capacity differently while keeping the MLP ratio that produced the best 10k signal.

## Verified Best 10k Result

Path:

```text
/workspace/sota_rascal/legs/2026-04-29_mikey_sp10240_full124_loop2_late050_mlp375_clip1305_8x
```

Key condition:

```text
vocab_size=10240
num_layers=11
model_dim=512
embedding_dim=512
mlp_mult=3.75
num_loops=2
loop_start=3
loop_end=5
enable_looping_at=0.50
matrix_clip_sigmas=13.05
seed=444
```

Best logged score:

```text
Total submission size quantized+brotli: 15954736 bytes
quantized_sliding_window val_bpb: 1.08229317
quantized_ttt val_bpb: 1.08066751
```

1x eval-only TTT5 on this same artifact improved to about:

```text
evalonly_quantized_ttt_epochs5 val_bpb: 1.08051259
```

## Size Evidence

These are actual pod log results, not estimates:

```text
10L MLP3.5: 14,242,610 bytes, bad quality
10L MLP4.0: 15,343,214 bytes, TTT 1.08242391
10L MLP4.35: 16,114,298 bytes, over cap by 114,298
11L MLP3.75: 15,954,736 bytes, TTT 1.08066751
11L MLP3.90 SpinQuant: 16,330,939 bytes, over cap
```

There is no verified completed plain `10k 11L MLP4.0` result in the current pod logs.

## Current/Recent Active Run

The current active/most recent experimental runner was:

```text
/workspace/sota_rascal/legs/2026-04-29_mikey_sp10240_full124_seqloop3_late55_68_80_9l_mlp4_clip1305_compilefix_8x
```

It is a high-risk 9L sequential loop3 test:

```text
vocab_size=10240
num_layers=9
mlp_mult=4.0
model_dim=512
num_loops=3
loop_schedule_fracs=(0.55, 0.68, 0.80)
compile_fullgraph=False
```

Watch command:

```bash
tail -f /workspace/sota_rascal/legs/2026-04-29_mikey_sp10240_full124_seqloop3_late55_68_80_9l_mlp4_clip1305_compilefix_8x/logs/mikey_sp10240_full124_seqloop3_late55_68_80_9l_mlp4_clip1305_compilefix_8x_seed444.txt
```

Observed warning spam:

```text
torch._dynamo hit config.recompile_limit
input_ids size mismatch expected 32 actual 30/31
```

Interpretation: compile/log noise from `dynamic=False` and rank-local batch size variation, not the main score issue.

Observed early score after first loop stage:

```text
4000/20000 val_loss: 3.1234 val_bpb: 1.1722
```

Do not over-interpret this at 5-6 minutes, but it is a loop-shock warning.

Prepared logfix variants, not launched by Codex:

```text
/workspace/sota_rascal/legs/2026-04-29_mikey_sp10240_full124_seqloop3_late55_68_80_9l_mlp4_clip1305_logfix_8x
/workspace/sota_rascal/legs/2026-04-29_mikey_sp10240_full124_seqloop3_late55_68_80_9l_mlp45_clip1305_logfix_8x
```

They set:

```text
compile_fullgraph=False
compile_dynamic=True
```

## Next Test To Prepare

User asked for a `10L loop2` at higher dim and corrected that MLP must be `3.75`.

Recommended next test car:

```text
2026-04-29_mikey_sp10240_full124_loop2_late050_10l_dim528_mlp375_clip1305_8x
```

Planned condition:

```text
source: clone 2026-04-29_mikey_sp10240_full124_loop2_late050_10l_mlp4_clip1305_8x
vocab_size=10240
num_layers=10
xsa_last_n=10
model_dim=528
embedding_dim=528
num_heads=8
num_kv_heads=4
head_dim=66
mlp_mult=3.75
num_loops=2
loop_start=3
loop_end=5
enable_looping_at=0.50
matrix_clip_sigmas=13.05
seed=444
build_seconds=600
eval_seconds=600
size_cap_bytes=16000000
```

Risk note: `head_dim=66` is even and valid for RoPE, but not a multiple of 8. This may be slower than 512. If tensor-core alignment is prioritized, use dim576, but that is a much larger size/step hit and should not be the first higher-dim try.

## Pod Details

8x pod:

```text
ssh -i /home/frosty40/.ssh/id_ed25519_apollo -p 9438 root@35.192.20.187
```

1x pod:

```text
ssh -i /home/frosty40/.ssh/id_ed25519_apollo -p 4997 root@34.48.171.202
```

## Operating Rules

- Do not stop or launch 8x runs unless explicitly told.
- User drives the 8x pod manually from Jupyter terminal.
- Preserve datasets, tokenizers, logs, and pulled artifacts.
- New tests must be dated self-contained `legs/YYYY-MM-DD_<name>/run.py` files.
- Do not use shell overrides for critical lab settings.
- If comparing scores, name the exact metric: `quantized`, `quantized_sliding_window`, `quantized_ttt`, or eval-only TTT.
- Do not recommend no-loop paths; user considers no-loop dead.
- Do not touch NGRAM Mikey; NGRAM is quarantined and not part of this lane.

