# 2026-04-30 Mined Day Report

## Evidence Roots

- Current 8x pod log pull: `pod_pulls/8x_35690435_20260430_logs_20260430_014905/`
- Best 10k artifact pull: `pod_pulls/8x_35002131_20260429_sp10240_mlp375_promising_20260429_202105/`
- 1x mechanics pull: `pod_pulls/1x_35778269_20260429_full_20260429_152155/`
- Axis miner: `scripts/mine_axis_deltas.py`

## Best Verified Standard 8x Lane

`2026-04-29_mikey_sp10240_full124_loop2_late050_mlp375_clip1305_8x`

| Metric | Value |
| --- | ---: |
| post-EMA BPB | `1.086793` |
| quantized BPB | `1.098550` |
| sliding BPB | `1.08229317` |
| TTT BPB | `1.08066751` |
| eval-only TTT5 BPB | `1.08051259` |
| total bytes | `15,954,736` |
| byte margin | `45,264` |

This remains the verified personal-record family.

## Completed 10k/8x Comparators

| Run | Post-EMA | Quant | Sliding/TTT | Bytes | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| 11L MLP3.75 loop2 late050 | `1.086793` | `1.098550` | `1.082293 / 1.080668` | `15,954,736` | best stable |
| 10L MLP4 loop2 late050 | `1.088100` | `1.103305` | cut before full eval | `15,341,787` | worse neural/quant despite bytes |
| 10L MLP4 + PPM byte | `1.087844` | `1.102560` | cut before full eval | `15,352,187` | PPM not worth applying to weak base |
| Whale 12L dim576 MLP4 int5 | `1.076487` | `1.112974` | cut | `16,701,823` | neural ceiling, quant/size failure |
| Whale 12L dim576 MLP3.75 attn6/mlp5 | `1.078456` | `1.107700` | cut | `17,587,249` | better quant, worse size |

## Inferred Axes

1. Quant policy was under-mined.
   - Whale had strong neural BPB and bad quant/size, so the missing axis was module-specific quant allocation.
   - `embed_bits`, per-module clip, mixed int5/int6/int7/int8, GPTQ calibration, and pass-through tensors should be checked before body changes when neural is good.

2. Whale is not primary, but it is not useless.
   - Whale proves a `1.076-1.078` neural body exists.
   - Its failure class is `size_overflow + quant_damage`.
   - It should feed quant-policy ideas, not be the main body under time pressure.

3. Mikey 10k MLP3.75 is the current carrier body.
   - Stable under cap by only `45KB`, so it needs byte-budget tools before capacity tools.
   - MLP micro-bumps are risky because the byte margin is tiny.

4. Loop repair is live but must be isolated.
   - Parent late050 with loop warmup is the proven best.
   - New late060/no-warm runs are useful, but they change loop schedule and cannot prove quant policy by themselves.

5. Dim pushes need FA3 legality first.
   - `dim528` failed because `head_dim=66` is not FA3-valid.
   - With `num_heads=8`, safe model dims are multiples of `64`: `512`, `576`, `640`, etc.

6. 1x mechanics evidence is useful, but not the scoring body.
   - EMA off saved about `0.054` BPB in short 1x no-loop mechanics.
   - WD `0.5` beat WD `0.3` by about `0.0014` BPB.
   - Loops at early activation caused a large step/throughput loss in that mechanics setting.
   - User decision still marks no-loop as dead for scoring; keep these as optimizer/runtime signals only.

7. Kernel work has a low ceiling here.
   - Cuda graphs/reduce-overhead can help at 8x grad_accum=1, but expected gain is small.
   - FP8, PACK_QKV, and compile autotune paths were bad or too expensive.

8. Muon/Shampoo is a hidden optimizer axis.
   - The current runner prints `muon_beta2=0.95`, but the `Muon` optimizer implementation does not use `muon_beta2`.
   - Current Muon is momentum plus Newton-Schulz polarization, effectively the `beta2=0` Shampoo-side update shown in the Muon/Shampoo equivalence diagram.
   - A real Shampoo accumulator / preconditioner EMA is therefore an untested axis, not an already-tested setting.
   - Do not treat previous `muon_beta2` log lines as evidence that beta2 preconditioning was tested.

## Active/Incomplete Runs At Pull Time

`2026-04-30_mikey_sp10240_mlp375_loop_late060_nowarm_8x`

- Pulled log only reached `500` steps.
- Same body as best 11L MLP3.75, but `enable_looping_at=0.60`, `loop_warmup_enabled=False`.

`2026-04-30_mikey_sp10240_mlp375_loop_late060_nowarm_emb7_mlpclip12_8x`

- Pulled/live log reached `3500` steps.
- Loop enabled at step `3567`, so final score was not yet available.
- This run changes both loop schedule and quant policy. It is a useful swing, but not a clean isolation of `embed_bits=7 + MLPClip12`.

## Missing Clean Test

If the current `emb7/mlpclip12` result is noisy or ambiguous, the clean next test is:

```text
best 11L MLP3.75 parent
same late050 loop activation
same loop warmup
only change embed_bits=7 and mlp_clip_sigmas=12.0
```

This isolates the newly discovered quant-policy axis from loop timing.

## Next Decision Rules

1. If `emb7/mlpclip12` lowers total bytes and does not worsen quant/sliding, promote quant policy immediately.
2. If late060/no-warm improves neural but quant remains bad, keep loop schedule and continue quant-policy search.
3. If late060/no-warm worsens neural, revert to late050/warmup and test quant policy alone.
4. Do not spend another 8x run on whale until a quant-policy fix proves it can reduce quant damage by at least `0.02` BPB or save the missing bytes.
5. Do not use dim528/dim536 again with FA3. Use dim512 or dim576+ only.
6. If quant policy fails and there is time for optimizer work, test a real Shampoo/preconditioner-EMA implementation separately from `muon_momentum_warmup`; current `muon_beta2` is only a logged no-op.

## Command Used

```bash
scripts/mine_axis_deltas.py 'pod_pulls/**/*.txt' 'pod_pulls/**/*.log' \
  --baseline pod_pulls/8x_35002131_20260429_sp10240_mlp375_promising_20260429_202105/mikey_sp10240_full124_loop2_late050_mlp375_clip1305_8x_seed444.txt \
  --top 80
```
