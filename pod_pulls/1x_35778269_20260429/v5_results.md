# v5 architecture/schedule sweep results

Hardware: 1x H100 80GB. Goal: ≤ 1.08 quantized val_bpb, ≤ 16MB submission, 10min build budget on competition's 8x H100 (≈ 80min here).

Methodology: short (10min, 600s) experiments on 1 GPU rank configs against each other. Final candidate validated with 80min run.

## Constraints
- Quant val_bpb ≤ 1.08 (target: beat)
- Compressed file ≤ 16MB
- Build wallclock ≤ 10min on 8x H100 (~80min on 1x H100)
- Eval+compression wallclock ≤ 10min

## Reference: v2 (mlp_mult=5.25 + smart loops)
- 80min/1GPU: raw bpb 1.0982, quant bpb 1.1081, file size 19MB ❌ over 16MB

## Experiments (10min/1x H100 each unless noted)

| ID | NUM_LOOPS | MUON_WMU | WARMDOWN | MLP_MULT | OTHER | tok/s | step@cap | val_bpb_quant | size MB |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 0 | 1500 | 0.72 | 3.0 | EMA=.9965 | 1098k | 821 | 1.269 | 13.5 |
| no_ema | 0 | 1500 | 0.72 | 3.0 | EMA=0 | 1093k | 818 | **1.215** | 13.5 |
| muon100 | 0 | 100 | 0.72 | 3.0 | EMA=0 | 1100k | 823 | 1.234 (regressed +0.019) | 13.5 |
| wd05 | 0 | 1500 | 0.5 | 3.0 | EMA=0 | 1099k | 822 | **1.212** (-0.003) | 13.5 |
| wd03 | 0 | 1500 | 0.3 | 3.0 | EMA=0 | 1101k | 824 | 1.214 | 13.5 |
| loops2 | 2 | 1500 | 0.5 | 3.0 | EMA=0,loop@0.05 | 750k | 561 | 1.233 (regressed) | 13.5 |

Looping cost: 822→561 steps. Per-step gain didn't compensate. Confirmed NUM_LOOPS=0 best for 10min budget.



Note: faster Muon momentum ramp HURTS in short runs. The 1500-step ramp keeps momentum at 0.92-0.96 throughout 800 steps, which apparently is the right zone for this model/data. Reaching 0.99 by step 100 over-momentums and hurts convergence.


**Key finding**: pre-EMA val_bpb=1.206 → post-EMA val_bpb=1.264. EMA hurts by 0.058 in short runs (decay 0.9965 averages over 285-step half-life, but training is only 821 steps so EMA pulls in bad early weights).


