# 1x Pod Evidence 2026-04-29

## Pull

- Pod: `35778269`, direct SSH `root@34.48.171.202 -p 4997`
- Local evidence root: `pod_pulls/1x_35778269_20260429/`
- Pulled: Mikey logs, top-level run logs, recursion logs, `v5_results.md`, and source bodies.
- Not pulled: model checkpoints and quantized model blobs.

## Source Hashes At Pull Time

These hashes identify the files present on the pod when evidence was pulled.
Older logs do not print source SHA256, so treat them as condition evidence from
their logged hyperparameters plus this pulled file snapshot, not as a proven
source-hash lock.

| Remote path | SHA256 |
|---|---|
| `/workspace/Mikey_II/train_gpt.py` | `6cf4abeececb632be2ecea92cc94b725c292e117d588ceb4d229384acf495a37` |
| `/workspace/Mikey_II_v2/train_gpt.py` | `a896ef91d5fa9f61757f85e96d3f84a847149b07ddeea088cbfed75390cbb3fb` |
| `/workspace/Mikey_II_v3/train_gpt.py` | `2251402e0cb87b3c5f3a85f90f29dd2d345d200b355256c51ae79d07b0e697e0` |
| `/workspace/Mikey_II_v5/train_gpt.py` | `62527f549e2bff3b46fe7c26be42fe69a7780ddb88365888bff77c31ab821246` |

## Completed 1x Mechanics Results

All rows below are `1x H100` mechanics evidence on `SP8192`, not standard 8x
leaderboard conditions and not direct SP5200 results.

| ID | loops | warmdown | EMA | Muon warmup | MLP | steps | raw bpb | quant bpb | bytes | tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Mikey_II` full | 2 | 0.72 | 0.9965 | 1500 | 3.0 | 5225 | 1.0981 | 1.11011717 | 13,579,951 | 865,055 |
| `Mikey_II_v2` full | 2 | 0.72 | 0.9965 | 1500 | 5.25 | 3407 | 1.0989 | 1.10812613 | 19,085,979 | 549,497 |
| `v5_baseline_noloop` | 0 | 0.72 | 0.9965 | 1500 | 3.0 | 821 | 1.2057 | 1.26926549 | 13,595,691 | 1,097,845 |
| `v5_no_ema` | 0 | 0.72 | 0.0 | 1500 | 3.0 | 818 | 1.2060 | 1.21545317 | 13,593,522 | 1,092,736 |
| `v5_muon100` | 0 | 0.72 | 0.0 | 100 | 3.0 | 823 | 1.2239 | 1.23359889 | 13,596,983 | 1,099,627 |
| `v5_wd05` | 0 | 0.5 | 0.0 | 1500 | 3.0 | 822 | 1.2033 | 1.21220168 | 13,594,517 | 1,099,248 |
| `v5_wd03` | 0 | 0.3 | 0.0 | 1500 | 3.0 | 824 | 1.2053 | 1.21357581 | 13,594,651 | 1,101,018 |
| `v5_loops2` | 2 | 0.5 | 0.0 | 1500 | 3.0 | 561 | 1.2253 | 1.23319879 | 13,597,605 | 749,694 |

## Calculations

- EMA off improved the short 10 minute quant result by `0.05381232` bpb versus baseline.
- `warmdown_frac=0.5` beat `warmdown_frac=0.3` by `0.00137413` quant bpb.
- `warmdown_frac=0.5` beat no-EMA baseline by `0.00325149` quant bpb.
- `muon_momentum_warmup_steps=100` regressed by `0.02139721` quant bpb versus `v5_wd05`.
- `NUM_LOOPS=2` at 10 minutes regressed by `0.02099711` quant bpb versus `v5_wd05`.
- Looping reduced 1x short-run steps from `822` to `561`, a `31.75%` step loss, and tok/s from `1,099,248` to `749,694`, a `31.80%` throughput loss.
- SP8192 to SP5200 tied embedding savings remain `1,531,904` parameters, enough to buy either moderate MLP width or a layer, but the 1x evidence does not support assuming loops pay for themselves in a 600s budget.

## Implication For SP5200 Queue

The strongest completed 1x 8k mechanics signal was:

```text
NUM_LOOPS=0, warmdown_frac=0.5, EMA=0, muon_warmup=1500, MLP_MULT=3.0
```

## Current Path Classification

As of the active 8x research decision, `NUM_LOOPS=0` is a dead scoring path.
The 1x no-loop rows remain useful only as mechanics evidence for EMA, warmdown,
and throughput; do not promote new no-loop scoring runners from them.

Loop optimization remains a live path. The PR1493 accepted result proves loops
can matter under the full scoring contract, so future loop work should optimize
loop topology, activation timing, quantization, and byte budget rather than drop
recurrence entirely.
