# PR1855 SP10240 CaseOps Repro 1x Mechanics Proxy

Run label: `mechanics_1x`.

This is not an official 8x leaderboard condition. It is a 1x H100 mechanics
proxy for the first true 10k CaseOps runner. It keeps the PR1855
CaseOps/LQER/pergroup/phased-TTT stack shape and changes only the execution
mechanics from 8x/600s to 1x/4800s.

## Source

- Parent runner: `legs/2026-04-30_pr1855_sp10240_caseops_repro_8x/run.py`
- Data builder: `scripts/prepare_sp10240_caseops_data.py`
- Dataset source on pod: `Frosty40/10k_caseops_golfer`

## Fixed Condition

- Dataset: `/workspace/SOTA_FINAL/data/datasets/fineweb10B_sp10240_caseops/datasets/datasets/fineweb10B_sp10240_lossless_caps_caseops_v1_reserved`
- Tokenizer: `/workspace/SOTA_FINAL/data/datasets/fineweb10B_sp10240_caseops/datasets/tokenizers/fineweb_10240_bpe_lossless_caps_caseops_v1_reserved.model`
- Vocab size: `10240`
- CaseOps sidecar: required, `fineweb_val_bytes_*.bin`
- Seed: `42`
- GPU count: `1`
- Build seconds: `4800`
- Eval/compression seconds: `4800`
- Size cap: `16000000`

## Model/Eval Policy

- PR1855 body: `num_layers=11`, `model_dim=512`, `mlp_mult=4.0`
- Loop: `num_loops=2`, `loop_start=3`, `loop_end=5`, `enable_looping_at=0.35`
- Winddown: `warmdown_frac=0.85`
- Quant/compression: pergroup, embed int7, matrix int6, LQER asym rank 4
- Eval: PR1855 phased LoRA TTT, prefix docs `2500`, phases `3`

## Command

```bash
cd /workspace/sota_rascal/legs/2026-04-30_pr1855_sp10240_caseops_repro_1x_mechanics
./launch_1x.sh
tail -f logs/pr1855_sp10240_caseops_repro_1x_seed42.txt
```
