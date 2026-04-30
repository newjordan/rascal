# PR1855 SP10240 CaseOps Repro Embed6 Size Cut

Run label: `new_experiment`, `standard_8x`.

This is the first size-cut response to the strong SP10240 CaseOps repro. It
keeps the PR1855 body shape, loop policy, LQER rank/top-k, pergroup compression,
and phased TTT fixed, and changes only `embed_bits: 7 -> 6`.

## Source

- Parent runner: `legs/2026-04-30_pr1855_sp8192_lqer_smeargate_repro_8x/run.py`
- Parent SHA256: `454f710d174be80f4603069ca952833d694f60d1d34c0c25703528323bc8878b`
- Data builder: `scripts/prepare_sp10240_caseops_data.py`

## Fixed Condition

- Dataset: `/workspace/SOTA_FINAL/data/datasets/fineweb10B_sp10240_caseops/datasets/datasets/fineweb10B_sp10240_lossless_caps_caseops_v1_reserved`
- Tokenizer: `/workspace/SOTA_FINAL/data/datasets/fineweb10B_sp10240_caseops/datasets/tokenizers/fineweb_10240_bpe_lossless_caps_caseops_v1_reserved.model`
- Vocab size: `10240`
- CaseOps sidecar: required, `fineweb_val_bytes_*.bin`
- Seed: `42`
- GPU count: `8`
- Build seconds: `600`
- Eval/compression seconds: `600`
- Size cap: `16000000`

## Model/Eval Policy

- PR1855 body: `num_layers=11`, `model_dim=512`, `mlp_mult=4.0`
- Loop: `num_loops=2`, `loop_start=3`, `loop_end=5`, `enable_looping_at=0.35`
- Winddown: `warmdown_frac=0.85`
- Quant/compression: pergroup, embed int6, matrix int6, LQER asym rank 4
- Eval: PR1855 phased LoRA TTT, prefix docs `2500`, phases `3`

## Command

```bash
cd /workspace/sota_rascal/legs/2026-04-30_pr1855_sp10240_caseops_repro_embed6_8x
./launch_8x.sh
tail -f logs/pr1855_sp10240_caseops_repro_embed6_8x_seed42.txt
```
