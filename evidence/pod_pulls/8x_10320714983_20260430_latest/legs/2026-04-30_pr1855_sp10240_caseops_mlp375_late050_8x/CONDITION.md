# PR1855 SP10240 CaseOps MLP3.75 Late050

Run label: `new_experiment`, `standard_8x`.

This runner combines the new SP10240 CaseOps tokenizer/data sidecar with the
strongest standard-10k neural body pressure from today: MLP3.75 and late050
loop2. It keeps PR1855 LQER/pergroup/phased-TTT compression/eval machinery.

## Source

- Donor runner: `legs/2026-04-30_pr1855_sp8192_lqer_smeargate_repro_8x/run.py`
- Donor SHA256: `454f710d174be80f4603069ca952833d694f60d1d34c0c25703528323bc8878b`
- Neural parent: `2026-04-29_mikey_sp10240_full124_loop2_late050_mlp375_clip1305_8x`
- Data builder: `scripts/prepare_sp10240_caseops_data.py`

## Fixed Condition

- Dataset: `/workspace/SOTA_FINAL/data/datasets/fineweb10B_sp10240_caseops/datasets/datasets/fineweb10B_sp10240_lossless_caps_caseops_v1_reserved`
- Tokenizer: `/workspace/SOTA_FINAL/data/datasets/fineweb10B_sp10240_caseops/datasets/tokenizers/fineweb_10240_bpe_lossless_caps_caseops_v1_reserved.model`
- Vocab size: `10240`
- CaseOps sidecar: required, `fineweb_val_bytes_*.bin`
- Seed: `444`
- GPU count: `8`
- Build seconds: `600`
- Eval/compression seconds: `600`
- Size cap: `16000000`

## Model/Eval Policy

- Body: `num_layers=11`, `model_dim=512`, `mlp_mult=3.75`
- Loop: `num_loops=2`, `loop_start=3`, `loop_end=5`, `enable_looping_at=0.50`
- Winddown: `warmdown_frac=0.85`
- Quant/compression: pergroup, embed int7, matrix int6, LQER asym rank 4
- Eval: PR1855 phased LoRA TTT, prefix docs `2500`, phases `3`

## Command

```bash
cd /workspace/sota_rascal/legs/2026-04-30_pr1855_sp10240_caseops_mlp375_late050_8x
./launch_8x.sh
tail -f logs/pr1855_sp10240_caseops_mlp375_late050_8x_seed444.txt
```
