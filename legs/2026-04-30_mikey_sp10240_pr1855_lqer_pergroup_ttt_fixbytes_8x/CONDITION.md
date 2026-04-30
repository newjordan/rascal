# Mikey SP10240 PR1855 LQER/Pergroup/TTT Port FixBytes

Run label: `new_experiment`, `standard_8x`.

This is not a PR1855 exact reproduction. It ports the PR1855 compression and
eval machinery onto the known-best standard SP10240 Mikey parent without
requiring CaseOps data.

This copy repairs the first smoke failure from the initial 10k port:
diagnostic eval now uses the standard SentencePiece byte-count LUT when
`caseops_enabled=False` and no `fineweb_val_bytes_*.bin` sidecar exists.

## Sources

- Parent 10k runner: `legs/2026-04-29_mikey_sp10240_full124_loop2_late050_mlp375_clip1305_8x/run.py`
- Parent SHA256: `b7a22bf7287ccefbad0ab879a201bd207060b97eec53a080e6ffd32da5248648`
- Donor PR1855 runner: `legs/2026-04-30_pr1855_sp8192_lqer_smeargate_repro_8x/run.py`
- Donor SHA256: `454f710d174be80f4603069ca952833d694f60d1d34c0c25703528323bc8878b`

## Fixed 10k Condition

- Dataset: `/workspace/SOTA_FINAL/data/datasets/fineweb10B_sp10240`
- Tokenizer: `/workspace/SOTA_FINAL/data/tokenizers/fineweb_10240_bpe.model`
- Vocab size: `10240`
- Train shards: `124`
- Val shards: `1`
- Seed: `444`
- GPU count: `8`
- Build seconds: `600`
- Eval/compression seconds: `600`
- Size cap: `16000000`

## Kept From Known-Best 10k Parent

- `num_layers=11`
- `model_dim=512`
- `mlp_mult=3.75`
- `num_loops=2`
- `loop_start=3`
- `loop_end=5`
- `enable_looping_at=0.50`
- `qk_gain_init=5.25`
- Standard SP10240 data/tokenizer, not CaseOps.

## Ported From PR1855

- `compressor=pergroup`
- LQER asymmetric rank correction: `lqer_enabled=1`, `lqer_asym_enabled=1`, `lqer_rank=4`, `lqer_top_k=3`
- PR1855 clip family: `embed_bits=7`, `mlp_clip_sigmas=11.5`, `attn_clip_sigmas=13.0`, `embed_clip_sigmas=14.0`
- Sparse attention gate and SmearGate enabled.
- PR1855 phased LoRA TTT path: prefix docs `2500`, phases `3`, LoRA rank `80`, Adam TTT beta2 `0.99`, weight decay `0.5`
- Fast compression settings: `gptq_reserve_seconds=0.5`, `gptq_calibration_batches=16`

## Deliberately Not Ported

- CaseOps tokenizer/data and byte sidecars. That requires building a 10k
  CaseOps tokenizer/dataset and is a separate data-prep lane.

## Command

```bash
cd /workspace/sota_rascal/legs/2026-04-30_mikey_sp10240_pr1855_lqer_pergroup_ttt_fixbytes_8x
./launch_8x.sh
tail -f logs/mikey_sp10240_pr1855_lqer_pergroup_ttt_fixbytes_8x_seed444.txt
```
