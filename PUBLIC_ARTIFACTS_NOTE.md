# Public Artifact Boundary

This repository snapshot is intended for source-level optimization agents.

The public GitHub repo intentionally excludes raw dataset shards and full model
checkpoints:

- `*.bin`
- `*.pt`
- `*.pth`
- `*.ckpt`
- `*.safetensors`
- archive bundles such as `*.tar`, `*.tgz`, `*.zip`, `*.zst`

Reason: the local checkout includes about 3 GB of binary run material, including
multiple full checkpoints over GitHub's 100 MB per-file limit. Those files are
not suitable for normal GitHub storage and should stay in pod pulls, Hugging
Face datasets, or other artifact storage.

Important local binary examples omitted from git:

- `pod_pulls/1x_35778269_20260429_full_20260429_152155/Mikey_II_v2/final_model.pt`
- `quarantine/2026-04-28_tokenizer_provenance/pod_artifacts/_artifacts/pod_artifacts/raphe_ii_seed42_final_model.pt`
- `pod_pulls/8x_35002131_20260429_sp10240_mlp375_promising_20260429_202105/final_model.pt`
- `pod_pulls/instance_35690435_20260430_pr1855_caseops_shards/`

Code, run definitions, condition notes, queue notes, audit docs, tokenizer
metadata, and small text evidence are kept in git for optimizer agents.
