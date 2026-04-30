# PR1855 SP8192 LQER SmearGate Repro

Condition source: openai/parameter-golf PR #1855, merge commit `510d03e0fc355406c9fd06f92d23b8c5aedea7fb`.

Run label: `exact_reproduction` once the CaseOps dataset and `lrzip` are present.

Primary metric: post-phased-TTT validation BPB. PR1855 reports seed 42 at `1.05989454` BPB and three-seed mean `1.06107587`.

Required data:

- `DATA_DIR=/workspace/SOTA_FINAL/data`
- `DATA_PATH=/workspace/SOTA_FINAL/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved`
- `TOKENIZER_PATH=/workspace/SOTA_FINAL/data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model`

Accepted log facts:

- seed 42 used `train_shards: 80`
- seed 42 used `val_tokens: 47851520`
- phased TTT reported `total_docs:50000`

Current pod blocker at creation time: `/workspace` had only 143M free, and the CaseOps dataset/tokenizer plus `lrzip` were absent.

Current status:

- `/workspace` has about 27G free after removing old non-active `Fartmagic` datasets.
- `lrzip` is installed on the pod.
- The tokenizer is staged on the pod.
- CaseOps data prep is running from the local canonical `docs_selected.jsonl` stream.
- The stream helper is locked to `--val-docs 50000 --max-train-shards 80` to match the accepted log.

Disk plan:

- Do not delete the active 10k dataset while an active 10k run is still in TTT.
- To prep this runner, free at least 30G if using `prep_caseops_streaming.sh`; it streams `docs_selected.jsonl` and only stores the generated CaseOps token shards.
- `Fartmagic` legacy datasets are the largest non-active candidates: standard SP4096 (~27G), SP1024 (~16G), and standard SP8192 (~15G).

Data gate:

- `launch_8x.sh` fails closed unless there are at least 80 train shards, at least one val shard, and matching `fineweb_val_bytes_*.bin` sidecars.

Run sequence after data and deps are ready:

```bash
cd /workspace/sota_rascal/legs/2026-04-30_pr1855_sp8192_lqer_smeargate_repro_8x
./launch_8x.sh
tail -f logs/pr1855_sp8192_lqer_smeargate_repro_8x_seed42.txt
```
