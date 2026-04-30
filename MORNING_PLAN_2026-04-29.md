# Morning Plan 2026-04-29

## Current Decision

Drop Rascal. It is reference-only now. Main lane pivots back to the Mikey foundation.

## Current Pod Allocation

The active scoring pod is the 8x Vast pod `35002131` (`8ball_1`).

- Direct SSH: `ssh -i /home/frosty40/.ssh/id_ed25519_apollo -p 9438 root@35.192.20.187`
- GPU state at handoff: all 8x H100 idle/cold, 0 MiB used.
- `/workspace/data` points at `/workspace/SOTA_FINAL/data`.
- Do not clear tokenizer assets or datasets unless explicitly requested.

The prior 4x/2+1 pod `34405931` was shut down and is not active.

Remote inventory on the 8x pod:

- `/workspace/data/tokenizers/fineweb_5200_bpe.model`: present, SHA256 `8bedf3c9e6e7a8ee55539c0ec2627b0c7a4c08782c26559bf007c8d07cd5e07f`
- `/workspace/data/tokenizers/fineweb_5200_bpe.vocab`: present, SHA256 `cfcfc1fb515f5745a3f092f78744a35439de836a81511a0d39a893ee635fa524`
- `/workspace/data/datasets/fineweb10B_sp5200`: present and validated, `137` train shards and `1` val shard.
- `/tmp/torchinductor_root`: cleared to free 55G of compile cache.

NGRAM-bearing Mikey legs are quarantined locally under
`quarantine/2026-04-29_ngram_mikey_no_touch/`. New Mikey work must start from
`quarantine/2026-04-28_tokenizer_provenance/candidate_submissions/Mikey_II/train_gpt.py`.

## SP5200 Morning Lane

Active test queue uses clean Mikey v5, SP5200, standard 8x timing, `WD=0.5`,
and `EMA=0`.

- tokenizer: `/workspace/data/tokenizers/fineweb_5200_bpe.model`
- tokenizer SHA256: `8bedf3c9e6e7a8ee55539c0ec2627b0c7a4c08782c26559bf007c8d07cd5e07f`
- vocab SHA256: `cfcfc1fb515f5745a3f092f78744a35439de836a81511a0d39a893ee635fa524`
- dataset: `/workspace/data/datasets/fineweb10B_sp5200`
- expected shard inventory: `137` train shards and `1` val shard.
- validation: `summary train=137 val=1 bad=[]`

Prepared runners:

- `legs/2026-04-29_mikey_II_v5_sp5200_loop2_11l_8x/run.py`
- `legs/2026-04-29_mikey_II_v5_sp5200_loop2_11l_mlp35_8x/run.py`
- `legs/2026-04-29_mikey_II_v5_sp5200_loop2_12l_8x/run.py`

First run command on the pod:

```bash
cd /workspace/sota_rascal/legs/2026-04-29_mikey_II_v5_sp5200_loop2_11l_8x
./launch_8x.sh
```
