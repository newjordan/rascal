# SP8192 Pivot Queue 2026-04-29

## Status

- Label: `standard_8x`
- Runner: `legs/2026-04-29_mikey_pr1493parity_sp8192_loop2_mlp4_8x/run.py`
- Remote runner: `/workspace/sota_rascal/legs/2026-04-29_mikey_pr1493parity_sp8192_loop2_mlp4_8x/run.py`
- `run.py` SHA256: `340b8351c86820b18b224bcf181ef85fcb4a145b9a12d4eb758fda728fd2ad57`
- `launch_8x.sh` SHA256: `cc84666ee592795ff4780b9c9faefa1008fc4d62a329a70683734e3dd65a9e6e`
- Data root: `/workspace/SOTA_FINAL/data`
- Dataset: `/workspace/SOTA_FINAL/data/datasets/fineweb10B_sp8192`
- Dataset status: `80` train shards, `1` val shard
- Tokenizer: `/workspace/SOTA_FINAL/data/tokenizers/fineweb_8192_bpe.model`
- Tokenizer SHA256: `0518df93315eaff6bc75ad4d9711b01c1b5e9ac32e8b3da603c040d8653f7f03`
- Vocab SHA256: `70089145f01fed1f8e376fae331c9160e4dc6453f0970f121bb95a9745347e5c`
- Val shard SHA256: `c4fd1d379ff3e99f7cb58398a73aba45da6434fa852b51212c744f3a55cac01b`
- Train 000000 SHA256: `d6dda65bbc989ee2452e83673f106d5004500d8cde6decc80a83373ced14efd3`
- Train 000079 SHA256: `3e8f5054c1e54f23163c07cdae0a660f1acea6312c4196ff496dbc737ab74dd7`

## Condition

This is the 8k pivot lane aligned to the accepted PR1493 public contract:

- `SEED=444`
- `VOCAB_SIZE=8192`
- `NUM_LAYERS=11`
- `XSA_LAST_N=11`
- `MLP_MULT=4.0`
- `NUM_LOOPS=2`
- `LOOP_START=3`
- `LOOP_END=5`
- `ENABLE_LOOPING_AT=0.35`
- `PARALLEL_RESIDUAL_START=7`
- `QK_GAIN_INIT=5.25`
- `WARMDOWN_FRAC=0.72`
- `EMA_DECAY=0.9965`
- `MUON_WD=0.095`
- `EMBED_WD=0.085`
- `ADAM_WD=0.02`
- `SLIDING_WINDOW_ENABLED=1`
- `TTT_ENABLED=1`

## Launch

Do not launch while the active 5200 `torchrun` is still holding GPUs.

```bash
cd /workspace/sota_rascal/legs/2026-04-29_mikey_pr1493parity_sp8192_loop2_mlp4_8x
./launch_8x.sh
```

Watch:

```bash
tail -f logs/mikey_pr1493parity_sp8192_loop2_mlp4_8x_seed444.txt
```

## Data Download Evidence

The SP8192 shard pull used the canonical Fartmagic helper with:

```bash
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
  /venv/main/bin/python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 80
```

Download log is under `/workspace/SOTA_FINAL/logs/sp8192_download_*.log`.
