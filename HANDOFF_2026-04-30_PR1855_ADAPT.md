# Handoff: PR1855 SOTA Adaptation

Timestamp: `2026-04-30T03:59:58Z`

## Situation

The active strategic pivot is to reproduce accepted PR1855:

- PR: `openai/parameter-golf#1855`
- Merge commit: `510d03e0fc355406c9fd06f92d23b8c5aedea7fb`
- Record path: `records/track_10min_16mb/2026-04-27_SP8192_LQER_SparseGate_BOSSmearFix_9HpStack_1.0611`
- Reported seed 42: `1.05989454` post-phased-TTT BPB
- Reported 3-seed mean: `1.06107587` post-phased-TTT BPB

This supersedes the local 10k Mikey/Rascal near-miss lane for now. The 10k lane was around `1.0805-1.081` best final TTT; PR1855 is the only currently known lane with a large enough margin to matter.

## Active Pod

Instance: `35690435`

SSH:

```bash
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/tmp/codex_vast_known_hosts -o ConnectTimeout=8 -i /home/frosty40/.ssh/id_ed25519_apollo -p 56335 root@206.125.32.60
```

Prepared pod leg:

```bash
/workspace/sota_rascal/legs/2026-04-30_pr1855_sp8192_lqer_smeargate_repro_8x
```

Prepared local leg:

```bash
/home/frosty40/sota_rascal/legs/2026-04-30_pr1855_sp8192_lqer_smeargate_repro_8x
```

## Current Live State

No training run was active at the snapshot. The only intentional active process is CaseOps data prep.

Data prep is running locally in tmux:

```bash
tmux attach -t pr1855_caseops
```

Data prep log:

```bash
tail -f /home/frosty40/sota_rascal/notes/runtime_logs/pr1855_caseops_tmux_20260430_035633.log
```

Snapshot readout:

- prep process active on pod: yes
- docs processed: at least `70000`
- train shards written: `1`
- val shards written: `4`
- val byte sidecars written: `4`
- pod free space: about `27G`
- current dataset size: about `172M`

The first `50000` docs are validation. Train shards start only after that, so low train count early is expected.

## Critical Corrections Already Made

Do not undo these.

1. `stream_prepare_caseops_data.py` now defaults to `--val-docs 50000`, not `10000`.
2. The active stream command passes `--val-docs 50000 --max-train-shards 80`.
3. `launch_8x.sh` fails closed unless:
   - `train >= 80`
   - `val >= 1`
   - `val_bytes == val`
4. `launch_8x.sh` uses a corrected val count that excludes `fineweb_val_bytes_*.bin`.
5. The bad partial 10k-val stream output was cleared before restarting the corrected stream.

Why this matters: accepted PR1855 seed42 log used:

- `train_shards: 80`
- `val_tokens: 47851520`
- phased TTT `total_docs:50000`

Anything built with 10k validation docs is not the accepted reproduction condition.

## Dataset Paths

Pod dataset target:

```bash
/workspace/SOTA_FINAL/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved
```

Pod tokenizer:

```bash
/workspace/SOTA_FINAL/data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
```

Local canonical source docs:

```bash
/home/frosty40/parameter-golf-lab/data/docs_selected.jsonl
```

## Status Commands

Pod data-prep status:

```bash
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/tmp/codex_vast_known_hosts -o ConnectTimeout=8 -i /home/frosty40/.ssh/id_ed25519_apollo -p 56335 root@206.125.32.60 'DATASET=/workspace/SOTA_FINAL/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved; train=$(find "$DATASET" -maxdepth 1 -type f -name "fineweb_train_*.bin" 2>/dev/null | wc -l); val=$(find "$DATASET" -maxdepth 1 -type f -name "fineweb_val_*.bin" ! -name "fineweb_val_bytes_*.bin" 2>/dev/null | wc -l); vb=$(find "$DATASET" -maxdepth 1 -type f -name "fineweb_val_bytes_*.bin" 2>/dev/null | wc -l); echo train=$train val=$val val_bytes=$vb; du -sh "$DATASET" 2>/dev/null || true; ps -eo pid,ppid,etimes,cmd | grep -E "stream_prepare|caseops" | grep -v grep || true; ps -eo pid,ppid,etimes,cmd | grep -E "torchrun|run.py" | grep -v grep || true; df -h /workspace'
```

Local stream offset:

```bash
ps -ef | grep stream_pr1855_caseops | grep -v grep
```

## Run Command After Data Prep

Only run after the data gate passes.

```bash
cd /workspace/sota_rascal/legs/2026-04-30_pr1855_sp8192_lqer_smeargate_repro_8x
./launch_8x.sh
tail -f logs/pr1855_sp8192_lqer_smeargate_repro_8x_seed42.txt
```

Expected early sanity signs from accepted seed42:

- `train_shards: 80`
- `val_tokens: 47851520`
- roughly `4945` steps in 600s
- post-EMA around `1.064`
- plain quant around `1.073`
- final phased TTT around `1.060`

If launch reports incomplete data, do not force it. Let prep finish or fix the dataset.

## Restarting Data Prep If It Dies

Use this only if the active prep has actually died. It clears only the PR1855 CaseOps generated shard directory.

```bash
tmux kill-session -t pr1855_caseops 2>/dev/null || true
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/tmp/codex_vast_known_hosts -o ConnectTimeout=8 -i /home/frosty40/.ssh/id_ed25519_apollo -p 56335 root@206.125.32.60 'kill $(pgrep -f "stream_prepare_caseops_data.py --docs -") 2>/dev/null || true; rm -f /workspace/SOTA_FINAL/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_*.bin'
LOG=/home/frosty40/sota_rascal/notes/runtime_logs/pr1855_caseops_tmux_$(date -u +%Y%m%d_%H%M%S).log
tmux new-session -d -s pr1855_caseops "cd /home/frosty40/sota_rascal && scripts/stream_pr1855_caseops_to_pod.sh > $LOG 2>&1"
echo "$LOG"
```

## Files Changed For This Pivot

- `legs/2026-04-30_pr1855_sp8192_lqer_smeargate_repro_8x/run.py`
- `legs/2026-04-30_pr1855_sp8192_lqer_smeargate_repro_8x/lossless_caps.py`
- `legs/2026-04-30_pr1855_sp8192_lqer_smeargate_repro_8x/prepare_caseops_data.py`
- `legs/2026-04-30_pr1855_sp8192_lqer_smeargate_repro_8x/stream_prepare_caseops_data.py`
- `legs/2026-04-30_pr1855_sp8192_lqer_smeargate_repro_8x/launch_8x.sh`
- `legs/2026-04-30_pr1855_sp8192_lqer_smeargate_repro_8x/CONDITION.md`
- `scripts/stream_pr1855_caseops_to_pod.sh`
- `notes/2026-04-30_live_decision_after_cut.md`

The accepted source files were copied from the PR1855 merge commit. Local helper changes are only for streaming local docs into the pod and fail-closing the run gate.

## Next Decisions

1. Finish CaseOps prep.
2. Run seed `42`.
3. If seed `42` reproduces within shouting distance, run seed `0`.
4. If seed `42` misses badly, first check:
   - FA3 runtime parity
   - `lrzip` present
   - `DATA_PATH` and `TOKENIZER_PATH`
   - `train_shards: 80`
   - `val_tokens: 47851520`
   - `CASEOPS_ENABLED=1`
   - `COMPRESSOR=pergroup`
   - `PHASED_TTT_PREFIX_DOCS=2500`

Do not branch into new architecture until the accepted PR1855 reproduction either runs cleanly or fails with a concrete condition mismatch.
