# Rascal lab

Parameter-golf experiment snapshot. The goal is a language model whose
compressed artifact fits in 16,000,000 bytes and trains in under 10 minutes on
8xH100. Scoring is bits per byte on FineWeb validation. This repository keeps
the trainers, run conditions, and lab tools. Dataset shards and checkpoints
stay outside git.

This snapshot has no `LICENSE` file.

## Layout

| Path | What it is |
| --- | --- |
| `legs/<name>/` | One recorded run. `run.py` and `CONDITION.md`, when present, are the run conditions. |
| `conditions/*.env` | Named condition artifacts for controls, proxies, and smoke runs. |
| `control_NOCHANGE/` | Frozen control trainers. Do not edit them in place. |
| `rascal_anchor_NEVERTOUCH.py` | Frozen anchor trainer. Do not edit it in place. |
| `rascal_loop/`, `midnight/`, `mom95_mlp45/`, `mom95_mlp45_kernel/`, `whale_donor/` | Active trainer lanes. |
| `scripts/` | Dataset, validation, and log-mining CLIs. |
| `rascal_lab/` | Shared checks those CLIs use. |
| `agent_hub/` | Current campaign index and the experiment ledger. Start here before reading old handoffs. |
| `quarantine/` | Preserved evidence, not a clean input. Read `QUARANTINE.md` before using it. |
| `pod_pulls/` | Text pulled back from pods. Nested checkouts are not vendored. |

Campaign notes with dates in the filename (`HANDOFF_*.md`, `ACTIVE_DECISION_*.md`, `agent_hub/ACTIVE_STATE_*.md`) are historical readouts. They are not a substitute for the leg that actually ran.

## Setup

The tooling tests need Python 3.11 or newer. This checkout was exercised with Python 3.12.3.

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest
```

`make test` runs the same suite. `make compile` syntax-checks the shared package, `scripts/`, and the active trainer lanes. The check only compiles those files.

Trainers import `torch`, `numpy`, and `sentencepiece`. Triton, FlashAttention, and zstandard are optional and already guarded in the trainers. Those packages come from the GPU pod image. The `.[dev]` extra is pytest. Checkpoints stay on the pod.

`sentencepiece` is only required for `scripts/prepare_sp10240_caseops_data.py`:

```bash
python3 -m pip install -e ".[data]"
```

## Tools

Shard header check. Train shards, validation token shards, and `fineweb_val_bytes_*.bin` sidecars are counted separately. A sidecar name also matches the glob `fineweb_val_*.bin`. The validator classifies filenames, which keeps sidecars out of the val count.

```bash
python3 scripts/validate_fineweb_shards.py DATASET_DIR
python3 scripts/validate_fineweb_shards.py --inventory DATASET_DIR
python3 scripts/validate_fineweb_shards.py DATASET_DIR --expect-train 80 --expect-val 1
```

Exit `0` when every shard header matches its file size and any `--expect-*` count matches. Exit `1` when a header or a count is wrong. Exit `2` when the directory or the shard set is missing. `--inventory` only prints:

```text
train=N
val=N
val_bytes=N
```

Log miner. The `best` column is the first present value among test-time-training, sliding-window, quantized, and post-EMA bits per byte. Rows sort by that value, lowest first. A byte margin of zero stays on the cap instead of being treated as missing. Changed hyperparameters are separated with semicolons so the markdown table keeps its columns.

```bash
python3 scripts/mine_axis_deltas.py logs/*.txt --baseline logs/control.txt --top 20
```

Experiment ledger. `agent_hub/experiments_2026-04-30.tsv` is the machine-readable campaign table. `rascal_lab.ledger` checks that a row marked `over_cap` is actually over 16,000,000 bytes, and that `complete` or `neural_miss` rows with a size are not.

### Data scripts

These talk to disks, Hugging Face, or a pod. CI skips them. They move local docs, tokenizer files, or dataset shards.

| Script | Role |
| --- | --- |
| `scripts/prepare_sp10240_caseops_data.py` | Build an SP10240 CaseOps tokenizer and uint16 shards from a local `docs_selected.jsonl`. |
| `scripts/build_sp10240_caseops_local.sh` | Wrapper. `DOCS_PATH` and `OUT_ROOT` override the defaults. The script runs from this repo, not a fixed home directory. |
| `scripts/build_sp5200_dataset.sh` | Pod dataset build. `PYTHON_BIN` defaults to `/venv/main/bin/python3` when that file exists, otherwise `python3`. |
| `scripts/download_sp10240_first80_from_hf.sh` | Fetch the first 80 SP10240 train shards plus one val shard, then header-check them. |
| `scripts/download_sp10240_full124_from_hf.sh` | Same for 124 train shards. |
| `scripts/upload_sp10240_caseops_to_hf.sh` | `check` (default) or `upload`. Counts sidecars separately and reports missing files before it looks for the `hf` CLI. |
| `scripts/stream_pr1855_caseops_to_pod.sh` | Stream local docs to the recorded pod command. `DOCS`, `KEY`, `HOST`, and `PORT` override the defaults. |
| `scripts/queue_10k_after_current_pr1855.sh` | Wait for a torchrun pid, then start one leg. `TENK_DIR` and `LOG_DIR` override the defaults. |
| `scripts/compression_sweep_4k_8x.py` | Reload a local checkpoint and sweep quant/compression pairs. It never trains. Legal size is `<= 16000000` bytes. |

Stopping a CaseOps build at `--max-train-shards` leaves a partial train buffer unwritten. The completed manifest's `tokens_train` is the number of tokens stored in shard files. `unwritten_train_tokens` is the dropped tail.

## Running an experiment

Read the condition file or the leg's `CONDITION.md` / `run.py` before launching. A 4x or 1x command is a mechanics proxy unless that condition says otherwise. Quarantined 4k tokenizer surfaces are not audited SP4096 evidence.

```bash
torchrun --standalone --nproc_per_node=8 rascal_loop/train_gpt_loop_8xgpu.py
torchrun --standalone --nproc_per_node=8 midnight/train_gpt_8xgpu.py
```

Full runs need the dataset, tokenizer, GPU count, and wallclock recorded for that condition. `*.bin`, `*.pt`, and other checkpoint or shard types are gitignored. See `PUBLIC_ARTIFACTS_NOTE.md`.

## Tests

`tests/` covers shard headers, the val/sidecar split, cap-stop token accounting, log ranking, byte-shuffle round trips, the 16,000,000-byte cap, the checked-in ledger, and the shell wrappers' failure paths. Pytest collected 69 tests here. The suite stays on local synthetic shards and the checked-in ledger.

GitHub Actions (`.github/workflows/ci.yml`) installs `.[dev]` on Python 3.12, runs `pytest`, and syntax-checks the active lanes.

## Rules that stay in force

- Do not edit `control_NOCHANGE/` or `rascal_anchor_NEVERTOUCH.py` unless that is the task.
- Do not restore quarantined files until `quarantine/2026-04-28_tokenizer_provenance/MANIFEST.md` says the condition is proven.
- Do not commit raw shards, checkpoints, or pod binaries. Compact text evidence belongs in `agent_hub/` or `notes/`.
- Leaderboard policy is summarized in `LEADERBOARD_RULES.md`.
