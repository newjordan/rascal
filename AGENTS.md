# Repository Guidelines

## Project Structure & Module Organization

This repository is a parameter-golf experiment lab, not a packaged Python
library. Top-level directories are experiment lanes, usually containing one
primary `train_gpt*.py` file plus optional `README.md` and `submission.json`
metadata. Stable anchors live in `control_NOCHANGE/` and
`rascal_anchor_NEVERTOUCH.py`; do not edit them in place. Active recorded run
conditions live in `conditions/*.env`. Quarantined 4k tokenizer/run surfaces,
pod logs, and submission candidates live under
`quarantine/2026-04-28_tokenizer_provenance/`.

## Build, Test, and Development Commands

There is no global build step. Run a specific experiment file directly.

```bash
torchrun --standalone --nproc_per_node=8 rascal_loop/train_gpt_loop_8xgpu.py
torchrun --standalone --nproc_per_node=8 midnight/train_gpt_8xgpu.py
python3 -m py_compile control_NOCHANGE/train_gpt_8xgpu.py
```

Use `python3 -m py_compile path/to/train_gpt.py` for a cheap syntax check after
surgical edits. Full validation requires the appropriate dataset, tokenizer,
GPU count, and wallclock from a recorded condition artifact.

## Coding Style & Naming Conventions

Python files use 4-space indentation and explicit environment-variable knobs
inside `Hyperparameters` classes. Prefer descriptive experiment directories
such as `mom95_mlp45/` or `rascal_loop/` and trainer names that encode GPU
count, e.g. `train_gpt_loop_8xgpu.py`. Keep condition-changing
values visible in the Python file or a committed `conditions/*.env`, not hidden
in shell wrappers. Do not restore 4k files from quarantine unless the manifest
condition is proven first.

## Testing Guidelines

No formal unit-test suite is present. Treat reproducible training and evaluation
logs as the test evidence. When comparing results, record the exact metric name
(`final_sliding_window_exact`, `final_int6_roundtrip_exact`, TTT metrics, etc.),
seed, tokenizer, dataset, GPU count, wallclock, and artifact bytes. A 4x run is
a mechanics proxy unless the condition file explicitly says otherwise.
Quarantined 4k results are not proper audited-SP4096 evidence.

## Commit & Pull Request Guidelines

Recent commit messages are short, imperative summaries with context, for
example `restructure: control -> control_NOCHANGE...` or `checkpoint: Mikey
shipped as new SOTA...`. PRs should state the experiment lane, parent/control,
changed fields, run command, metric, artifact size, and linked logs or
submission PR. Do not include untracked pod artifacts unless they are named
evidence for the change.

## Agent-Specific Instructions

Fail closed on unknown lab conditions. Before preparing, launching, or
interpreting a control, baseline, proxy, or comparator, read the relevant
`conditions/*.env`, `README.md`, log, or source trainer. Never modify
`control_NOCHANGE/` or `rascal_anchor_NEVERTOUCH.py` unless explicitly asked.
Read `QUARANTINE.md` before touching quarantined files.

For current multi-pod coordination, start in `agent_hub/`. It holds the compact
active-state index, current experiment ledger, pod commands, cut rules, and
next-run order. Treat it as a routing layer; the named `legs/<test>/run.py` and
`CONDITION.md` files remain the source of truth for run conditions.
