# Quarantine Manifest - 2026-04-28 Tokenizer Provenance

## Purpose

This quarantine preserves unproven 4k-vocab experiment surfaces without deleting
evidence. The trigger was tokenizer/dataset provenance drift discovered on
2026-04-28 and recorded in:

- `../../4K_TOKENIZER_INTEGRITY_AUDIT_2026-04-28.md`

Do not use files in this quarantine as clean leaderboard evidence without a new
condition audit.

## Why Quarantined

The audited SP4096 assets have:

- tokenizer SHA256: `6b0337698df13acdb58e5b05446ef26d9d7e4ca5545dc5e13ad5815a7ec904e0`
- vocab SHA256: `925129ee4771d6f0cdb83d1284ca023cfb37360fb7b759c966cbce911ab44ad9`
- validation tokens: `45516437`

The local `4k_vocab_lib/` tokenizer artifacts and Apr 27 run logs did not
match that audited manifest. Existing 4k results are therefore
`quarantined_4k_shaped_signal`, not proper audited-SP4096 records.

## Contents

- `suspect_4k_surfaces/`:
  tracked and untracked 4k tokenizer, 4k runner, and Apr 26/27 run-log surfaces.
  This also includes the old 4k condition file moved out of active
  `conditions/`.
- `candidate_submissions/`:
  Donnie, Mikey, Raphe, and related submission candidates that depend on the
  unproven 4k condition.
- `pod_artifacts/`:
  pulled logs, pod artifacts, and temporary pod build files.
- `misc_experiments/`:
  unrelated or exploratory untracked experiment folders moved out of the root.
- `generated_cache/`:
  ignored Python cache folders moved out of active source directories.

At quarantine time this folder contained 157 files and 79 directories.

## Clean Root Boundary

The active repo root should now be limited to:

- rule/context docs: `LEADERBOARD_RULES.md`, `AGENTS.md`
- audit docs: `4K_TOKENIZER_INTEGRITY_AUDIT_2026-04-28.md`,
  `AUDIT_Mikey_Raphe_2026-04-27.md`, `RESEARCH_REPORT_2026-04-27.md`
- active non-4k recorded conditions: `conditions/`
- frozen/control lineage: `control_NOCHANGE/`, `rascal_anchor_NEVERTOUCH.py`
- non-4k tracked baselines: `midnight/`, `mom95_mlp45/`,
  `mom95_mlp45_kernel/`, `rascal_loop/`, `whale_donor/`
- tooling: `scripts/`, `vast_connect_instructions.md`

## Recovery Rule

Before anything leaves quarantine, prove the exact condition:

```text
TOKENIZER_SHA256=6b0337698df13acdb58e5b05446ef26d9d7e4ca5545dc5e13ad5815a7ec904e0
VOCAB_SHA256=925129ee4771d6f0cdb83d1284ca023cfb37360fb7b759c966cbce911ab44ad9
VAL_TOKENS=45516437
TRAIN_SHARDS=143
WORLD_SIZE=8
MAX_WALLCLOCK_SECONDS=600
METRIC=<exact metric name, on decompressed artifact if reporting artifact score>
```

If any field differs, restore only into a new explicitly labeled condition
folder, not into the clean root.
