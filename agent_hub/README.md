# Rascal Agent Hub

This folder is the compact coordination surface for multi-agent pod work.
Agents should start here before scanning old chat, old handoffs, or broad
`legs/` history.

## Current Files

- `ACTIVE_STATE_2026-04-30.md`: live strategic state, pod commands, current
  best findings, cut rules, and next-run order.
- `experiments_2026-04-30.tsv`: machine-readable run ledger for the current
  CaseOps/PR1855 SP10240 campaign.

## Operating Rule

Treat these notes as a routing index, not as a replacement for the actual
runner files. Before launching or comparing a result, open the named
`legs/<test>/run.py` and `CONDITION.md`, then check the live pod log.

Do not commit pod binaries, datasets, checkpoints, `*.bin`, `*.pt`, or raw
`logs/`. Put compact text results and commands here instead.

## Coordination Rule

Use wide 1x/6x scatter only long enough to identify the best axis. Once the
slow 6x lanes produce enough post-EMA/quant evidence, pivot future work to
focused `4x` and `2x` mechanics tests that answer one specific question faster.
Do not keep burning broad scatter after the axis is known.
