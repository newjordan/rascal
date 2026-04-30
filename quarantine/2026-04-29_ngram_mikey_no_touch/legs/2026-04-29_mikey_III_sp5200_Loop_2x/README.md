# Mikey III SP5200 Loop 2x

Prepared morning lane for the 5200-tokenizer Mikey III loop test.

Status on the 4x pod as of setup:

- Tokenizer present: `/workspace/data/tokenizers/fineweb_5200_bpe.model`
- Tokenizer vocab size verified: `5200`
- Dataset missing: `/workspace/data/datasets/fineweb10B_sp5200`
- Launch is intentionally blocked until shard validation passes.

Build dataset first:

```bash
cd /workspace/sota_rascal
./scripts/build_sp5200_dataset.sh
```

Run after the dataset exists and validates:

```bash
cd /workspace/sota_rascal/legs/2026-04-29_mikey_III_sp5200_Loop_2x
./launch_2x_gpu23.sh
```

Watch:

```bash
tail -f "$(ls -t logs/mikey_III_sp5200_Loop_2x_seed444_*.txt | head -1)"
```

This is a `new_experiment` / `mechanics_proxy`, not an 8x reproduction.
