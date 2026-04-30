# 4K Tokenizer Integrity Audit - 2026-04-28

## Verdict

The existing Apr 27 `sota_rascal` 4k runs must be treated as **quarantined
4k-shaped evidence**, not proven proper audited-SP4096 records.

The model work may still contain a real signal, but the tokenizer/dataset
provenance is not clean enough to call these official 4k-vocab leaderboard
runs without a rerun.

## Evidence

The audited SP4096 package lives in:

- `/home/frosty40/sota_crawler/.claude/worktrees/tokenizer-4k-build/tokenizer_4k/`
- `/home/frosty40/parameter-golf-lab/data/tokenizers/`
- `/home/frosty40/parameter-golf-lab/data/datasets/fineweb10B_sp4096/`

Audited manifest values:

- model SHA256: `6b0337698df13acdb58e5b05446ef26d9d7e4ca5545dc5e13ad5815a7ec904e0`
- vocab SHA256: `925129ee4771d6f0cdb83d1284ca023cfb37360fb7b759c966cbce911ab44ad9`
- validation shard tokens: `45516437`
- total tokens: `14315946905`
- gates G1-G7: PASS in `tokenizer_4k/docs/GATES.md`

The tokenizer copy committed into this repo at `4k_vocab_lib/` does **not**
match those audited artifacts:

- `4k_vocab_lib/fineweb_4096_bpe.model` SHA256:
  `e5b31cf3d3bd87f68998ded73e8f7f9bdaea998bee4899461ab4f4d47f7b4987`
- `4k_vocab_lib/fineweb_4096_bpe.vocab` SHA256:
  `695f624cb475c14b050cb2ad7fc793465bdcd11b91cd86344ed6b220bdeac9e3`
- committed in `7f646cd` on Apr 26 with Claude co-author metadata

The Apr 27 4k run logs also disagree with the audited manifest:

- run logs repeatedly print `val_loader ... tokens:45514752`
- audited SP4096 val shard header is `45516437`

That difference means the Apr 27 pod runs are not proven to have used the
audited tokenizer/dataset pair, even though they used a `fineweb10B_sp4096`
path and 143 train shards.

## What This Means

Do not describe Mikey, Raphe, Donnie, or Donnie_TTT as clean proper 4k-vocab
records unless they are rerun against the audited manifest assets above.

Acceptable wording for current results:

- `quarantined 4k-shaped signal`
- `not tokenizer-proven`
- `requires audited-SP4096 rerun`

Unacceptable wording:

- `proper 4k run`
- `audited tokenizer run`
- `clean SP4096 record`

## Clean Recovery Condition

A proper 4k-vocab run must print or record these before training starts:

```text
TOKENIZER_PATH=<absolute path>/fineweb_4096_bpe.model
TOKENIZER_SHA256=6b0337698df13acdb58e5b05446ef26d9d7e4ca5545dc5e13ad5815a7ec904e0
VOCAB_SHA256=925129ee4771d6f0cdb83d1284ca023cfb37360fb7b759c966cbce911ab44ad9
DATA_PATH=<absolute path>/fineweb10B_sp4096
VAL_TOKENS=45516437
TRAIN_SHARDS=143
WORLD_SIZE=8
MAX_WALLCLOCK_SECONDS=600
METRIC=final_sliding_window_ttt_exact on decompressed artifact model
```

If any field differs, label the run as a new tokenizer condition or proxy, not
the audited SP4096 run.

## Immediate Next Step

Freeze architecture search. First rerun the best known stack on the audited
SP4096 assets only. The highest-value target is Donnie_TTT because it already
has artifact-sliding and TTT evaluation on the decompressed model path.
