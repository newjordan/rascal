# NGRAM Mikey Quarantine

Date: 2026-04-29

This quarantine holds Mikey runners that were prepared from NGRAM-bearing
bodies. They are not approved for active runs because the source bodies still
contain NGRAM, trigram, complement, and bigram-hash code paths, even when the
env disables them.

## Quarantined Files

- `legs/2026-04-29_mikey_1p1_sp5200_2x/run.py`
  - SHA256: `2024095b6445eb93aba4a3a16e8a1c5ca5cfe1cc56fddda21512085accae9669`
- `legs/2026-04-29_mikey_1p1_sp4096_1x_kernel_smoke/run.py`
  - SHA256: `c92b25d485ff0a34bf8a9d6c9dd41fd41df98436dfe6e6edf43746c0d83eae7b`
- `legs/2026-04-29_mikey_III_8k_Loop_2x/run.py`
  - SHA256: `3681e3c0557575b712cd01409dfb043b86cd6549bbffe363b0ee40970e0588d1`
- `legs/2026-04-29_mikey_III_sp5200_Loop_2x/run.py`
  - SHA256: `c84fc1d71af0bf9f0890aa1b3446d3d85fe518d486c8e3b7795845c2ddcfb74a`
- `conditions/mikey_1p1_sp5200_2x.env`
  - SHA256: `2d0fc12756967b9b3077835086cb88502c8f847b4812553b6c7843e6cf71e2e4`
- `conditions/mikey_1p1_sp4096_1x_kernel_smoke.env`
  - SHA256: `c89102010cd2d4df24fbe254a74b1784a06ee5161d2234db78959bba58bf44ba`
- `conditions/mikey_III_8k_Loop_2x.env`
  - SHA256: `1853888e9d9fdb60b05f2141d5e1772cf8d39a21c001d4a2385a4390ecd76abd`
- `conditions/mikey_III_sp5200_Loop_2x.env`
  - SHA256: `a4427c2fe75273a8b32a325fe38fa666e30c1d3d5fa4f6dd011634e3109b225a`

## Evidence

`rg "NGRAM|ngram|TrainNgram|BigramHash|TRIGRAM|COMPLEMENT"` hits in the
quarantined `run.py` include:

- `TRIGRAM` env field
- `NGRAM_EVAL_ORDER` env field
- `TrainNgramTracker`
- `BigramHashEmbedding`
- `eval_val_sliding_hashed_ngram`

## Clean Mikey Source

Use this file for clean Mikey work instead:

- `quarantine/2026-04-28_tokenizer_provenance/candidate_submissions/Mikey_II/train_gpt.py`
  - SHA256: `6cf4abeececb632be2ecea92cc94b725c292e117d588ceb4d229384acf495a37`
  - Verified with the same `rg` pattern: no hits.
