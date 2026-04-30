# Rascal 4k MLP Math Sweep

Condition source: `legs/2026-04-28_raphe_1p1_sp4096_2x/run.py`
Run label: `mechanics_proxy`

Current 4k Rascal body:

- `VOCAB_SIZE=4096`
- `NUM_LAYERS=10`
- `MODEL_DIM=512`
- `NUM_HEADS=8`
- `NUM_KV_HEADS=4`
- `MLP_MULT=3.0`
- `BIGRAM_VOCAB_SIZE=2048`
- `BIGRAM_DIM=128`
- `VE_ENABLED=1`
- `VE_DIM=128`
- Logged params: `26,598,484`
- Accepted Rascal PR #1120 reference curve:
  - seed 444 step 4000 val_bpb: `1.1985`
  - seed 444 wallclock stop step 6593 val_bpb: `1.1343`
  - seed 444 post-EMA diagnostic val_bpb: `1.1333`
  - seed 444 final sliding-window exact val_bpb: `1.10986874`

The parameter algebra matches the live log exactly:

- banks: `qo + kv + mlp = 23,592,960`
- token embedding: `2,097,152`
- bigram path: `327,681`
- value embedding path: `557,059`
- scalars/skips/smear: `23,632`
- total: `26,598,484`

## Sweep

`eff_MB` is an ideal mixed-int pressure estimate, not final brotli artifact size.
`compute_rel` is relative to current `10L D512 MLP3.0`.

| Config | Params | eff_MB | compute_rel | Note |
| --- | ---: | ---: | ---: | --- |
| `L10 D512 MLP3.0` | 26.60M | 18.33 | 1.000 | current |
| `L10 D512 MLP3.25` | 27.91M | 19.15 | 1.056 | smallest direct MLP bump |
| `L10 D512 MLP3.5` | 29.22M | 19.97 | 1.111 | best next direct MLP test |
| `L9 D512 MLP4.0` | 28.96M | 19.70 | 1.100 | high-MLP, loses one layer |
| `L10 D544 MLP2.75` | 28.30M | 19.55 | 1.066 | width-balanced alternative |
| `L10 D544 MLP3.25` | 31.26M | 21.40 | 1.192 | aggressive width+MLP |
| `L12 D480 MLP3.0` | 27.75M | 19.09 | 1.055 | depth test; changes XSA layer pattern |

## Recommendation

Next run should use `NUM_LAYERS=10 MODEL_DIM=512 MLP_MULT=3.5`.

Reason: it isolates the missing 4k MLP question without changing depth, head shape,
value embedding width, XSA pattern, or tokenizer/data. It is only about 11% heavier
than the current run and adds 2.62M params, mostly in int5 MLP banks.

Do not compare step 4000 to the `~1.1337` late-run/post-EMA region. In the accepted
Rascal PR #1120 log, step 4000 was still about `1.1985`; `~1.1337` appears after
the 600s wallclock stop and EMA. The active 2x proxy step-4000 `1.1977` is therefore
on-curve, not a failure signal by itself.

If that is neutral or positive, the next more aggressive candidate is
`NUM_LAYERS=10 MODEL_DIM=544 MLP_MULT=3.25`.
