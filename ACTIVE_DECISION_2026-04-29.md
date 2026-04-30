# Active Decision 2026-04-29

## Drop Rascal

Rascal is frozen as reference-only evidence. Do not launch more Rascal variants, Rascal-loop variants, Rascal kernel integrations, or Rascal 13L legalization work unless explicitly reopened.

Reason:

- Best legal Rascal found: 4k 12L brotli mixed-int, `final_sliding_window_exact=0.86441066`, size `15,653,512`.
- Better 13L Rascal was oversized: `final_sliding_window_exact=0.85957256`, size `16,765,032`.
- 8k loop Rascal/Mikey-style run was not competitive after quantization: `final_int6_roundtrip_exact=1.15178198`, size `17,599,005`.
- D64 custom attention did not beat FA3 on the exact active shape; kernel work should not drive the next scoring lane.

## Active Lane

Pivot back to the clean Mikey foundation.

Loop-path update:

- `NUM_LOOPS=0` is now classified as a dead scoring path.
- No-loop runs may remain as diagnostics/watch-only mechanics evidence, but they
  should not be used as the next scoring branch.
- Loop optimization is live: preserve PR1493-style recurrence and optimize the
  loop topology, activation timing, quantization, and byte budget around it.

Immediate priorities:

1. Use `quarantine/2026-04-28_tokenizer_provenance/candidate_submissions/Mikey_II/train_gpt.py` as the base body for new scoring attempts.
2. Keep 4k proven tokenizer/data as the reliable lane unless a complete 5200 dataset exists and validates.
3. Treat SP5200 as prepared but blocked on dataset build/free space.
4. Preserve Rascal logs/artifacts only for comparison and fallback.

NGRAM-bearing Mikey runners are quarantined under
`quarantine/2026-04-29_ngram_mikey_no_touch/` and are not active sources.

## SP10240 Current Decisions

Active anchor is now the 10L/10k Mikey family. The user-reported best current
world run is the 10L SP10240 line, so new 8x neural-build swings should stay
centered on 10L/10k unless explicitly redirected.

Marked loser:

- `legs/2026-04-29_mikey_sp10240_full124_loop2_l5_7_late050_11l_mlp375_spinquant_8x/`
- Condition: SP10240 full124, 11L, MLP3.75, loop2 moved to L5-7, late050,
  partial SpinQuant, seed 444.
- Result signal: post-EMA `1.09275396`, quantized `1.10411885`,
  sliding `1.08787812`, total bytes `15,967,297`.
- Decision: do not continue the 11L L5-7 recurrence-placement branch as a
  scoring path. Use 1x only for eval/winddown mechanics; use 8x for 10L/10k
  neural-body swings.

## 8x SP5200 Pivot

The current 8x pod target is clean Mikey v5 on SP5200, not Rascal.

- Pod: `35002131`, direct SSH `root@35.192.20.187 -p 9438`
- First runner: `legs/2026-04-29_mikey_II_v5_sp5200_loop2_11l_8x/run.py`
- Follow-up runners: `11l_mlp35_8x`, then `12l_8x`
- Tokenizer SHA256: `8bedf3c9e6e7a8ee55539c0ec2627b0c7a4c08782c26559bf007c8d07cd5e07f`
- Required dataset: `/workspace/data/datasets/fineweb10B_sp5200`
- Required inventory: `137` train shards and `1` val shard; validated on pod with `bad=[]`
- See `RUN_QUEUE_2026-04-29_SP5200_8X.md` for commands.

Last full pod save:

- `/home/frosty40/sota_rascal_pod_saves/sota_rascal_4x_shutdown_20260429_065632.tgz`
- SHA256: `c357f40a5b6296e3f782fbbf9913c3ba33682dfbef24c85e190e953892537c92`
