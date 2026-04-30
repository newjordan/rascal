# 2026-04-30 Live Decision After Cut

## Current State

Pulled logs from `instance_35690435` into:

`pod_pulls/instance_35690435_20260430_020532/`

## Read From Latest Logs

Known personal-best band:

- parent: `2026-04-29_mikey_sp10240_full124_loop2_late050_mlp375_clip1305_8x`
- total bytes: `15,954,736`
- quantized sliding BPB: `1.08229317`
- quantized TTT BPB: `1.08066751`
- eval-only TTT5 BPB: `1.08051259`

Failed/confounded run:

- run: `mikey_sp10240_mlp375_loop_late060_nowarm_emb7_mlpclip12_8x_seed444`
- post-EMA BPB: `1.08936674`
- quant BPB: `1.10448894`
- sliding BPB: `1.08809943`
- TTT BPB: `1.08492921`
- total bytes: `15,599,874`

Interpretation: late060/no-warm lost too much quality. It does not prove `embed_bits=7 + MLPClip12` is bad because it changed loop timing and warmup at the same time.

Shampoo run:

- run: `mikey_sp10240_mlp375_loop_late050_emb7_mlpclip12_shampoo_8x_seed444`
- early tok/s around `3.8M`, much slower than the `7-8M` normal path
- cut before comparable readout

Interpretation: real optimizer axis exists, but exact inverse roots are too slow for the current paid-run lane unless converted to a cheaper update cadence or lower-cost approximation.

## Next Paid Run

Run the clean quant-policy isolation:

```bash
cd /workspace/sota_rascal/legs/2026-04-30_mikey_sp10240_mlp375_loop_late050_emb7_mlpclip12_8x && ./launch_8x.sh
```

Why: it keeps the best late050 loop schedule and loop warmup, changing only:

- `embed_bits: 8 -> 7`
- `mlp_clip_sigmas: unset -> 12.0`

## Cut Criteria

Keep only if it stays near the parent curve:

- at final neural/post-EMA: needs to be near `1.0868`, not `1.089+`
- plain quant should not jump deep above `1.10`
- final sliding/TTT must beat or threaten `1.0805-1.0807`

If it fails, drop the quant-policy isolation as a primary lane and move back to body/loop changes.

## Cut: Clean Quant-Policy Isolation

Run:

`2026-04-30_mikey_sp10240_mlp375_loop_late050_emb7_mlpclip12_8x`

Observed before user cut:

- post-EMA BPB: `1.08710925`
- plain quant BPB: `1.10196082`
- sliding BPB: `1.08562639`
- total bytes: `15,600,561`

Decision: useful byte recovery, not a winner path. It saved about 350KB versus
the best parent but crossed the current `1.10` plain-quant cut gate and did not
threaten the best `1.0805-1.0807` TTT band.

## Active Follow-Up

Run:

`2026-04-30_mikey_sp10240_9l_mlp4_seqloop3_late55_68_80_maxstep_8x`

Why: move away from tiny quant-policy tuning and test the bigger loop/max-step
theory directly: 9L body, lower base params, MLP4, and staged 3-loop pressure at
late fractions `0.55`, `0.68`, and `0.80`.

Live read: first standard val was off-curve but early for this hypothesis:

- `4000/20000 val_bpb: 1.1695`

Context: at the `4000` val, only loop stage 1 had just activated
(`step:3823`). Stage 2 activated at `step:4473`; stage 3 activated at
`step:4986`. This test is explicitly late-payoff, so do not cut solely on the
`4000` read. Judge it on the final/post-EMA and quant read after all loop
stages have had time to act.

Decision: if the final neural/post-EMA remains clearly off the SOTA parent band,
cut and return to the SOTA parent lane.

Final useful read:

- post-EMA BPB: `1.09623498`
- compressed total bytes: `14,122,131`
- plain quant BPB: `1.11010192`

Decision: bust as a contender. It proved the 9L seqloop3 body is very small
and leaves about 1.88MB of byte slack, but neural quality and plain quant are
too far behind the SOTA parent.

## Three-Branch Queue

1. Sucks / return to SOTA:
   `2026-04-29_mikey_sp10240_full124_loop2_late050_mlp375_clip1305_8x`
   is the known best parent: 10k vocab, 11L, MLP3.75, loop2 late050, clip13.05.

2. Good but not strong enough:
   `2026-04-30_mikey_sp10240_9l_dim576_mlp375_seqloop3_late55_68_80_maxstep_8x`
   is prepared as a stronger copy of the active 9L max-step test. It keeps
   seqloop3 but moves width to FA3-legal dim576 and lowers MLP to 3.75.

3. 8k vocab fallback:
   `2026-04-29_mikey_sp8192_loop2_l5_7_late050_mlp3955_8x`
   is prepared on the pod as the 8k revisit. Use it if 10k keeps quant-spinning
   or if the SOTA parent cannot beat the 1.0805 band.

## Earlier Seqloop Prepared

Prepared and uploaded:

`2026-04-30_mikey_sp10240_9l_mlp4_seqloop3_early45_60_72_maxstep_8x`

Reason: if the active late schedule shows the right qualitative behavior but
does not have enough time for stage 3 to settle, this keeps the same 9L MLP4
sequential loop3 body and moves loop stages from `0.55/0.68/0.80` to
`0.45/0.60/0.72`.

Use only if the active late schedule looks directionally promising. If the
final/post-EMA is poor, return to the SOTA parent instead.

## Step-Scheduled 10L Prepared

Prepared and uploaded:

`2026-04-30_mikey_sp10240_10l_mlp4_seqloop3_step1500_3000_4500_maxstep_8x`

Reason: user requested the first loop on at `1500` steps. This is a 10L MLP4
escalation of the seqloop3 idea with explicit step thresholds:

- loop stage 1: step `1500`
- loop stage 2: step `3000`
- loop stage 3: step `4500`

The schedule is implemented inside `run.py` rather than hidden shell overrides.

Final useful read:

- `4000` val BPB: `1.1247`
- final raw val BPB at wallclock: `1.0909`
- post-EMA BPB: `1.09940234`
- compressed total bytes: `15,342,516`
- plain quant BPB: `1.11292914`

Decision: bust as contender. It showed real neural curve improvement versus
9L loop experiments, but EMA and quantization regressed hard. Do not continue
staged loop3 body variants unless quant policy is explicitly changed.

## Nimble Backburner Rule

Current active lane:

`2026-04-30_mikey_sp10240_9l_mlp4_seqloop3_early45_60_72_maxstep_8x`

Do not let the loop branch consume the whole run queue. Keep the SOTA parent as
the default backburner/recovery lane:

`2026-04-29_mikey_sp10240_full124_loop2_late050_mlp375_clip1305_8x`

Decision rule:

- if early seqloop is clearly bad at `4000`, cut and run SOTA parent
- if early seqloop is merely okay but final/post-EMA is not near the SOTA band,
  run SOTA parent before more loop variants
- only run the 10L step1500 copy if early seqloop shows a real directional gain
  that just needs depth

## SOTA-Line Next Concept

Concept:

`SOTA parent loop-timing repair`

Parent:

`2026-04-29_mikey_sp10240_full124_loop2_late050_mlp375_clip1305_8x`

Hypothesis: the known 11L SP10240 MLP3.75 SOTA parent already has enough neural
quality and size fit; the next high-leverage change is not another body swap but
giving its existing loop2 path more settling time. Move loop activation from the
late `0.50` wallclock gate to an explicit earlier step gate around `1500` while
keeping tokenizer, layers, MLP, quant clip, EMA, TTT, and byte policy fixed.

Decision metric: final post-EMA must stay in the SOTA parent band, and
plain-quant/sliding/TTT must threaten the `1.0805-1.0807` best band. The `4000`
read is only a shock/sanity check.

## PR1855 SOTA Adaptation

New accepted target:

`openai/parameter-golf#1855`

Merge commit:

`510d03e0fc355406c9fd06f92d23b8c5aedea7fb`

Record:

`records/track_10min_16mb/2026-04-27_SP8192_LQER_SparseGate_BOSSmearFix_9HpStack_1.0611`

Why this becomes primary: it reports post-phased-TTT BPB `1.05989454` for
seed `42` and three-seed mean `1.06107587`, far past the local Mikey/Rascal
near-miss band.

Prepared exact-source leg:

`2026-04-30_pr1855_sp8192_lqer_smeargate_repro_8x`

Status:

- exact accepted `train_gpt.py` copied as `run.py`
- accepted tokenizer copied into the leg
- `lossless_caps.py` and `prepare_caseops_data.py` copied from the same commit
- local and pod syntax checks passed
- uploaded to active pod at
  `/workspace/sota_rascal/legs/2026-04-30_pr1855_sp8192_lqer_smeargate_repro_8x`

Pod blockers at upload time:

- `/workspace` full: `102G/102G`, only ~`142M` free

Resolved blocker:

- removed old non-active `/workspace/Fartmagic/data/datasets/fineweb10B_sp{1024,4096,8192}`
- `/workspace` now has about `27G` free
- `lrzip` is installed on the pod for PR1855 `COMPRESSOR=pergroup`

Active adaptation status:

- CaseOps tokenizer is on the pod
- accepted PR1855 runner is on the pod
- CaseOps dataset is being built by streaming local
  `/home/frosty40/parameter-golf-lab/data/docs_selected.jsonl` into the pod
- streaming prep has proven it can write train shards on the pod
- accepted PR1855 seed42 log used `train_shards: 80`, `val_tokens: 47851520`,
  and phased TTT `total_docs:50000`
- stream helper is corrected to `--val-docs 50000 --max-train-shards 80`
- do not run `launch_8x.sh` until the dataset directory has both
  `fineweb_train_*.bin` and `fineweb_val_*.bin`
- the first bad partial stream used a 10k-val default and was cleared from
  the pod before the corrected stream restarted

Accepted PR1855 run command after data prep:

```bash
cd /workspace/sota_rascal/legs/2026-04-30_pr1855_sp8192_lqer_smeargate_repro_8x
./launch_8x.sh
tail -f logs/pr1855_sp8192_lqer_smeargate_repro_8x_seed42.txt
```

Data-prep status command:

```bash
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/tmp/codex_vast_known_hosts -o ConnectTimeout=8 -i /home/frosty40/.ssh/id_ed25519_apollo -p 56335 root@206.125.32.60 'ps -eo pid,ppid,etimes,cmd | grep -E "stream_prepare|caseops" | grep -v grep; df -h /workspace; find /workspace/SOTA_FINAL/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved -maxdepth 1 -type f 2>/dev/null | wc -l'
```
- `lrzip` missing, required for `COMPRESSOR=pergroup`
- required CaseOps dataset missing:
  `/workspace/SOTA_FINAL/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved`

Prepared data helper:

`prep_caseops_streaming.sh`

It streams `docs_selected.jsonl` from HF and writes CaseOps shards directly, so
we do not need to store the 48GB raw doc file on the pod. Still needs roughly
30GB free for generated shards and working room.

Do not delete the active 10k dataset while the current SOTA loop run is still
in TTT. Largest non-active-looking candidates are the legacy Fartmagic standard
datasets:

- `/workspace/Fartmagic/data/datasets/fineweb10B_sp4096` ~`27G`
- `/workspace/Fartmagic/data/datasets/fineweb10B_sp1024` ~`16G`
- `/workspace/Fartmagic/data/datasets/fineweb10B_sp8192` ~`15G`

Next action after explicit delete authorization or current-run completion:

1. free at least `30G`
2. install `lrzip`
3. run `prep_caseops_streaming.sh`
4. run `launch_8x.sh`
