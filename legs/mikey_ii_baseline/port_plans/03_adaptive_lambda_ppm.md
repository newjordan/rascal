# Port plan: PR #1795 — adaptive-λ byte-PPM mixture (SP4096 → SP8192)

## 1. Technique summary
Eval-time mixture: spread the NN's per-token logprob uniformly over each token's UTF-8 bytes (`q_NN_byte`), build an online byte-level PPM-D order-4 predictor (`q_PPM_byte`), mix in probability space `q_mix = λ·q_NN + (1−λ)·q_PPM`. λ is per-byte adaptive: descend prefix orders O→0 and freeze `cf = max_count/total` at the deepest context with data; `λ = 0.05 if cf > 0.9 else 0.9` — PPM dominates only when its local context is very confident. The "strict-legal gate" means `cf` is computed from prefix + PPM state alone, frozen *before* any `d.get(observed_byte)` lookup, so two hypothetical next bytes get identical λ. The earlier #1785 used `cf = P_PPM(observed_byte)` (target-conditioned, illegal) — flagged by @nprime06; #1795 is the rebuild.

## 2. PR base + scope
+2246 / -0 LOC, 6 files in `records/.../2026-04-23_SP4096_PPM_AdaptiveMix/`. **Eval-time only.** NN stack is verbatim @clarkkev 2026-04-01 SP4096 (PR #1334). Source addition: `_ppm_mixture_bpb` (~55 lines, PR `train_gpt.py` line 1422) plus ~30 lines of gather + mix logic in `eval_val_sliding` (PR line 1604–1640), gated by `PPM_MIX_ENABLED=1`. Training, GPTQ, brotli — untouched.

## 3. Code surface in Mikey_II
`/home/frosty40/sota_rascal/legs/mikey_ii_baseline/train_gpt.py`:
- `eval_val_sliding` lives at **line 329–339** (one-liner-style; per-window scoring loop is line 336, per-sequence accumulator with `tgt`/`prev` is line 337).
- `_loss_bpb` at **line 317** computes the headline `val_bpb`.
- `train_and_eval` at **line 445** calls `timed_eval('quantized_sliding_window', eval_val_sliding, …)` — the value PR #1848 reports as 0.86548.
- Graft: extend line 337 to collect `lp_chunks`/`tgt_chunks` (negative scored NLL + tgt ids); after `dist.all_reduce` (line 338), gather to rank 0, call `_ppm_mixture_bpb(tga, lpa, val_data.sp)`, override returned `val_bpb`. Distributed gather pattern: PR lines 1611–1628.
- Mikey_II has loops (`looping_active=True`, line 443) and TTT (`eval_val_ttt` line 340). Run mixture on the post-quant looping path; if TTT enabled, mix on TTT-adapted scored stream.

## 4. Vocab port cost
Byte-PPM is **vocab-agnostic** — operates on UTF-8 bytes from `id_to_piece` (PR lines 1424–1430). NN-spread depends only on per-token byte length, which Mikey_II's `base_bytes_lut` already provides (line 337). λ schedule (`cf > 0.9 → λ=0.05`) is **likely robust** to vocab change: 0.9 thresholds PPM's own state, not NN entropy. Order-4 was tuned for SP4096 (~1.39 bytes/tok); SP8192 ≈ 3.0 bytes/tok per Mikey_II's accounting, so the byte stream is **~2× longer** for the same val token count → PPM tables grow more, order-4 may underfit (try O=5 if budget allows), λ-threshold may want a sweep. Effort: ~2 hr code + sweep `{O=4, O=5}` × `{T ∈ 0.7, 0.8, 0.9}`.

## 5. Comparison to PR #1835 / #1850
- **#1835** (SP8192, 1.00136): same binary-λ gate, order-5, identical schedule, but on a 3M-token *subset*. "adaptive-λ" and "binary-λ" are functionally equivalent. Already SP8192-ported.
- **#1850** (SP8192, 1.00495): same idea, **full-val + native C scorer** + order-4. Confirms full-val eval cost is the binding constraint at SP8192.
- **#1795 is not strictly additive over #1835/#1850 — it is the same family.** Differentiators: (a) outcome-independent gate proof, (b) full-val basis, (c) vocab=4k. For Mikey_II: **port #1850's native C scorer + #1795's strict-legal gate code, not #1835's Python-on-subset variant.** If `01_ppm_byte_mixture.md` (static-λ) is going ahead, this `03` should layer adaptive-λ on top as a one-line gate change.

## 6. Hyperparams to expose

| name | default | range | what it does |
|---|---|---|---|
| `ppm_mix_enabled` | `False` | bool | toggle mixture on top of sliding val_bpb |
| `ppm_order` | `4` | 3–5 | PPM-D context depth; 5 ≈ +0.02 bpb but +15s eval |
| `ppm_lambda_hi` | `0.9` | 0.7–0.95 | λ when PPM not confident (NN-dominated) |
| `ppm_lambda_lo` | `0.05` | 0.01–0.2 | λ when PPM confident (PPM-dominated) |
| `ppm_conf_threshold` | `0.9` | 0.5–0.95 | `cf = max/total` cutoff |
| `ppm_native_enabled` | `True` | bool | use #1850's C scorer; Python fallback |

Bake into `Hyperparameters` defaults (project rule: no top-of-file CONDITION dicts that write `os.environ`).

## 7. Compute requirement
**Eval-only, no retraining.** PR #1795 reports 485–521s SP4096 full-val (~152.6 MB bytes, pure Python PPM). Mikey_II at SP8192 has the same byte count (tokenizer-invariant) but fewer tokens, so NN forward shortens and PPM dominates eval. Expect **eval ~500–600s with Python PPM — at the cap.** **Port #1850's native C scorer up front.** Smoke at 1×H100 with `EVAL_SEQ_LEN=2048` first; on 8×H100 the 600s cap may force order-4 over order-5 (same as #1850 chose).

## 8. Expected gain stacked on Mikey_II
**Critical caveat:** #1795/#1835/#1850 all start from a **1.087 NN-only byte-BPB floor** and recover ~0.07–0.13 BPB. Mikey_II is at **0.86548** — already 0.22 below their NN-only floor. PPM wins on rare-byte exact-repeats (URLs, code identifiers, formatting); a stronger NN absorbs many of those bits already, so marginal Δ shrinks. **Best estimate: −0.005 to −0.020 BPB**, i.e. 0.846–0.860. Adaptive-λ over static-λ is second-order — #1835's README notes "meta-mix learning per-expert weights regressed" — so a 2-region binary gate is the empirical sweet spot. Net: **marginal-but-real on a 0.865 floor; expect partial saturation, not a 0.07 swing.**

## 9. A/B test recipe
1. **Seed 1337:** clone Mikey_II → `mikey_ii_adaplam_ppm_seed1337`, add `_ppm_mixture_bpb` + gather plumbing in `eval_val_sliding`, `ppm_mix_enabled=True`. Run 8×H100, confirm eval < 600s, log `[ppm_mix]` line. Compare `quantized_sliding_window val_bpb` vs Mikey_II 0.86548 same seed.
2. If Δ ≥ −0.005 (0.005-nat bar): **3-seed mean** {1337, 42, 7} (Mikey_II's leaderboard seed set).
3. **Side-by-side with `01_ppm_byte_mixture.md`** (static-λ): both on seed 1337 first; if static ties within 0.001, drop adaptive (Occam). If adaptive wins by ≥ 0.002, promote to 3-seed.
4. Tag: `mikey_ii_adaplam_ppm_seed<N>`. Logs: `_run_logs/`.

## 10. Risk flags
- **Vocab mismatch.** SP4096 tuning. λ schedule is PPM-state-keyed (first-order vocab-agnostic) but **order needs re-tuning at 8k**. Risk: medium.
- **Strict-legal gate compliance.** Use #1795 gate exactly, not #1785's target-conditioned form. `cf_mx`/`cf_tot` captured **before** `d.get(x, 0)` (PR line 1457). Audit: zero `x` references between `cf_seen = True` and the `cf[i] = ...` write. Risk: low if copied verbatim.
- **Eval-attribution risk.** Per eval-audit memo (Mikey/Raphe had base_model-vs-eval_model bug): mixture must run on the **same scored stream** that produces `quantized_sliding_window val_bpb` — `eval_model` (post-quant, looping active), not `compiled_model`/`base_model`. PR's integration is correctly inside `eval_val_sliding(base_model)` where `base_model` is the dequantized eval model passed by `train_and_eval` line 442. Mirror that contract.
- **Eval budget.** Python PPM at SP8192 full-val may exceed 600s; native C scorer (#1850) is the safety net. Plan `PPM_NATIVE_ENABLED=1` day 1.
- **Organizer ruling pending.** #1795 README states submission withdrawn if score-first online PPM ruled illegal. Score-first TTT precedent (PR #1493 in our memo) supports legality, but durability is conditional.
