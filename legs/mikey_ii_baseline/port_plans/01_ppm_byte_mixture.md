# Port plan: PR #1835 / #1850 — PPM-D byte mixture (SP8192)

## 1. Technique summary
At eval time, spread the sliding-window NN per-token NLL uniformly across each token's UTF-8 bytes (`q_NN_byte`); in parallel run online byte-level PPM-D (Cleary-Witten 1984) over the same stream — descend prefix orders O→0, deepest-matching context, escape via `unique/(total+unique)`. Mix in probability space `q = λ·q_NN + (1−λ)·q_PPM`. λ is a binary gate on PPM's top-symbol confidence: `cf ≥ 0.9 ⇒ λ_lo=0.05` (trust PPM), else `λ_hi=0.9` (trust NN). Score-first: counts incremented *after* the byte's `−log q` is logged. The 1.06 → 1.00 win is byte-level surprisal recovered on local-repetitive contexts (URLs, identifiers, numeric literals) that a 16 MB transformer cannot afford to memorize.

## 2. PR comparison
| axis | #1835 (anmarhindi, 1.00136) | #1850 (someone114514, 1.00495) |
|---|---|---|
| scorer | pure Python loop, dict-of-dict counts | runtime-compiled C (`gcc -O3`), open-addressed tables |
| coverage | 3M-token subset (`PPM_SUBSET_TOKENS=3_000_000`), rank 0 only | full val, distributed gather of per-rank `(tgt, prev, nll)` files in `/tmp` |
| order | 5 | 4 |
| extras | exposes `ppm_use_meta_mix` (3-expert NN/PPM/token-PPM mixture, off by default), runs alongside a LoRA sliding-TTT | strips the Python reference and TTT to fit the 16MB cap; ships `ppm_log_cache_size`, `sliding_batch_seqs` |
| LoC added | ~1485 (includes LoRA TTT, meta-mix variants) | ~810 (PPM only) |
| same gate | yes — `λ_hi=0.9, λ_lo=0.05, T=0.9` | yes — identical |

**Port #1850's variant.** It is the cleaner mechanical port: native C makes full-val tractable inside the 600s cap; no LoRA-TTT clutter to disentangle from the PPM contribution; Mikey_II already ships its own TTT and we don't want #1835's competing implementation. We give up #1835's 0.0036 bpb edge but that gap is from order=5 over order=4 plus subset noise, not from a methodological win — and the meta-mix knob in #1835 is documented to regress.

## 3. PR base + scope
- **#1835** sits on PR #1493 (legal TTT). +1485 / −0 LoC, 8 added files. Subsystem: eval-time mixture **plus** a LoRA-style sliding TTT (separate concern, do not port).
- **#1850** sits on the merged 2026-04-09 SP8192 legal TTT record (1.0810 base). +810 / −0 LoC, 6 added files. Subsystem: eval-time mixture only.
Both are pure eval-time additions: no training-loop, optimizer, GPTQ, brotli, or model-architecture changes. Port surface is `ValidationData.__init__`, `eval_val_sliding`, and `Hyperparameters`.

## 4. Code surface in Mikey_II
`/home/frosty40/sota_rascal/legs/mikey_ii_baseline/train_gpt.py`:
- **`ValidationData.__init__` (line 16–20)** — append `self.token_bytes_py = build_token_bytes_lut(self.sp, h.vocab_size)`; add the helper at module level.
- **`eval_val_sliding` (line 329–339)** — headline path. In the per-window loop (line 337) collect rank-local `nll_np / tgt_np / prev_np` alongside the loss accumulators. After `dist.all_reduce` (338): write per-rank files to `/tmp/pg_ppm_<RUN_ID>/rank{N}.bin`, concat in position order on rank 0, call `_ppm_mixture_bpb_native(tgt, prev, nll, val_data.token_bytes_py, has_leading_np, is_boundary_np, order, λ_hi, λ_lo, T, log_cache_size)`. Override the returned `val_bpb` with `mix_bpb`; synthesize `val_loss = mix_bpb · log(2) · bytes/tokens` so `timed_eval`'s log line (378) is consistent.
- **New top-level helpers** (~75 lines from PR #1850 lines 343–415): `_build_native_ppm_lib` (writes `.c` to `tempfile.gettempdir()`, sha1-keyed, `gcc -O3`), `_ppm_mixture_bpb_native` (ctypes shim returning `(mix_bpb, ppm_only, nn_byte_bpb, nn_token_bpb, bytes, gate_high_frac)`).
- **`train_and_eval` (line 444–445)** — no signature change. `timed_eval('quantized_sliding_window', …)` at 445 already passes `eval_model` (the post-quant dequantized model from line 442) — correct attribution preserved. TTT at 446–449 re-deserializes into `ttt_model`, unaffected. PPM-on-TTT later: replicate the same hooks inside `eval_val_ttt` (line 340).

## 5. Hyperparams to expose
Bake into `Hyperparameters` dataclass (line 7); per project rule, no `os.environ` block at top-of-file:

| name | default | range | what it does |
|---|---|---|---|
| `ppm_enabled` | `False` | bool | master toggle for the byte-PPM mixture |
| `ppm_native_enabled` | `True` | bool | use the C scorer (set False to fall back to Python ref for parity audits) |
| `ppm_order` | `4` | 3–6 | PPM-D context depth; 5 likely +~0.003 bpb at +~15s eval |
| `ppm_lambda_hi` | `0.9` | 0.7–0.95 | mix weight on NN when PPM confidence is *below* threshold |
| `ppm_lambda_lo` | `0.05` | 0.0–0.2 | mix weight on NN when PPM confidence is *above* threshold |
| `ppm_conf_threshold` | `0.9` | 0.7–0.95 | gate trigger on PPM top-symbol prob at deepest matching context |
| `ppm_log_cache_size` | `1048576` | 2¹⁵–2²² | size of integer-log LUT inside C scorer; affects throughput, not result |
| `sliding_batch_seqs` | `32` | 8–64 | sliding eval batch (already implicitly 32 in Mikey_II line 445; expose for tuning) |

## 6. Compute requirement
Original: 8×H100 SXM, 600s cap, ITERATIONS=20000, eval+PPM all under 600s — same as Mikey_II's existing canonical schedule. The PPM scorer itself is single-threaded CPU; #1850's C path runs in ~25–60s on full SP8192 val (~10M tokens). **Eval-only, no extra GPU compute, no extra training data.** First run on a fresh pod incurs a one-time `gcc -O3` build (~2s, cached by sha1 in `tempfile.gettempdir()`). Mikey_II's data path is already SP8192 fineweb so no shard work.

## 7. Expected gain stacked on Mikey_II
Mikey_II = 0.86548 (3-seed mean, SP8192, 11L+loops+TTT+mixed-int+brotli, post-quant sliding+TTT). The PRs land at ~1.00 from a 1.06 base on the **non-Donnie / pre-Mikey** SP8192 stack. Three orthogonality questions:
- **Loops:** orthogonal — PPM operates on the post-quant scored byte stream, agnostic to the model's depth/topology.
- **Brotli + mixed-int quant:** orthogonal — quant only changes which `nll[i]` values feed into the mixture.
- **TTT:** *additive but partially redundant.* Both TTT and PPM-D recover surprisal from local-context regularities; expect ~50–70% of #1850's standalone delta to survive on top of TTT. #1835 stacks PPM on top of legal-TTT and reports +0.08 bpb still landing — i.e., the redundancy is real but small.
- **Mixed-int:** orthogonal.

#1850 reports +0.0761 bpb (from 1.0810 → 1.00495) on a stack without #1413's full TTT. On Mikey_II the effective lift is bounded by TTT-byte-overlap; estimate **0.020–0.045 bpb** (i.e., 0.865 → ~0.82–0.84). That alone would beat Donnie (0.87480) and reset the world record. Lower-bound case (~0.85): still publishable, still beats Mikey_II floor. Upper-bound case is contingent on PPM finding bytes TTT did not adapt against; we don't know without a run.

## 8. A/B test recipe
Single canonical seed first, **8×H100 canonical schedule** (600s cap, ITERATIONS=20000) — the eval delta is what we are measuring and the PPM scorer is full-val so subset-on-1xH100 would not be apples-to-apples with Mikey_II's 0.86548.

```
RUN_ID=mikey_ii_ppm_byte_seed1337  SEED=1337  PPM_ENABLED=1  PPM_NATIVE_ENABLED=1
  PPM_ORDER=4  PPM_LAMBDA_HI=0.9  PPM_LAMBDA_LO=0.05  PPM_CONF_THRESHOLD=0.9
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Decision rule: if seed-1337 `quantized_sliding_window val_bpb` ≤ 0.855 (i.e., ≥ 0.010 bpb under Mikey_II — twice the contest 0.005-nat threshold), launch seeds 42 + 7 for the 3-seed mean. Tag `mikey_ii_ppm_byte_seed{1337,42,7}`.

## 9. Risk flags
- **Eval attribution.** PPM must consume the model that `timed_eval` passes in. Line 445 passes `eval_model` (post-quant, dequantized, line 442) ✓. Keep PPM logic *inside* `eval_val_sliding`; do not stash NLL across calls. TTT (line 446–449) re-deserializes a fresh `ttt_model` — if PPM is later enabled there it must re-collect, not reuse.
- **Vocab dependency.** Both PRs SP8192; Mikey_II SP8192 → straight port. C scorer keys on byte windows (vocab-agnostic); only `token_bytes_py` is vocab-derived (SentencePiece piece bytes).
- **Score-first contract.** PR #1850's C `score_byte` increments counts *after* `mix_nll -= log_mix`. Audit the ported version preserves this ordering — reversal is illegal per Issue #1017.
- **`val_bpb` semantics.** `_loss_bpb` (line 317) computes bpb from `loss_sum/token_count * tokens/bytes`. The mixture returns `mix_bpb` directly; synthesize `val_loss = mix_bpb · log2 · bytes/tokens` so `timed_eval`'s log line stays internally consistent.
- **C build hardening.** Need `gcc` in PATH on eval host (Im_sorry_pod_setup.sh installs it). On build failure, fall back to `ppm_native_enabled=False` (Python ref) with a warning rather than crashing the run.
- **Eval budget.** PPM C scorer adds ~25–60s single-threaded CPU on rank 0 after sliding finishes. Confirm in dry-run with `MAX_WALLCLOCK_SECONDS=600` that total fits.
