# Port plan: PR #1758 — PreQuant TTT LR=1e-3 + Unfrozen

## 1. Technique summary

**PreQuant TTT** is a fine-tune pass run on the *float* model on held-out legal val tokens before GPTQ — distinct from eval-time TTT on the dequantized model. PR #1758 inherits the pass from PR #1735/#1738 (21-epoch cosine AdamW, 32768-token chunks, parallel across GPUs). **Unfrozen** = `PREQUANT_TTT_FREEZE_BLOCKS=0`, dropping the inherited `=2` (which held the lowest two transformer blocks fixed). **LR=1e-3** replaces `5e-4`: at 5e-4 the cosine flattens by epoch 12; at 1e-3 it descends monotonically through epoch 21.

## 2. Mikey_II's existing TTT

Mikey_II has **only post-quant eval-time TTT**. `eval_val_ttt` (`train_gpt.py:340-377`) runs on the dequantized model from `deserialize` (called at `train_gpt.py:449`). It uses **SGD+momentum** (line 345), sets `requires_grad_(True)` on **all** parameters (line 344) — already fully unfrozen — at `ttt_lr=0.005`, `ttt_epochs=3`, `ttt_momentum=0.9`. **Mikey_II has no PreQuant TTT phase**: training goes EMA → GPTQ → eval (+ optional eval-TTT). PR #1758's "unfrozen" delta does not apply (already unfrozen) and "LR=1e-3" applies to a phase that doesn't exist here.

## 3. PR base + scope

PR #1758 net delta = **2 lines** (`os.environ.setdefault` for `PREQUANT_TTT_LR` and `PREQUANT_TTT_FREEZE_BLOCKS`) prepended to PR #1738's lzma-packed blob. Pure hparam-flip on top of #1738; the prequant-TTT machinery (loop, freeze-mask, AdamW, cosine schedule) lives inside #1738. Stacking on Mikey_II requires **porting the entire prequant-TTT subsystem from #1738** (~150-300 LOC: optimizer, freeze masking, chunked train loop, parallel reduce) — structural, not hparam.

## 4. Code surface in Mikey_II

- TTT param unfreeze: `train_gpt.py:344` — already unfreezes all params.
- TTT optimizer + LR: `train_gpt.py:345` (`SGD(...,lr=h.ttt_lr,momentum=h.ttt_momentum)`).
- TTT cosine: `train_gpt.py:361-362`.
- TTT call site: `train_gpt.py:446-449` (post-GPTQ).
- Hyperparams declared on `Hyperparameters` line 7: `ttt_enabled`, `ttt_lr=.005`, `ttt_epochs=3`, `ttt_momentum=.9`, `ttt_chunk_tokens=32768`.
- **No prequant TTT call site.** A new pass would slot between EMA (line 440) and `serialize` (line 440).

## 5. Hyperparams to expose

| field | default |
|---|---|
| `prequant_ttt_enabled: bool` | `False` |
| `prequant_ttt_lr: float` | `1e-3` |
| `prequant_ttt_epochs: int` | `21` |
| `prequant_ttt_freeze_blocks: int` | `0` |
| `prequant_ttt_wd: float` | `0.0` |
| `prequant_ttt_grad_clip: float` | `1.0` |
| `prequant_ttt_chunk_tokens: int` | `32768` |

Bake into `Hyperparameters` line 7 — no top-of-file `os.environ` block.

## 6. Compute

Original: 8× H100 SXM, 600s train + 600s eval. Prequant TTT log shows ~21.6s/epoch × 21 ≈ **454s** of 8-GPU wallclock for the prequant pass. On float weights this is ~2× post-quant TTT cost; stacking forces ~7+ min off main-train under the 600s cap.

## 7. Expected gain stacked on Mikey_II

**Most-likely-redundant of the 5 candidates.** PR #1758 delta is 1.03540 → 1.02840 (−0.0070) on a non-eval-TTT base. Mikey_II already has eval-time TTT (unfrozen, cosine) at 0.86548. The PR's two lifts are (a) more-aggressive LR for the prequant pass, (b) unfreezing — Mikey_II already does (b) for its post-quant TTT and has no prequant phase. A +0.007 on a 1.03 base is unlikely to translate to measurable gain on a 0.865 base where TTT residual is largely extracted. **Expected: null or small negative (wallclock theft from main train).**

## 8. A/B test recipe

The user-suggested cheap A/B (toggle unfrozen + LR=1e-3 on existing TTT) is **not actually meaningful**: Mikey_II's eval TTT is already unfrozen, and `ttt_lr=0.005` was tuned for SGD+momentum with 3 epochs, not AdamW with 21. A meaningful prequant-TTT test requires the structural port.

Cheap proxy (3 seeds each, tag `mikey_ii_ttt_unfrozen_seed{43,44,45}`):
- A: Mikey_II as-is.
- B: sweep `ttt_lr ∈ {0.001, 0.002, 0.01}` × `ttt_epochs ∈ {1, 3, 6}` on existing post-quant TTT.

If no B point beats A by ≥ 0.005 at p<0.01, post-quant TTT is saturated → prequant port is dead. Cost: ~9 × 12 min = ~110 min on 8× H100.

## 9. Risk flags

- **Mikey_II already does this (loud).** Unfrozen is Mikey_II's default for post-quant TTT (`train_gpt.py:344`).
- **Different optimizer family.** PR #1758 prequant = AdamW + 21-epoch cosine; Mikey_II eval TTT = SGD+momentum + 3-epoch. LR=1e-3 does not transfer.
- **Negative stack.** Mikey_II's `ttt_lr=0.005` is tuned for SGD+3-epoch; flipping toward AdamW values can regress.
- **Wallclock theft.** ~450s prequant phase under 600s cap forces main-train cuts.
- **Recommendation: deprioritize.** Run the cheap proxy first; port full subsystem only if post-quant TTT is shown saturated AND there is wallclock slack.
