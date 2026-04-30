# Port plan: PR #1898 — Partial SpinQuant + GPTQ

## 1. Technique summary

SpinQuant (Liu et al. 2024) inserts an orthogonal rotation R on the input side of a linear: replace `F.linear(x, W)` with `F.linear(x @ R, W @ R)` — exact identity, but the new W has its outliers smeared across columns, so per-channel quantizers see a tighter dynamic range and incur less GPTQ damage. PR #1898 uses a **static Hadamard** rotation (Sylvester-Hadamard padded to next power of 2, multiplied by a random sign diagonal, QR re-orthonormalised) regenerated deterministically from `(SPINQUANT_SEED, tag)` — **zero serialized bytes**. **Partial (start_layer=5)** = layers 0–4 untouched, layers 5–10 rotated. No learned rotations.

## 2. Mikey_II's existing quant

Mikey_II is GPTQ int6 weights + int8 token-embeddings + brotli — no Hadamard anywhere. Pipeline (`/home/frosty40/sota_rascal/legs/mikey_ii_baseline/train_gpt.py`):
- `Hyperparameters` line 7: `matrix_bits=6`, `embed_bits=8`, `matrix_clip_sigmas=12.85`, `embed_clip_sigmas=20`, `gptq_calibration_batches=64`, `gptq_reserve_seconds=12`, `compressor='brotli'`.
- `collect_hessians` line 221–249: forward hooks accumulate `H = x.T @ x` into per-linear buffers over 64 calibration batches.
- `gptq_quantize_weight` line 250–256 + `gptq_mixed_quantize` line 257–264: GPTQ Cholesky path; bits/clip-sigma is per-key (`embed_*` for `tok_emb`, else `matrix_*`).
- `serialize` line 305–312: collects Hessians → `gptq_mixed_quantize(sd_cpu, hessians, h)` → `_compress` (brotli with `_byte_shuffle` line 295). No rotation between Hessian collection and quantization.
- `deserialize` line 313–316: `dequantize_mixed → eval_model.load_state_dict(deq_state, strict=True)`. No post-load rotation hook.

## 3. PR base + scope

PR #1898 is +16 384 / −0 across 15 files (open, unmerged). Stack: PR #1851 base (CaseOps + SmearGate-BOS + LQER-Asym + Phased TTT) + PR #1855 hparam greedy + Partial SpinQuant. **The SpinQuant delta itself is small**: ~200 LOC inside `train_gpt.py` — `_hadamard_rotation`, `install_spinquant_rotations`, `_spinquant_rotate_sd_and_H` plus 4 forward-hook lines in `CausalSelfAttention.forward` / `MLP.forward` and 2 lines each in serialize/deserialize. **No new tensors are serialized** (rotations regenerated from seed); buffers are non-persistent. **Eval-only mathematically** — `_sq_active=False` during training so Dynamo constant-folds the branch away.

## 4. Code surface in Mikey_II

Rotation is applied **pre-quant on weights** (`W_rot = W @ R` in CPU fp32 inside `serialize`) and **online on activations** during the post-deserialize forward pass (`x @ R` before each rotated linear). Exact lines to change in `train_gpt.py`:
- **77–78** `CastedLinear` → add class flag `_sq_active = False`.
- **101** `CausalSelfAttention.forward` → before `c_q/c_k/c_v` apply `R_attn_in`; before `self.proj(y)` apply `R_attn_proj_in`.
- **106** `MLP.forward` → before `self.fc(x)` apply `R_mlp_in`; before `self.proj(...)` apply `R_mlp_proj_in`.
- **308** `serialize` → between `collect_hessians(...)` and `gptq_mixed_quantize(...)`, call `_spinquant_rotate_sd_and_H(sd_cpu, hessians, h)`.
- **316** `deserialize` → after `load_state_dict`, call `install_spinquant_rotations(eval_model, h, start_layer=h.spinquant_start_layer)` and set `CastedLinear._sq_active=True`.
- Add the three primitive functions (`_stable_seed`, `_hadamard_rotation`, `install_spinquant_rotations`, `_spinquant_rotate_sd_and_H`).

**No retrain required** — base model is trained unrotated; rotation is post-hoc on the checkpoint at quant time.

## 5. Hyperparams to expose

Add to `Hyperparameters` (line 7) as dataclass defaults — bake values, no top-of-file env dicts:
- `spinquant_enabled: bool = True`
- `spinquant_seed: int = 20260416`
- `spinquant_start_layer: int = 5` (for 11-layer Mikey_II this rotates layers 5–10, 6/11 layers, 24 weight modules — twice PR #1898's 12 because Mikey_II has more layers in scope)
- `spinquant_rotation_kind: str = "hadamard"` (only hadamard implemented; placeholder for future learned)
- `spinquant_pre_gptq: bool = True` (composition order; see §6)

## 6. Composition with GPTQ

Order is fixed and correct as written: (1) `collect_hessians` on **unrotated** activations; (2) `_spinquant_rotate_sd_and_H` rotates `W → W @ R` and `H → R.T H R` so the Hessian matches the rotated weight in the rotated frame; (3) `gptq_mixed_quantize` runs on rotated `(W, H)` pairs; (4) brotli compresses the quantized rotated weights. At eval, `x @ R` is applied online, recovering `F.linear(x, W) ≡ F.linear(x @ R, W @ R)` to fp precision before quantization noise. Mikey_II's GPTQ path (`gptq_quantize_weight` line 250) is generic in `(W, H)`, so it accepts rotated pairs unchanged.

## 7. Compute requirement

PR #1898 ran 8×H100 SXM, ~596s training cap, ~493–500s eval (TTT-heavy). The SpinQuant addition itself adds **negligible compute**: rotation generation is one Hadamard build + QR per tag (4 tags, milliseconds), bake is 4 `(in_dim, in_dim)` matmuls per rotated layer (CPU fp32, < 1s for 11 layers @ dim 512/1536), and forward adds one `dim×dim` matmul per rotated linear (4 per layer × 6 layers = 24 extra matmuls per token). **No learned-rotation pretrain step.** Static Hadamard, applied at quant time. Should run inside Mikey_II's existing GPTQ reserve (12s).

## 8. Expected gain stacked on Mikey_II

Mikey_II = 0.86548. PR #1898 = 1.06614 vs PR #1851 = 1.06128 → **regression of +0.00486** when stacked on top of an EMBED_BITS=6 push (the gain came from enabling the EMBED_BITS=6 budget shift, not from rotation per se; the rotation buys back damage from a bit drop). On Mikey_II, embeddings are already at int8 with a 16 MB cap that fits fine. The relevant question: does Hadamard reduce GPTQ-int6 damage on the matrix weights? Empirically, SpinQuant typically buys 0.01–0.05 nats on int4–int6 mixed quant. **Realistic estimate on Mikey_II: −0.005 to +0.005 BPB.** Likely a wash unless paired with a bit-drop (e.g. matrix_bits 6→5 or embed_bits 8→6) where the saved bytes can fund another technique.

## 9. A/B test recipe

Pure quant-only A/B (no retrain):
1. Train Mikey_II baseline once, save `final_model.pt`.
2. **Run A:** standard `serialize` → record post-GPTQ + post-EMA `val_bpb`.
3. **Run B:** same checkpoint, `spinquant_enabled=True`, `spinquant_start_layer=5` → record `val_bpb`.
4. Sweep `spinquant_start_layer ∈ {0, 3, 5, 7, 9}` (full vs partial) and `matrix_bits ∈ {6, 5}` to find the bit-drop budget where Hadamard pays for itself.
5. If a setting beats 0.86548 single-seed, lift to canonical 3-seed (1337, 42, 2024). Tag: `mikey_ii_spinquant_seed<N>`.

## 10. Risk flags

- **Hadamard padding:** Mikey_II `model_dim=512` (= 2^9) and `hidden_dim = 3 × 512 = 1536` (not power-of-2). Code pads to next power-of-2 (2048) then crops with `[:n, :n]` after QR — works, but the QR-orthonormalised slice is no longer pure Hadamard. PR #1898 has the same situation (model_dim=512, hidden_dim=2048 there, also padded). Math still holds (R is orthogonal); just slightly less analytically clean.
- **Brotli/byte-shuffle composition:** rotated weights have different value distributions than originals — brotli ratio may shift. PR #1898 reports ~200 KB extra entropy for partial vs ~1 MB for full. Budget headroom needed (~200–400 KB) on Mikey_II's ~15 MB artifact.
- **Eval-attribution bug class:** rotations install on `eval_model` but `_sq_active` is a `CastedLinear` class flag — global. If a future audit reuses `base_model` post-`deserialize` (e.g. for sliding-window or TTT eval), forward will rotate against unrotated banks → silent corruption. Audit requires: zero `base_model` forward calls after `_sq_active=True` is set, or a `del base_model` immediately after `serialize` returns.
- **Loop blocks:** Mikey_II loops layers 3–5 (line 7 `loop_start=3, loop_end=5`). `start_layer=5` would rotate the looped exit layer once but visit it multiple times per forward — correct as long as the rotation is applied at every visit (it is, since it lives in `forward`).
- **TTT interaction:** Mikey_II's `ttt_enabled=False` by default; if TTT is enabled later, gradient flow through `x @ R` is fine (R is a constant buffer), but TTT updates would apply in the rotated frame.
