# Port plan: PR #1698 — GatedDeltaNet (FLA) + score-first TTT

## 1. Technique summary
GDN is an O(n) linear-attention layer: a gated delta-rule recurrence that low-rank-updates a matrix-valued state from (q,k,v), modulated by a per-token forget gate. No KV cache. FLA (`flash-linear-attention` / `fla-core`) ships chunked Triton kernels. "Legal score-first TTT" is the same per-chunk score-then-SGD protocol Mikey_II already runs (`inference_mode` score, then 3 epochs SGD on the 32K-token chunk, freeze first 2 blocks).

## 2. PR base + scope
arsenis-cmd, +3386 / -0, on top of PR #1687 (resouer's `K_KVShare_Wider` GDN at 1.04090). One record dir: `architectures.py` (709), `train_gdn_7k.py` (1361), `configs.py` (316). Deps: `flash-linear-attention==0.4.2`, `fla-core==0.4.2`, `triton==3.2.0`, `transformers==5.5.4`, `tokenizers==0.22.2`, `safetensors==0.7.0`, `zstandard`. PR is non-compliant: 16,600,916 B > 16 MB decimal cap (sergeevii123); dexhunter flagged a GDN-family LUT byte double-count (#1719) — real canonical likely ~1.189.

## 3. Code surface in Mikey_II
`/home/frosty40/sota_rascal/legs/mikey_ii_baseline/train_gpt.py`, 469 lines:
- `CausalSelfAttention`, 91-103 — direct call to `flash_attn_3_func`. Replace with FLA `GatedDeltaNet(hidden_size, head_dim, num_heads, expand_v=1, mode="chunk", layer_idx=i)`.
- `Block`, 107-113 — `attn_out = self.attn(...)` → `recurrent_out = self.recurrent(self.attn_norm(x_in))[0]` (FLA returns a tuple).
- `Rotary` (79-87), `apply_rotary_emb` (88-90) — gone from GDN path; GDN's short-conv replaces RoPE.
- `GPT.__init__`, 114-137 — block instantiation (121), `block.attn.rope_dims` patch (122-124), and XSA-last-N loop (127-128) all gate on `attention_kind`. XSA meaningless under GDN.
- `forward_logits`, 144-159 — interface unchanged.
- `eval_val_ttt`, 340-377 — works on `base_model.parameters()`. Hardcoded grad clip `1.` at 374; first-N freeze unimplemented — lift both to hparams.
- `restore_fp32_params` (216), `collect_hessians` (221), `gptq_mixed_quantize` (257) — GDN's k/v/q/g/b_proj are linear, CastedLinear coverage holds. Short-conv Conv1d small — passthrough fp16 path at 261 (`numel()<=65536`) covers it.

## 4. Library + dependency
`pip install --no-deps flash-linear-attention==0.4.2 fla-core==0.4.2`. PR pins `triton==3.2.0`; pod is `vastai/pytorch:cuda-13.0.2-auto`, torch 2.11.0+cu130, FA3 wheel. FLA wheels target torch 2.6-2.9 / cu12x — cu130 install needs `--no-deps` and likely fights triton ≥3.5 ABI on chunked kernels. Fallback `FLA_USE_NAIVE=1` (architectures.py:162). GDN is linear in T — memory-comfortable on 1×H100 80GB.

## 5. Hyperparams to expose
On `Hyperparameters` (line 7), baked defaults, no env dicts:
- `attention_kind: str = "mha"` ("mha" | "gdn")
- `gdn_head_dim: int = 64`
- `gdn_expand_v: int = 1`
- `gdn_use_short_conv: bool = True`
- `gdn_allow_neg_eigval: bool = False`
- `gdn_mode: str = "chunk"`
- `ttt_freeze_blocks: int = 2`
- `ttt_grad_clip: float = 1.0` (currently hardcoded at 374)

## 6. Compute requirement
PR: 8×H100 80GB SXM, ~2,377 steps in 600 s, ~320 s eval (200 s TTT). 1×H100 smoke feasible — linear attention is memory-light at T=2048. Smoke pattern: `world_size=1`, `grad_accum_steps=8`, `train_batch_tokens=786432`. Canonical needs 8×H100; 1×H100 number not directly comparable.

## 7. Expected gain stacked on Mikey_II
Skeptical. GDN sits at 1.01 vs prior softmax floor 1.08 (+0.07 at that tier). Mikey_II is **0.86548** with softmax + loops + brotli + mixed-int + TTT. NOT orthogonal: (a) Mikey_II's encoder/decoder skip-loop (131-137) co-tuned to per-block KV softmax; looping a GDN block re-runs recurrent state in an unstudied way. (b) Headline 1.01 plausibly ~1.19 canonical (LUT bug) — true arch-vs-softmax delta much smaller. (c) PR's TTT delta is -0.01, which Mikey_II's TTT already captures. **Prior: regression of +0.05 to +0.20 BPB.** 1-seed smoke only.

## 8. A/B test recipe
Stage 1 (~30 min): `mikey_ii_gdn_seed1337` on 1×H100, full 600 s. Validate fwd+bwd+TTT, GDN k_proj/v_proj quantize cleanly. Compare to matched 1×H100 `mikey_ii_baseline_seed1337` smoke. Stage 2 (only if Stage 1 within +0.05 of baseline): `mikey_ii_gdn_seed{1337,42,2025}` on 8×H100, 600 s, 3-seed mean.

## 9. Risk flags
- **Arch-swap: HIGH.** Loop + skip gates + parallel residual + XSA all co-tuned to softmax MHA. GDN drops c_q/c_k/c_v, RoPE, q_gain, XSA at once.
- **Library install: HIGH on cu130.** FLA 0.4.2 not built against cu130/torch 2.11. `--no-deps` + `FLA_USE_NAIVE=1` fallback.
- **TTT redundancy.** Mikey_II already has score-first TTT — GDN-specific TTT marginal ~0. Architecture is the only real delta.
- **Compliance taint.** Don't inherit PR's 16 MB miss or LUT bug. Keep Mikey_II's `build_sentencepiece_luts` (21-30) and serialize path.
- **Loop × recurrence unstudied.** Either `num_loops=0` for GDN or treat looping as a separate axis after baseline.
