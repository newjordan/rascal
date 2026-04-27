"""Compression-only sweep for the 4k-vocab Rascal final_model.pt.

Loads /workspace/rascal/final_model.pt (post-EMA weights) ONCE, then for each
{quant policy x compression algorithm} combo:
  1. Quantizes the export state dict (per the policy).
  2. Serializes and compresses the quant blob to a temp file.
  3. Reloads, dequantizes, runs val_bpb (and sliding-window val_bpb).
  4. Records (combo, artifact_bytes, val_bpb_post_quant, val_bpb_sliding).

Sorted by val_bpb_sliding ascending; the best LEGAL (<=16 MiB) combo is
reported. NEVER retrains. NEVER mutates the checkpoint or the trainer.

Run on the 8xH100 pod (single-process is fine -- model is ~25M params):

  cd /workspace/rascal
  python3 /workspace/rascal/scripts/compression_sweep_4k_8x.py \
      --checkpoint /workspace/rascal/final_model.pt \
      --trainer /workspace/rascal/4k_vocab_rascal/train_gpt_4K_8xgpu.py \
      --val-files '/workspace/rascal/data/datasets/fineweb10B_sp4096/fineweb_val_*.bin' \
      --tokenizer /workspace/rascal/data/tokenizers/fineweb_4096_bpe.model

Or distributed (8 GPUs, faster sliding-window eval):
  torchrun --standalone --nproc_per_node=8 \
      /workspace/rascal/scripts/compression_sweep_4k_8x.py [args...]
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Callable

import torch
import torch.distributed as dist

# ---- Constants -------------------------------------------------------------

LEGAL_LIMIT_BYTES = 16 * 1024 * 1024  # 16 MiB
PR1120_BPB = 1.10987
TODAY_TARGET_BPB = 0.8672  # current 4k Rascal pre-compression sliding val_bpb

_BSHF_MAGIC = b"BSHF"

# Naming patterns for the 5 "naive" matrix tensors that participate in
# per-tensor quant policy choices. Everything else is left as INT8 default
# (or passthrough for tiny / control tensors).
NAIVE_LAYER_NAMES = (
    "tok_emb.weight",     # embedding (tied -> also lm_head)
    "qo_bank",            # attn out projection (and Q)
    "kv_bank",            # attn K/V projection
    "mlp_up_bank",        # mlp fc-up
    "mlp_down_bank",      # mlp fc-down (most quant-tolerant per 11d signals)
)

# Mapping from finer-grained "tier" labels (used in policy specs) to actual
# state-dict tensor names. The qo_bank holds both Q and Out projections in a
# single 3D tensor, so we can't cleanly split it -- treat it as one "attn_proj"
# bucket and treat kv_bank as the "qkv" bucket.
TIER_TO_TENSORS = {
    "mlp_proj": ("mlp_down_bank",),
    "mlp_fc":   ("mlp_up_bank",),
    "mlp":      ("mlp_up_bank", "mlp_down_bank"),
    "qkv":      ("kv_bank",),
    "attn_proj": ("qo_bank",),
    "attn":     ("qo_bank", "kv_bank"),
    "embed":    ("tok_emb.weight",),
}


def _classify_naive(name: str) -> str | None:
    """Match a state_dict key against the 5 named "naive" tensors."""
    for key in NAIVE_LAYER_NAMES:
        if key in name:
            return key
    return None


# ---- Trainer module loader -------------------------------------------------

def load_trainer_module(trainer_path: Path):
    """Import the train_gpt_4K_8xgpu.py file as a module without invoking
    main(). We need: GPT class, Hyperparameters, eval_val, eval_val_sliding,
    load_validation_tokens, build_sentencepiece_luts, _classify_param,
    quantize_int6_per_row, quantize_float_tensor, restore_low_dim_params_to_fp32,
    CONTROL_TENSOR_NAME_PATTERNS, CastedLinear.
    """
    trainer_path = trainer_path.resolve()
    if not trainer_path.is_file():
        raise FileNotFoundError(f"Trainer file not found: {trainer_path}")
    # The trainer reads env vars at import time (Hyperparameters defaults),
    # but does NOT call main() at import. Adding parent dir to sys.path lets
    # any sibling imports work (none are needed here).
    sys.path.insert(0, str(trainer_path.parent))
    spec = importlib.util.spec_from_file_location("rascal_trainer_4k_8x", trainer_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load spec from {trainer_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---- Quantization (int5 / int6 / int8 mixed) -------------------------------

def quantize_per_row_clipped(t: torch.Tensor, clip_range: int, naive_quantize_int6: Callable) -> tuple[torch.Tensor, torch.Tensor]:
    """Reuse the trainer's quantize_int6_per_row for any clip_range (defaults
    to 31 = int6 [-31,31], use 15 for int5 [-15,15]).
    The trainer's function already accepts clip_range as a keyword arg."""
    return naive_quantize_int6(t, clip_range=clip_range)


def mixed_quantize_with_policy(
    state_dict: dict,
    policy: dict,
    *,
    trainer,
) -> tuple[dict, dict]:
    """Apply a mixed-precision quant policy to the state dict.

    `policy` maps each of NAIVE_LAYER_NAMES -> {'int5','int6','int8'}.
    Anything else (tiny tensors, control tensors, other Linear weights) goes
    through the existing INT8 default path identical to the trainer's
    int6_cats={'mlp','attn','aux','embed'} setup.

    Returns (result, meta) suitable for the matching dequantize routine.
    """
    quantize_int6_per_row = trainer.quantize_int6_per_row
    quantize_float_tensor = trainer.quantize_float_tensor
    CONTROL_TENSOR_NAME_PATTERNS = trainer.CONTROL_TENSOR_NAME_PATTERNS

    result: dict[str, torch.Tensor] = {}
    meta: dict[str, object] = {}
    counts = {"int5": 0, "int6": 0, "int8": 0, "passthrough": 0}

    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()

        # Tiny tensors -> passthrough (matches trainer)
        if not t.is_floating_point() or t.numel() <= trainer.INT8_KEEP_FLOAT_MAX_NUMEL:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough"
            counts["passthrough"] += 1
            continue
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
            result[name] = t.float()
            meta[name] = "passthrough_ctrl"
            counts["passthrough"] += 1
            continue

        # Identify which "naive" tensor this is (if any) for policy lookup.
        naive_key = _classify_naive(name)
        bit_choice = policy.get(naive_key, "int8") if naive_key is not None else "int8"

        # Reshape banks (3D) to 2D for per-row quant (matches trainer's
        # ndim>=1 branch which does t.reshape(-1, t.shape[-1])).
        if t.ndim > 2:
            t_2d = t.reshape(-1, t.shape[-1])
        else:
            t_2d = t

        if bit_choice == "int5":
            q, s = quantize_int6_per_row(t_2d, clip_range=15)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int5", "clip_range": 15}
            counts["int5"] += 1
        elif bit_choice == "int6":
            q, s = quantize_int6_per_row(t_2d, clip_range=31)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int6", "clip_range": 31}
            counts["int6"] += 1
        else:  # int8 default
            q, s = quantize_float_tensor(t_2d)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int8"}
            counts["int8"] += 1

    return result, meta, counts


def dequantize_with_policy(result: dict, meta: dict, template_sd: dict) -> dict:
    """Mirror of trainer.dequantize_mixed_int6 but tolerates int5 entries
    (they share the per-row scale layout -- the int5 quantizer stored q in
    int8 with values in [-15,15], so reconstruction is the same product)."""
    out: dict[str, torch.Tensor] = {}
    for name, orig in template_sd.items():
        info = meta.get(name)
        if info is None:
            continue
        orig_dtype = orig.dtype
        if info in ("passthrough", "passthrough_ctrl", "passthrough_fp16"):
            t = result[name]
            if t.dtype == torch.float16 and orig_dtype in (torch.float32, torch.bfloat16):
                t = t.to(orig_dtype)
            out[name] = t
            continue
        q, s = result[name + ".q"], result[name + ".scale"]
        if s.ndim > 0:
            val = (q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))).to(orig_dtype)
        else:
            val = (q.float() * float(s.item())).to(orig_dtype)
        out[name] = val.reshape(orig.shape) if val.shape != orig.shape else val
    return out


# ---- Compression backends --------------------------------------------------

def _byte_shuffle(data: bytes, stride: int = 2) -> bytes:
    if stride <= 1 or len(data) < stride:
        return data
    import numpy as np
    src = np.frombuffer(data, dtype=np.uint8)
    n = len(src)
    out = np.empty(n, dtype=np.uint8)
    dest_off = 0
    for pos in range(stride):
        chunk = src[pos::stride]
        out[dest_off:dest_off + len(chunk)] = chunk
        dest_off += len(chunk)
    return _BSHF_MAGIC + bytes([stride]) + out.tobytes()


def _byte_unshuffle(data: bytes) -> bytes:
    if len(data) < 5 or data[:4] != _BSHF_MAGIC:
        return data
    stride = data[4]
    if stride < 2:
        return data[5:]
    import numpy as np
    payload = np.frombuffer(data, dtype=np.uint8, offset=5)
    n = len(payload)
    out = np.empty(n, dtype=np.uint8)
    src_off = 0
    for pos in range(stride):
        chunk_len = n // stride + (1 if pos < n % stride else 0)
        out[pos::stride][:chunk_len] = payload[src_off:src_off + chunk_len]
        src_off += chunk_len
    return out.tobytes()


def compress_blob(raw: bytes, scheme: str) -> bytes:
    """scheme is one of: 'zstd', 'brotli_bshf'."""
    if scheme == "zstd":
        import zstandard
        return zstandard.ZstdCompressor(level=22).compress(raw)
    if scheme == "brotli_bshf":
        import brotli
        shuffled = _byte_shuffle(raw, stride=2)
        return brotli.compress(shuffled, quality=11)
    raise ValueError(f"unknown scheme: {scheme}")


def decompress_blob(blob: bytes, scheme: str) -> bytes:
    if scheme == "zstd":
        import zstandard
        return zstandard.ZstdDecompressor().decompress(blob)
    if scheme == "brotli_bshf":
        import brotli
        raw = brotli.decompress(blob)
        return _byte_unshuffle(raw)
    raise ValueError(f"unknown scheme: {scheme}")


# ---- Sweep matrix ----------------------------------------------------------

def policy_uniform_int6() -> dict:
    return {k: "int6" for k in NAIVE_LAYER_NAMES}


def policy_all_int5() -> dict:
    # Embedding stays int8 (vocab matters more than per-dim resolution would
    # suggest; embeds in PR1120-tier runs are typically less quant-tolerant).
    p = {k: "int5" for k in NAIVE_LAYER_NAMES}
    p["tok_emb.weight"] = "int8"
    return p


def policy_mlp_int5() -> dict:
    # mlp_proj + mlp_fc int5 (most tolerant tier per 11d signals);
    # qkv + attn_proj int6 (less tolerant); embed int8 (least tolerant).
    return {
        "mlp_down_bank": "int5",  # mlp_proj
        "mlp_up_bank":   "int5",  # mlp_fc
        "kv_bank":       "int6",  # qkv
        "qo_bank":       "int6",  # attn_proj (and Q)
        "tok_emb.weight": "int8",
    }


def policy_attn_int5() -> dict:
    # The "less likely to win" combo (attn is least quant-tolerant) -- run for
    # falsification value: confirms the 11d ordering on this checkpoint.
    return {
        "kv_bank":       "int5",  # qkv
        "qo_bank":       "int5",  # attn_proj
        "mlp_up_bank":   "int6",
        "mlp_down_bank": "int6",
        "tok_emb.weight": "int8",
    }


def policy_mlp_proj_int5_only() -> dict:
    # Just mlp_proj (most tolerant) drops to int5; everything else int6.
    p = policy_uniform_int6()
    p["mlp_down_bank"] = "int5"
    return p


def build_sweep() -> list[dict]:
    """Returns a list of combos. Each combo is a dict with name/policy/scheme."""
    combos = [
        # baseline -- exactly matches what the trainer produced
        {"name": "baseline_int6_zstd",      "policy": policy_uniform_int6(),       "scheme": "zstd"},
        {"name": "uniform_int6_brotli",     "policy": policy_uniform_int6(),       "scheme": "brotli_bshf"},

        # mlp_proj-only step-down (smallest perturbation at int5)
        {"name": "mlp_proj_int5_zstd",      "policy": policy_mlp_proj_int5_only(), "scheme": "zstd"},
        {"name": "mlp_proj_int5_brotli",    "policy": policy_mlp_proj_int5_only(), "scheme": "brotli_bshf"},

        # mlp tier (proj+fc) int5
        {"name": "mlp_int5_zstd",           "policy": policy_mlp_int5(),           "scheme": "zstd"},
        {"name": "mlp_int5_brotli",         "policy": policy_mlp_int5(),           "scheme": "brotli_bshf"},

        # all 5 naive layers int5 (embed kept int8)
        {"name": "all_int5_zstd",           "policy": policy_all_int5(),           "scheme": "zstd"},
        {"name": "all_int5_brotli",         "policy": policy_all_int5(),           "scheme": "brotli_bshf"},

        # attn_only int5 (canary -- expect to LOSE bpb; if wins, signal is wrong)
        {"name": "attn_int5_zstd",          "policy": policy_attn_int5(),          "scheme": "zstd"},
        {"name": "attn_int5_brotli",        "policy": policy_attn_int5(),          "scheme": "brotli_bshf"},
    ]
    return combos


# ---- Distributed setup -----------------------------------------------------

def is_distributed() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def setup_distributed() -> tuple[int, int, int, torch.device]:
    if is_distributed():
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        rank, world_size, local_rank = 0, 1, 0
        torch.cuda.set_device(0)
        device = torch.device("cuda", 0)
    return rank, world_size, local_rank, device


def is_master() -> bool:
    return (not is_distributed()) or int(os.environ.get("RANK", "0")) == 0


def log(msg: str) -> None:
    if is_master():
        print(msg, flush=True)


# ---- Main sweep ------------------------------------------------------------

def build_eval_model(trainer, args, device):
    """Construct an eval-mode GPT identical to the one the trainer exports."""
    GPT = trainer.GPT
    CastedLinear = trainer.CastedLinear
    eval_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        mtp_num_heads=0, mtp_loss_weight=0.0,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
        dtg=args.dtg_enabled, ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
        gated_attention=args.gated_attention, value_residual=args.value_residual,
    ).to(device).bfloat16()
    eval_model.qo_bank.data = eval_model.qo_bank.data.float()
    eval_model.kv_bank.data = eval_model.kv_bank.data.float()
    eval_model.mlp_up_bank.data = eval_model.mlp_up_bank.data.float()
    eval_model.mlp_down_bank.data = eval_model.mlp_down_bank.data.float()
    for m in eval_model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    trainer.restore_low_dim_params_to_fp32(eval_model)
    return eval_model


def run_combo(
    combo: dict,
    *,
    trainer,
    args,
    sd_cpu: dict,
    template_sd: dict,
    device: torch.device,
    rank: int,
    world_size: int,
    grad_accum_steps: int,
    val_tokens,
    base_bytes_lut,
    has_leading_space_lut,
    is_boundary_token_lut,
    do_sliding: bool,
    eval_seq_len: int,
    artifact_dir: Path,
) -> dict:
    name = combo["name"]
    policy = combo["policy"]
    scheme = combo["scheme"]
    log(f"\n[combo] {name}: policy={policy} scheme={scheme}")

    # 1. quantize on rank-0 only (cpu work) and broadcast artifact bytes after
    #    write so all ranks read the same blob.
    artifact_path = artifact_dir / f"sweep_{name}.bin"
    if is_master():
        t0 = time.perf_counter()
        result, meta, counts = mixed_quantize_with_policy(sd_cpu, policy, trainer=trainer)
        buf = io.BytesIO()
        torch.save({"w": result, "m": meta}, buf)
        raw = buf.getvalue()
        try:
            blob = compress_blob(raw, scheme)
        except ImportError as e:
            log(f"[combo] {name}: SKIPPED (missing compressor: {e})")
            return {
                "name": name, "policy": policy, "scheme": scheme,
                "bytes": -1, "legal": False,
                "val_bpb_post_quant": float("nan"),
                "val_bpb_sliding": float("nan"),
                "counts": {}, "error": f"missing compressor: {e}",
            }
        with open(artifact_path, "wb") as f:
            f.write(blob)
        artifact_bytes = len(blob)
        quant_secs = time.perf_counter() - t0
        log(f"[combo] {name}: counts={counts} raw={len(raw)} compressed={artifact_bytes} ({quant_secs:.1f}s)")
    else:
        artifact_bytes = 0
        counts = {}

    if is_distributed():
        dist.barrier()
        # Broadcast artifact_bytes from master so all ranks agree.
        sz_t = torch.tensor([artifact_bytes], dtype=torch.long, device=device)
        dist.broadcast(sz_t, src=0)
        artifact_bytes = int(sz_t.item())

    legal = (0 < artifact_bytes <= LEGAL_LIMIT_BYTES)

    # 2. all ranks read+decompress+dequant -> load eval model -> eval val_bpb
    with open(artifact_path, "rb") as f:
        blob_disk = f.read()
    raw_disk = decompress_blob(blob_disk, scheme)
    quant_state = torch.load(io.BytesIO(raw_disk), map_location="cpu")
    deq_state = dequantize_with_policy(quant_state["w"], quant_state["m"], template_sd)

    eval_model = build_eval_model(trainer, args, device)
    eval_model.load_state_dict(deq_state, strict=True)

    # 3. fast post-quant val_bpb
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    q_val_loss, q_val_bpb = trainer.eval_val(
        args, eval_model, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        eval_seq_len=eval_seq_len,
    )
    torch.cuda.synchronize()
    fast_secs = time.perf_counter() - t1
    log(f"[combo] {name}: val_bpb_post_quant={q_val_bpb:.6f} ({fast_secs:.1f}s)")

    # 4. sliding-window val_bpb (the leaderboard metric)
    sw_val_bpb = float("nan")
    if do_sliding and args.eval_stride > 0 and args.eval_stride < eval_seq_len:
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        try:
            sw_val_loss, sw_val_bpb = trainer.eval_val_sliding(
                args, eval_model, rank, world_size, device,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                stride=args.eval_stride,
                eval_seq_len=eval_seq_len,
            )
        except Exception as e:
            log(f"[combo] {name}: sliding eval FAILED: {e}")
            sw_val_bpb = float("nan")
        torch.cuda.synchronize()
        slide_secs = time.perf_counter() - t2
        log(f"[combo] {name}: val_bpb_sliding={sw_val_bpb:.6f} ({slide_secs:.1f}s)")

    del eval_model, deq_state, quant_state
    torch.cuda.empty_cache()

    return {
        "name": name, "policy": policy, "scheme": scheme,
        "bytes": artifact_bytes, "legal": legal,
        "val_bpb_post_quant": float(q_val_bpb),
        "val_bpb_sliding": float(sw_val_bpb),
        "counts": counts,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="/workspace/rascal/final_model.pt",
                   help="Path to the saved final_model.pt (post-EMA weights).")
    p.add_argument("--trainer", default="/workspace/rascal/4k_vocab_rascal/train_gpt_4K_8xgpu.py",
                   help="Path to the trainer .py file (imported for GPT class + helpers).")
    p.add_argument("--val-files", default="/workspace/rascal/data/datasets/fineweb10B_sp4096/fineweb_val_*.bin",
                   help="Glob for fineweb val shards.")
    p.add_argument("--tokenizer", default="/workspace/rascal/data/tokenizers/fineweb_4096_bpe.model",
                   help="Path to the SentencePiece model.")
    p.add_argument("--artifact-dir", default="/workspace/rascal/sweep_artifacts",
                   help="Where to write per-combo .bin blobs.")
    p.add_argument("--no-sliding", action="store_true",
                   help="Skip sliding-window eval (just post-quant val_bpb -- ~10x faster).")
    p.add_argument("--eval-stride", type=int, default=64,
                   help="Sliding-window stride (matches the trainer's leaderboard run).")
    p.add_argument("--combos", default="",
                   help="Comma-separated combo-name allowlist (empty = run all).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print sweep matrix and exit without running.")
    p.add_argument("--val-batch-size", type=int, default=0,
                   help="Override VAL_BATCH_SIZE (else use trainer default).")
    args_cli = p.parse_args()

    # --- import the trainer module -----------------------------------------
    trainer_path = Path(args_cli.trainer)
    log(f"loading trainer from {trainer_path}")
    trainer = load_trainer_module(trainer_path)

    # --- distributed setup --------------------------------------------------
    rank, world_size, local_rank, device = setup_distributed()
    log(f"world_size={world_size} rank={rank} device={device}")

    # --- override env so Hyperparameters picks up the right paths -----------
    os.environ["DATA_PATH"] = str(Path(args_cli.val_files).parent)
    os.environ["TOKENIZER_PATH"] = args_cli.tokenizer
    if args_cli.val_batch_size > 0:
        os.environ["VAL_BATCH_SIZE"] = str(args_cli.val_batch_size)
    args = trainer.Hyperparameters()
    # Force the val-files glob to whatever the user passed (tolerates absolute
    # paths different from the data_path-relative default).
    args.val_files = args_cli.val_files
    args.tokenizer_path = args_cli.tokenizer
    # Disable expensive on-eval n-gram bookkeeping for the sweep
    args.ngram_eval_order = 0
    args.skip_final_eval = False
    args.eval_stride = args_cli.eval_stride
    grad_accum_steps = max(1, int(os.environ.get("GRAD_ACCUM_STEPS", "1")))

    log(f"vocab_size={args.vocab_size} num_layers={args.num_layers} model_dim={args.model_dim}")
    log(f"val_files={args.val_files}")
    log(f"tokenizer={args.tokenizer_path}")

    # --- print sweep matrix (always) ---------------------------------------
    sweep = build_sweep()
    if args_cli.combos:
        keep = set(args_cli.combos.split(","))
        sweep = [c for c in sweep if c["name"] in keep]

    log("\n=== SWEEP MATRIX ===")
    for c in sweep:
        log(f"  {c['name']:<28s} scheme={c['scheme']:<12s} policy={c['policy']}")
    log(f"=== {len(sweep)} combos total ===\n")

    if args_cli.dry_run:
        log("dry-run: exiting without running combos.")
        return

    # --- load checkpoint (rank 0 reads, broadcast via barrier) -------------
    ckpt_path = Path(args_cli.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint missing: {ckpt_path}")
    log(f"loading checkpoint: {ckpt_path} ({ckpt_path.stat().st_size:,} bytes)")
    sd_full = torch.load(ckpt_path, map_location="cpu")
    # The trainer already excludes mtp_heads when saving final_model.pt, but
    # double-check (be defensive against accidentally re-using a base state).
    sd_cpu = {k: v for k, v in sd_full.items() if "mtp_heads" not in k}
    log(f"checkpoint keys: {len(sd_cpu)} (excluded {len(sd_full) - len(sd_cpu)} mtp_* tensors)")

    # template_sd is what the eval model expects (same shapes/dtypes as sd_cpu)
    template_sd = sd_cpu

    # --- prepare val data and luts (each rank loads its own copy) ----------
    sp = trainer.spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"vocab mismatch: ckpt={args.vocab_size}, sp={int(sp.vocab_size())}")

    eval_seq_len = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    val_seq_len = max(args.train_seq_len, eval_seq_len)
    val_tokens = trainer.load_validation_tokens(args.val_files, val_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = trainer.build_sentencepiece_luts(
        sp, args.vocab_size, device,
    )
    log(f"val tokens: {val_tokens.numel():,}  eval_seq_len={eval_seq_len}  stride={args.eval_stride}")

    # --- prepare artifact dir ----------------------------------------------
    artifact_dir = Path(args_cli.artifact_dir)
    if is_master():
        artifact_dir.mkdir(parents=True, exist_ok=True)
    if is_distributed():
        dist.barrier()

    # --- run sweep ----------------------------------------------------------
    results = []
    do_sliding = not args_cli.no_sliding
    sweep_t0 = time.perf_counter()

    for i, combo in enumerate(sweep):
        log(f"\n=== combo {i+1}/{len(sweep)}: {combo['name']} ===")
        try:
            r = run_combo(
                combo,
                trainer=trainer, args=args,
                sd_cpu=sd_cpu, template_sd=template_sd,
                device=device, rank=rank, world_size=world_size,
                grad_accum_steps=grad_accum_steps,
                val_tokens=val_tokens,
                base_bytes_lut=base_bytes_lut,
                has_leading_space_lut=has_leading_space_lut,
                is_boundary_token_lut=is_boundary_token_lut,
                do_sliding=do_sliding,
                eval_seq_len=eval_seq_len,
                artifact_dir=artifact_dir,
            )
        except Exception as e:
            log(f"[combo] {combo['name']}: ERROR -- {e}")
            traceback.print_exc()
            r = {
                "name": combo["name"], "policy": combo["policy"], "scheme": combo["scheme"],
                "bytes": -1, "legal": False,
                "val_bpb_post_quant": float("nan"),
                "val_bpb_sliding": float("nan"),
                "counts": {}, "error": str(e),
            }
        results.append(r)

    sweep_secs = time.perf_counter() - sweep_t0
    log(f"\n=== sweep complete in {sweep_secs:.1f}s ({sweep_secs/60:.1f} min) ===")

    if not is_master():
        if is_distributed():
            dist.barrier()
        return

    # --- final report -------------------------------------------------------
    def sort_key(r):
        v = r.get("val_bpb_sliding")
        if v is None or math.isnan(v):
            v = r.get("val_bpb_post_quant", float("inf"))
        if v is None or math.isnan(v):
            v = float("inf")
        return v

    results_sorted = sorted(results, key=sort_key)
    log("\n=== RESULTS (sorted by val_bpb_sliding ascending) ===")
    log(f"{'combo':<28s} {'bytes':>10s} {'legal':>6s} {'post_quant':>12s} {'sliding':>12s} {'delta_pr1120':>13s}")
    for r in results_sorted:
        bpb_post = r["val_bpb_post_quant"]
        bpb_sw = r["val_bpb_sliding"]
        delta = (bpb_sw if not math.isnan(bpb_sw) else bpb_post) - PR1120_BPB
        log(
            f"combo={r['name']:<22s}  "
            f"bytes={r['bytes']:<10d}  "
            f"legal={'yes' if r['legal'] else 'no':<3s}  "
            f"val_bpb={bpb_post:.6f}  "
            f"val_bpb_sliding={bpb_sw:.6f}  "
            f"delta_vs_pr1120={delta:+.6f}"
        )

    legal_with_score = [
        r for r in results_sorted
        if r["legal"] and not math.isnan(sort_key(r))
    ]
    if legal_with_score:
        best = legal_with_score[0]
        bpb_use = best["val_bpb_sliding"] if not math.isnan(best["val_bpb_sliding"]) else best["val_bpb_post_quant"]
        log("")
        log(f"BEST LEGAL: combo={best['name']} bytes={best['bytes']} val_bpb_sliding={bpb_use:.6f}")
        if bpb_use < PR1120_BPB:
            log(f"  -> BEATS PR1120 ({PR1120_BPB:.6f}) by {PR1120_BPB - bpb_use:.6f} bpb!")
        else:
            log(f"  -> Does NOT beat PR1120 ({PR1120_BPB:.6f}); short by {bpb_use - PR1120_BPB:.6f} bpb.")
    else:
        log("\nBEST LEGAL: <none>  (no combo fit under 16 MiB AND produced a numeric val_bpb)")

    if is_distributed():
        dist.barrier()


if __name__ == "__main__":
    main()
