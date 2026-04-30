#!/usr/bin/env python3
"""Mine run logs for hidden experiment axes and failure classes."""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path


CAP = 16_000_000
AXES = [
    "vocab_size",
    "num_layers",
    "model_dim",
    "embedding_dim",
    "num_heads",
    "num_kv_heads",
    "mlp_mult",
    "num_loops",
    "loop_start",
    "loop_end",
    "enable_looping_at",
    "loop_warmup_enabled",
    "parallel_residual_start",
    "warmup_steps",
    "warmdown_frac",
    "train_batch_tokens",
    "grad_accum_steps",
    "muon_wd",
    "adam_wd",
    "embed_wd",
    "ema_decay",
    "matrix_bits",
    "embed_bits",
    "matrix_clip_sigmas",
    "mlp_clip_sigmas",
    "embed_clip_sigmas",
    "gptq_calibration_batches",
    "gptq_reserve_seconds",
    "compressor",
    "ttt_enabled",
    "ttt_epochs",
    "ttt_lr",
    "eval_stride",
]


def _float(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def parse_file(path: Path) -> dict:
    text = path.read_text(errors="replace")
    cfg: dict[str, str] = {}
    metrics: dict[str, float | int | str | None] = {"path": str(path)}

    for key, val in re.findall(r"^\s{2}([A-Za-z_][\w]*):\s*(.+?)\s*$", text, re.M):
        cfg[key] = val
    for key, val in re.findall(r"^\s*([A-Za-z_][\w]*?)=([^\s,]+)", text, re.M):
        cfg.setdefault(key, val)

    run_id = cfg.get("run_id")
    if not run_id:
        run_id = path.stem
    metrics["run_id"] = run_id

    patterns = {
        "post_ema_bpb": r"pre-quantization post-ema val_loss:[0-9.]+ val_bpb:([0-9.]+)",
        "quant_bpb": r"^quantized val_loss:[0-9.]+ val_bpb:([0-9.]+)",
        "sliding_bpb": r"quantized_sliding_window val_loss:[0-9.]+ val_bpb:([0-9.]+)",
        "ttt_bpb": r"quantized_ttt val_loss:[0-9.]+ val_bpb:([0-9.]+)",
    }
    for key, pat in patterns.items():
        match = re.search(pat, text, re.M)
        metrics[key] = _float(match.group(1)) if match else None

    for key, pat in {
        "total_bytes": r"Total submission size quantized\+brotli:\s*(\d+)",
        "quant_model_bytes": r"Serialized model quantized\+brotli:\s*(\d+)",
        "model_params": r"model_params:(\d+)",
    }.items():
        match = re.search(pat, text)
        metrics[key] = int(match.group(1)) if match else None

    toks = [float(x) for x in re.findall(r"tok/s:\s*([0-9.]+)", text)]
    metrics["last_tok_s"] = int(toks[-1]) if toks else None
    metrics["cfg"] = cfg
    metrics["best_bpb"] = metrics["ttt_bpb"] or metrics["sliding_bpb"] or metrics["quant_bpb"] or metrics["post_ema_bpb"]
    metrics["quant_damage"] = (
        metrics["quant_bpb"] - metrics["post_ema_bpb"]
        if metrics["quant_bpb"] is not None and metrics["post_ema_bpb"] is not None
        else None
    )
    metrics["byte_margin"] = CAP - metrics["total_bytes"] if metrics["total_bytes"] is not None else None
    metrics["failure_class"] = classify(metrics)
    return metrics


def classify(row: dict) -> str:
    classes: list[str] = []
    if row.get("byte_margin") is not None and row["byte_margin"] < 0:
        classes.append("size_overflow")
    if row.get("quant_damage") is not None and row["quant_damage"] > 0.02:
        classes.append("quant_damage")
    if row.get("post_ema_bpb") is not None and row["post_ema_bpb"] > 1.09:
        classes.append("neural_quality")
    if row.get("last_tok_s") is not None and row["last_tok_s"] < 5_000_000:
        classes.append("runtime_step_loss")
    return "+".join(classes) if classes else "needs_compare"


def fmt(v: object) -> str:
    if v is None:
        return "NA"
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def diff_axes(base: dict, row: dict) -> list[str]:
    out = []
    bcfg = base["cfg"]
    rcfg = row["cfg"]
    for axis in AXES:
        bv = bcfg.get(axis)
        rv = rcfg.get(axis)
        if bv != rv and (bv is not None or rv is not None):
            out.append(f"{axis}:{bv}->{rv}")
    return out


def expand_inputs(items: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in items:
        matches = glob.glob(item, recursive=True)
        paths.extend(Path(m) for m in matches if Path(m).is_file())
    return sorted(set(paths))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="log files or globs")
    ap.add_argument("--baseline", help="baseline log path")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    paths = expand_inputs(args.logs)
    rows = [parse_file(p) for p in paths]
    rows = [r for r in rows if r.get("best_bpb") is not None or r.get("total_bytes") is not None]
    rows.sort(key=lambda r: (r["best_bpb"] is None, r["best_bpb"] or 999, -(r["byte_margin"] or -999999999)))

    baseline = parse_file(Path(args.baseline)) if args.baseline else (rows[0] if rows else None)

    print("| run | best | post | quant | q_damage | bytes_margin | class | axes_vs_base |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    for row in rows[: args.top]:
        axes = ", ".join(diff_axes(baseline, row)[:12]) if baseline else ""
        print(
            "| {run} | {best} | {post} | {quant} | {qd} | {margin} | {klass} | {axes} |".format(
                run=row["run_id"],
                best=fmt(row["best_bpb"]),
                post=fmt(row["post_ema_bpb"]),
                quant=fmt(row["quant_bpb"]),
                qd=fmt(row["quant_damage"]),
                margin=fmt(row["byte_margin"]),
                klass=row["failure_class"],
                axes=axes,
            )
        )


if __name__ == "__main__":
    main()
