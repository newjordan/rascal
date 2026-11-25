"""Mine trainer logs for changed hyperparameters and failure classes."""

from __future__ import annotations

import glob
import math
import re
from pathlib import Path

from rascal_lab.limits import SUBMISSION_CAP_BYTES

AXES = (
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
)

QUANT_DAMAGE_LIMIT = 0.02
NEURAL_BPB_LIMIT = 1.09
SLOW_TOKENS_PER_SECOND = 5_000_000

_METRIC_PATTERNS = {
    "post_ema_bpb": r"pre-quantization post-ema val_loss:[0-9.]+ val_bpb:([0-9.]+)",
    "quant_bpb": r"^quantized val_loss:[0-9.]+ val_bpb:([0-9.]+)",
    "sliding_bpb": r"quantized_sliding_window val_loss:[0-9.]+ val_bpb:([0-9.]+)",
    "ttt_bpb": r"quantized_ttt val_loss:[0-9.]+ val_bpb:([0-9.]+)",
}


def _float(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def first_present(*values: float | None) -> float | None:
    for value in values:
        if value is not None:
            return value
    return None


def parse_log(path: Path) -> dict:
    text = path.read_text(errors="replace")
    cfg: dict[str, str] = {}
    metrics: dict = {"path": str(path)}

    for key, val in re.findall(r"^\s{2}([A-Za-z_][\w]*):\s*(.+?)\s*$", text, re.M):
        cfg[key] = val
    for key, val in re.findall(r"^\s*([A-Za-z_][\w]*?)=([^\s,]+)", text, re.M):
        cfg.setdefault(key, val)

    metrics["run_id"] = cfg.get("run_id") or path.stem
    for key, pattern in _METRIC_PATTERNS.items():
        match = re.search(pattern, text, re.M)
        metrics[key] = _float(match.group(1)) if match else None

    for key, pattern in {
        "total_bytes": r"Total submission size quantized\+brotli:\s*(\d+)",
        "quant_model_bytes": r"Serialized model quantized\+brotli:\s*(\d+)",
        "model_params": r"model_params:(\d+)",
    }.items():
        match = re.search(pattern, text)
        metrics[key] = int(match.group(1)) if match else None

    rates = [float(item) for item in re.findall(r"tok/s:\s*([0-9.]+)", text)]
    metrics["last_tok_s"] = rates[-1] if rates else None
    metrics["cfg"] = cfg
    metrics["best_bpb"] = first_present(
        metrics["ttt_bpb"],
        metrics["sliding_bpb"],
        metrics["quant_bpb"],
        metrics["post_ema_bpb"],
    )
    if metrics["quant_bpb"] is not None and metrics["post_ema_bpb"] is not None:
        metrics["quant_damage"] = metrics["quant_bpb"] - metrics["post_ema_bpb"]
    else:
        metrics["quant_damage"] = None
    if metrics["total_bytes"] is not None:
        metrics["byte_margin"] = SUBMISSION_CAP_BYTES - metrics["total_bytes"]
    else:
        metrics["byte_margin"] = None
    metrics["failure_class"] = classify(metrics)
    return metrics


def classify(row: dict) -> str:
    classes: list[str] = []
    margin = row.get("byte_margin")
    if margin is not None and margin < 0:
        classes.append("size_overflow")
    damage = row.get("quant_damage")
    if damage is not None and damage > QUANT_DAMAGE_LIMIT:
        classes.append("quant_damage")
    post = row.get("post_ema_bpb")
    if post is not None and post > NEURAL_BPB_LIMIT:
        classes.append("neural_quality")
    rate = row.get("last_tok_s")
    if rate is not None and rate < SLOW_TOKENS_PER_SECOND:
        classes.append("runtime_step_loss")
    return "+".join(classes) if classes else "needs_compare"


def diff_axes(base: dict, row: dict) -> list[str]:
    changes = []
    base_cfg = base["cfg"]
    row_cfg = row["cfg"]
    for axis in AXES:
        before = base_cfg.get(axis)
        after = row_cfg.get(axis)
        if before != after and (before is not None or after is not None):
            changes.append(f"{axis}:{before}->{after}")
    return changes


def sort_rows(rows: list[dict]) -> list[dict]:
    """Lower bits-per-byte first. Equal scores keep the larger byte margin first.

    Missing scores sort last. A margin of zero is a real margin (the artifact
    is exactly on the cap) and must not be treated as missing.
    """

    def key(row: dict) -> tuple:
        score = row["best_bpb"]
        margin = row["byte_margin"]
        missing_margin = margin is None
        margin_rank = -(margin if margin is not None else 0)
        return (score is None, score if score is not None else math.inf, missing_margin, margin_rank)

    return sorted(rows, key=key)


def expand_inputs(items: list[str]) -> tuple[list[Path], list[str]]:
    paths: list[Path] = []
    missing: list[str] = []
    for item in items:
        matches = [Path(match) for match in glob.glob(item, recursive=True) if Path(match).is_file()]
        if not matches:
            missing.append(item)
        paths.extend(matches)
    return sorted(set(paths)), missing


def fmt(value: object) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _cell(value: object) -> str:
    return fmt(value).replace("|", "/")


def format_table(rows: list[dict], baseline: dict | None) -> list[str]:
    lines = [
        "| run | best | post | quant | q_damage | bytes_margin | class | axes_vs_base |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        axes = "; ".join(diff_axes(baseline, row)[:12]) if baseline else ""
        lines.append(
            "| {run} | {best} | {post} | {quant} | {damage} | {margin} | {klass} | {axes} |".format(
                run=_cell(row["run_id"]),
                best=_cell(row["best_bpb"]),
                post=_cell(row["post_ema_bpb"]),
                quant=_cell(row["quant_bpb"]),
                damage=_cell(row["quant_damage"]),
                margin=_cell(row["byte_margin"]),
                klass=_cell(row["failure_class"]),
                axes=_cell(axes),
            )
        )
    return lines


def class_counts(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for name in str(row["failure_class"]).split("+"):
            counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))
