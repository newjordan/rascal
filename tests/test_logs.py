from __future__ import annotations

from pathlib import Path

import pytest

from rascal_lab.logs import classify, diff_axes, parse_log, sort_rows


def _write(path: Path, body: str) -> Path:
    path.write_text(body)
    return path


def test_best_bpb_keeps_an_explicit_zero(tmp_path: Path):
    path = _write(
        tmp_path / "zero.txt",
        "\n".join(
            [
                "  run_id: zero",
                "quantized_ttt val_loss:0.0 val_bpb:0.0",
                "pre-quantization post-ema val_loss:1.2 val_bpb:1.200000",
            ]
        ),
    )
    row = parse_log(path)
    assert row["ttt_bpb"] == 0.0
    assert row["best_bpb"] == 0.0


def test_sort_treats_a_zero_margin_as_on_cap(tmp_path: Path):
    def row(name: str, score: str, total: str | None, rate: str = "6000000") -> dict:
        lines = [
            f"  run_id: {name}",
            f"pre-quantization post-ema val_loss:1.0 val_bpb:{score}",
            f"tok/s: {rate}",
        ]
        if total is not None:
            lines.append(f"Total submission size quantized+brotli: {total}")
        return parse_log(_write(tmp_path / f"{name}.txt", "\n".join(lines)))

    rows = sort_rows(
        [
            row("on_cap", "1.0", "16000000"),
            row("room", "1.0", "15900000"),
            row("overflow", "1.0", "16000001"),
            row("missing", "1.0", None),
            row("better", "0.9", "16000001"),
        ]
    )
    assert [item["run_id"] for item in rows] == [
        "better",
        "room",
        "on_cap",
        "overflow",
        "missing",
    ]
    assert rows[2]["byte_margin"] == 0
    assert rows[2]["failure_class"] == "needs_compare"


def test_classify_thresholds(tmp_path: Path):
    text = "\n".join(
        [
            "  run_id: damaged",
            "  vocab_size: 10240",
            "  num_layers: 9",
            "pre-quantization post-ema val_loss:2.0 val_bpb:1.100000",
            "quantized val_loss:2.1 val_bpb:1.130000",
            "Total submission size quantized+brotli: 16000001",
            "tok/s: 4999999.2",
            "tok/s: 8920225.5",
        ]
    )
    row = parse_log(_write(tmp_path / "damaged.txt", text))
    assert row["quant_damage"] == pytest.approx(0.03)
    assert row["last_tok_s"] == 8920225.5
    assert row["failure_class"] == "size_overflow+quant_damage+neural_quality"

    exact = classify(
        {
            "byte_margin": 0,
            "quant_damage": 0.02,
            "post_ema_bpb": 1.09,
            "last_tok_s": 5_000_000,
        }
    )
    assert exact == "needs_compare"

    slow = classify({"byte_margin": None, "quant_damage": None, "post_ema_bpb": None, "last_tok_s": 4999999.2})
    assert slow == "runtime_step_loss"


def test_diff_axes_reports_only_changed_keys(tmp_path: Path):
    base = parse_log(
        _write(
            tmp_path / "base.txt",
            "  run_id: base\n  vocab_size: 10240\n  num_layers: 11\npre-quantization post-ema val_loss:1 val_bpb:1.0\n",
        )
    )
    other = parse_log(
        _write(
            tmp_path / "other.txt",
            "  run_id: other\n  vocab_size: 10240\n  num_layers: 10\npre-quantization post-ema val_loss:1 val_bpb:1.1\n",
        )
    )
    assert diff_axes(base, other) == ["num_layers:11->10"]

