from __future__ import annotations

import struct
from pathlib import Path

from rascal_lab.cli import mine_main, validate_main
from rascal_lab.shards import HEADER_BYTES, SHARD_MAGIC, SHARD_VERSION, write_uint16_shard


def _header(tokens: int) -> bytes:
    blob = bytearray(HEADER_BYTES)
    struct.pack_into("<3i", blob, 0, SHARD_MAGIC, SHARD_VERSION, tokens)
    return bytes(blob)


def test_validate_counts_sidecars_separately(tmp_path: Path, capsys):
    write_uint16_shard(tmp_path / "fineweb_train_000000.bin", [1, 2])
    write_uint16_shard(tmp_path / "fineweb_val_000000.bin", [3])
    write_uint16_shard(tmp_path / "fineweb_val_bytes_000000.bin", [4, 5, 6])
    code = validate_main([str(tmp_path)])
    captured = capsys.readouterr()
    assert code == 0
    assert "summary train=1 val=1 val_bytes=1 bad=[]" in captured.out
    assert "kind=val_bytes" in captured.out
    assert captured.err == ""


def test_validate_expect_rejects_a_glob_inflated_val_count(tmp_path: Path, capsys):
    write_uint16_shard(tmp_path / "fineweb_val_000000.bin", [1])
    write_uint16_shard(tmp_path / "fineweb_val_bytes_000000.bin", [2])
    code = validate_main([str(tmp_path), "--expect-val", "2"])
    captured = capsys.readouterr()
    assert code == 1
    assert "val=1 expected=2" in captured.err
    inventory = validate_main(["--inventory", str(tmp_path), "--expect-val", "1", "--expect-val-bytes", "1"])
    assert inventory == 0


def test_validate_missing_dir_and_empty_dir(tmp_path: Path, capsys):
    assert validate_main([str(tmp_path / "missing")]) == 2
    assert validate_main([str(tmp_path)]) == 2
    capsys.readouterr()
    assert validate_main(["--inventory", str(tmp_path)]) == 0


def test_validate_reports_a_corrupt_shard(tmp_path: Path, capsys):
    (tmp_path / "fineweb_train_000000.bin").write_bytes(_header(4))
    code = validate_main([str(tmp_path), "--expect-train", "1"])
    captured = capsys.readouterr()
    assert code == 1
    assert "reason=size_mismatch" in captured.out
    assert "bad=['fineweb_train_000000.bin']" in captured.out


def test_mine_prints_a_table_and_warns_on_unmatched_globs(tmp_path: Path, capsys):
    log = tmp_path / "lane.txt"
    log.write_text(
        "\n".join(
            [
                "  run_id: lane",
                "  num_layers: 10",
                "pre-quantization post-ema val_loss:1.0 val_bpb:1.050000",
                "Total submission size quantized+brotli: 16000000",
                "tok/s: 7000000",
            ]
        )
    )
    code = mine_main([str(log), str(tmp_path / "missing-*.log"), "--top", "5"])
    captured = capsys.readouterr()
    assert code == 0
    assert "no files matched:" in captured.err
    assert "| lane | 1.050000 | 1.050000 | NA | NA | 0 | needs_compare |" in captured.out
    assert "# classes needs_compare=1" in captured.out
    assert "# scanned=1 scored=1 shown=1 baseline=lane" in captured.out


def test_mine_rejects_a_missing_baseline(tmp_path: Path):
    log = tmp_path / "lane.txt"
    log.write_text("pre-quantization post-ema val_loss:1 val_bpb:1.0\n")
    assert mine_main([str(log), "--baseline", str(tmp_path / "nope.txt")]) == 2
    assert mine_main([str(tmp_path / "none-*.txt")]) == 2
