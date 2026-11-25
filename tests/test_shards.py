from __future__ import annotations

import struct
from pathlib import Path

import pytest

from rascal_lab.shards import (
    HEADER_BYTES,
    SHARD_MAGIC,
    SHARD_VERSION,
    check_shard,
    classify_shard_name,
    expected_file_size,
    inventory_dir,
    subtract_unwritten_train_tokens,
    write_uint16_shard,
)


def test_classify_keeps_byte_sidecars_out_of_val():
    assert classify_shard_name("fineweb_val_000000.bin") == "val"
    assert classify_shard_name("fineweb_val_bytes_000000.bin") == "val_bytes"
    assert classify_shard_name("fineweb_train_000012.bin") == "train"
    assert classify_shard_name("notes.txt") is None


def test_inventory_does_not_follow_the_val_glob(tmp_path: Path):
    (tmp_path / "fineweb_train_000000.bin").write_bytes(b"")
    (tmp_path / "fineweb_val_000000.bin").write_bytes(b"")
    (tmp_path / "fineweb_val_bytes_000000.bin").write_bytes(b"")
    (tmp_path / "fineweb_val_000001.bin").write_bytes(b"")
    inventory = inventory_dir(tmp_path)
    assert [path.name for path in inventory.train] == ["fineweb_train_000000.bin"]
    assert [path.name for path in inventory.val] == [
        "fineweb_val_000000.bin",
        "fineweb_val_000001.bin",
    ]
    assert [path.name for path in inventory.val_bytes] == ["fineweb_val_bytes_000000.bin"]
    globbed = sorted(path.name for path in tmp_path.glob("fineweb_val_*.bin"))
    assert "fineweb_val_bytes_000000.bin" in globbed
    assert len(globbed) == 3


def test_write_and_check_roundtrip(tmp_path: Path):
    path = tmp_path / "fineweb_train_000000.bin"
    tokens = [1, 256, 65535]
    write_uint16_shard(path, tokens)
    blob = path.read_bytes()
    assert len(blob) == expected_file_size(len(tokens))
    fields = struct.unpack_from("<256i", blob)
    assert (fields[0], fields[1], fields[2]) == (SHARD_MAGIC, SHARD_VERSION, 3)
    assert blob[HEADER_BYTES:] == struct.pack("<3H", *tokens)
    ok, message = check_shard(path)
    assert ok, message
    assert "kind=train" in message
    assert "ok=True" in message


def test_empty_values_do_not_create_a_shard(tmp_path: Path):
    path = tmp_path / "fineweb_train_000000.bin"
    write_uint16_shard(path, [])
    assert not path.exists()


def test_rejects_values_outside_uint16(tmp_path: Path):
    path = tmp_path / "fineweb_train_000000.bin"
    with pytest.raises(ValueError, match="uint16"):
        write_uint16_shard(path, [65536])


def test_short_header_negative_tokens_and_size_mismatch(tmp_path: Path):
    short = tmp_path / "fineweb_val_000000.bin"
    short.write_bytes(b"\x00" * 16)
    ok, message = check_shard(short)
    assert not ok
    assert "reason=short_header" in message

    negative = tmp_path / "fineweb_val_000001.bin"
    header = bytearray(HEADER_BYTES)
    struct.pack_into("<3i", header, 0, SHARD_MAGIC, SHARD_VERSION, -1)
    negative.write_bytes(header)
    ok, message = check_shard(negative)
    assert not ok
    assert "reason=negative_tokens" in message

    mismatch = tmp_path / "fineweb_train_000000.bin"
    header = bytearray(HEADER_BYTES)
    struct.pack_into("<3i", header, 0, SHARD_MAGIC, SHARD_VERSION, 4)
    mismatch.write_bytes(header)
    ok, message = check_shard(mismatch)
    assert not ok
    assert "reason=size_mismatch expected=1032" in message

    bad_magic = tmp_path / "fineweb_train_000001.bin"
    header = bytearray(HEADER_BYTES)
    struct.pack_into("<3i", header, 0, 1, SHARD_VERSION, 0)
    bad_magic.write_bytes(header)
    ok, message = check_shard(bad_magic)
    assert not ok
    assert "reason=bad_header" in message


def test_cap_stop_drops_the_unwritten_tail():
    stats = {"tokens_train": 4250}
    subtract_unwritten_train_tokens(stats, 2250)
    assert stats["tokens_train"] == 2000
    assert stats["unwritten_train_tokens"] == 2250


def test_cap_stop_rejects_a_tail_larger_than_the_count():
    with pytest.raises(ValueError, match="exceed"):
        subtract_unwritten_train_tokens({"tokens_train": 10}, 11)
