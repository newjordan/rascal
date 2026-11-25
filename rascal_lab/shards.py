"""FineWeb uint16 shard headers and dataset inventory.

A shard is a 1024-byte little-endian header (256 int32s) followed by uint16
tokens. ``header[0]`` is magic ``20240520``, ``header[1]`` is version ``1``,
and ``header[2]`` is the token count. Validation byte sidecars use the same
layout and are named ``fineweb_val_bytes_*.bin``. Their names also match the
glob ``fineweb_val_*.bin``, so inventory must classify names rather than globs.
"""

from __future__ import annotations

import struct
import sys
from array import array
from dataclasses import dataclass
from pathlib import Path

SHARD_MAGIC = 20240520
SHARD_VERSION = 1
HEADER_INTS = 256
HEADER_BYTES = HEADER_INTS * 4
TOKEN_BYTES = 2
UINT16_MAX = 0xFFFF


def classify_shard_name(name: str) -> str | None:
    """Return ``train``, ``val``, ``val_bytes``, or ``None``.

    Byte sidecars are checked before validation token shards. The sidecar
    prefix is a prefix of the validation-shard prefix, so the opposite order
    would count sidecars as validation shards.
    """
    if not name.endswith(".bin"):
        return None
    if name.startswith("fineweb_val_bytes_"):
        return "val_bytes"
    if name.startswith("fineweb_val_"):
        return "val"
    if name.startswith("fineweb_train_"):
        return "train"
    return None


@dataclass(frozen=True)
class ShardInventory:
    train: tuple[Path, ...]
    val: tuple[Path, ...]
    val_bytes: tuple[Path, ...]

    def counts(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "val": len(self.val),
            "val_bytes": len(self.val_bytes),
        }

    def lines(self) -> list[str]:
        counts = self.counts()
        return [
            f"train={counts['train']}",
            f"val={counts['val']}",
            f"val_bytes={counts['val_bytes']}",
        ]


def inventory_dir(root: Path) -> ShardInventory:
    if not root.is_dir():
        raise FileNotFoundError(root)
    groups: dict[str, list[Path]] = {"train": [], "val": [], "val_bytes": []}
    for path in root.iterdir():
        if not path.is_file():
            continue
        kind = classify_shard_name(path.name)
        if kind is not None:
            groups[kind].append(path)
    for paths in groups.values():
        paths.sort()
    return ShardInventory(
        train=tuple(groups["train"]),
        val=tuple(groups["val"]),
        val_bytes=tuple(groups["val_bytes"]),
    )


def expected_file_size(token_count: int) -> int:
    if token_count < 0:
        raise ValueError(f"token count must be >= 0, got {token_count}")
    return HEADER_BYTES + token_count * TOKEN_BYTES


def header_bytes(token_count: int) -> bytes:
    if token_count < 0 or token_count > 0x7FFFFFFF:
        raise ValueError(f"token count does not fit in int32: {token_count}")
    fields = [0] * HEADER_INTS
    fields[0] = SHARD_MAGIC
    fields[1] = SHARD_VERSION
    fields[2] = token_count
    return struct.pack(f"<{HEADER_INTS}i", *fields)


def write_uint16_shard(path: Path, values) -> None:
    """Write one shard. An empty value sequence writes nothing."""
    tokens = [int(value) for value in values]
    if not tokens:
        return
    try:
        payload = array("H", tokens)
    except OverflowError as exc:
        raise ValueError(f"value outside uint16 range: {exc}") from exc
    if sys.byteorder != "little":
        payload.byteswap()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header_bytes(len(payload)) + payload.tobytes())


def check_shard(path: Path) -> tuple[bool, str]:
    kind = classify_shard_name(path.name) or "unknown"
    size = path.stat().st_size
    with path.open("rb") as handle:
        header = handle.read(HEADER_BYTES)
    if len(header) != HEADER_BYTES:
        return False, _line(path.name, kind, size, None, False, "short_header")
    fields = struct.unpack_from(f"<{HEADER_INTS}i", header)
    magic, version, tokens = fields[0], fields[1], fields[2]
    if tokens < 0:
        return False, _line(path.name, kind, size, tokens, False, "negative_tokens")
    if magic != SHARD_MAGIC or version != SHARD_VERSION:
        reason = f"bad_header magic={magic} version={version}"
        return False, _line(path.name, kind, size, tokens, False, reason)
    expected = expected_file_size(tokens)
    if size != expected:
        reason = f"size_mismatch expected={expected}"
        return False, _line(path.name, kind, size, tokens, False, reason)
    return True, _line(path.name, kind, size, tokens, True, "")


def _line(name: str, kind: str, size: int, tokens: int | None, ok: bool, reason: str) -> str:
    token_text = "NA" if tokens is None else str(tokens)
    text = f"{name}: kind={kind} size={size} tokens={token_text} ok={ok}"
    if not ok:
        text += f" reason={reason}"
    return text


def subtract_unwritten_train_tokens(stats: dict, unwritten: int) -> None:
    """Drop buffered tokens that a shard cap did not flush.

    ``tokens_train`` counts every training token seen. Stopping at
    ``max_train_shards`` leaves a partial buffer on disk unwritten. The
    completed manifest has to report tokens that were actually stored, plus
    ``unwritten_train_tokens`` for the dropped tail.
    """
    if unwritten < 0:
        raise ValueError(f"unwritten token count must be >= 0, got {unwritten}")
    current = int(stats["tokens_train"])
    if unwritten > current:
        raise ValueError(
            f"unwritten tokens {unwritten} exceed counted train tokens {current}"
        )
    stats["tokens_train"] = current - unwritten
    stats["unwritten_train_tokens"] = unwritten
