"""Byte-shuffle used before brotli in the compression sweep.

The on-disk framing is the four bytes ``BSHF``, one stride byte, then the
strided planes concatenated. Inputs shorter than the stride, and strides below
2, are returned unchanged so the historical sweep round-trips those blobs.
"""

from __future__ import annotations

BSHF_MAGIC = b"BSHF"


def byte_shuffle(data: bytes, stride: int = 2) -> bytes:
    if stride <= 1 or len(data) < stride:
        return data
    if stride > 255:
        raise ValueError(f"stride must fit in one byte, got {stride}")
    planes = [data[pos::stride] for pos in range(stride)]
    return BSHF_MAGIC + bytes((stride,)) + b"".join(planes)


def byte_unshuffle(data: bytes) -> bytes:
    if len(data) < 5 or data[:4] != BSHF_MAGIC:
        return data
    stride = data[4]
    payload = data[5:]
    if stride < 2:
        return payload
    size = len(payload)
    restored = bytearray(size)
    offset = 0
    remainder = size % stride
    base = size // stride
    for pos in range(stride):
        chunk_len = base + (1 if pos < remainder else 0)
        restored[pos::stride] = payload[offset : offset + chunk_len]
        offset += chunk_len
    if offset != size:
        raise ValueError(f"byte shuffle payload did not partition: used {offset} of {size}")
    return bytes(restored)
