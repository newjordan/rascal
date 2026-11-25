from __future__ import annotations

import pytest

from rascal_lab.compress import byte_shuffle, byte_unshuffle
from rascal_lab.limits import SUBMISSION_CAP_BYTES, is_legal_artifact


@pytest.mark.parametrize("stride", [2, 3, 4, 7])
@pytest.mark.parametrize("length", [0, 1, 2, 3, 5, 16, 17, 64, 255, 256])
def test_byte_shuffle_roundtrip(stride: int, length: int):
    payload = bytes((index * 17 + stride) % 256 for index in range(length))
    assert byte_unshuffle(byte_shuffle(payload, stride=stride)) == payload


def test_short_input_is_left_unframed():
    payload = b"\x00\x01"
    assert byte_shuffle(payload, stride=8) == payload
    assert byte_unshuffle(payload) == payload


def test_unshuffle_ignores_plain_bytes():
    payload = b"not a shuffled blob"
    assert byte_unshuffle(payload) == payload


def test_submission_cap_is_sixteen_million_bytes_not_mib():
    assert SUBMISSION_CAP_BYTES == 16_000_000
    assert is_legal_artifact(16_000_000)
    assert not is_legal_artifact(16_000_001)
    assert not is_legal_artifact(16 * 1024 * 1024)
    assert not is_legal_artifact(0)
    assert not is_legal_artifact(-1)
