"""Submission size cap used across the lab.

Trainers record ``SIZE_CAP_BYTES=16000000``. That is 16 million bytes, not
16 MiB (16 * 1024 * 1024 = 16777216). Artifacts in the gap between those two
numbers are over the lab cap.
"""

from __future__ import annotations

SUBMISSION_CAP_BYTES = 16_000_000


def is_legal_artifact(num_bytes: int) -> bool:
    """Return whether a compressed submission size is inside the lab cap.

    Zero and negative sizes are not legal artifacts. The cap itself is legal,
    matching the log miner's rule that only a negative byte margin is overflow.
    """
    return 0 < num_bytes <= SUBMISSION_CAP_BYTES
