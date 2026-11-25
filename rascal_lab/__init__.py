"""Shared checks for the Rascal parameter-golf lab.

Training entry points stay in their experiment directories. This package holds
the small, testable pieces those scripts and the ``scripts/`` CLIs share:
shard headers, submission-size checks, log mining, and the experiment ledger.
"""

from rascal_lab.limits import SUBMISSION_CAP_BYTES, is_legal_artifact

__all__ = ["SUBMISSION_CAP_BYTES", "is_legal_artifact"]
