"""Read the agent-hub experiment ledger and check it against the size cap."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from rascal_lab.limits import SUBMISSION_CAP_BYTES

REQUIRED_COLUMNS = (
    "test_id",
    "type",
    "parent",
    "key_delta",
    "status",
    "post_ema_bpb",
    "quant_bpb",
    "ttt_bpb",
    "total_bytes",
    "path",
    "next_action",
)

# Statuses that claim a finished artifact is inside the submission cap.
UNDER_CAP_STATUSES = frozenset({"complete", "neural_miss"})


def _optional_float(text: str) -> float | None:
    if text in {"", "NA"}:
        return None
    return float(text)


def _optional_int(text: str) -> int | None:
    if text in {"", "NA"}:
        return None
    return int(text)


@dataclass(frozen=True)
class LedgerRow:
    test_id: str
    type: str
    parent: str
    key_delta: str
    status: str
    post_ema_bpb: float | None
    quant_bpb: float | None
    ttt_bpb: float | None
    total_bytes: int | None
    path: str
    next_action: str

    @property
    def byte_margin(self) -> int | None:
        if self.total_bytes is None:
            return None
        return SUBMISSION_CAP_BYTES - self.total_bytes


def load_ledger(path: Path) -> list[LedgerRow]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} missing columns: {', '.join(missing)}")
        rows = []
        for raw in reader:
            if not raw.get("test_id"):
                continue
            rows.append(
                LedgerRow(
                    test_id=raw["test_id"],
                    type=raw["type"],
                    parent=raw["parent"],
                    key_delta=raw["key_delta"],
                    status=raw["status"],
                    post_ema_bpb=_optional_float(raw["post_ema_bpb"]),
                    quant_bpb=_optional_float(raw["quant_bpb"]),
                    ttt_bpb=_optional_float(raw["ttt_bpb"]),
                    total_bytes=_optional_int(raw["total_bytes"]),
                    path=raw["path"],
                    next_action=raw["next_action"],
                )
            )
    return rows


def cap_conflicts(rows: list[LedgerRow], cap: int = SUBMISSION_CAP_BYTES) -> list[str]:
    """Return human-readable rows whose status disagrees with ``total_bytes``."""
    problems: list[str] = []
    for row in rows:
        if row.total_bytes is None:
            continue
        over = row.total_bytes > cap
        if row.status == "over_cap" and not over:
            problems.append(
                f"{row.test_id} status=over_cap but total_bytes={row.total_bytes} is within {cap}"
            )
        if row.status in UNDER_CAP_STATUSES and over:
            problems.append(
                f"{row.test_id} status={row.status} but total_bytes={row.total_bytes} exceeds {cap}"
            )
    return problems
