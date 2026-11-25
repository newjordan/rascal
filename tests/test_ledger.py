from __future__ import annotations

from pathlib import Path

from rascal_lab.ledger import cap_conflicts, load_ledger

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "agent_hub" / "experiments_2026-04-30.tsv"


def test_checked_in_ledger_matches_the_size_cap_and_local_legs():
    rows = load_ledger(LEDGER)
    assert rows
    assert cap_conflicts(rows) == []
    local = [row for row in rows if row.path.startswith("legs/")]
    assert local
    missing = [row.path for row in local if not (ROOT / row.path).is_dir()]
    assert missing == []
    remote = [row for row in rows if row.path.startswith("remote:")]
    assert remote


def test_cap_conflict_detector_flags_a_finished_row_over_the_cap():
    rows = load_ledger(LEDGER)
    sample = rows[0]
    broken = type(sample)(
        test_id="synthetic",
        type=sample.type,
        parent=sample.parent,
        key_delta=sample.key_delta,
        status="complete",
        post_ema_bpb=sample.post_ema_bpb,
        quant_bpb=sample.quant_bpb,
        ttt_bpb=sample.ttt_bpb,
        total_bytes=16_000_001,
        path=sample.path,
        next_action=sample.next_action,
    )
    problems = cap_conflicts([broken])
    assert len(problems) == 1
    assert "synthetic" in problems[0]
