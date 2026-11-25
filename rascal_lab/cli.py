"""Command-line entry points shared by ``scripts/``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rascal_lab.logs import class_counts, expand_inputs, format_table, parse_log, sort_rows
from rascal_lab.shards import ShardInventory, check_shard, inventory_dir


def _expect_errors(inventory: ShardInventory, args: argparse.Namespace) -> list[str]:
    expected = {
        "train": args.expect_train,
        "val": args.expect_val,
        "val_bytes": args.expect_val_bytes,
    }
    actual = inventory.counts()
    errors = []
    for name, want in expected.items():
        if want is None:
            continue
        got = actual[name]
        if got != want:
            errors.append(f"{name}={got} expected={want}")
    return errors


def validate_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="validate_fineweb_shards.py",
        description=(
            "Check FineWeb shard headers. Validation token shards and "
            "fineweb_val_bytes sidecars are counted separately."
        ),
    )
    parser.add_argument("dataset_dir", help="Directory of fineweb_*.bin shards")
    parser.add_argument(
        "--inventory",
        action="store_true",
        help="Print train/val/val_bytes counts and do not read headers",
    )
    parser.add_argument("--expect-train", type=int, default=None)
    parser.add_argument("--expect-val", type=int, default=None)
    parser.add_argument("--expect-val-bytes", type=int, default=None)
    args = parser.parse_args(argv)

    root = Path(args.dataset_dir)
    if not root.is_dir():
        print(f"missing dataset dir: {root}", file=sys.stderr)
        return 2
    try:
        inventory = inventory_dir(root)
    except FileNotFoundError:
        print(f"missing dataset dir: {root}", file=sys.stderr)
        return 2

    files = (*inventory.train, *inventory.val, *inventory.val_bytes)
    if args.inventory:
        print("\n".join(inventory.lines()))
        errors = _expect_errors(inventory, args)
        if errors:
            print("inventory mismatch: " + "; ".join(errors), file=sys.stderr)
            return 1
        return 0

    if not files:
        print(f"no fineweb shards found in {root}", file=sys.stderr)
        return 2

    bad: list[str] = []
    for path in files:
        ok, message = check_shard(path)
        print(message)
        if not ok:
            bad.append(path.name)
    counts = inventory.counts()
    print(
        "summary train={train} val={val} val_bytes={val_bytes} bad={bad}".format(
            train=counts["train"],
            val=counts["val"],
            val_bytes=counts["val_bytes"],
            bad=bad,
        )
    )
    errors = _expect_errors(inventory, args)
    for error in errors:
        print(f"count mismatch: {error}", file=sys.stderr)
    if bad or errors:
        return 1
    return 0


def mine_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mine_axis_deltas.py",
        description="Compare trainer logs and classify size, quant, quality, and speed failures.",
    )
    parser.add_argument("logs", nargs="+", help="Log files or globs")
    parser.add_argument("--baseline", help="Baseline log path")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args(argv)
    if args.top < 0:
        print("--top must be >= 0", file=sys.stderr)
        return 2

    paths, missing = expand_inputs(args.logs)
    for item in missing:
        print(f"no files matched: {item}", file=sys.stderr)
    if not paths:
        return 2

    rows = [parse_log(path) for path in paths]
    rows = [row for row in rows if row.get("best_bpb") is not None or row.get("total_bytes") is not None]
    rows = sort_rows(rows)

    baseline = None
    if args.baseline:
        baseline_path = Path(args.baseline)
        if not baseline_path.is_file():
            print(f"missing baseline log: {baseline_path}", file=sys.stderr)
            return 2
        baseline = parse_log(baseline_path)
    elif rows:
        baseline = rows[0]

    shown = rows[: args.top]
    baseline_name = baseline["run_id"] if baseline else "NA"
    print(f"# scanned={len(paths)} scored={len(rows)} shown={len(shown)} baseline={baseline_name}")
    if not rows:
        print("no scored runs", file=sys.stderr)
        return 1
    print("\n".join(format_table(shown, baseline)))
    counts = " ".join(f"{name}={count}" for name, count in class_counts(shown).items())
    print(f"# classes {counts}")
    return 0
