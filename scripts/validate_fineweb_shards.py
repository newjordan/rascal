#!/usr/bin/env python3
from __future__ import annotations

import struct
import sys
from pathlib import Path


def check_file(path: Path) -> tuple[bool, str]:
    size = path.stat().st_size
    with path.open("rb") as f:
        header = f.read(1024)
    if len(header) != 1024:
        return False, f"{path.name}: short header size={size}"
    fields = struct.unpack("<256i", header)
    magic, version, tokens = fields[0], fields[1], fields[2]
    expected = 1024 + tokens * 2
    if magic != 20240520 or version != 1:
        return False, f"{path.name}: bad header magic={magic} version={version}"
    if size != expected:
        return False, f"{path.name}: size={size} expected={expected} tokens={tokens}"
    return True, f"{path.name}: size={size} tokens={tokens} ok=True"


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: validate_fineweb_shards.py DATASET_DIR", file=sys.stderr)
        return 2
    root = Path(sys.argv[1])
    if not root.is_dir():
        print(f"missing dataset dir: {root}", file=sys.stderr)
        return 2
    files = sorted(root.glob("fineweb_train_*.bin")) + sorted(root.glob("fineweb_val_*.bin"))
    if not files:
        print(f"no fineweb shards found in {root}", file=sys.stderr)
        return 2
    bad = []
    for path in files:
        ok, msg = check_file(path)
        print(msg)
        if not ok:
            bad.append(path.name)
    train_count = len(list(root.glob("fineweb_train_*.bin")))
    val_count = len(list(root.glob("fineweb_val_*.bin")))
    print(f"summary train={train_count} val={val_count} bad={bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
