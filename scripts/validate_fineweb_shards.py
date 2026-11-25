#!/usr/bin/env python3
"""Check FineWeb shard headers and count train, val, and val-byte files apart."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rascal_lab.cli import validate_main


if __name__ == "__main__":
    raise SystemExit(validate_main())
