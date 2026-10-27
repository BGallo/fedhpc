"""Recompute the exact weighted-sum reference points of instance files and
store them, with each file's sha256, in data/reference_points.json (read by
fedhpc.data.reference_points).

Usage: uv run python scripts/compute_reference_points.py [INSTANCE.json ...]
(default: every instance already in the table).
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import date
from pathlib import Path

from fedhpc.data import REFERENCE_POINTS, Instance
from fedhpc.formulations import configure_env
from fedhpc.pareto import weighted_sum_reference


def main() -> None:
    table = json.loads(REFERENCE_POINTS.read_text())
    paths = [Path(p) for p in sys.argv[1:]] or [
        REFERENCE_POINTS.parent / f"{stem}.json" for stem in table if not stem.startswith("_")
    ]
    configure_env()
    for path in paths:
        t0 = time.perf_counter()
        (f1_T, f2_T, f1_0, f2_0), _ = weighted_sum_reference(
            Instance.from_file(path), min_cost_reference=True, MIPGap=0)
        table[path.stem] = dict(
            f1_T=f1_T, f2_T=f2_T, f1_0=f1_0, f2_0=f2_0,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            computed=date.today().isoformat(),
        )
        print(f"{path.stem}: f1_T={f1_T:.4f} f2_T={f2_T:.6f} f1_0={f1_0:.4f} "
              f"f2_0={f2_0:.6f} ({time.perf_counter() - t0:.1f}s)")
    REFERENCE_POINTS.write_text(json.dumps(table, indent=2) + "\n")


if __name__ == "__main__":
    main()
