"""Reproducible single-thread LP-first pilot on the Kramer archive.

Extract the archive and pass its parent directory as the first argument.
Writes one JSON record per run, including failures, so interrupted runs survive.
The MIP backend is HiGHS in both configurations; only the master LP differs.
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fedhpc-matplotlib")
from bench_kramer_fair import horizon, inst_path, read_instance
from bench_prjwc import fed_instance
from fedhpc.lpfirst import LPFirstOptions, solve_lpfirst

root = Path(sys.argv[1])
out = Path(sys.argv[2])
out.parent.mkdir(parents=True, exist_ok=True)
done = set()
if out.exists():
    for line in out.read_text().splitlines():
        row = json.loads(line)
        done.add((row["instance"], row["lp_backend"]))

for m in (2, 5):
    for release in (0, 2, 4, 6, 8, 9):
        path = inst_path(root, 20, m, release, 1)
        p, w, r, machines = read_instance(path)
        for lp in ("highs", "fedlp"):
            if (path.stem, lp) in done:
                continue
            row = {"instance": path.stem, "lp_backend": lp, "mip_backend": "highs",
                   "time_limit_s": 30, "threads": 1}
            start = time.monotonic()
            try:
                inst = fed_instance(p, w, r, machines, horizon(p, r, machines))
                stats = {}
                opt = LPFirstOptions(backend="highs", lp_backend=lp, mip_threads=1,
                                     lp_threads=1, lns_sweeps=0)
                result = solve_lpfirst(inst, 1.0, 0.0, time_limit=30, options=opt, stats=stats)
                offset = sum(a * b for a, b in zip(w, r))
                row.update(status=result.status, objective=result.objective + offset,
                           bound=result.bound + offset, how=result.how, stats=stats)
            except Exception:
                row.update(status="error", error=traceback.format_exc())
            row["wall_s"] = time.monotonic() - start
            with out.open("a") as f:
                f.write(json.dumps(row, default=str) + "\n")
            print(json.dumps(row, default=str), flush=True)
