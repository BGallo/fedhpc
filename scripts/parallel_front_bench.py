"""Exact Pareto front in parallel (lpfirst on HiGHS): the box search
(pareto.parallel_pareto_frontier, ``--method box``) or the single-phase
ε-constraint search (pareto.parallel_epsilon_frontier, ``--method eps``, with ``--direction``).

    uv run python scripts/parallel_front_bench.py data/fedhpc_known_runtime_offline_10min.json \
        --workers 14 --budget 7200 --per-solve 300 --method eps

Writes results/parallel_front/<instance>_<method>_w<workers>.json: the points (f1, f2,
status), the search counters and the wall time. Each point is also appended to
``<instance>_<method>_w<workers>.points.jsonl`` (with its discovery time) as soon as it
is proven, so an interrupted run keeps its points. A partial front (budget hit or
a dropped box) is flagged by ``complete: false``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from fedhpc import lpfirst, pareto
from fedhpc.data import Instance

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("instance")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--per-solve", type=float, default=None)
    ap.add_argument("--method", choices=("box", "eps"), default="box")
    ap.add_argument("--direction", choices=("adaptive", "f1", "f2"), default="adaptive",
                    help="eps only: which single-phase solves to use")
    args = ap.parse_args()
    inst = Instance.from_file(args.instance)
    opts = lpfirst.LPFirstOptions(backend="highs", lns_sweeps=0, lp_threads=1, mip_threads=1)
    st: dict = {}
    t0 = time.perf_counter()
    kw = {"direction": args.direction} if args.method == "eps" else {}
    tag = args.method + (f"-{args.direction}" if args.method == "eps" else "")
    stem = f"{Path(args.instance).stem}_{tag}_w{args.workers}"
    stream = ROOT / "results" / "parallel_front" / f"{stem}.points.jsonl"
    stream.parent.mkdir(parents=True, exist_ok=True)
    with stream.open("w") as fh:
        def on_point(s) -> None:
            fh.write(json.dumps(dict(t=round(time.perf_counter() - t0, 2), f1=s.f1, f2=s.f2,
                                     status=s.status)) + "\n")
            fh.flush()
        search = (pareto.parallel_epsilon_frontier if args.method == "eps"
                  else pareto.parallel_pareto_frontier)
        front = search(inst, args.workers, lpfirst_options=opts, time_budget=args.budget,
                       per_solve_time_limit=args.per_solve, stats=st, on_point=on_point, **kw)
    wall = time.perf_counter() - t0
    out = ROOT / "results" / "parallel_front" / f"{stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        instance=args.instance, method=args.method, direction=kw.get("direction"), workers=args.workers, budget=args.budget, wall=wall,
        complete=st["open_boxes"] == 0 and st["n_unresolved"] == 0, stats=st,
        points=[dict(f1=s.f1, f2=s.f2, status=s.status) for s in front]), indent=1))
    print(f"{len(front)} points in {wall:.0f} s; {st}")


if __name__ == "__main__":
    main()
