"""BPC vs VRPSolver on the same models (``fedhpc.vrpsolver``).

Default suite: the Rm||TWCT+TMUC instances (Chen et al., EJOR 2024) of
``scripts/bpc_bench.py``, objective f1 + f2. Both engines get the same start
sets and the same heuristic cutoff, so the comparison is between the solvers.

    uv run python scripts/vrpsolver_bench.py --out results/vrpsolver_bench/rm.jsonl \\
        --instances rm_M2N10_1 rm_M5N25_1 --time-limit 600

Needs VRPSolver (``BAPCOD_RCSP_LIB`` built with CLP or CPLEX, Julia or ``FEDHPC_JULIA``; see
``docs/vrpsolver_integration.md``). ``--dry-run`` only builds the VRPSolver
models, which needs Julia alone.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bpc_bench import rm_instance  # noqa: E402

from fedhpc import bpc, vrpsolver  # noqa: E402

DEFAULT = ["rm_M2N10_1", "rm_M2N10_3", "rm_M2N20_3", "rm_M5N25_1", "rm_M8N40_1"]


def load(name: str):
    m = re.fullmatch(r"rm_M(\d+)N(\d+)_(\d+)", name)
    if not m:
        raise SystemExit(f"unknown instance {name} (expected rm_M<m>N<n>_<i>)")
    return rm_instance(*map(int, m.groups()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--instances", nargs="*", default=DEFAULT)
    ap.add_argument("--time-limit", type=float, default=600.0)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-rank1", action="store_true")
    ap.add_argument("--skip-bpc", action="store_true")
    args = ap.parse_args()

    if not args.dry_run:
        ok, why = vrpsolver.available()
        if not ok:
            raise SystemExit(f"VRPSolver not available: {why}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
    w1, w2 = 1.0, 1.0
    for name in args.instances:
        inst = load(name)
        prep = bpc._prepare(inst)
        seeds = bpc.heuristic_schedules(inst)
        rec: dict = {"instance": name, "n": len(inst.jobs), "m": len(inst.instance_types)}
        model = vrpsolver.export_model(inst, w1, w2, prep=prep, seed_schedules=seeds)
        rec["placements"] = sum(len(pt["arcs"]) for pt in model["path_types"])
        if args.dry_run:
            t = time.perf_counter()
            res = vrpsolver.run_julia(model, dry_run=True)
            rec["vrp_build_s"] = time.perf_counter() - t
            rec.update({f"vrp_{k}": v for k, v in res.items() if k != "log"})
        else:
            if not args.skip_bpc:
                st: dict = {}
                t = time.perf_counter()
                try:
                    obj, _a = bpc._branch_and_price(inst, w1, w2, prep=prep, seed_schedules=seeds,
                                                    time_limit=args.time_limit, stats=st)
                    rec["bpc"] = {"objective": obj, "proven": True}
                except bpc.BPCTimeLimit as e:
                    rec["bpc"] = {"objective": e.objective, "bound": e.bound, "proven": False}
                rec["bpc"].update(time_s=time.perf_counter() - t, nodes=st.get("nodes"),
                                  root_bound=st.get("root_bound"))
            st = {}
            sol = vrpsolver.solve_vrpsolver(inst, w1, w2, time_limit=args.time_limit,
                                            rank1_cuts=not args.no_rank1, stats=st,
                                            prep=prep, seed_schedules=seeds)
            stats = st.get("stats", {})
            rec["vrpsolver"] = {"objective": sol.objective, "status": sol.status,
                                "time_s": st.get("wall_s"), "solve_s": st.get("solve_s"),
                                "root_bound": stats.get("bcRecRootDb"),
                                "best_bound": stats.get("bcRecBestDb"),
                                "nodes": stats.get("bcCountNodeProc")}
        print(json.dumps(rec), flush=True)
        if args.out:
            with args.out.open("a") as f:
                f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
