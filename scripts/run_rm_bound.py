"""Certify ILS-SP solutions on Rm||TWCT+TMUC with the Lagrangian / CG lower bound.

For each instance: ILS-SP for ``--ils-time`` s (UB), then ``cg_bound.lp_bound``
for up to ``--bound-time`` s (LB). Reports the certified gap (UB − LB)/UB and
whether UB is proven optimal (LB == UB, costs are integral).

Usage
-----
    uv run --with openpyxl python scripts/run_rm_bound.py \
        [--classes V1_F1] [--sizes M10N300,M15N450] [--ils-time 5] \
        [--bound-time 120] [--out Rm_(TWCT+TMUC)/bound_results.csv]
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_rm_ilssp import load_optima  # noqa: E402
from run_rm_twct_tmuc import ROOT, load_instances, load_reference  # noqa: E402

from fedhpc import cg_bound, ilssp  # noqa: E402
from fedhpc import rm_benchmark as rm  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", default="")
    ap.add_argument("--sizes", default="", help="comma list of M{m}N{n} prefixes")
    ap.add_argument("--ils-time", type=float, default=5.0)
    ap.add_argument("--bound-time", type=float, default=120.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "bound_results.csv"))
    args = ap.parse_args()

    insts = load_instances()
    ref = load_reference()
    opt = load_optima()
    classes = set(filter(None, args.classes.split(",")))
    sizes = tuple(filter(None, args.sizes.split(",")))

    def key(k):
        m, n, i = map(int, re.match(r"M(\d+)N(\d+)_(\d+)", k[1]).groups())
        return (k[0], m, n, i)

    keys = [k for k in sorted(insts, key=key)
            if (not classes or k[0] in classes) and (not sizes or k[1].startswith(sizes))]

    cols = ["class", "id", "m", "n", "ub", "lb", "gap_pct", "proven", "subgradient_lb",
            "lagrangian", "lp_converged", "cg_iters", "known_opt", "bap_ofv", "ils_s",
            "bound_s"]
    proven = 0
    gaps = []
    with open(args.out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(cols)
        for cls, iid in keys:
            data = rm.parse(insts[cls, iid])
            m, n = len(data[2]), len(data[1])
            res = ilssp.solve(data, args.ils_time, seed=args.seed,
                              sp_interval=max(args.ils_time / 10, 0.1))
            b = cg_bound.lp_bound(data, res.cost, res.columns, incumbent=res.assignment,
                                  time_limit=args.bound_time)
            gap = 100 * (res.cost - b.lb) / res.cost
            gaps.append(gap)
            proven += b.proven_optimal
            o = opt.get((cls, iid), math.nan)
            if not math.isnan(o):
                assert b.lb <= o + 1e-6, f"invalid bound on {cls} {iid}: {b.lb} > {o}"
            bap = ref.get((cls, iid), (math.nan, math.nan))[1]
            wr.writerow([cls, iid, m, n, res.cost, b.lb, round(gap, 6), int(b.proven_optimal),
                         round(b.subgradient_lb, 3), round(b.lagrangian, 3),
                         int(b.lp_optimal), b.iterations, o, bap, round(res.time_s, 2),
                         round(b.time_s, 2)])
            f.flush()
            print(f"{cls} {iid:12s} ub={res.cost:.0f} lb={b.lb} gap={gap:.5f}% "
                  f"{'PROVEN' if b.proven_optimal else '      '} bound {b.time_s:6.1f}s "
                  f"(cg it {b.iterations})", flush=True)
    if gaps:
        print(f"\n{len(gaps)} instances: proven optimal {proven}, "
              f"mean gap {sum(gaps)/len(gaps):.5f}%, max {max(gaps):.5f}%")


if __name__ == "__main__":
    main()
