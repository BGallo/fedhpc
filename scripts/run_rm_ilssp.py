"""Run the ILS-SP matheuristic on the Rm||TWCT+TMUC benchmark set.

Compares against the fedhpc Gurobi optima (``fedhpc_gurobi_results.csv``) and
the paper's BaP values/CPU times (``Experimental_results_of_BaPwithHeuristic*.xlsx``).

Usage
-----
    uv run --with openpyxl python scripts/run_rm_ilssp.py \
        [--classes V1_F2] [--sizes M10N300,M15N450] [--time-limit 30] \
        [--no-sp] [--limit 20] [--out Rm_(TWCT+TMUC)/ilssp_results.csv]
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_rm_twct_tmuc import ROOT, load_instances, load_reference  # noqa: E402

from fedhpc import ilssp  # noqa: E402
from fedhpc import rm_benchmark as rm  # noqa: E402


def load_optima() -> dict[tuple[str, str], float]:
    out = {}
    path = ROOT / "fedhpc_gurobi_results.csv"
    if path.exists():
        with open(path) as f:
            for r in csv.DictReader(f):
                if r["mip_status"] == "optimal":
                    out[r["class"], r["id"]] = float(r["mip_obj"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", default="")
    ap.add_argument("--sizes", default="", help="comma list of M{m}N{n} prefixes")
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--sp-interval", type=float, default=None)
    ap.add_argument("--no-sp", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="max instances")
    ap.add_argument("--stop-at-opt", action="store_true",
                    help="stop as soon as the known optimum is reached")
    ap.add_argument("--out", default=str(ROOT / "ilssp_results.csv"))
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
    if args.limit:
        keys = keys[:args.limit]

    cols = ["class", "id", "m", "n", "opt", "bap_ofv", "ilssp", "gap_opt_pct",
            "gap_bap_pct", "time_s", "time_to_best_s", "iters", "pool", "sp_calls",
            "sp_impr", "machines_used"]
    gaps = []
    with open(args.out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(cols)
        for cls, iid in keys:
            data = rm.parse(insts[cls, iid])
            m, n = len(data[2]), len(data[1])
            o = opt.get((cls, iid), math.nan)
            bap = ref.get((cls, iid), (math.nan, math.nan))[1]
            res = ilssp.solve(data, args.time_limit, seed=args.seed, use_sp=not args.no_sp,
                              sp_interval=args.sp_interval,
                              target=o if args.stop_at_opt and not math.isnan(o) else None)
            g_opt = 100 * (res.cost - o) / o if not math.isnan(o) else math.nan
            g_bap = 100 * (res.cost - bap) / bap if not math.isnan(bap) else math.nan
            if not math.isnan(g_opt):
                gaps.append(g_opt)
            wr.writerow([cls, iid, m, n, o, bap, res.cost, round(g_opt, 4), round(g_bap, 4),
                         round(res.time_s, 2), round(res.time_to_best_s, 2), res.iterations,
                         res.pool_size, res.sp_calls, res.sp_improvements,
                         len(set(res.assignment))])
            f.flush()
            print(f"{cls} {iid:12s} ilssp={res.cost:.0f} opt={o:.0f} gap={g_opt:7.4f}% "
                  f"tbest={res.time_to_best_s:6.2f}s it={res.iterations} pool={res.pool_size} "
                  f"sp={res.sp_improvements}/{res.sp_calls}", flush=True)
    if gaps:
        hit = sum(g < 1e-9 for g in gaps)
        print(f"\n{len(gaps)} with known optimum: mean gap {sum(gaps)/len(gaps):.4f}%, "
              f"max {max(gaps):.4f}%, optimal {hit}/{len(gaps)}")


if __name__ == "__main__":
    main()
