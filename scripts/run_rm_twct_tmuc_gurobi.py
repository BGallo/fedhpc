"""Solve the Rm||TWCT+TMUC benchmarks with fedhpc's Gurobi MIP (space-time
formulation extended with job weights and fixed machine-usage costs).

Per instance:
1. Upper bound UB = best of fedhpc's C++ weighted memetic solver (fixed costs
   invisible to it, see ``run_rm_twct_tmuc.py``), the greedy constructive, and
   every single-machine schedule — all WSPT-sequenced. The best one is also
   the MIP start.
2. Per-machine horizons ``H_k = rm_benchmark.machine_horizons(data, UB)`` —
   provably no solution of cost ≤ UB loads machine k beyond H_k, so the
   time-indexed MIP loses no optimal solution.
3. ``rm_benchmark.solve_total_cost`` minimises f1 + f2 = TWCT + TMUC.

Usage
-----
    uv run --with openpyxl python scripts/run_rm_twct_tmuc_gurobi.py \
        [--max-n 30] [--max-m 5] [--time-limit 300] [--max-x-vars 5000000]
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_rm_twct_tmuc import ROOT, greedy, load_instances, load_reference, run_fedhpc  # noqa: E402

from fedhpc import formulations, rm_benchmark as rm  # noqa: E402
from fedhpc.model import Solution  # noqa: E402

FIELDS = ["class", "id", "m", "n", "horizon", "n_x_vars", "ref_lb", "ref_ofv",
          "heur_ub", "mip_status", "mip_obj", "mip_bound", "mip_gap_pct",
          "gap_vs_ref_pct", "machines_used", "heur_s",
          "build_s", "solve_s"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", default="")
    ap.add_argument("--max-n", type=int, default=10**9)
    ap.add_argument("--max-m", type=int, default=15)
    ap.add_argument("--only", default="", help="regex on instance id")
    ap.add_argument("--time-limit", type=float, default=None,
                    help="Gurobi TimeLimit per solve (default: none, run to optimality)")
    ap.add_argument("--threads", type=int, default=0, help="0 = Gurobi default (all cores)")
    ap.add_argument("--nodefile-start", type=float, default=4.0,
                    help="GB of B&B node memory before Gurobi spills nodes to disk")
    ap.add_argument("--max-x-vars", type=int, default=0,
                    help="skip models with more x vars (default 0: no cap)")
    ap.add_argument("--out", default=str(ROOT / "fedhpc_gurobi_results.csv"))
    args = ap.parse_args()

    formulations.configure_env(verbose=False)
    insts = load_instances()
    ref = load_reference()
    classes = set(args.classes.split(",")) if args.classes else None
    only = re.compile(args.only) if args.only else None

    def dims(iid):
        return tuple(map(int, re.match(r"M(\d+)N(\d+)_(\d+)", iid).groups()))

    keys = sorted(
        (k for k in insts
         if (classes is None or k[0] in classes)
         and dims(k[1])[0] <= args.max_m and dims(k[1])[1] <= args.max_n
         and (only is None or only.search(k[1]))),
        key=lambda k: (dims(k[1])[0] * dims(k[1])[1], k[0], dims(k[1])),
    )

    out_path = Path(args.out)
    done: set[tuple[str, str]] = set()
    if out_path.exists():
        with open(out_path) as f:
            done = {(r["class"], r["id"]) for r in csv.DictReader(f)}
    new_file = not out_path.exists()
    with open(out_path, "a", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            wr.writeheader()
        for key in keys:
            if key in done:
                continue
            cls, iid = key
            data = rm.parse(insts[key])
            p, w, U, L = data
            n, m = len(w), len(U)

            # 1. upper bound / MIP start
            t0 = time.perf_counter()
            g_mach = greedy(p, w, U, L)
            g_load = max(sum(p[j][k] for j in range(n) if g_mach[j] == k) for k in range(m))
            h_mach, _f1, _f2 = run_fedhpc(
                p, w, U, L, horizon=max(math.ceil(1.5 * g_load), max(map(max, p))),
                pop_size=32, n_gen=60, seed=42, n_threads=args.threads)
            cands = [h_mach, g_mach] + [[k] * n for k in range(m)]
            mach = min(cands, key=lambda a: rm.evaluate(data, a))
            ub = rm.evaluate(data, mach)
            heur_s = time.perf_counter() - t0

            # 2. valid per-machine horizons
            Hk = rm.machine_horizons(data, ub)
            inst = rm.to_instance(data, Hk)
            n_x = sum(len(inst.T[j, k]) for j in range(n) for k in inst.F[j])
            lb_ref, ofv_ref = ref.get(key, (float("nan"), float("nan")))
            row = {"class": cls, "id": iid, "m": m, "n": n, "horizon": max(Hk),
                   "n_x_vars": n_x, "ref_lb": lb_ref, "ref_ofv": ofv_ref,
                   "heur_ub": ub, "heur_s": round(heur_s, 3)}

            # 3. MIP
            if args.max_x_vars and n_x > args.max_x_vars:
                row["mip_status"] = "skipped_too_large"
            else:
                hint = Solution("hint", None, None, None, rm.wspt_schedule(data, mach), {})
                limits = {} if args.time_limit is None else {"TimeLimit": args.time_limit}
                sol, bound = rm.solve_total_cost(
                    inst, hint=hint, Threads=args.threads, MIPGap=0,
                    NodefileStart=args.nodefile_start, **limits)
                row.update(mip_status=sol.status, build_s=round(sol.build_time, 2),
                           solve_s=round(sol.solve_time, 2))
                if sol.f1 is not None:
                    mach_mip = [sol.assignment[j][0] for j in range(n)]
                    # MIP start times may idle; the WSPT re-score is never worse
                    obj = min(sol.f1 + sol.f2, rm.evaluate(data, mach_mip))
                    row.update(
                        mip_obj=obj, mip_bound=bound,
                        mip_gap_pct=round(100 * (obj - bound) / obj, 4),
                        gap_vs_ref_pct=round(100 * (obj - ofv_ref) / ofv_ref, 4)
                        if ofv_ref == ofv_ref else "",
                        machines_used=len(set(mach_mip)))
            wr.writerow(row)
            fh.flush()
            print(f"{cls} {iid:>10}  H={row['horizon']:<5} x={row['n_x_vars']:<8} ref={ofv_ref:>10.0f} "
                  f"heur={ub:>10.0f}  mip={row.get('mip_obj', float('nan')):>10.0f} "
                  f"[{row['mip_status']}, gap {row.get('mip_gap_pct', '-')}%, "
                  f"vs ref {row.get('gap_vs_ref_pct', '-')}%]  "
                  f"used={row.get('machines_used', '-')}/{m}  "
                  f"{row.get('build_s', 0)}+{row.get('solve_s', 0)}s", flush=True)


if __name__ == "__main__":
    main()
