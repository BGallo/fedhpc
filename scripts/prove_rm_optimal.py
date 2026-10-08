"""Prove (or refute) optimality of ILS-SP solutions on Rm||TWCT+TMUC with Gurobi.

Reduced-cost column enumeration + exact set-partitioning MIP:

1. ILS-SP gives UB and its assignment.
2. A long Polyak subgradient run gives duals β and the Lagrangian bound
   L(β) = Σβ + Σ_k v_k,  v_k = min(0, min_S c(k,S) − β(S)).
3. Any solution of cost ≤ UB − 1 costs L + Σ_{k used}(rc_k(S_k) − v_k)
   + Σ_{k unused}(−v_k), all terms ≥ 0, so each of its columns has
   rc_k(S) ≤ v_k + (UB − 1 − L). The C++ enumerator lists *every* such column
   (load ≤ machine_horizons(UB), itself valid for cost ≤ UB).
4. Gurobi solves the set-partitioning MIP over those columns with
   Cutoff = UB − 1/2 on all cores. Infeasible ⇒ UB is optimal; otherwise
   the MIP returns a strictly better solution.

Usage
-----
    uv run --with openpyxl python scripts/prove_rm_optimal.py V1_F1/M15N450_1 ...
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_rm_twct_tmuc import load_instances  # noqa: E402

from fedhpc import _rmprice  # type: ignore[attr-defined]  # noqa: E402
from fedhpc import cg_bound, ilssp  # noqa: E402
from fedhpc import rm_benchmark as rm  # noqa: E402


def lagrangian(pricer, beta, H):
    mins, _ = pricer.price(beta, [0.0] * len(H), H, 0)
    v = [min(0.0, x) for x in mins]
    return sum(beta) + sum(v), v


def best_duals(data, pricer, H, ub, assignment, budget, log):
    """Several subgradient schedules from the cost-share duals; keep the best."""
    beta0 = cg_bound._cost_shares(data, assignment)
    best_beta, best_l = beta0, -math.inf
    deadline = time.perf_counter() + budget
    for theta, patience in [(0.1, 100), (0.1, 400), (0.05, 1000), (0.02, 3000)]:
        if time.perf_counter() >= deadline:
            break
        start = best_beta if best_l > -math.inf else beta0
        beta, lb, _ = cg_bound._subgradient(pricer, start, H, ub, 10**7, deadline=deadline,
                                            theta=theta, patience=patience, theta_min=1e-10)
        log(f"  subgradient θ={theta} patience={patience}: L = {lb:.3f}")
        if lb > best_l:
            best_beta, best_l = beta, lb
        if math.ceil(best_l - 1e-6) >= ub:
            break
    return best_beta


def prove(cls, iid, text, args, log):
    data = rm.parse(text)
    p, w, U, L = data
    n, m = len(w), len(U)
    t0 = time.perf_counter()
    res = ilssp.solve(data, args.ils_time, seed=args.seed, sp_interval=args.ils_time / 10)
    ub = int(round(res.cost))
    assert rm.evaluate(data, res.assignment) == ub
    log(f"{cls} {iid}: n={n} m={m}  UB = {ub}  (ILS-SP {res.time_s:.1f}s)")

    H = rm.machine_horizons(data, ub)
    pricer = _rmprice.Pricer(p, w, U, L)
    beta = best_duals(data, pricer, H, ub, res.assignment, args.sg_time, log)
    lag, v = lagrangian(pricer, beta, H)
    gap = ub - 1 - lag
    log(f"  L(β) = {lag:.3f}   UB − 1 − L = {gap:.3f}")
    if gap < -1e-6:
        log("  PROVEN OPTIMAL by the Lagrangian bound alone")
        return "proven", ub, ub

    t1 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=m) as ex:
        outs = list(ex.map(
            lambda k: pricer.enumerate(k, beta, H[k], v[k] + gap, args.max_cols), range(m)))
    cols = []
    for k, (ck, complete) in enumerate(outs):
        if not complete:
            log(f"  machine {k}: enumeration hit max_cols={args.max_cols}; aborting")
            return "enumeration-overflow", ub, math.ceil(lag - 1e-6)
        cols.extend((k, js, c) for js, c, _ in ck)
    log(f"  enumerated {len(cols)} columns in {time.perf_counter() - t1:.1f}s "
        f"(per machine: {[len(o[0]) for o in outs]})")

    mdl = gp.Model("sp_proof")
    mdl.Params.Threads = 0                 # all cores
    mdl.Params.Cutoff = ub - 0.5           # only solutions with cost ≤ UB − 1
    mdl.Params.MIPGap = 0.0
    mdl.Params.MIPGapAbs = 0.999           # integral objective
    mdl.Params.MIPFocus = 3                # proof: push the bound
    mdl.Params.Presolve = 1                # aggressive presolve stalls on huge column sets
    mdl.Params.LogFile = str(args.log_dir / f"gurobi_{cls}_{iid}.log")
    x = mdl.addVars(len(cols), vtype=GRB.BINARY, obj=[float(c) for _, _, c in cols])
    by_job = [[] for _ in range(n)]
    by_mach = [[] for _ in range(m)]
    for s, (k, js, _) in enumerate(cols):
        by_mach[k].append(s)
        for j in js:
            by_job[j].append(s)
    for j in range(n):
        if not by_job[j]:
            log(f"  job {j} is in no enumerated column ⇒ no solution below UB")
            return "proven", ub, ub
        mdl.addConstr(gp.quicksum(x[s] for s in by_job[j]) == 1)
    for k in range(m):
        if by_mach[k]:
            mdl.addConstr(gp.quicksum(x[s] for s in by_mach[k]) <= 1)
    mdl.optimize()
    log(f"  Gurobi status {mdl.Status}  runtime {mdl.Runtime:.1f}s  nodes {mdl.NodeCount:.0f}")
    if mdl.Status in (GRB.INFEASIBLE, GRB.CUTOFF) or (
            mdl.Status == GRB.OPTIMAL and mdl.ObjVal > ub - 0.5):
        log(f"  PROVEN OPTIMAL: no solution with cost ≤ {ub - 1}  "
            f"(total {time.perf_counter() - t0:.1f}s)")
        return "proven", ub, ub
    if mdl.SolCount > 0:
        assign = [-1] * n
        for s, (k, js, _) in enumerate(cols):
            if x[s].X > 0.5:
                for j in js:
                    assign[j] = k
        better = rm.evaluate(data, assign)
        log(f"  BETTER SOLUTION FOUND: {better:.0f} < UB {ub}")
        return "improved", better, math.ceil(mdl.ObjBound - 1e-6)
    return f"status-{mdl.Status}", ub, math.ceil(lag - 1e-6)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("instances", nargs="+", help="CLASS/ID, e.g. V1_F1/M15N450_1")
    ap.add_argument("--ils-time", type=float, default=30.0)
    ap.add_argument("--sg-time", type=float, default=300.0)
    ap.add_argument("--max-cols", type=int, default=30_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-dir", type=Path, default=Path("."))
    args = ap.parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    insts = load_instances()

    def log(msg):
        print(msg, flush=True)

    summary = []
    for spec in args.instances:
        cls, iid = spec.split("/")
        summary.append((spec, *prove(cls, iid, insts[cls, iid], args, log)))
    print("\nsummary:")
    for spec, status, ub, lb in summary:
        print(f"  {spec:18s} {status:22s} UB {ub}  LB {lb}")


if __name__ == "__main__":
    main()
