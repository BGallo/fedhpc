"""Gap of the memetic weighted-sum metaheuristic (moea.weighted_solve) vs the
exact Gurobi weighted-sum solve, at the SAME lambda and the SAME reference
points, on the two big 10-minute instances.

Both solvers minimise the identical scalar
    g(lam) = (lam/(f1_0-f1_T))*f1 + ((1-lam)/f2_T)*f2      s.t. f1 <= f1_0
so the reported number is pure optimisation quality on one fixed objective.

  gap%     = 100 * (g_heur - g_gurobi_incumbent) / g_gurobi_incumbent
  opt_gap% = Gurobi's own MIPGap (incumbent vs dual bound); >0 means the MIP
             did NOT prove optimality within the time limit, so g_gurobi is
             only an upper bound and a negative gap% just means the heuristic
             found a better incumbent than Gurobi did in the budget.
"""
from __future__ import annotations

import os
import sys
import time

from gurobipy import GRB

from fedhpc.data import Instance
from fedhpc.formulations import SpaceTimeFormulation, configure_env
from fedhpc.model import set_mip_start
from fedhpc.moea import heuristic_weighted_reference_points, weighted_solve
from fedhpc.pareto import _reference_points

sys.stdout.reconfigure(line_buffering=True)

INSTANCES = sys.argv[1:] or [
    "data/fedhpc_known_runtime_offline_10min.json",
    "data/pos_congestion_known_runtime_offline_10min.json",
]
LAMS = [float(x) for x in os.environ.get("LAMS", "0.25,0.5,0.75").split(",")]
TL = float(os.environ.get("TL", "300"))
REF_TL = float(os.environ.get("REF_TL", "300"))
POP = int(os.environ.get("POP", "24"))
NGEN = int(os.environ.get("NGEN", "40"))


def gurobi_weighted(inst, lam, f1_T, f2_T, f1_0, time_limit, hint=None):
    """min the normalised weighted sum; returns (f1, f2, g, incumbent_obj,
    dual_bound, mip_gap, wall_s). Mirrors model.solve_weighted_sum.
    `hint` — a Solution used as the MIP start (so Gurobi never does worse)."""
    fmt = SpaceTimeFormulation()
    mdl, vars_ = fmt.build(inst)
    mdl.setParam("OutputFlag", 0)
    mdl.setParam("TimeLimit", time_limit)
    x = vars_["x"]
    if hint is not None:
        set_mip_start(inst, x, hint)
    f1e, f2e = fmt.f1_expr(inst, x), fmt.f2_expr(inst, x)
    mdl.addConstr(f1e <= f1_0, name="pareto_region")
    degenerate = f1_0 <= f1_T + 1e-8 or f2_T <= 1e-8
    if degenerate:
        mdl.setObjective(f2e, GRB.MINIMIZE)
    else:
        mdl.setObjective((lam / (f1_0 - f1_T)) * f1e + ((1 - lam) / f2_T) * f2e,
                         GRB.MINIMIZE)
    t = time.time()
    mdl.optimize()
    wall = time.time() - t
    if mdl.SolCount == 0:
        return None
    # recover f1, f2 from the incumbent
    f1 = 0.0
    f2 = 0.0
    for j in inst.jobs:
        for m_id in inst.F[j.id]:
            for tt in inst.T[j.id, m_id]:
                xv = x[j.id, m_id, tt].X
                if xv > 0.5:
                    f1 += tt + inst.p_occ[j.id, m_id] - j.arrival
                f2 += inst.c[j.id, m_id] * xv
    g = f2 if degenerate else (lam / (f1_0 - f1_T)) * f1 + ((1 - lam) / f2_T) * f2
    try:
        mgap = mdl.MIPGap
    except Exception:  # noqa: BLE001
        mgap = float("nan")
    return f1, f2, g, mdl.ObjVal, mdl.ObjBound, mgap, wall


def main():
    configure_env(verbose=False)
    for path in INSTANCES:
        inst = Instance.from_file(path)
        name = path.split("/")[-1]
        print(f"\n{'='*104}\n{name}  —  {len(inst.jobs)} jobs, "
              f"{len(inst.instance_types)} types, H={inst.horizon}\n{'='*104}")

        t = time.time()
        ref_src = "gurobi"
        try:
            f1_T, f2_T, f1_0 = _reference_points(
                inst, OutputFlag=0, TimeLimit=REF_TL, MIPGap=1e-4)
        except Exception as e:  # noqa: BLE001
            print(f"  exact reference points failed ({type(e).__name__}: {e}) "
                  f"-> heuristic estimates")
            f1_T, f2_T, f1_0 = heuristic_weighted_reference_points(inst)
            ref_src = "heuristic"
        print(f"  reference points [{ref_src}]: f1_T={f1_T:.1f}  f2_T={f2_T:.2f}  "
              f"f1_0={f1_0:.1f}   ({time.time()-t:.1f}s)")

        hdr = (f"  {'lam':>5} | {'Gur f1':>9} {'Gur f2':>9} {'g_gur':>10} {'MIPgap':>8} "
               f"{'t(s)':>7} | {'heur f1':>9} {'heur f2':>9} {'g_heur':>10} {'t(s)':>6} | "
               f"{'g_heur vs g_gur':>15} {'g_heur vs bound':>15}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        hint_gurobi = os.environ.get("HINT", "0") == "1"
        for lam in LAMS:
            t = time.time()
            he = weighted_solve(inst, lam, f1_T=f1_T, f2_T=f2_T, f1_0=f1_0,
                                pop_size=POP, n_gen=NGEN)
            t_he = time.time() - t

            gr = gurobi_weighted(inst, lam, f1_T, f2_T, f1_0, TL,
                                 hint=he if hint_gurobi else None)
            g_he = (he.f2 if (f1_0 <= f1_T + 1e-8 or f2_T <= 1e-8)
                    else (lam / (f1_0 - f1_T)) * he.f1 + ((1 - lam) / f2_T) * he.f2)

            if gr is None:
                print(f"  {lam:>5.2f} | {f'no incumbent in {TL}s':>38} | "
                      f"{he.f1:>9.1f} {he.f2:>9.2f} {g_he:>10.4f} {t_he:>6.1f} |")
                continue
            f1g, f2g, g_gur, _obj, bound, mgap, wall = gr
            vs_gur = 100.0 * (g_he - g_gur) / abs(g_gur) if abs(g_gur) > 1e-12 else float("nan")
            vs_bnd = 100.0 * (g_he - bound) / abs(bound) if abs(bound) > 1e-12 else float("nan")
            og = f"{mgap*100:6.2f}%" if mgap == mgap else "   n/a"
            print(f"  {lam:>5.2f} | {f1g:>9.1f} {f2g:>9.2f} {g_gur:>10.4f} {og:>8} "
                  f"{wall:>7.1f} | {he.f1:>9.1f} {he.f2:>9.2f} {g_he:>10.4f} {t_he:>6.1f} | "
                  f"{vs_gur:>+14.2f}% {vs_bnd:>+14.2f}%")


if __name__ == "__main__":
    main()
