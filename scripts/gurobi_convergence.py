"""Log Gurobi's incumbent/bound convergence over wall time for the weighted-sum
MIP (same scalarisation as weighted_heur_vs_gurobi.py), run towards proven
optimality, and record the mono-objective memetic heuristic's single
(time, objective) point for the same lambda.

Writes one JSON log per (instance, lambda) to pareto_runs/convergence/.

  uv run python scripts/gurobi_convergence.py [instances...]

Env vars:
  LAMS  comma list of lambda values (default 0.25,0.5,0.75)
  TL    Gurobi wall-clock limit per solve, seconds (default: uncapped --
        Gurobi runs until it proves optimality, MIPGap 1e-6).
  POP/NGEN  heuristic population/generations (default 24/40)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from gurobipy import GRB

from fedhpc.data import Instance
from fedhpc.formulations import SpaceTimeFormulation, configure_env
from fedhpc.moea import weighted_solve

sys.stdout.reconfigure(line_buffering=True)

INSTANCES = sys.argv[1:] or [
    "data/fedhpc_known_runtime_offline_10min.json",
    "data/pos_congestion_known_runtime_offline_10min.json",
]
LAMS = [float(x) for x in os.environ.get("LAMS", "0.25,0.5,0.75").split(",")]
TL = float(os.environ.get("TL", str(GRB.INFINITY)))
POP = int(os.environ.get("POP", "24"))
NGEN = int(os.environ.get("NGEN", "40"))
OUT_DIR = Path("pareto_runs/convergence")

# Exact reference points already computed (REF_TL=300, MIPGap=1e-4) in
# weighted_heur_vs_gurobi.py on 2026-09-16 -- reused verbatim since they are
# deterministic for a fixed instance/formulation, saving ~12 minutes.
REF_POINTS = {
    "fedhpc_known_runtime_offline_10min.json": (11151.0, 361.96, 16577.0),
    "pos_congestion_known_runtime_offline_10min.json": (44163.7, 648.53, 51251.7),
}


def gurobi_weighted_logged(inst, lam, f1_T, f2_T, f1_0, time_limit, mip_gap=1e-6):
    fmt = SpaceTimeFormulation()
    mdl, vars_ = fmt.build(inst)
    mdl.setParam("OutputFlag", 0)
    mdl.setParam("TimeLimit", time_limit)
    mdl.setParam("MIPGap", mip_gap)
    x = vars_["x"]
    f1e, f2e = fmt.f1_expr(inst, x), fmt.f2_expr(inst, x)
    mdl.addConstr(f1e <= f1_0, name="pareto_region")
    degenerate = f1_0 <= f1_T + 1e-8 or f2_T <= 1e-8
    if degenerate:
        mdl.setObjective(f2e, GRB.MINIMIZE)
    else:
        mdl.setObjective((lam / (f1_0 - f1_T)) * f1e + ((1 - lam) / f2_T) * f2e,
                         GRB.MINIMIZE)

    log: list[tuple[float, float, float]] = []

    def cb(model, where):
        if where == GRB.Callback.MIP:
            rt = model.cbGet(GRB.Callback.RUNTIME)
            objbst = model.cbGet(GRB.Callback.MIP_OBJBST)
            objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)
            log.append((rt, objbst, objbnd))

    t0 = time.time()
    mdl.optimize(cb)
    wall = time.time() - t0

    if mdl.SolCount == 0:
        log.append((wall, None, mdl.ObjBound))
        return {
            "log": log, "f1": None, "f2": None, "g": None, "bound": mdl.ObjBound,
            "mip_gap": float("nan"), "wall": wall, "status": mdl.Status,
            "degenerate": degenerate,
        }

    log.append((wall, mdl.ObjVal, mdl.ObjBound))

    f1 = 0.0
    f2 = 0.0
    for j in inst.jobs:
        for m_id in inst.F[j.id]:
            for tt in inst.T[j.id, m_id]:
                xv = x[j.id, m_id, tt].X
                if xv > 0.5:
                    f1 += tt + inst.p_occ[j.id, m_id] - j.arrival
                f2 += inst.c[j.id, m_id] * xv
    return {
        "log": log, "f1": f1, "f2": f2, "g": mdl.ObjVal, "bound": mdl.ObjBound,
        "mip_gap": mdl.MIPGap, "wall": wall, "status": mdl.Status,
        "degenerate": degenerate,
    }


def main():
    configure_env(verbose=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in INSTANCES:
        inst = Instance.from_file(path)
        name = path.split("/")[-1]
        f1_T, f2_T, f1_0 = REF_POINTS[name]
        print(f"\n{'='*90}\n{name}  —  {len(inst.jobs)} jobs  "
              f"ref: f1_T={f1_T} f2_T={f2_T} f1_0={f1_0}\n{'='*90}")

        for lam in LAMS:
            out_path = OUT_DIR / f"{name.replace('.json', '')}_lam{lam}.json"
            print(f"  lam={lam}: solving heuristic...", end=" ")
            t = time.time()
            he = weighted_solve(inst, lam, f1_T=f1_T, f2_T=f2_T, f1_0=f1_0,
                                 pop_size=POP, n_gen=NGEN)
            t_he = time.time() - t
            g_he = (he.f2 if (f1_0 <= f1_T + 1e-8 or f2_T <= 1e-8)
                    else (lam / (f1_0 - f1_T)) * he.f1 + ((1 - lam) / f2_T) * he.f2)
            print(f"g={g_he:.4f} t={t_he:.1f}s")

            tl_str = "uncapped" if TL >= GRB.INFINITY else f"cap {TL:.0f}s"
            print(f"  lam={lam}: solving Gurobi ({tl_str})...", end=" ")
            t = time.time()
            gr = gurobi_weighted_logged(inst, lam, f1_T, f2_T, f1_0, TL)
            g_str = f"{gr['g']:.4f}" if gr["g"] is not None else "no incumbent"
            print(f"g={g_str} bound={gr['bound']:.4f} "
                  f"gap={gr['mip_gap']*100:.2f}% wall={gr['wall']:.1f}s "
                  f"({len(gr['log'])} ticks)")

            payload = {
                "instance": name, "lam": lam,
                "f1_T": f1_T, "f2_T": f2_T, "f1_0": f1_0,
                "heuristic": {"f1": he.f1, "f2": he.f2, "g": g_he, "time": t_he},
                "gurobi": gr,
            }
            out_path.write_text(json.dumps(payload, indent=2))
            print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
