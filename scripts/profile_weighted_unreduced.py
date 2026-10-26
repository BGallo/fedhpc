"""Profile the plain (no dominance reductions) SpaceTime weighted-sum solve.

Per lambda: cProfile of Formulation.build, model size, and a Gurobi callback
timeline (presolve done, root LP done, first incumbent, optimal), plus the
Gurobi log in LOGDIR. Mirrors model.solve_weighted_sum (f1 <= f1_0 cap).
Reference points default to the proven values for fedhpc_known (the
reductions are exact, so they are identical with or without them); set
REFS=exact to recompute them with the plain model.

    uv run python scripts/profile_weighted_unreduced.py [instance.json]
"""
from __future__ import annotations

import cProfile
import io
import json
import os
import pstats
import sys
import time

import gurobipy as gp
from gurobipy import GRB

from fedhpc.data import Instance
from fedhpc.formulations import SpaceTimeFormulation, configure_env
from fedhpc.pareto import _reference_points

sys.stdout.reconfigure(line_buffering=True)

PATH = sys.argv[1] if len(sys.argv) > 1 else "data/fedhpc_known_runtime_offline_10min.json"
LAMS = [float(x) for x in os.environ.get("LAMS", "0.1,0.25,0.5,0.75,0.9").split(",")]
LOGDIR = os.environ.get("LOGDIR", "profile_logs")
OUT = os.environ.get("OUT", os.path.join(LOGDIR, "profile.jsonl"))
KNOWN_REFS = {"fedhpc_known_runtime_offline_10min": (11151.04, 361.963, 16577.04)}


def plain():
    return SpaceTimeFormulation(collapse_unbounded=False, pareto_unbounded=False,
                                rejection_deadlines=False, lex_deadlines=False)


def main():
    configure_env(verbose=False)
    os.makedirs(LOGDIR, exist_ok=True)
    inst = Instance.from_file(PATH)
    name = os.path.basename(PATH).removesuffix(".json")
    if os.environ.get("REFS") == "exact" or name not in KNOWN_REFS:
        t = time.perf_counter()
        refs = _reference_points(inst, plain(), OutputFlag=0, MIPGap=1e-6)
        print(f"refs (plain model) {refs}  {time.perf_counter() - t:.1f}s")
    else:
        refs = KNOWN_REFS[name]
    f1_T, f2_T, f1_0 = refs
    print(f"{name}: {len(inst.jobs)} jobs, {len(inst.instance_types)} types, "
          f"H={inst.horizon}; refs f1_T={f1_T} f2_T={f2_T} f1_0={f1_0}")

    with open(OUT, "a") as fh:
        for lam in LAMS:
            fmt = plain()
            prof = cProfile.Profile()
            t0 = time.perf_counter()
            prof.enable()
            mdl, vars_ = fmt.build(inst, objective=None)
            x = vars_["x"]
            f1e, f2e = fmt.f1_expr(inst, x), fmt.f2_expr(inst, x)
            mdl.addConstr(f1e <= f1_0, name="pareto_region")
            mdl.setObjective((lam / (f1_0 - f1_T)) * f1e + ((1 - lam) / f2_T) * f2e,
                             GRB.MINIMIZE)
            mdl.update()
            prof.disable()
            build = time.perf_counter() - t0

            s = io.StringIO()
            pstats.Stats(prof, stream=s).sort_stats("cumulative").print_stats(12)
            with open(os.path.join(LOGDIR, f"{name}_lam{lam}_build.prof.txt"), "w") as pf:
                pf.write(s.getvalue())

            mdl.setParam("OutputFlag", 1)
            mdl.setParam("LogToConsole", 0)
            mdl.setParam("LogFile", os.path.join(LOGDIR, f"{name}_lam{lam}_gurobi.log"))
            mdl.setParam("MIPGap", 1e-6)

            ev = {}

            def cb(model, where):
                now = time.perf_counter() - t1
                if where == GRB.Callback.PRESOLVE:
                    ev.setdefault("presolve_start", now)
                elif where == GRB.Callback.SIMPLEX and "presolve_done" not in ev:
                    ev["presolve_done"] = now
                elif where == GRB.Callback.MIP:
                    ev.setdefault("root_done", now)
                elif where == GRB.Callback.MIPSOL:
                    ev.setdefault("first_incumbent", now)
                    ev["last_incumbent"] = now

            t1 = time.perf_counter()
            mdl.optimize(cb)
            solve = time.perf_counter() - t1

            rec = dict(inst=name, lam=lam, status=mdl.Status, obj=mdl.ObjVal,
                       bound=mdl.ObjBound, gap=mdl.MIPGap, nodes=mdl.NodeCount,
                       simplex_iters=mdl.IterCount, work=mdl.Work,
                       cols=mdl.NumVars, rows=mdl.NumConstrs, nnz=mdl.NumNZs,
                       f1=f1e.getValue(), f2=f2e.getValue(),
                       build=build, solve=solve, events=ev)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"lam={lam:<4} status={mdl.Status} obj={mdl.ObjVal:.6f} gap={mdl.MIPGap:.1e} "
                  f"f1={rec['f1']:.2f} f2={rec['f2']:.4f} | {mdl.NumVars/1e6:.2f}M cols "
                  f"{mdl.NumConstrs/1e3:.0f}k rows {mdl.NumNZs/1e6:.1f}M nnz | build {build:.1f}s "
                  f"solve {solve:.1f}s nodes {mdl.NodeCount:.0f} | "
                  + " ".join(f"{k}={v:.1f}" for k, v in ev.items()))
            del mdl, vars_, x, f1e, f2e


if __name__ == "__main__":
    main()
