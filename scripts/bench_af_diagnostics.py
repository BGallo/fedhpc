"""Factorial diagnostics for the paper AF and FED-HPC flow formulations.

Separates the modeling choices that the basic AF-vs-fed_st benchmark bundles:
full versus normal-pattern job arcs, unit versus jump idle arcs, equality versus
covering assignment rows, and integer versus continuous idle flow.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

from gurobipy import GRB

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_prjwc import build_af, fed_instance, generate, horizon  # noqa: E402

from fedhpc import formulations  # noqa: E402
from fedhpc.formulations import SpaceTimeFormulation  # noqa: E402


VARIANTS = {
    "paper_af": None,
    "grid_eq_int": {},
    "grid_ge_cont": dict(cover_ge=True, continuous_idle=True),
    "normal_ge_cont": dict(cover_ge=True, continuous_idle=True, normal_patterns=True),
    "compressed_eq": dict(compressed_flow=True),
    "compressed_ge": dict(compressed_flow=True, cover_ge=True),
}

FIELDS = [
    "case", "variant", "seed", "n_vars", "n_int", "n_cons", "n_nz",
    "pre_vars", "pre_int", "pre_cons", "pre_nz", "build_s", "presolve_s",
    "lp", "obj", "bound", "status", "nodes", "simplex_it", "solve_s",
]


def make_model(name, p, w, r, m, T, env):
    if name == "paper_af":
        return build_af(p, w, r, m, T, env)
    inst = fed_instance(p, w, r, m, T)
    fmt = SpaceTimeFormulation(**VARIANTS[name])
    mdl, vs = fmt.build(inst)
    mdl.setObjective(fmt.f1_expr(inst, vs["x"]) + sum(a * b for a, b in zip(w, r)))
    return mdl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="100:50:2:0.2,100:50:2:1.0")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--time-limit", type=float, default=300)
    ap.add_argument("--out", default="pareto_runs/af_diagnostics.csv")
    args = ap.parse_args()

    formulations.configure_env(False)
    env = formulations._shared_env
    out = Path(args.out)
    new = not out.exists()
    with out.open("a", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            wr.writeheader()
        for raw in args.cases.split(","):
            pmax, n, m, alpha = raw.split(":")
            pmax, n, m, alpha = int(pmax), int(n), int(m), float(alpha)
            rng_seed = n * 1_000_003 + m * 10_007 + int(alpha * 10) * 101 + pmax
            p, w, r = generate(n, m, alpha, pmax, 10, rng_seed)
            T = horizon(p, r, m)
            case = f"p{pmax}_n{n}_m{m}_a{alpha}"
            for name in args.variants.split(","):
                t0 = time.perf_counter()
                mdl = make_model(name, p, w, r, m, T, env)
                mdl.update()
                build_s = time.perf_counter() - t0
                raw_stats = (mdl.NumVars,
                             sum(v.VType != GRB.CONTINUOUS for v in mdl.getVars()),
                             mdl.NumConstrs, mdl.NumNZs)

                # Presolve once: its reduced dimensions explain much more than
                # raw pseudo-polynomial size when formulations share an LP bound.
                for k, v in {"OutputFlag": 0, "Threads": 1, "MIPGap": 0,
                             "TimeLimit": args.time_limit}.items():
                    mdl.setParam(k, v)
                t0 = time.perf_counter()
                pre = mdl.presolve()
                presolve_s = time.perf_counter() - t0
                pre.update()
                pre_stats = (pre.NumVars,
                             sum(v.VType != GRB.CONTINUOUS for v in pre.getVars()),
                             pre.NumConstrs, pre.NumNZs)
                pre.dispose()

                rel = mdl.relax()
                rel.setParam("OutputFlag", 0)
                rel.setParam("Method", 2)
                rel.optimize()
                lp = rel.ObjVal if rel.Status == GRB.OPTIMAL else None
                rel.dispose()

                for seed in map(int, args.seeds.split(",")):
                    mdl.reset()
                    mdl.setParam("Seed", seed)
                    t0 = time.perf_counter()
                    mdl.optimize()
                    solve_s = time.perf_counter() - t0
                    has = mdl.SolCount > 0
                    row = dict(
                        case=case, variant=name, seed=seed,
                        n_vars=raw_stats[0], n_int=raw_stats[1],
                        n_cons=raw_stats[2], n_nz=raw_stats[3],
                        pre_vars=pre_stats[0], pre_int=pre_stats[1],
                        pre_cons=pre_stats[2], pre_nz=pre_stats[3],
                        build_s=round(build_s, 4), presolve_s=round(presolve_s, 4),
                        lp=lp, obj=mdl.ObjVal if has else None,
                        bound=mdl.ObjBound, status={GRB.OPTIMAL: "optimal",
                            GRB.TIME_LIMIT: "time_limit"}.get(mdl.Status, str(mdl.Status)),
                        nodes=int(mdl.NodeCount), simplex_it=int(mdl.IterCount),
                        solve_s=round(solve_s, 4),
                    )
                    wr.writerow(row)
                    fh.flush()
                    print(case, name, f"seed={seed}", row["status"],
                          f"raw={raw_stats[0]}/{raw_stats[2]}/{raw_stats[3]}",
                          f"pre={pre_stats[0]}/{pre_stats[2]}/{pre_stats[3]}",
                          f"nodes={row['nodes']} t={row['solve_s']}", flush=True)
                mdl.dispose()


if __name__ == "__main__":
    main()
