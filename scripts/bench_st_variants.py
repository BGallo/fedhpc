"""Cheap SpaceTimeFormulation improvements + the C++ memetic heuristic, on a
subset of Kramer et al. (2020) P|r_j|Σw_jC_j instances and the fedhpc
``fedhpc_known_runtime_offline_10min`` instance.

Solvers (all single-threaded, like the paper's B&P):

- ``st_base``  SpaceTimeFormulation()                      (fedhpc default)
- ``st_cy``    continuous_idle=True
- ``st_ge``    cover_ge=True
- ``st_np``    normal_patterns=True
- ``st_all``   all three
- ``st_af``    compressed normal-pattern graph with jump loss arcs
- ``meta``     moea weighted memetic, weighted_solve defaults (pop 32, 60 gen)
- ``meta_long`` same, pop 64 / 300 generations

Objectives: paper instances minimise Σ w_j C_j (f1 + Σ w_j r_j); the offline
instance minimises the λ=0.5 weighted sum g = λ f1/(f1_0−f1_T) + (1−λ) f2/f2_T
subject to f1 ≤ f1_0, with the exact reference points stored in
``pareto_runs/convergence/fedhpc_known_runtime_offline_10min_lam0.5.json``.

Usage
-----
    uv run python scripts/bench_st_variants.py --only paper --workers 3
    uv run python scripts/bench_st_variants.py --only offline
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from gurobipy import GRB

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_prjwc import fed_instance, generate, horizon  # noqa: E402

from fedhpc import formulations, moea  # noqa: E402
from fedhpc.data import Instance  # noqa: E402
from fedhpc.formulations import SpaceTimeFormulation  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# (p_max, n, m, alpha) — chosen from prjwc_p*_m*.csv: fed_st needed 30–1000 s,
# plus the p_max=1000 root-stall instance.
PAPER_SUBSET = [
    (100, 50, 2, 0.2), (100, 50, 2, 0.6), (100, 50, 5, 0.6),
    (100, 100, 2, 0.2), (100, 100, 3, 0.2), (100, 100, 5, 0.2),
    (1000, 50, 2, 1.0), (1000, 50, 2, 1.5), (1000, 50, 3, 0.2), (1000, 50, 5, 0.2),
    (1000, 50, 2, 0.2),
]
OFFLINE = "fedhpc_known_runtime_offline_10min"
MIP_VARIANTS = {
    "st_base": {},
    "st_cy": dict(continuous_idle=True),
    "st_ge": dict(cover_ge=True),
    "st_np": dict(normal_patterns=True),
    "st_all": dict(continuous_idle=True, cover_ge=True, normal_patterns=True),
    "st_af": dict(cover_ge=True, compressed_flow=True),
}
META = {
    "meta": dict(pop_size=32, n_gen=60),
    "meta_long": dict(pop_size=64, n_gen=300),
    "meta_wspt": dict(pop_size=32, n_gen=60, decode_order=2),
    "meta_wspt_long": dict(pop_size=64, n_gen=300, decode_order=2),
}


# ---------------------------------------------------------------------------

class Case:
    """An instance plus its scalar objective g(f1, f2) and MIP objective."""

    def __init__(self, name: str, inst: Instance, w1: float, w2: float,
                 const: float, f1_cap: float | None):
        self.name, self.inst = name, inst
        self.w1, self.w2, self.const, self.f1_cap = w1, w2, const, f1_cap

    def g(self, f1: float, f2: float) -> float:
        return self.w1 * f1 + self.w2 * f2 + self.const


def paper_case(pmax, n, m, alpha) -> Case:
    seed = n * 1_000_003 + m * 10_007 + int(alpha * 10) * 101 + 0 * 7 + pmax
    p, w, r = generate(n, m, alpha, pmax, 10, seed)
    inst = fed_instance(p, w, r, m, horizon(p, r, m))
    return Case(f"p{pmax}_n{n}_m{m}_a{alpha}", inst, 1.0, 0.0,
                sum(wj * rj for wj, rj in zip(w, r)), None)


def offline_case() -> Case:
    inst = Instance.from_file(ROOT / "data" / f"{OFFLINE}.json")
    ref = json.load(open(ROOT / "pareto_runs" / "convergence" / f"{OFFLINE}_lam0.5.json"))
    lam, f1_T, f2_T, f1_0 = ref["lam"], ref["f1_T"], ref["f2_T"], ref["f1_0"]
    return Case(OFFLINE, inst, lam / (f1_0 - f1_T), (1 - lam) / f2_T, 0.0, f1_0)


# ---------------------------------------------------------------------------

def run_mip(case: Case, opts: dict, time_limit: float) -> dict:
    fmt = SpaceTimeFormulation(**opts)
    t0 = time.perf_counter()
    mdl, vars_ = fmt.build(case.inst)
    x = vars_["x"]
    f1e, f2e = fmt.f1_expr(case.inst, x), fmt.f2_expr(case.inst, x)
    if case.f1_cap is not None:
        mdl.addConstr(f1e <= case.f1_cap, name="pareto_region")
    mdl.setObjective(case.w1 * f1e + case.w2 * f2e + case.const, GRB.MINIMIZE)
    mdl.update()
    build_s = time.perf_counter() - t0
    for k, v in {"OutputFlag": 0, "Threads": 1, "MIPGap": 0,
                 "TimeLimit": time_limit}.items():
        mdl.setParam(k, v)
    t1 = time.perf_counter()
    mdl.optimize()
    solve_s = time.perf_counter() - t1
    has = mdl.SolCount > 0
    n_int = sum(1 for v in mdl.getVars() if v.VType != GRB.CONTINUOUS)
    return dict(
        n_vars=mdl.NumVars, n_int=n_int, n_cons=mdl.NumConstrs,
        root_bound=None, obj=mdl.ObjVal if has else None,
        bound=mdl.ObjBound if has else None,
        status={GRB.OPTIMAL: "optimal", GRB.TIME_LIMIT: "time_limit"}.get(
            mdl.Status, str(mdl.Status)),
        nodes=int(mdl.NodeCount), build_s=round(build_s, 3), solve_s=round(solve_s, 3))


def run_meta(case: Case, kw: dict) -> dict:
    inst = case.inst
    kw = dict(kw)
    decode_order = kw.pop("decode_order", 1)
    t0 = time.perf_counter()
    asgn, f1, f2, _g, _ls = moea._weighted_raw(
        inst, case.w1, case.w2, case.f1_cap if case.f1_cap is not None else 1e18,
        seed=42, n_threads=1, ls_moves=8, restart_patience=6, shortlist=24,
        xover_mode=2, decode_order=decode_order, **kw)
    dt = time.perf_counter() - t0
    return dict(n_vars=None, n_int=None, n_cons=None, root_bound=None,
                obj=case.g(f1, f2), bound=None, status="heuristic", nodes=None,
                build_s=0.0, solve_s=round(dt, 3))


FIELDS = ["case", "solver", "n_vars", "n_int", "n_cons", "root_bound", "obj",
          "bound", "status", "nodes", "build_s", "solve_s"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["paper", "offline", "all"], default="all")
    ap.add_argument("--solvers", default=",".join([*MIP_VARIANTS, *META]))
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--time-limit", type=float, default=1800.0)
    ap.add_argument("--out", default=str(ROOT / "pareto_runs" / "st_variants.csv"))
    args = ap.parse_args()

    formulations.configure_env(verbose=False)
    specs: list[tuple] = []
    if args.only in ("paper", "all"):
        specs += [("paper", s) for s in PAPER_SUBSET]
    if args.only in ("offline", "all"):
        specs.append(("offline", None))
    solvers = args.solvers.split(",")
    jobs = [(sp, so) for sp in specs for so in solvers]
    jobs = jobs[args.worker::args.workers]

    out = Path(args.out)
    done = set()
    if out.exists():
        with open(out) as f:
            done = {(r["case"], r["solver"]) for r in csv.DictReader(f)}
    cache: dict = {}
    for (kind, spec), solver in jobs:
        key = spec if kind == "paper" else OFFLINE
        if key not in cache:
            cache.clear()
            cache[key] = paper_case(*spec) if kind == "paper" else offline_case()
        case = cache[key]
        if (case.name, solver) in done:
            continue
        res = (run_mip(case, MIP_VARIANTS[solver], args.time_limit)
               if solver in MIP_VARIANTS else run_meta(case, META[solver]))
        new = not out.exists()
        with open(out, "a", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=FIELDS)
            if new:
                wr.writeheader()
            wr.writerow(dict(case=case.name, solver=solver, **res))
        print(f"{case.name:<22} {solver:<10} obj={res['obj']} bnd={res['bound']} "
              f"{res['status']:<10} vars={res['n_vars']} int={res['n_int']} "
              f"{res['build_s']}+{res['solve_s']}s", flush=True)


if __name__ == "__main__":
    main()
