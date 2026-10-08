"""Benchmark fedhpc's MIP formulations against Kramer, Dell'Amico, Feillet & Iori
(2020), "Scheduling jobs with release dates on identical parallel machines by
minimizing the total weighted completion time" (arXiv 2005.10097, P|r_j|Σw_jC_j).

P|r_j|Σw_jC_j is fedhpc with a single instance type of capacity m, arrivals
a_j = r_j, weights w_j and zero cost; fedhpc's f1 = Σ w_j (C_j − r_j) differs
from Σ w_j C_j only by the constant Σ w_j r_j. Four models are compared on the
same instances, horizon and Gurobi settings:

- ``TI``        paper eq. (2)–(5): x_jt, one capacity row per t.
- ``AF``        paper eq. (6)–(10): m-unit flow over the normal-pattern graph
                of Algorithm 1 (job arcs binary, loss arcs continuous).
- ``fed_occ``   fedhpc ``OccupancyFormulation`` (≈ TI with occupancy vars w_mt).
- ``fed_st``    fedhpc ``SpaceTimeFormulation`` (≈ AF on the full grid 0..T,
                integer idle vars y_mt).
- ``fed_af``    fedhpc compressed normal-pattern graph with continuous jump
                loss arcs (the closest like-for-like implementation of AF).

Instances follow the paper's §5.1 generator: p_j ~ U[1, p_max],
w_j ~ U[1, w_max], r_j ~ U[0, ⌊α·n·(p_max+1)/2 / m⌋]; horizon T from eq. (1).
All models run single-threaded with no MIP start (the paper gave its models an
ILS upper bound; omitted here so every model starts from the same nothing).

Usage
-----
    uv run python scripts/bench_prjwc.py [--n 20,50] [--m 2,3,5,10]
        [--alpha 0.2,0.6,1.0,1.5,2.0,3.0] [--pmax 100] [--time-limit 600]
        [--models TI,AF,fed_occ,fed_st] [--out results.csv]
"""
from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

from fedhpc import formulations
from fedhpc.data import Instance, InstanceType, Job
from fedhpc.formulations import OccupancyFormulation, SpaceTimeFormulation

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Instances
# ---------------------------------------------------------------------------

def generate(n: int, m: int, alpha: float, pmax: int, wmax: int, seed: int):
    rng = random.Random(seed)
    p = [rng.randint(1, pmax) for _ in range(n)]
    w = [rng.randint(1, wmax) for _ in range(n)]
    rmax = int(alpha * n * (pmax + 1) / 2 / m)
    r = [rng.randint(0, rmax) for _ in range(n)]
    return p, w, r


def horizon(p, r, m) -> int:
    """Paper eq. (1): T = ⌊Σp/m + (m−1)/m·p_max⌋ + r_max."""
    return int(sum(p) / m + (m - 1) / m * max(p)) + max(r)


# ---------------------------------------------------------------------------
# Paper models
# ---------------------------------------------------------------------------

def build_ti(p, w, r, m, T, env):
    n = len(p)
    mdl = gp.Model("TI", env=env)
    x = {(j, t): mdl.addVar(vtype=GRB.BINARY)
         for j in range(n) for t in range(r[j], T - p[j] + 1)}
    mdl.setObjective(gp.quicksum(w[j] * (t + p[j]) * v for (j, t), v in x.items()))
    for j in range(n):
        mdl.addConstr(gp.quicksum(x[j, t] for t in range(r[j], T - p[j] + 1)) >= 1)
    for t in range(T + 1):
        terms = [x[j, s] for j in range(n)
                 for s in range(max(r[j], t + 1 - p[j]), min(t, T + 1 - p[j]) + 1)
                 if (j, s) in x]
        if terms:
            mdl.addConstr(gp.quicksum(terms) <= m)
    return mdl


def normal_pattern_graph(p, r, T):
    """Paper Algorithm 1: node set N and job arcs A_j, loss arcs A_0."""
    n = len(p)
    R = sorted(set(r) | {T})
    P = bytearray(T + 1)
    for q in R:
        P[q] = 1
    arcs = []                           # (q, q+p_j, j)
    for q in range(T + 1):
        if P[q]:
            for j in range(n):
                if q >= r[j] and q + p[j] <= T:
                    P[q + p[j]] = 1
                    arcs.append((q, q + p[j], j))
    nodes = [q for q in range(T + 1) if P[q]]
    loss, i = [], 0
    for q in nodes:
        while i < len(R) and q >= R[i]:
            i += 1
        if i < len(R):
            loss.append((q, R[i]))
    return nodes, arcs, loss


def build_af(p, w, r, m, T, env):
    n = len(p)
    nodes, arcs, loss = normal_pattern_graph(p, r, T)
    mdl = gp.Model("AF", env=env)
    x = {a: mdl.addVar(vtype=GRB.BINARY) for a in arcs}
    z = {a: mdl.addVar(lb=0, ub=m) for a in loss}
    # arc (q, q+p_j, j) completes job j at q+p_j
    mdl.setObjective(gp.quicksum(w[j] * rr * v for (q, rr, j), v in x.items()))
    by_job = [[] for _ in range(n)]
    out_ = {q: [] for q in nodes}
    in_ = {q: [] for q in nodes}
    for (q, rr, j), v in x.items():
        by_job[j].append(v)
        out_[q].append(v)
        in_[rr].append(v)
    for (q, rr), v in z.items():
        out_[q].append(v)
        in_[rr].append(v)
    for j in range(n):
        mdl.addConstr(gp.quicksum(by_job[j]) >= 1)
    rmin = min(r)
    for q in nodes:
        rhs = m if q == rmin else (-m if q == T else 0)
        mdl.addConstr(gp.quicksum(out_[q]) - gp.quicksum(in_[q]) == rhs)
    return mdl


# ---------------------------------------------------------------------------
# fedhpc models
# ---------------------------------------------------------------------------

def fed_instance(p, w, r, m, T) -> Instance:
    big = float("inf")
    itype = InstanceType(id=0, kind="machine", perf=1.0, io_time=0.0, deploy=0.0,
                         cost_io=0.0, cost_vm=0.0, cost_stor=0.0,
                         cpu=big, mem=big, stor=big, capacity=m)
    jobs = [Job(id=j, arrival=float(r[j]), exec_time=float(p[j]), io_volume=0.0,
                cpu=0.0, mem=0.0, stor=0.0, weight=float(w[j])) for j in range(len(p))]
    inst = Instance(jobs=jobs, instance_types=[itype], horizon=T, budget=big)
    inst.build()
    return inst


def build_fed(fmt, p, w, r, m, T):
    inst = fed_instance(p, w, r, m, T)
    mdl, vars_ = fmt.build(inst)
    # f1 = Σ w (C − r); add Σ w r back so every model reports Σ w C.
    mdl.setObjective(fmt.f1_expr(inst, vars_["x"]) + sum(wj * rj for wj, rj in zip(w, r)))
    return mdl


MODELS = {
    "TI": lambda p, w, r, m, T, env: build_ti(p, w, r, m, T, env),
    "AF": lambda p, w, r, m, T, env: build_af(p, w, r, m, T, env),
    "fed_occ": lambda p, w, r, m, T, env: build_fed(OccupancyFormulation(), p, w, r, m, T),
    "fed_st": lambda p, w, r, m, T, env: build_fed(SpaceTimeFormulation(), p, w, r, m, T),
    "fed_af": lambda p, w, r, m, T, env: build_fed(
        SpaceTimeFormulation(cover_ge=True, compressed_flow=True), p, w, r, m, T
    ),
}


def run(name, p, w, r, m, T, env, time_limit):
    t0 = time.perf_counter()
    mdl = MODELS[name](p, w, r, m, T, env)
    mdl.update()
    build_s = time.perf_counter() - t0
    for k, v in {"OutputFlag": 0, "Threads": 1, "MIPGap": 0, "TimeLimit": time_limit}.items():
        mdl.setParam(k, v)
    rel = mdl.relax()
    rel.setParam("Method", 2)   # barrier for the large LPs
    rel.optimize()
    lp = rel.ObjVal if rel.Status == GRB.OPTIMAL else None
    t1 = time.perf_counter()
    mdl.optimize()
    solve_s = time.perf_counter() - t1
    has = mdl.SolCount > 0
    return dict(model=name, n_vars=mdl.NumVars, n_cons=mdl.NumConstrs,
                lp=lp, obj=mdl.ObjVal if has else None,
                bound=mdl.ObjBound if has else None,
                status={GRB.OPTIMAL: "optimal", GRB.TIME_LIMIT: "time_limit"}.get(
                    mdl.Status, str(mdl.Status)),
                nodes=int(mdl.NodeCount), build_s=round(build_s, 3),
                solve_s=round(solve_s, 3))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", default="20,50")
    ap.add_argument("--m", default="2,3,5,10")
    ap.add_argument("--alpha", default="0.2,0.6,1.0,1.5,2.0,3.0")
    ap.add_argument("--pmax", type=int, default=100)
    ap.add_argument("--wmax", type=int, default=10)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--time-limit", type=float, default=600.0)
    ap.add_argument("--models", default="TI,AF,fed_occ,fed_st")
    ap.add_argument("--out", default=str(ROOT / "pareto_runs" / "prjwc_bench.csv"))
    args = ap.parse_args()

    formulations.configure_env(verbose=False)
    env = formulations._shared_env
    models = args.models.split(",")
    fields = ["n", "m", "alpha", "rep", "pmax", "T", "model", "n_vars", "n_cons",
              "lp", "obj", "bound", "status", "nodes", "build_s", "solve_s"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with open(out) as f:
            done = {(r["n"], r["m"], r["alpha"], r["rep"], r["pmax"], r["model"])
                    for r in csv.DictReader(f)}
    new = not out.exists()
    with open(out, "a", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        if new:
            wr.writeheader()
        for n in map(int, args.n.split(",")):
            for m in map(int, args.m.split(",")):
                for alpha in map(float, args.alpha.split(",")):
                    for rep in range(args.reps):
                        seed = (n * 1_000_003 + m * 10_007 + int(alpha * 10) * 101
                                + rep * 7 + args.pmax)
                        p, w, r = generate(n, m, alpha, args.pmax, args.wmax, seed)
                        T = horizon(p, r, m)
                        for name in models:
                            key = (str(n), str(m), str(alpha), str(rep), str(args.pmax), name)
                            if key in done:
                                continue
                            res = run(name, p, w, r, m, T, env, args.time_limit)
                            row = dict(n=n, m=m, alpha=alpha, rep=rep, pmax=args.pmax,
                                       T=T, **res)
                            wr.writerow(row)
                            fh.flush()
                            print(f"n={n:<3} m={m:<2} a={alpha:<3} T={T:<6} {name:<8} "
                                  f"vars={res['n_vars']:<8} lp={res['lp'] or float('nan'):>11.1f} "
                                  f"obj={res['obj'] or float('nan'):>11.1f} "
                                  f"bnd={res['bound'] or float('nan'):>11.1f} "
                                  f"{res['status']:<10} nodes={res['nodes']:<7} "
                                  f"{res['build_s']}+{res['solve_s']}s", flush=True)


if __name__ == "__main__":
    main()
