"""Fair comparison of Kramer et al. (2020) TI/AF against fedhpc's space-time models
on the paper's own public instances (github.com/ahkramer/PrwC, Table 3 subset).

Models (all solve P|r_j|Σw_jC_j; objective reported as Σ w_j C_j):

- ``TI``      paper eq. (2)–(5), full time grid            (theirs, no reduction)
- ``AF``      paper eq. (6)–(10), normal-pattern arc flow  (theirs, reduced)
- ``fed_st``  fedhpc SpaceTimeFormulation(), full grid     (ours, no reduction)
- ``fed_af``  fedhpc SpaceTimeFormulation(cover_ge, compressed_flow) (ours, reduced)

fedhpc's own dominance rules R1–R3/lex need an unbounded (cloud) type, so they are
inactive on these instances; the only reduction that applies is normal patterns.

Every run: one thread, pinned to its own physical core, MIPGap 0, Seed 0, fixed
time limit. Optional warm start = the same heuristic schedule for every model
(list-scheduling + insertion ILS for n/5 s, as the paper's ILS budget).

Subcommands
-----------
    tune   coordinate-descent search of Gurobi parameters per model on a held-out
           tuning set (instance id 3), equal budget for every model
    run    final runs on the test set (ids 1, 2) for given configs
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from multiprocessing import Pool, Queue
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

ALPHA_CODE = {0: 0.2, 2: 0.6, 4: 1.0, 6: 1.5, 8: 2.0, 9: 3.0}
CPUS = [0, 2, 4, 6, 12, 14]          # one hyperthread per physical P-core
MODELS = ["TI", "AF", "fed_st", "fed_af"]

FIELDS = ["phase", "config", "model", "inst", "n", "m", "alpha", "id", "T", "params",
          "warm", "ws_obj", "ws_s", "n_vars", "n_cons", "n_nz", "lp", "root_bound",
          "root_s", "obj", "bound", "gap", "status", "nodes", "first_sol_s",
          "best_sol_s", "build_s", "solve_s", "cpu"]


# ---------------------------------------------------------------------------
# Instances
# ---------------------------------------------------------------------------

def read_instance(path: Path):
    tok = path.read_text().split()
    n, m = int(tok[0]), int(tok[1])
    vals = list(map(int, tok[3:3 + 3 * n]))
    p, r, w = vals[0::3], vals[1::3], vals[2::3]
    return p, w, r, m


def inst_path(inst_dir: Path, n, m, z, i):
    return inst_dir / "Instances_2020_ok" / "p_max_100" / f"N{n}" / f"N{n}_M{m}_R{z}_{i}.dat"


def horizon(p, r, m) -> int:
    return int(sum(p) / m + (m - 1) / m * max(p)) + max(r)


# ---------------------------------------------------------------------------
# Warm-start heuristic: list scheduling + insertion ILS (same start for all models)
# ---------------------------------------------------------------------------

def decode(order, p, w, r, m):
    free = [0] * m
    start = [0] * len(p)
    obj = 0
    for j in order:
        k = min(range(m), key=lambda k: (max(free[k], r[j]), k))
        s = max(free[k], r[j])
        start[j] = s
        free[k] = s + p[j]
        obj += w[j] * free[k]
    return obj, start


def heuristic(p, w, r, m, T, budget, seed=0):
    rng = random.Random(seed)
    n = len(p)
    seeds = [sorted(range(n), key=f) for f in (
        lambda j: (r[j], p[j] / w[j]),
        lambda j: (r[j] + p[j]) / w[j],
        lambda j: p[j] / w[j],
        lambda j: r[j] + p[j],
    )]
    t0 = time.perf_counter()

    def ok(st):
        return all(st[j] + p[j] <= T for j in range(n))

    best = None
    for o in seeds:
        v, st = decode(o, p, w, r, m)
        if ok(st) and (best is None or v < best[0]):
            best = (v, o)
    cur_v, cur = best
    gbest = best
    while time.perf_counter() - t0 < budget:
        improved = True
        while improved and time.perf_counter() - t0 < budget:   # first-improvement insertion LS
            improved = False
            for _ in range(4 * n):
                i, k = rng.randrange(n), rng.randrange(n)
                if i == k:
                    continue
                o = cur[:]
                o.insert(k, o.pop(i))
                v, st = decode(o, p, w, r, m)
                if v < cur_v and ok(st):
                    cur_v, cur, improved = v, o, True
                    break
        if cur_v < gbest[0]:
            gbest = (cur_v, cur)
        cur = gbest[1][:]                                           # kick from the best
        for _ in range(max(2, n // 20)):
            i, k = rng.randrange(n), rng.randrange(n)
            cur[i], cur[k] = cur[k], cur[i]
        cur_v, st = decode(cur, p, w, r, m)
        if not ok(st):
            cur_v, cur = gbest
    v, st = decode(gbest[1], p, w, r, m)
    return v, st


# ---------------------------------------------------------------------------
# Models (returning start -> variable maps for warm starts)
# ---------------------------------------------------------------------------

def build(name, p, w, r, m, T):
    import gurobipy as gp
    from gurobipy import GRB
    from fedhpc import formulations
    from fedhpc.formulations import SpaceTimeFormulation
    import bench_prjwc as bp

    env = formulations._shared_env
    n = len(p)
    if name == "TI":
        mdl = bp.build_ti(p, w, r, m, T, env)
        mdl.update()
        vs = mdl.getVars()
        keys = [(j, t) for j in range(n) for t in range(r[j], T - p[j] + 1)]
        return mdl, {k: v for k, v in zip(keys, vs)}
    if name == "AF":
        mdl = bp.build_af(p, w, r, m, T, env)
        mdl.update()
        _, arcs, _ = bp.normal_pattern_graph(p, r, T)
        vs = mdl.getVars()
        return mdl, {(j, q): v for (q, _rr, j), v in zip(arcs, vs)}
    fmt = (SpaceTimeFormulation() if name == "fed_st"
           else SpaceTimeFormulation(cover_ge=True, compressed_flow=True))
    inst = bp.fed_instance(p, w, r, m, T)
    mdl, vars_ = fmt.build(inst)
    mdl.setObjective(fmt.f1_expr(inst, vars_["x"]) + sum(a * b for a, b in zip(w, r)))
    mdl.update()
    return mdl, {(j, t): v for (j, _m, t), v in vars_["x"].items()}


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

def _init(q: Queue):
    cpu = q.get()
    os.sched_setaffinity(0, {cpu})
    os.environ["FAIR_CPU"] = str(cpu)
    from fedhpc import formulations
    formulations.configure_env(verbose=False)


def run_task(task):
    import gurobipy as gp
    from gurobipy import GRB

    p, w, r, m = read_instance(Path(task["path"]))
    T = horizon(p, r, m)
    t0 = time.perf_counter()
    mdl, xmap = build(task["model"], p, w, r, m, T)
    build_s = time.perf_counter() - t0
    params = {"OutputFlag": 0, "Threads": 1, "MIPGap": 0, "Seed": 0,
              "TimeLimit": task["time_limit"], **task["params"]}
    for k, v in params.items():
        mdl.setParam(k, v)

    lp = None
    if task.get("lp"):
        rel = mdl.relax()
        rel.setParam("TimeLimit", task["time_limit"])
        rel.optimize()
        lp = rel.ObjVal if rel.Status == GRB.OPTIMAL else None
        rel.dispose()

    ws_obj = ws_s = None
    if task["warm"]:
        h0 = time.perf_counter()
        ws_obj, st = heuristic(p, w, r, m, T, budget=len(p) / 5)
        ws_s = time.perf_counter() - h0
        for v in mdl.getVars():
            v.Start = GRB.UNDEFINED
        miss = 0
        for j, s in enumerate(st):
            v = xmap.get((j, s))
            if v is None:
                miss += 1
            else:
                v.Start = 1.0
        if miss:
            raise RuntimeError(f"{task['model']}: {miss} warm-start starts not in model")

    info = {"root_bound": None, "root_s": None, "first": None, "best": None}

    def cb(model, where):
        if where == GRB.Callback.MIP:
            nd = model.cbGet(GRB.Callback.MIP_NODCNT)
            if nd == 0:
                info["root_bound"] = model.cbGet(GRB.Callback.MIP_OBJBND)
            elif info["root_s"] is None:
                info["root_s"] = model.cbGet(GRB.Callback.RUNTIME)
        elif where == GRB.Callback.MIPSOL:
            rt = model.cbGet(GRB.Callback.RUNTIME)
            if info["first"] is None:
                info["first"] = rt
            info["best"] = rt
        elif where == GRB.Callback.MIPNODE and model.cbGet(GRB.Callback.MIPNODE_NODCNT) == 0:
            info["root_bound"] = model.cbGet(GRB.Callback.MIPNODE_OBJBND)

    t1 = time.perf_counter()
    mdl.optimize(cb)
    solve_s = time.perf_counter() - t1
    has = mdl.SolCount > 0
    status = {GRB.OPTIMAL: "optimal", GRB.TIME_LIMIT: "time_limit"}.get(mdl.Status, str(mdl.Status))
    if status == "optimal" and info["root_s"] is None:
        info["root_s"] = solve_s
    obj = mdl.ObjVal if has else None
    bound = mdl.ObjBound
    if info["root_bound"] is None:
        info["root_bound"] = bound
    row = dict(task["meta"], T=T, params=json.dumps(task["params"], sort_keys=True),
               warm=int(task["warm"]), ws_obj=ws_obj,
               ws_s=None if ws_s is None else round(ws_s, 2),
               n_vars=mdl.NumVars, n_cons=mdl.NumConstrs, n_nz=mdl.NumNZs, lp=lp,
               root_bound=info["root_bound"], root_s=info["root_s"], obj=obj, bound=bound,
               gap=None if obj is None else (obj - bound) / obj,
               status=status, nodes=int(mdl.NodeCount), first_sol_s=info["first"],
               best_sol_s=info["best"], build_s=round(build_s, 3), solve_s=round(solve_s, 3),
               cpu=os.environ.get("FAIR_CPU"))
    mdl.dispose()
    return row


# ---------------------------------------------------------------------------
# Driver helpers
# ---------------------------------------------------------------------------

def make_task(inst_dir, phase, config, model, n, m, z, i, params, warm, tl, lp=False):
    path = inst_path(inst_dir, n, m, z, i)
    return dict(path=str(path), model=model, params=params, warm=warm, time_limit=tl, lp=lp,
                meta=dict(phase=phase, config=config, model=model, inst=path.stem, n=n, m=m,
                          alpha=ALPHA_CODE[z], id=i))


def key_of(row):
    return (row["phase"], row["config"], row["model"], row["inst"])


def execute(tasks, out: Path, workers: int):
    done = set()
    if out.exists():
        with open(out) as f:
            done = {key_of(r) for r in csv.DictReader(f)}
    todo = [t for t in tasks if key_of({**t["meta"]}) not in done]
    random.Random(1).shuffle(todo)          # interleave models/configs over time
    print(f"{len(todo)} runs to do ({len(tasks) - len(todo)} already in {out.name})", flush=True)
    if not todo:
        return
    new = not out.exists()
    q = Queue()
    for c in CPUS[:workers]:
        q.put(c)
    with open(out, "a", newline="") as fh, Pool(workers, _init, (q,)) as pool:
        wr = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            wr.writeheader()
        for k, row in enumerate(pool.imap_unordered(run_task, todo), 1):
            wr.writerow(row)
            fh.flush()
            print(f"[{k}/{len(todo)}] {row['phase']}:{row['config']:<10} {row['model']:<7} "
                  f"{row['inst']:<16} {row['status']:<10} obj={row['obj']} "
                  f"nodes={row['nodes']} b={row['build_s']}s s={row['solve_s']}s", flush=True)


def load(out: Path):
    with open(out) as f:
        return list(csv.DictReader(f))


def score(rows, tl):
    """Shifted geometric mean (shift 10 s) of solve time, time-outs as 2·TL (PAR2)."""
    ts = [float(r["solve_s"]) if r["status"] == "optimal" else 2 * tl for r in rows]
    return math.exp(sum(math.log(t + 10) for t in ts) / len(ts)) - 10


# ---------------------------------------------------------------------------
# Tuning (coordinate descent, one pass, same candidate grid for every model)
# ---------------------------------------------------------------------------

GRID = [
    ("Method", [-1, 1, 2]),          # root LP algorithm (concurrent is moot at 1 thread)
    ("DegenMoves", [-1, 0]),
    ("Presolve", [-1, 0, 2]),
    ("MIPFocus", [0, 1, 2, 3]),
    ("Cuts", [-1, 0, 2]),
    ("Heuristics", [0.05, 0.0, 0.2]),
    ("Symmetry", [-1, 0, 2]),
]
TUNE_SET = [(n, m, z, 3) for n in (50, 100) for m in (2, 5) for z in (0, 4, 9)]


def cfg_name(params):
    return "default" if not params else ",".join(f"{k}={v}" for k, v in sorted(params.items()))


def tune(args):
    inst_dir, out, tl = Path(args.inst_dir), Path(args.out), args.time_limit
    models = args.models.split(",")
    best = {mdl: {} for mdl in models}
    # round 0: defaults, plus the warm-start pilot on the same instances
    tasks = []
    for mdl in models:
        for (n, m, z, i) in TUNE_SET:
            tasks.append(make_task(inst_dir, "tune", "default", mdl, n, m, z, i, {}, False, tl))
            tasks.append(make_task(inst_dir, "tune", "default+ws", mdl, n, m, z, i, {}, True, tl))
    execute(tasks, out, args.workers)
    for pname, values in GRID:
        tasks = []
        for mdl in models:
            for v in values:
                params = dict(best[mdl])
                if v == GRID_DEFAULT[pname]:
                    params.pop(pname, None)
                else:
                    params[pname] = v
                for (n, m, z, i) in TUNE_SET:
                    tasks.append(make_task(inst_dir, "tune", cfg_name(params), mdl, n, m, z, i,
                                           params, False, tl))
        execute(tasks, out, args.workers)
        rows = load(out)
        for mdl in models:
            scores = {}
            for v in values:
                params = dict(best[mdl])
                if v == GRID_DEFAULT[pname]:
                    params.pop(pname, None)
                else:
                    params[pname] = v
                rs = [r for r in rows if r["phase"] == "tune" and r["model"] == mdl
                      and r["config"] == cfg_name(params)]
                scores[v] = (score(rs, tl), params)
            cur = scores[best[mdl].get(pname, GRID_DEFAULT[pname])][0]
            v_best = min(scores, key=lambda v: scores[v][0])
            if scores[v_best][0] < 0.98 * cur:          # keep incumbent unless >2 % better
                best[mdl] = scores[v_best][1]
            print(f"  {mdl:<7} {pname:<11} " + "  ".join(
                f"{v}:{s:.1f}" for v, (s, _) in scores.items()) + f"  -> {best[mdl]}", flush=True)
    (out.parent / "tuned_params.json").write_text(json.dumps(best, indent=2, sort_keys=True))
    print(json.dumps(best, indent=2))


GRID_DEFAULT = {k: v[0] for k, v in GRID}


def final(args):
    inst_dir, out, tl = Path(args.inst_dir), Path(args.out), args.time_limit
    tuned = json.loads(Path(args.tuned).read_text())
    configs = args.configs.split(",")
    tasks = []
    for mdl in args.models.split(","):
        for cfg in configs:
            params = {} if cfg.startswith("default") else tuned[mdl]
            warm = cfg.endswith("+ws")
            for n in (20, 50, 100):
                for m in (2, 3, 5, 10):
                    for z in ALPHA_CODE:
                        for i in (1, 2):
                            tasks.append(make_task(inst_dir, "test", cfg, mdl, n, m, z, i,
                                                   params, warm, tl, lp=True))
    execute(tasks, out, args.workers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tune", "run"])
    ap.add_argument("--inst-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--time-limit", type=float, default=1800.0)
    ap.add_argument("--tuned", default=None)
    ap.add_argument("--configs", default="default,tuned,tuned+ws")
    args = ap.parse_args()
    tune(args) if args.cmd == "tune" else final(args)


if __name__ == "__main__":
    main()
