"""Benchmark fedhpc.mip against HiGHS and Gurobi on identical MPS files.

Instances
---------
- Kramer et al. (2020) P|r_j|sum w_j C_j: the time-indexed (TI) and
  normal-pattern arc-flow (AF) models of scripts/bench_prjwc.py, written to
  MPS by ``build`` (needs the archive extracted: --kramer-dir).
- MIPLIB 2017 benchmark instances of the scheduling and packing families
  (benchmarks/mip/miplib2017/benchmark-v2/).

Protocol: every solver gets one thread (fedlp's own threads too), the same
time limit and relative gap 1e-4, and each run is a subprocess pinned to one
core (a hard kill at 1.5x the limit + 60 s backstops a solver that ignores
its limit). Results append to a JSONL file; finished (instance, solver)
pairs are skipped, so an interrupted run resumes.

    uv run python scripts/mip_bench.py build --kramer-dir /tmp/fedhpc_kramer
    uv run python scripts/mip_bench.py run --set kramer20 --time-limit 60 --label k20
    uv run python scripts/mip_bench.py report --label k20

Report: per instance, each solver's status, objective, bound, gap and time;
then per solver the count solved to the gap, the shifted geometric mean of
time (shift 1 s, unsolved at the limit) and the mean primal-dual gap.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results" / "mip_bench"
MPS = OUT / "mps"
MIPLIB = ROOT / "benchmarks" / "mip" / "miplib2017" / "benchmark-v2"
CPUS = [0, 2, 4, 6, 12, 14]           # one hyperthread per physical P-core (as bench_kramer_fair)

KRAMER_GRID = {
    "kramer20": [(20, m, z, 1) for m in (2, 5, 10) for z in (0, 4, 9)],
    "kramer50": [(50, m, z, 1) for m in (2, 5, 10) for z in (0, 4, 9)],
    "kramer100": [(100, m, z, 1) for m in (2, 5, 10) for z in (0, 4, 9)],
}
MIPLIB_SETS = {
    "miplib_sched": ["csched007", "csched008", "mcsched", "lectsched-5-obj", "nursesched-sprint02",
                     "physiciansched3-3", "gfd-schedulen180f7d50m30k18"],
    "miplib_pack": ["bppc4-08", "fhnw-binpack4-4", "fhnw-binpack4-48"],
    # general MIPLIB 2017 benchmark instances of small to medium size, from
    # families where presolve (aggregation, probing) is known to matter:
    # lot sizing, fixed charge, network design, knapsack/assignment structure
    "miplib_general": ["neos5", "gen-ip054", "timtab1", "mas76", "mas74", "gen-ip002", "glass4",
                       "assign1-5-8", "mad", "tr12-30", "exp-1-500-5-5", "binkar10_1", "beasleyC3",
                       "p200x1188c", "pg", "mc11", "pg5_34", "lotsize", "neos17", "mik-250-20-75-4",
                       "sp150x300d", "neos-911970", "50v-10", "neos859080"],
}
SOLVERS = ("fedmip", "highs", "gurobi")


def instances(name: str) -> list[Path]:
    if name in KRAMER_GRID:
        return [MPS / f"{model}_N{n}_M{m}_R{z}_{i}.mps.gz"
                for (n, m, z, i) in KRAMER_GRID[name] for model in ("AF", "TI")]
    if name in MIPLIB_SETS:
        return [MIPLIB / f"{s}.mps.gz" for s in MIPLIB_SETS[name]]
    raise SystemExit(f"unknown set {name}")


# --------------------------------------------------------------------- build
def build(args) -> None:
    import bench_kramer_fair as bk
    from fedhpc import formulations
    formulations.configure_env()
    MPS.mkdir(parents=True, exist_ok=True)
    for grid in KRAMER_GRID.values():
        for (n, m, z, i) in grid:
            p, w, r, mm = bk.read_instance(bk.inst_path(Path(args.kramer_dir), n, m, z, i))
            T = bk.horizon(p, r, mm)
            for model in ("AF", "TI"):
                out = MPS / f"{model}_N{n}_M{m}_R{z}_{i}.mps.gz"
                if out.exists():
                    continue
                mdl, _ = bk.build(model, p, w, r, mm, T)
                mdl.ModelName = out.name.split(".")[0]
                mdl.write(str(out))
                print("wrote", out.name, flush=True)


# ----------------------------------------------------------------------- run
def _solve(solver: str, path: str, tl: float, gap: float) -> dict:
    t0 = time.perf_counter()
    if solver.startswith("fedmip"):
        # "fedmip" or "fedmip[k=v;k=v]" (MipOptions; presolve.<k>=v goes to presolve())
        import ast
        from fedhpc.mip import read_mps, solve
        kw, pre = {}, {}
        if "[" in solver:
            for kv in solver[solver.index("[") + 1:-1].split(";"):
                k, v = kv.split("=")
                val = ast.literal_eval(v)
                if k.startswith("presolve."):
                    pre[k[9:]] = val
                else:
                    kw[k] = val
        P = read_mps(path)
        r = solve(P, time_limit=tl, gap_rel=gap, verbose=False, lp_options={"threads": 1},
                  presolve_options=pre, **kw)
        st = r.stats
        return dict(status=r.status, objective=r.objective, bound=r.bound, nodes=r.nodes,
                    seconds=time.perf_counter() - t0,
                    extra={k: st.get(k) for k in ("root_bound_no_cuts", "root_bound", "cuts_added", "presolve",
                                                  "rc_fixed", "restart",
                                                  "lp_iterations", "lp_seconds", "strong_iterations",
                                                  "heuristic_solutions", "root_lp_algorithm", "time",
                                                  "network_rows", "implied_integers")})
    if solver == "highs":
        import highspy
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("threads", 1)
        h.setOptionValue("time_limit", float(tl))
        h.setOptionValue("mip_rel_gap", gap)
        h.readModel(path)
        h.run()
        info = h.getInfo()
        ms = h.getModelStatus()
        name = h.modelStatusToString(ms).lower()
        status = {"optimal": "optimal", "infeasible": "infeasible", "time limit reached": "time_limit"}.get(name, name)
        has = info.primal_solution_status == 2
        return dict(status=status, objective=info.objective_function_value if has else None,
                    bound=info.mip_dual_bound, nodes=int(info.mip_node_count),
                    seconds=time.perf_counter() - t0)
    if solver == "gurobi":
        import gurobipy as gp
        from fedhpc import formulations
        formulations.configure_env()
        g = gp.read(path, env=formulations._shared_env)
        g.Params.OutputFlag = 0
        g.Params.Threads = 1
        g.Params.TimeLimit = tl
        g.Params.MIPGap = gap
        g.optimize()
        status = {2: "optimal", 3: "infeasible", 9: "time_limit"}.get(g.Status, f"status{g.Status}")
        return dict(status=status, objective=g.ObjVal if g.SolCount else None, bound=g.ObjBound,
                    nodes=int(g.NodeCount), seconds=time.perf_counter() - t0)
    raise ValueError(solver)


def _worker(task, q):
    solver, path, tl, gap, cpu = task
    try:
        os.sched_setaffinity(0, {cpu})
    except (AttributeError, OSError):
        pass
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        q.put(_solve(solver, path, tl, gap))
    except Exception as e:  # noqa: BLE001 - recorded, the bench goes on
        import traceback
        q.put(dict(status="error", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-2000:]))


def run_one(solver, path, tl, gap, cpu) -> dict:
    """One run in a fresh process. The result is read before joining (a
    child cannot exit while its queue's feeder still holds data); a child
    alive past the hard limit is killed."""
    import queue as _queue
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=((solver, str(path), tl, gap, cpu), q))
    t0 = time.perf_counter()
    p.start()
    hard = 1.5 * tl + 60
    try:
        res = q.get(timeout=hard)
    except _queue.Empty:
        res = None
    p.join(10)
    if p.is_alive():
        p.kill()
        p.join()
    if res is None:
        return dict(status="killed" if time.perf_counter() - t0 >= hard else "crashed",
                    exitcode=p.exitcode, seconds=time.perf_counter() - t0)
    return res


def run(args) -> None:
    out = OUT / f"{args.label}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["instance"], r["solver"]))
    tasks = []
    for s in args.set:
        for path in instances(s):
            if not path.exists():
                print("missing", path, file=sys.stderr)
                continue
            for solver in args.solvers:
                key = (path.name.split(".")[0], solver)
                if key not in done:
                    tasks.append((key, s, solver, path))
    print(f"{len(tasks)} runs, {args.workers} at a time, {args.time_limit:g} s each", flush=True)
    from concurrent.futures import ThreadPoolExecutor
    cpus = CPUS[: args.workers]
    free = list(cpus)
    import threading
    lock = threading.Lock()

    def go(t):
        key, s, solver, path = t
        with lock:
            cpu = free.pop()
        try:
            r = run_one(solver, path, args.time_limit, args.gap, cpu)
        finally:
            with lock:
                free.append(cpu)
        r.update(instance=key[0], solver=solver, set=s, time_limit=args.time_limit, gap_target=args.gap)
        with lock:
            with out.open("a") as f:
                f.write(json.dumps(r, default=str) + "\n")
        print(f"{key[0]:34s} {solver:7s} {r.get('status')!s:11s} obj {r.get('objective')} "
              f"bound {r.get('bound')} {r.get('seconds', 0):.1f}s", flush=True)

    with ThreadPoolExecutor(max_workers=len(cpus)) as ex:
        list(ex.map(go, tasks))


# -------------------------------------------------------------------- report
def _gap(r) -> float:
    o, b = r.get("objective"), r.get("bound")
    if o is None or b is None:
        return 1.0
    if not (math.isfinite(o) and math.isfinite(b)):
        return 1.0
    return min(1.0, abs(o - b) / max(abs(o), 1e-9))


def report(args) -> None:
    rows = [json.loads(line) for line in (OUT / f"{args.label}.jsonl").read_text().splitlines()]
    by = {}
    for r in rows:
        by.setdefault(r["instance"], {})[r["solver"]] = r
    seen = {s for v in by.values() for s in v}
    solvers = sorted(seen, key=lambda s: (s not in SOLVERS, SOLVERS.index(s) if s in SOLVERS else 0, s))
    print(f"{'instance':34s} " + " ".join(f"{s:>30s}" for s in solvers))
    for inst in sorted(by):
        cells = []
        for s in solvers:
            r = by[inst].get(s)
            if r is None:
                cells.append(f"{'-':>30s}")
                continue
            o = r.get("objective")
            cells.append(f"{r['status'][:7]:>8s} {o if o is None else round(o, 2)!s:>11s} "
                         f"{_gap(r):7.1e} {r.get('seconds', 0):5.0f}s")
        print(f"{inst:34s} " + " ".join(cells))
    print()
    for s in solvers:
        rs = [v[s] for v in by.values() if s in v]
        tl = rs[0].get("time_limit", 60) if rs else 60
        solved = [r for r in rs if r["status"] == "optimal"]
        sgm = math.exp(sum(math.log(min(r.get("seconds", tl), tl) + 1.0) if r["status"] == "optimal"
                           else math.log(tl + 1.0) for r in rs) / max(1, len(rs))) - 1.0
        mg = sum(_gap(r) for r in rs) / max(1, len(rs))
        print(f"{s:8s} solved {len(solved):3d}/{len(rs):3d}  shifted geomean time {sgm:8.2f}s  mean gap {mg:.2e}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--kramer-dir", default="/tmp/fedhpc_kramer")
    r = sub.add_parser("run")
    r.add_argument("--set", nargs="+", default=["kramer20"])
    r.add_argument("--solvers", nargs="+", default=list(SOLVERS))
    r.add_argument("--time-limit", type=float, default=60.0)
    r.add_argument("--gap", type=float, default=1e-4)
    r.add_argument("--workers", type=int, default=6)
    r.add_argument("--label", required=True)
    rp = sub.add_parser("report")
    rp.add_argument("--label", required=True)
    args = ap.parse_args()
    {"build": build, "run": run, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
