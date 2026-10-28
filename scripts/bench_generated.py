"""Benchmark the generated instance classes (fedhpc.instance_gen): exact
anchors and weighted-sum solves (Gurobi) against the dispatch heuristic and
the list-scheduling baselines, recording quality and wall time.

Per instance, one JSON line in results/generated_bench/<run>.jsonl:
  lex_f1  — Gurobi lex-min (f1, f2): f1_T, f2_T, time
  lex_f2  — Gurobi lex-min (f2, f1): f2_0, f1_0, time
  per λ   — Gurobi weighted sum (with the reference points above), the
            dispatch heuristic (weighted_solve_dispatch) and the best
            threshold-bursting baseline on the same scalar g(λ)
  baselines — greedy / on-prem-only / threshold sweep (f1, f2, time)

Instances are solved in parallel worker processes, each pinned to THREADS
Gurobi / heuristic threads so the timings are comparable.

    uv run python scripts/bench_generated.py [--names-from manifest] [--workers 4]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

LAMS = (0.25, 0.5, 0.75)
THETAS = (0, 1, 2, 3, 6, 12, 24, 48, 96)


def _g(lam, f1, f2, ref):
    f1_T, f2_T, f1_0 = ref
    if f1_0 <= f1_T + 1e-9 or f2_T <= 1e-9:
        return f2
    return lam * (f1 - f1_T) / (f1_0 - f1_T) + (1 - lam) * f2 / f2_T


def run_one(name: str, threads: int, time_limit: float) -> dict:
    from fedhpc import baselines
    from fedhpc import instance_gen as g
    from fedhpc.data import Instance
    from fedhpc.model import solve_f1, solve_f2, solve_weighted_sum
    from fedhpc.moea import weighted_solve_dispatch

    path = g.DATA / "generated" / f"{name}.json"
    inst = Instance.from_file(path) if path.exists() else g.build(name)
    spec = g.InstanceSpec.parse(name)
    grb = dict(OutputFlag=0, Threads=threads, TimeLimit=time_limit, MIPGap=0)
    rec = dict(name=name, cls=spec.cls, workload=spec.workload, counts=list(spec.counts),
               loads="".join(spec.loads), n_jobs=len(inst.jobs),
               n_running=sum(inst.n_running.values()), onprem_nodes=sum(spec.counts))

    t = time.perf_counter(); s1 = solve_f1(inst, **grb)
    rec["lex_f1"] = dict(status=s1.status, f1=s1.f1, f2=s1.f2, t=time.perf_counter() - t)
    t = time.perf_counter(); s2 = solve_f2(inst, hint=None, **grb)
    rec["lex_f2"] = dict(status=s2.status, f1=s2.f1, f2=s2.f2, t=time.perf_counter() - t)
    for s in (s1, s2):
        inst.check_schedule(s.assignment)
    ref = (s1.f1, s1.f2, s2.f1)

    base = {}
    t = time.perf_counter(); b = baselines.greedy_earliest_completion(inst)
    base["greedy"] = dict(f1=b.f1, f2=b.f2, t=time.perf_counter() - t)
    t = time.perf_counter(); b = baselines.onprem_only_spt(inst)
    base["onprem"] = dict(f1=b.f1, f2=b.f2, t=time.perf_counter() - t,
                          complete=len(b.assignment) == len(inst.jobs))
    sweep = []
    t = time.perf_counter()
    for th in THETAS:
        b = baselines.threshold_bursting(inst, th)
        sweep.append(dict(theta=th, f1=b.f1, f2=b.f2))
    base["threshold"] = dict(points=sweep, t=time.perf_counter() - t)
    rec["baselines"] = base

    rec["lams"] = {}
    for lam in LAMS:
        r = {}
        t = time.perf_counter()
        sw = solve_weighted_sum(inst, lam, *ref, f2_0=s2.f2, **grb)
        r["gurobi"] = dict(status=sw.status, f1=sw.f1, f2=sw.f2, g=_g(lam, sw.f1, sw.f2, ref),
                           t=time.perf_counter() - t)
        inst.check_schedule(sw.assignment)
        t = time.perf_counter()
        hd = weighted_solve_dispatch(inst, lam, f1_T=ref[0], f2_T=ref[1], f1_0=ref[2],
                                     n_threads=threads)
        r["dispatch"] = dict(f1=hd.f1, f2=hd.f2, g=_g(lam, hd.f1, hd.f2, ref),
                             t=time.perf_counter() - t)
        inst.check_schedule(hd.assignment)
        best = min(sweep, key=lambda p: _g(lam, p["f1"], p["f2"], ref))
        r["threshold"] = dict(theta=best["theta"], f1=best["f1"], f2=best["f2"],
                              g=_g(lam, best["f1"], best["f2"], ref), t=base["threshold"]["t"])
        rec["lams"][str(lam)] = r
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--manifest", action="store_true", help="all instances in data/generated")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--time-limit", type=float, default=600)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "generated_bench" / "bench.jsonl")
    args = ap.parse_args()
    names = list(args.names)
    if args.manifest:
        names += sorted(json.loads((ROOT / "data" / "generated" / "manifest.json").read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        done = {json.loads(l)["name"] for l in args.out.read_text().splitlines() if l.strip()}
    todo = [n for n in names if n not in done]
    # slowest-first: pos before fedhpc, class B before A
    todo.sort(key=lambda n: (("pos" not in n), not n.startswith("B"), n))
    print(f"{len(todo)} to run ({len(done)} already in {args.out.name})", flush=True)
    with ProcessPoolExecutor(args.workers) as ex, open(args.out, "a") as f:
        futs = {ex.submit(run_one, n, args.threads, args.time_limit): n for n in todo}
        for k, fut in enumerate(as_completed(futs), 1):
            n = futs[fut]
            try:
                rec = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"[{k}/{len(todo)}] {n} FAILED {type(e).__name__}: {e}", flush=True)
                continue
            f.write(json.dumps(rec) + "\n"); f.flush()
            ts = rec["lex_f1"]["t"] + rec["lex_f2"]["t"] + sum(
                r["gurobi"]["t"] for r in rec["lams"].values())
            print(f"[{k}/{len(todo)}] {n} gurobi {ts:.0f}s", flush=True)


if __name__ == "__main__":
    main()
