"""Test why weighted_solve_dispatch is far from optimal on the generated
pos instances (results/generated_bench/).

Part 1 (--part flags): the decoder's on-prem tie-break — default (lowest type
index = smallest shape first), DF_TIE_MOST_FREE, DF_TIE_BEST_FIT, and DF_NO_DEADLINE (``auto`` jobs never
offloaded by the decoder's per-job deadline rule) — on every
A·pos and B·pos instance at λ = 0.25 / 0.5 / 0.75, gap to the exact g* from
bench.jsonl.

Part 2 (--part decomp): on a subset of A·pos instances, re-solve the weighted
sum exactly and split g(heuristic) − g* by job: per job
g_j = w1·w_j·(C_j − a_j) + w2·c_j, grouped into jobs that only fit the large
on-prem type ("big-only") and the rest, and record how many small-fitting
jobs each solution places on the large type.

    uv run python scripts/dispatch_tiebreak_ablation.py --part flags
    uv run python scripts/dispatch_tiebreak_ablation.py --part decomp
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "results" / "generated_bench"
BENCH = OUT / "bench.jsonl"

NO_DEADLINE, MOST_FREE, BEST_FIT, MIX_RANK = 1, 2, 4, 64
VARIANTS = {"default": MIX_RANK, "most_free": MIX_RANK | MOST_FREE, "best_fit": MIX_RANK | BEST_FIT,
            "no_deadline": MIX_RANK | NO_DEADLINE}
LAMS = ("0.25", "0.5", "0.75")


def _weights(lam, ref):
    f1_T, f2_T, f1_0 = ref
    return lam / (f1_0 - f1_T), (1 - lam) / f2_T


def _g(lam, f1, f2, ref):
    f1_T, f2_T, f1_0 = ref
    return lam * (f1 - f1_T) / (f1_0 - f1_T) + (1 - lam) * f2 / f2_T


def _inst(name):
    from fedhpc import instance_gen as g
    from fedhpc.data import Instance
    return Instance.from_file(g.DATA / "generated" / f"{name}.json")


def run_flags(rec, variant, threads):
    from fedhpc.moea import weighted_solve_dispatch
    inst = _inst(rec["name"])
    ref = (rec["lex_f1"]["f1"], rec["lex_f1"]["f2"], rec["lex_f2"]["f1"])
    out = dict(name=rec["name"], cls=rec["cls"], variant=variant, lams={})
    for lam in LAMS:
        t = time.perf_counter()
        s = weighted_solve_dispatch(inst, float(lam), f1_T=ref[0], f2_T=ref[1], f1_0=ref[2],
                                    flags=VARIANTS[variant], n_threads=threads)
        inst.check_schedule(s.assignment)
        g = _g(float(lam), s.f1, s.f2, ref)
        gstar = rec["lams"][lam]["gurobi"]["g"]
        out["lams"][lam] = dict(g=g, gstar=gstar, gap=100 * (g - gstar) / gstar,
                                t=time.perf_counter() - t)
    return out


def _split(inst, sol, lam, ref, big):
    """Per-group Σ g_j and placements of a solution."""
    w1, w2 = _weights(lam, ref)
    small_ids = {j.id for j in inst.jobs if 0 in inst.F[j.id]}   # fits the small type
    grp = dict(big_only=0.0, small=0.0)
    n = dict(small_on_big=0, big_only_cloud=0, big_only_wait=0.0, small_cloud=0)
    for j in inst.jobs:
        m, t = sol.assignment[j.id]
        gj = w1 * j.weight * (t + inst.p_occ[j.id, m] - j.arrival) + w2 * inst.c[j.id, m]
        key = "small" if j.id in small_ids else "big_only"
        grp[key] += gj
        cloud = inst.instance_types[m].capacity is None
        if key == "small":
            n["small_on_big"] += m == big
            n["small_cloud"] += cloud
        else:
            n["big_only_cloud"] += cloud
            n["big_only_wait"] += t - max(0, math.ceil(j.arrival))
    return grp, n, len(inst.jobs) - len(small_ids)


def run_decomp(rec, threads):
    from fedhpc.model import solve_weighted_sum
    from fedhpc.moea import weighted_solve_dispatch
    inst = _inst(rec["name"])
    ref = (rec["lex_f1"]["f1"], rec["lex_f1"]["f2"], rec["lex_f2"]["f1"])
    big = 1                                     # class A: type 1 = c64
    out = dict(name=rec["name"], lams={})
    for lam in LAMS:
        ex = solve_weighted_sum(inst, float(lam), *ref, f2_0=rec["lex_f2"]["f2"],
                                OutputFlag=0, Threads=threads, MIPGap=0, TimeLimit=600)
        he = weighted_solve_dispatch(inst, float(lam), f1_T=ref[0], f2_T=ref[1], f1_0=ref[2],
                                     n_threads=threads)
        ge, ne, n_big = _split(inst, ex, float(lam), ref, big)
        gh, nh, _ = _split(inst, he, float(lam), ref, big)
        out["lams"][lam] = dict(status=ex.status, n_big_only=n_big,
                                g_exact=_g(float(lam), ex.f1, ex.f2, ref),
                                g_heur=_g(float(lam), he.f1, he.f2, ref),
                                g_bench_heur=rec["lams"][lam]["dispatch"]["g"],
                                exact=dict(groups=ge, counts=ne), heur=dict(groups=gh, counts=nh))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("flags", "decomp"), required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    rows = [json.loads(l) for l in BENCH.read_text().splitlines() if l.strip()]
    pos = [r for r in rows if r["workload"] == "pos" and r["lex_f1"]["f2"] > 0]
    path = OUT / f"dispatch_ablation_{args.part}.jsonl"
    done = set()
    if path.exists():
        done = {(d["name"], d.get("variant")) for d in map(json.loads, path.read_text().splitlines())}
    jobs = []
    if args.part == "flags":
        jobs = [(run_flags, (r, v, args.threads)) for r in pos for v in ("most_free", "best_fit", "no_deadline")
                if (r["name"], v) not in done]
    else:
        a = sorted((r for r in pos if r["cls"] == "A"), key=lambda r: r["name"])
        subset = a[::7][:12]
        jobs = [(run_decomp, (r, args.threads)) for r in subset if (r["name"], None) not in done]
    print(f"{len(jobs)} tasks", flush=True)
    with ProcessPoolExecutor(args.workers) as ex, open(path, "a") as f:
        futs = {ex.submit(fn, *a): a[0]["name"] for fn, a in jobs}
        for k, fut in enumerate(as_completed(futs), 1):
            try:
                f.write(json.dumps(fut.result()) + "\n"); f.flush()
                print(f"[{k}/{len(jobs)}] {futs[fut]}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[{k}/{len(jobs)}] {futs[fut]} FAILED {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
