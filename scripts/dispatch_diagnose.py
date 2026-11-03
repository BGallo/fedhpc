"""Why is weighted_solve_dispatch far from the exact weighted-sum optimum on
the generated instances (results/generated_bench/bench_io_noreduc.jsonl)?

Per (instance, λ) of a sample, against the exact optimum (re-solved with the
reduced SpaceTime model — same optimum as the bench's plain one, faster, and
it gives the assignment the bench did not store):

  dispatch       the bench configuration, with its final genome and stats
  oracle_auto    decode of the exact on-prem/cloud split, on-prem jobs `auto`
  oracle_forced  … on-prem jobs forced on-prem (gene 2)
  oracle_order   … forced, dispatched in the exact solution's start order
  why            per-reason count of the cloud jobs in each decode
                 (gene / deadline offload at placement / while waiting /
                 horizon / leftover), via _ext.dispatch_decode

If the oracle decodes are near g* the decoder can represent the optimum and
the search misses it; if not, the decoder is the ceiling, and oracle_order
tells whether the dispatch order is what breaks it.

    uv run python scripts/dispatch_diagnose.py [--workers 3] [--threads 4]
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
BENCH = ROOT / "results" / "generated_bench" / "bench_io_noreduc.jsonl"
OUT = ROOT / "results" / "generated_bench" / "dispatch_study"

SAMPLE = ["A_pos_n100_lE_iM", "A_pos_n100_lF_iM", "A_pos_n300_lH_iM", "A_pos_n500_lE_iM",
          "B_pos_n100_lE_iM", "B_pos_n100_lF_iM", "B_pos_n300_lH_iM",
          "A_fedhpc_n100_lE_iM", "B_fedhpc_n100_lH_iM"]
LAMS = ("0.25", "0.5", "0.75")
WHY = ("onprem", "gene", "off_place", "off_wait", "horizon", "leftover")


def _g(lam, f1, f2, ref):
    f1_T, f2_T, f1_0 = ref
    return lam * (f1 - f1_T) / (f1_0 - f1_T) + (1 - lam) * f2 / f2_T


def _split(lam, f1, f2, ref):
    """(turnaround term, cost term) of g."""
    f1_T, f2_T, f1_0 = ref
    return lam * (f1 - f1_T) / (f1_0 - f1_T), (1 - lam) * f2 / f2_T


def _cloud_ids(inst):
    return {m.id for m in inst.instance_types if m.capacity is None}


def run_one(rec, threads):
    from fedhpc import _moea as ext
    from fedhpc import instance_gen as gen
    from fedhpc.data import Instance
    from fedhpc.model import solve_weighted_sum
    from fedhpc.moea import _dispatch_jobs, _heuristic_weights, _init_occ, _type_cap, \
        weighted_solve_dispatch

    inst = Instance.from_file(gen.DATA / "generated" / f"{rec['name']}.json")
    ref = (rec["lex_f1"]["f1"], rec["lex_f1"]["f2"], rec["lex_f2"]["f1"])
    cloud = _cloud_ids(inst)
    jobs = _dispatch_jobs(inst)
    tc, occ = _type_cap(inst), _init_occ(inst)
    out = dict(name=rec["name"], cls=rec["cls"], workload=rec["workload"], count=rec["count"],
               load=rec["load"], io=rec["io"], lams={})
    exact_store = {}
    for lam in LAMS:
        L = float(lam)
        _, (w1, w2, f1_cap) = _heuristic_weights(inst, L, *ref, 0.0, seed=42, n_threads=threads)
        r = dict(g_bench=rec["lams"][lam]["gurobi"]["g"], g_bench_dispatch=rec["lams"][lam]["dispatch"]["g"])

        t = time.perf_counter()
        ex = solve_weighted_sum(inst, L, *ref, f2_0=0.0, OutputFlag=0, Threads=threads,
                                TimeLimit=600, MIPGap=0)
        r["exact"] = dict(status=ex.status, f1=ex.f1, f2=ex.f2, g=_g(L, ex.f1, ex.f2, ref),
                          split=_split(L, ex.f1, ex.f2, ref), t=time.perf_counter() - t,
                          n_cloud=sum(m in cloud for m, _ in ex.assignment.values()))
        exact_store[lam] = {str(j): list(v) for j, v in ex.assignment.items()}

        t = time.perf_counter()
        hd, st = weighted_solve_dispatch(inst, L, f1_T=ref[0], f2_T=ref[1], f1_0=ref[2],
                                         n_threads=threads, return_stats=True)
        r["dispatch"] = dict(f1=hd.f1, f2=hd.f2, g=_g(L, hd.f1, hd.f2, ref),
                             split=_split(L, hd.f1, hd.f2, ref), t=time.perf_counter() - t,
                             n_cloud=sum(m in cloud for m, _ in hd.assignment.values()),
                             stats={k: v for k, v in st.items() if k not in ("trace", "gen")},
                             trace=st["trace"])
        mu = st["mu"]

        def decode(genome, order=()):
            asgn, f1, f2, _pen, why, cv = ext.dispatch_decode(
                H=inst.horizon, jobs=jobs, type_cap=tc, init_occ=occ, w1=w1, w2=w2,
                f1_cap=f1_cap, gen=genome, mu=mu, flags=64, order=list(order))
            cnt = {w: 0 for w in WHY}
            for x in why:
                cnt[WHY[x]] += 1
            return dict(f1=f1, f2=f2, g=_g(L, f1, f2, ref), split=_split(L, f1, f2, ref), cv=cv,
                        why=cnt, n_cloud=sum(m in cloud for m, _ in asgn))

        idx = {j.id: k for k, j in enumerate(inst.jobs)}
        ex_cloud = [0] * len(inst.jobs)
        for jid, (m, _s) in ex.assignment.items():
            ex_cloud[idx[jid]] = int(m in cloud)
        r["dispatch_redecode"] = decode(st["gen"])
        r["oracle_auto"] = decode([1 if c else 0 for c in ex_cloud])
        r["oracle_forced"] = decode([1 if c else 2 for c in ex_cloud])
        start = {idx[jid]: s for jid, (_m, s) in ex.assignment.items()}
        order = sorted(range(len(inst.jobs)), key=lambda k: (ex_cloud[k], start[k], inst.jobs[k].arrival))
        r["oracle_order"] = decode([1 if c else 2 for c in ex_cloud], order)

        # which jobs differ: exact on-prem but heuristic cloud, and the reverse
        d_cloud = [int(m in cloud) for m, _ in (hd.assignment[j.id] for j in inst.jobs)]
        def agg(ks):
            if not ks:
                return dict(n=0)
            ex_t = [inst.jobs[k].exec_time for k in ks]
            cost = [min(inst.c[inst.jobs[k].id, m] for m in inst.F[inst.jobs[k].id] if m in cloud) for k in ks]
            return dict(n=len(ks), exec_med=sorted(ex_t)[len(ks) // 2], cloud_cost_sum=sum(cost))
        r["diff"] = dict(
            heur_cloud_exact_onprem=agg([k for k in range(len(d_cloud)) if d_cloud[k] and not ex_cloud[k]]),
            exact_cloud_heur_onprem=agg([k for k in range(len(d_cloud)) if ex_cloud[k] and not d_cloud[k]]))
        out["lams"][lam] = r
        print(f"{rec['name']} λ={lam} g*={r['exact']['g']:.4f} D={r['dispatch']['g']:.4f} "
              f"auto={r['oracle_auto']['g']:.4f} forced={r['oracle_forced']['g']:.4f} "
              f"order={r['oracle_order']['g']:.4f}", flush=True)
    (OUT / "exact").mkdir(parents=True, exist_ok=True)
    (OUT / "exact" / f"{rec['name']}.json").write_text(json.dumps(exact_store))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("names", nargs="*")
    args = ap.parse_args()
    bench = {r["name"]: r for r in map(json.loads, BENCH.read_text().splitlines())}
    names = args.names or SAMPLE
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "diagnose.jsonl"
    with ProcessPoolExecutor(args.workers) as ex, open(path, "a") as f:
        futs = {ex.submit(run_one, bench[n], args.threads): n for n in names}
        for fut in as_completed(futs):
            try:
                f.write(json.dumps(fut.result()) + "\n"); f.flush()
            except Exception as e:  # noqa: BLE001
                print(f"{futs[fut]} FAILED {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
