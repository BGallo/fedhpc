"""Ablation of weighted_solve_dispatch on the generated instances: one knob
away from the bench configuration per variant, gap to the exact g* of
results/generated_bench/bench_io_noreduc.jsonl, wall time and search stats.

Variants (hypothesis tested in brackets, see dispatch_study/README.md):
  default      bench configuration (flags 64, K 256, 60 kicks of 16)
  budget4      K 1024, 240 kicks                       [Q1 search budget]
  n_mu3        start from the 3 best mu values          [Q1 search budget]
  no_ls        best-mu decode only, no local search    [Q1 / P: what LS buys]
  no_kick      local search, no ILS kicks               [P3 kick cost]
  K64          64 candidates per LS step                [P2 wasted decodes]
  no_deadline  auto jobs never offloaded by the decoder [Q3 deadline rule]
  serial       serial SGS decoder                       [Q4 sequencing]
  best_of      better of dispatch and serial SGS        [Q4 sequencing]
  most_free    pool tie-break: most free capacity       [Q5 pool choice]
  memetic      weighted_solve_memetic defaults (adapt 7)[Q6 search framework]
  edf          indifference-deadline EDF priority only  [Q4 fix]
  edf_add      EDF as an extra start beside the best mu [Q4 fix]
  edf_nodl, edf_add_nodl   … combined with no_deadline   [Q3+Q4]
  edf_add_nodl_K64         … with 64 candidates per step [P2 on the fix]
  edfP[_K64]               EDF start without deadline offload, mu start with it
  edfP_K64_patK            … stop kicks after K without improvement [P3]

    uv run python scripts/dispatch_ablation.py [--variants a,b] [--workers 3] [--threads 4]
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
from dispatch_diagnose import BENCH, LAMS, OUT, SAMPLE, _g  # noqa: E402

NO_DEADLINE, MOST_FREE, NO_KICK, MIX, SERIAL, BEST_OF, EDF, EDF_ADD, EDF_NODL = \
    1, 2, 32, 64, 256, 512, 2048, 4096, 8192
# every variant is one change from the bench_io_noreduc configuration (the
# pre-2026-11-03 defaults), not from the current weighted_solve_dispatch defaults
BASE = dict(K=256, K_max=0, flags=MIX, kick_patience=0)   # kicks split over starts (default)
VARIANTS = {
    "default": {},
    "budget4": dict(K=1024, n_kicks=240),
    "n_mu3": dict(n_mu=3, n_kicks=180),
    "no_ls": dict(ls=False),
    "no_kick": dict(flags=MIX | NO_KICK),
    "K64": dict(K=64),
    "no_deadline": dict(flags=MIX | NO_DEADLINE),
    "serial": dict(flags=MIX | SERIAL),
    "best_of": dict(flags=MIX | BEST_OF),
    "most_free": dict(flags=MIX | MOST_FREE),
    "memetic": "memetic",
    "edf": dict(flags=MIX | EDF),
    "edf_add": dict(flags=MIX | EDF_ADD),
    "edf_nodl": dict(flags=MIX | EDF | NO_DEADLINE),
    "edf_add_nodl": dict(flags=MIX | EDF_ADD | NO_DEADLINE),
    "edf_add_nodl_K64": dict(flags=MIX | EDF_ADD | NO_DEADLINE, K=64),
    "edfP": dict(flags=MIX | EDF_ADD | EDF_NODL),
    "edfP_K64": dict(flags=MIX | EDF_ADD | EDF_NODL, K=64),
    "edfP_K64_pat3": dict(flags=MIX | EDF_ADD | EDF_NODL, K=64, kick_patience=3),
    "edfP_K64_pat5": dict(flags=MIX | EDF_ADD | EDF_NODL, K=64, kick_patience=5),
    "edfP_K64_pat10": dict(flags=MIX | EDF_ADD | EDF_NODL, K=64, kick_patience=10),
}


def run_one(rec, variant, threads):
    from fedhpc import instance_gen as gen
    from fedhpc.data import Instance
    from fedhpc.moea import weighted_solve_dispatch, weighted_solve_memetic

    inst = Instance.from_file(gen.DATA / "generated" / f"{rec['name']}.json")
    ref = (rec["lex_f1"]["f1"], rec["lex_f1"]["f2"], rec["lex_f2"]["f1"])
    out = dict(name=rec["name"], cls=rec["cls"], workload=rec["workload"], count=rec["count"],
               load=rec["load"], io=rec["io"], variant=variant, lams={})
    kw = VARIANTS[variant] if VARIANTS[variant] == "memetic" else {**BASE, **VARIANTS[variant]}
    for lam in LAMS:
        gstar = rec["lams"][lam]["gurobi"]["g"]
        t = time.perf_counter()
        if kw == "memetic":
            sol, st = weighted_solve_memetic(inst, float(lam), f1_T=ref[0], f2_T=ref[1], f1_0=ref[2],
                                             n_threads=threads, return_stats=True)
        else:
            sol, st = weighted_solve_dispatch(inst, float(lam), f1_T=ref[0], f2_T=ref[1], f1_0=ref[2],
                                              n_threads=threads, return_stats=True, **kw)
        dt = time.perf_counter() - t
        inst.check_schedule(sol.assignment)
        g = _g(float(lam), sol.f1, sol.f2, ref)
        out["lams"][lam] = dict(g=g, gstar=gstar, gap=100 * (g - gstar) / max(gstar, 1e-9), t=dt,
                                f1=sol.f1, f2=sol.f2,
                                stats={k: v for k, v in st.items()
                                       if isinstance(v, (int, float))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path, default=OUT / "ablation.jsonl")
    args = ap.parse_args()
    bench = {r["name"]: r for r in map(json.loads, BENCH.read_text().splitlines())}
    OUT.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        done = {(r["name"], r["variant"]) for r in map(json.loads, args.out.read_text().splitlines())}
    tasks = [(n, v) for v in args.variants.split(",") for n in SAMPLE if (n, v) not in done]
    print(f"{len(tasks)} tasks", flush=True)
    with ProcessPoolExecutor(args.workers) as ex, open(args.out, "a") as f:
        futs = {ex.submit(run_one, bench[n], v, args.threads): (n, v) for n, v in tasks}
        for k, fut in enumerate(as_completed(futs), 1):
            n, v = futs[fut]
            try:
                rec = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"[{k}/{len(tasks)}] {n} {v} FAILED {type(e).__name__}: {e}", flush=True)
                continue
            f.write(json.dumps(rec) + "\n"); f.flush()
            print(f"[{k}/{len(tasks)}] {n} {v} " + " ".join(
                f"{l}:{x['gap']:.1f}%/{x['t']:.0f}s" for l, x in rec["lams"].items()), flush=True)


if __name__ == "__main__":
    main()
