"""Score the MOEAs on the generated instances against a small, exact,
*sampled* Pareto front.

The full true front of a generated instance has hundreds of points (f1 spans
~1-60k integer steps), so instead of enumerating it we sample it exactly:

  * the two lex anchors (f1_T, f2_T) and (f1_0, f2_0),
  * K ε-constraint points  min f2 s.t. f1 <= ε  (lex f2,f1), ε evenly in f1,
  * K ε-constraint points  min f1 s.t. f2 <= ε  (lex f1,f2), ε evenly in f2.

Every certified point is individually Pareto-optimal, so the sample is a
subset of the true front: an EA point dominated by a sample point is
dominated by the true front (the reverse is not true — the reported
dominated counts are LOWER bounds). Unproven solves are dropped.

Per (instance, algorithm, seed) we report, normalised by the sample's
ideal/nadir: card, exact hits, dominated (# and %), dominates (must be 0),
IGD+, eps+, and the anchor misses (how far the EA's best f1 / best f2 is from
the exact anchors).  Fronts are cached, so re-runs only redo the EAs.

usage: compare_moea_vs_sampled_front.py OUT_DIR INSTANCE.json [...]
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from fedhpc.data import Instance
from fedhpc.formulations import SpaceTimeFormulation, configure_env
from fedhpc.metrics import compare_fronts
from fedhpc.moea import (moead_frontier, moead_sgs_frontier, nsga2_frontier,
                         nsga2_sgs_frontier, nsga3_frontier, nsga3_sgs_frontier)
from fedhpc.pareto import _make_lexmin_bounded

K = int(os.environ.get("K", 8))              # ε points per direction
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,7,123").split(",")]
POP, NGEN = int(os.environ.get("POP", 200)), int(os.environ.get("NGEN", 300))
PER_SOLVE = float(os.environ.get("PER_SOLVE", 300))

ALGOS = {
    "NSGA-II": lambda i, s: nsga2_frontier(i, pop_size=POP, n_gen=NGEN, seed=s),
    "NSGA-III": lambda i, s: nsga3_frontier(i, pop_size=POP, n_divisions=POP - 1, n_gen=NGEN, seed=s),
    "MOEA/D": lambda i, s: moead_frontier(i, n_weights=POP, n_gen=NGEN, seed=s),
    "NSGA-II-SGS": lambda i, s: nsga2_sgs_frontier(i, pop_size=POP, n_gen=NGEN, seed=s),
    "NSGA-III-SGS": lambda i, s: nsga3_sgs_frontier(i, pop_size=POP, n_divisions=POP - 1, n_gen=NGEN, seed=s),
    "MOEA/D-SGS": lambda i, s: moead_sgs_frontier(i, n_weights=POP, n_gen=NGEN, seed=s),
}


def _nd(pts):
    pts = sorted(set(pts))
    out, best = [], float("inf")
    for f1, f2 in pts:
        if f2 < best - 1e-9:
            out.append((f1, f2))
            best = f2
    return out


def sampled_front(inst, path_cache):
    if os.path.exists(path_cache):
        return json.load(open(path_cache))
    lb, _ = _make_lexmin_bounded(inst, SpaceTimeFormulation(), {"OutputFlag": 0},
                                 per_solve_time_limit=PER_SOLVE, n_seed_retries=2)
    log, pts = [], []

    def solve(tag, *args):
        t = time.time()
        s = lb(*args)
        dt = time.time() - t
        log.append(dict(tag=tag, status=s.status, f1=s.f1, f2=s.f2, t=dt))
        print(f"    {tag:14} {s.status:9} f1={s.f1} f2={s.f2} ({dt:.1f}s)", flush=True)
        if s.status == "optimal":
            pts.append((s.f1, s.f2))
        return s

    a = solve("anchor_f1", "f1", "f2")
    b = solve("anchor_f2", "f2", "f1")
    # An unproven anchor_f2 still gives a usable f1 range end (its incumbent);
    # every ε point below is certified on its own, so the sample stays exact.
    if a.status == "optimal" and b.f1 is not None and b.f1 > a.f1 + 1:
        for k in range(1, K + 1):
            eps = a.f1 + (b.f1 - a.f1) * k / (K + 1)
            solve(f"f1<={eps:.0f}", "f2", "f1", "f1", eps)
        for k in range(1, K + 1):
            eps = b.f2 + (a.f2 - b.f2) * k / (K + 1)
            solve(f"f2<={eps:.1f}", "f1", "f2", "f2", eps)
    res = dict(points=_nd(pts), log=log,
               anchors=dict(f1_T=a.f1, f2_at_f1_T=a.f2, f1_0=b.f1, f2_0=b.f2))
    json.dump(res, open(path_cache, "w"), indent=1)
    return res


def run(out_dir, path):
    name = os.path.basename(path).removesuffix(".json")
    inst = Instance.from_file(path)
    print(f"\n=== {name}: {len(inst.jobs)} jobs, {len(inst.instance_types)} types, "
          f"H={inst.horizon}", flush=True)
    t0 = time.time()
    fr = sampled_front(inst, os.path.join(out_dir, "fronts", name + ".json"))
    ref = np.array(fr["points"], float)
    print(f"  sampled front: {len(ref)} exact points ({time.time()-t0:.0f}s)", flush=True)
    ideal, nadir = ref.min(0), ref.max(0)
    span = np.where(nadir - ideal > 1e-9, nadir - ideal, 1.0)
    rows = []
    for algo, fn in ALGOS.items():
        for seed in SEEDS:
            t = time.time()
            front = fn(inst, seed)
            dt = time.time() - t
            ea = np.array([(s.f1, s.f2) for s in front if s.f1 is not None], float)
            m = compare_fronts(ea, ref, ideal=ideal, nadir=nadir)
            row = dict(instance=name, algo=algo, seed=seed, t=dt, n_ref=len(ref),
                       **{k: (float(v) if isinstance(v, (float, np.floating)) else int(v))
                          for k, v in m.items()},
                       f1_miss=float((ea[:, 0].min() - ideal[0]) / span[0]),
                       f2_miss=float((ea[:, 1].min() - ideal[1]) / span[1]),
                       f1_miss_abs=float(ea[:, 0].min() - ideal[0]),
                       f2_miss_abs=float(ea[:, 1].min() - ideal[1]),
                       # per sampled point: EA's best f2 at f1 <= f1*, gap / f2 span
                       cost_gaps=[float(((ea[ea[:, 0] <= p0 + 1e-6, 1].min() - p1) / span[1])
                                        if np.any(ea[:, 0] <= p0 + 1e-6) else np.inf)
                                  for p0, p1 in ref])
            rows.append(row)
            print(f"  {algo:12} s{seed:<4} card={m['card']:4d} exact={m['exact']:3d} "
                  f"dom={m['dominated']:4d} ({100*m['dominated']/max(m['card'],1):5.1f}%) "
                  f"dominates={m['dominates']} IGD+={m['igd_plus']:.4f} eps+={m['eps_plus']:.4f} "
                  f"f1miss={row['f1_miss_abs']:.0f} f2miss={row['f2_miss_abs']:.1f} ({dt:.1f}s)",
                  flush=True)
    with open(os.path.join(out_dir, "results.jsonl"), "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    out_dir, paths = sys.argv[1], sys.argv[2:]
    os.makedirs(os.path.join(out_dir, "fronts"), exist_ok=True)
    configure_env(verbose=False)
    done = set()
    rp = os.path.join(out_dir, "results.jsonl")
    if os.path.exists(rp):
        done = {json.loads(l)["instance"] for l in open(rp)}
    for p in paths:
        if os.path.basename(p).removesuffix(".json") in done:
            continue
        run(out_dir, p)


if __name__ == "__main__":
    main()
