"""A/B the last committed MOEA/D change (commit 83d1561 "tentatively improve
EA" — the scalar_ls_interval scalarised-local-search polish) on the
pos_congestion instance, vs the pre-commit behaviour.

scalar_ls_interval == 0 reproduces the pre-commit MOEA/D output byte-for-byte
(per its own docstring / the _ext binding default), so no git checkout /
rebuild is needed — just call moead_frontier both ways.

  OFF  = scalar_ls_interval=0    (pre-83d1561)
  ON   = scalar_ls_interval=-30  (current moead_frontier default)

Scored against the partial exact front in
pareto_runs/pos_congestion_10min_map_checkpoint.json (5 proven points).
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from fedhpc.data import Instance
from fedhpc.formulations import configure_env
from fedhpc.moea import moead_frontier
from fedhpc.pareto import _filter_dominated, _solution_from_dict

INSTANCE = "data/pos_congestion_known_runtime_offline_10min.json"
CHECKPOINT = "pareto_runs/pos_congestion_10min_map_checkpoint.json"
POP = int(os.environ.get("POP", 200))
NGEN = int(os.environ.get("NGEN", 300))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,1,7").split(",")]


def known_front():
    d = json.load(open(CHECKPOINT))
    sols = _filter_dominated([_solution_from_dict(s) for s in d["solutions"]])
    return np.array(sorted((s.f1, s.f2) for s in sols), dtype=float)


def _sdom(a, b):
    return np.all(a <= b) and np.any(a < b)


def score(ea, kn, ideal, span):
    e = (ea - ideal) / span
    k = (kn - ideal) / span
    igd = float(np.mean([np.linalg.norm(e - kk, axis=1).min() for kk in k])) if len(e) else np.inf
    eps = float(max(min(max(ei - kk) for ei in e) for kk in k)) if len(e) else np.inf
    dom = sum(1 for ei in ea if any(_sdom(kk, ei) for kk in kn))
    eadom = sum(1 for ei in ea if any(_sdom(ei, kk) for kk in kn))
    exact = sum(1 for ei in ea
                if np.any(np.all(np.abs(kn - ei) <= 1e-4 + 1e-4 * np.abs(kn), axis=1)))
    return dict(card=len(ea), exact=exact, dom=dom, eadom=eadom, igd=igd, eps=eps)


def main():
    configure_env(verbose=False)
    inst = Instance.from_file(INSTANCE)
    kn = known_front()
    ideal, nadir = kn.min(0), kn.max(0)
    span = np.where(nadir - ideal > 1e-10, nadir - ideal, 1.0)
    print(f"{INSTANCE}  |  known front: {len(kn)} proven pts  "
          f"f1∈[{kn[:,0].min():.0f},{kn[:,0].max():.0f}] f2∈[{kn[:,1].min():.1f},{kn[:,1].max():.1f}]")
    print(f"pop={POP} n_gen={NGEN} seeds={SEEDS}\n")

    hdr = f"  {'cfg':22} {'seed':>4} {'card':>5} {'exact':>5} {'dom_by_known':>12} {'dom%':>6} {'ea_dom':>6} {'IGD':>8} {'eps+':>8} {'t(s)':>7}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    agg = {}
    for label, sli in [("OFF (pre-83d1561)", 0), ("ON  (scalar_ls=-30)", -30)]:
        rows = []
        for sd in SEEDS:
            t = time.time()
            front = moead_frontier(inst, n_weights=POP, n_gen=NGEN, seed=sd,
                                   scalar_ls_interval=sli)
            dt = time.time() - t
            ea = np.array([(s.f1, s.f2) for s in front if s.f1 is not None], dtype=float)
            m = score(ea, kn, ideal, span)
            rows.append((m, dt))
            print(f"  {label:22} {sd:>4} {m['card']:>5} {m['exact']:>5} {m['dom']:>12} "
                  f"{100*m['dom']/max(m['card'],1):>5.1f}% {m['eadom']:>6} "
                  f"{m['igd']:>8.4f} {m['eps']:>8.4f} {dt:>7.1f}")
        agg[label] = rows
    print()
    for label, rows in agg.items():
        ms = [r[0] for r in rows]
        print(f"  {label:22}  mean  card={np.mean([m['card'] for m in ms]):.1f}  "
              f"dom%={np.mean([100*m['dom']/max(m['card'],1) for m in ms]):.1f}  "
              f"IGD={np.mean([m['igd'] for m in ms]):.4f}  "
              f"eps+={np.mean([m['eps'] for m in ms]):.4f}  "
              f"t={np.mean([r[1] for r in rows]):.1f}s")


if __name__ == "__main__":
    main()
