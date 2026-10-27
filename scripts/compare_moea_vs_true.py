"""Test the evolutionary MOEAs (NSGA-II, NSGA-III, MOEA/D) against the exact
true Pareto frontier computed by `true_pareto_frontier` (exact box splitting).

For each instance:
  1. compute the complete, provably-exact true front,
  2. run each EA,
  3. score the EA front against the true one with normalised indicators.

Indicators (objectives normalised by the *true* front's ideal/nadir):
  card        cardinality of the returned front
  exact       # EA points coinciding (<1e-6) with a true Pareto point
  dominated   # EA points strictly dominated by some true point  (should be 0)
  IGD         mean dist from each TRUE point to nearest EA point   (↓, 0 = perfect coverage+convergence)
  GD          mean dist from each EA point to nearest TRUE point   (↓, 0 = every EA point on the front)
  eps+        additive epsilon indicator: how far the EA front must
              shift to weakly dominate the whole true front         (↓, 0 = covers it)
  HVR         hypervolume(EA) / hypervolume(true)                    (↑, 1 = matches)
"""
from __future__ import annotations

import sys
import time

import numpy as np

from fedhpc.data import Instance
from fedhpc.formulations import configure_env
from fedhpc.metrics import compare_fronts
from fedhpc.moea import moead_frontier, nsga2_frontier, nsga3_frontier
from fedhpc.pareto import true_pareto_frontier

INSTANCES = sys.argv[1:]
if not INSTANCES:
    sys.exit("usage: compare_moea_vs_true.py INSTANCE.json [...]  "
             "(small instances: the exact front is enumerated in full)")

POP, NGEN, SEED = 200, 300, 42


def _pts(front):
    return np.array([(s.f1, s.f2) for s in front if s.f1 is not None], dtype=float)


def score(ea_pts, true_pts, ideal, span):
    """metrics.compare_fronts, normalised by the true front's ideal/nadir."""
    return compare_fronts(ea_pts, true_pts, ideal=ideal, nadir=ideal + span)


def run_instance(path: str):
    inst = Instance.from_file(path)
    print(f"\n{'=' * 78}\n{path}  —  {len(inst.jobs)} jobs, "
          f"{len(inst.instance_types)} types, H={inst.horizon}\n{'=' * 78}")

    t0 = time.time()
    true_front = true_pareto_frontier(
        inst, verbose=False, OutputFlag=0,
        per_solve_time_limit=120.0, time_budget=900.0,
    )
    true_pts = _pts(true_front)
    print(f"true front : {len(true_pts):3d} points   "
          f"f1 in [{true_pts[:,0].min():.2f}, {true_pts[:,0].max():.2f}]   "
          f"f2 in [{true_pts[:,1].min():.2f}, {true_pts[:,1].max():.2f}]   "
          f"({time.time()-t0:.1f}s)")

    ideal = true_pts.min(axis=0)
    nadir = true_pts.max(axis=0)
    span = np.where(nadir - ideal > 1e-10, nadir - ideal, 1.0)

    algos = {
        "NSGA-II ": lambda: nsga2_frontier(inst, pop_size=POP, n_gen=NGEN, seed=SEED),
        "NSGA-III": lambda: nsga3_frontier(inst, pop_size=POP, n_divisions=POP - 1,
                                           n_gen=NGEN, seed=SEED),
        "MOEA/D  ": lambda: moead_frontier(inst, n_weights=POP, n_gen=NGEN, seed=SEED),
    }

    hdr = f"  {'algo':8}  {'card':>4}  {'exact':>5}  {'domd':>4}  {'IGD':>9}  {'GD':>9}  {'eps+':>9}  {'HVR':>7}  {'t(s)':>6}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name, fn in algos.items():
        ts = time.time()
        front = fn()
        dt = time.time() - ts
        ea_pts = _pts(front)
        m = score(ea_pts, true_pts, ideal, span)
        print(f"  {name}  {m['card']:>4}  {m['exact']:>5}  {m['dominated']:>4}  "
              f"{m['igd']:>9.5f}  {m['gd']:>9.5f}  {m['eps_plus']:>9.5f}  "
              f"{m['hvr']:>7.4f}  {dt:>6.1f}")


def main():
    configure_env(verbose=False)
    for p in INSTANCES:
        try:
            run_instance(p)
        except Exception as e:
            print(f"  !! {p}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
