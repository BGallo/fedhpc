"""Benchmark the classical priority-key + non-delay-SGS chromosome (pymoo's
generic NSGA-II / NSGA-III / MOEA-D) against the same pre-computed exact
Pareto frontier used by scripts/compare_moea_vs_known_front.py, so the two
representations are scored with the identical metric definitions.

Usage: uv run --with pymoo python scripts/compare_priority_sgs_vs_known.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import compare_moea_vs_known_front as cmkf  # noqa: E402

from fedhpc.data import Instance  # noqa: E402
from priority_sgs_problem import SGSScheduleProblem  # noqa: E402

from pymoo.algorithms.moo.nsga2 import NSGA2  # noqa: E402
from pymoo.algorithms.moo.nsga3 import NSGA3  # noqa: E402
from pymoo.algorithms.moo.moead import MOEAD  # noqa: E402
from pymoo.util.ref_dirs import get_reference_directions  # noqa: E402
from pymoo.optimize import minimize  # noqa: E402

POP = int(os.environ.get("POP", 100))
NGEN = int(os.environ.get("NGEN", 300))
SEED = int(os.environ.get("SEED", 42))
N_PARTITIONS = POP - 1  # das-dennis(2 obj, p partitions) -> p+1 points


def feasible_front(res):
    """(f1,f2) for every feasible, non-dominated individual pymoo returns."""
    if res.F is None:
        return np.empty((0, 2))
    if res.G is not None:
        feas = np.all(res.G <= 1e-6, axis=1)
        return res.F[feas]
    return res.F


def main():
    inst = Instance.from_file(cmkf.INSTANCE)
    known = cmkf.load_known_front()
    kpts = cmkf._pts(known)
    print(f"{cmkf.INSTANCE}\n  {len(inst.jobs)} jobs, {len(inst.instance_types)} types, "
          f"H={inst.horizon}, B={inst.budget}")
    print(f"  known exact front: {len(kpts)} proven-optimal points")
    print(f"    f1 in [{kpts[:,0].min():.2f}, {kpts[:,0].max():.2f}]   "
          f"f2 in [{kpts[:,1].min():.2f}, {kpts[:,1].max():.2f}]")

    ideal = kpts.min(axis=0)
    nadir = kpts.max(axis=0)
    span = np.where(nadir - ideal > 1e-10, nadir - ideal, 1.0)

    prob = SGSScheduleProblem(inst)
    ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=N_PARTITIONS)

    algos = {
        "NSGA-II ": lambda: NSGA2(pop_size=POP),
        "NSGA-III": lambda: NSGA3(pop_size=len(ref_dirs), ref_dirs=ref_dirs),
        "MOEA/D  ": lambda: MOEAD(ref_dirs=ref_dirs, n_neighbors=20),
    }

    print(f"\n  priority-key + non-delay-SGS (pymoo), pop={POP} (NSGA-III/MOEA-D use "
          f"{len(ref_dirs)} ref dirs), n_gen={NGEN}, seed={SEED}")
    hdr = (f"  {'algo':8}  {'card':>4}  {'exact':>5}  {'dom_by_known':>12}  "
           f"{'ea_dom':>6}  {'IGD':>8}  {'eps+':>8}  {'(GD)':>8}  {'(HVR)':>7}  {'t(s)':>7}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    results = {}
    for name, make in algos.items():
        ts = time.time()
        res = minimize(prob, make(), ("n_gen", NGEN), seed=SEED, verbose=False)
        dt = time.time() - ts
        ea_pts = feasible_front(res)

        m = cmkf.score(ea_pts, kpts, ideal, span)
        print(f"  {name}  {m['card']:>4}  {m['exact']:>5}  {m['dom_by_known']:>12}  "
              f"{m['ea_dominates']:>6}  {m['igd']:>8.5f}  {m['eps_plus']:>8.5f}  "
              f"{m['gd']:>8.5f}  {m['hvr']:>7.4f}  {dt:>7.1f}")
        results[name.strip()] = dict(m=m, dt=dt, res=res)

    print("\n  Reference — current C++ job_slots-index scheme (pop=400, n_gen=2000):")
    print("    NSGA-II    337      0           331       0   0.07530   0.09306   0.07991   0.8771      5.0")
    print("    NSGA-III   287      0           284       0   0.07894   0.09587   0.08649   0.8722      5.6")
    print("    MOEA/D     334      0           332       0   0.06100   0.07919   0.06969   0.9034     11.5")
    return results


if __name__ == "__main__":
    main()
