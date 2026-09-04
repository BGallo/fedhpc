"""Same benchmark as compare_priority_sgs_vs_known.py, but the pymoo initial
population is seeded with domain-aware heuristic genomes (translated from
ga_common.hpp's make_heuristic_seeds()) instead of pymoo's default uniform-
random init, isolating "seeding" as the one changed variable.

Usage: uv run --with pymoo python scripts/compare_priority_sgs_seeded_vs_known.py
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
from priority_sgs_problem import SGSScheduleProblem, build_candidates  # noqa: E402

from pymoo.algorithms.moo.nsga2 import NSGA2  # noqa: E402
from pymoo.algorithms.moo.nsga3 import NSGA3  # noqa: E402
from pymoo.algorithms.moo.moead import MOEAD  # noqa: E402
from pymoo.core.sampling import Sampling  # noqa: E402
from pymoo.operators.sampling.rnd import FloatRandomSampling  # noqa: E402
from pymoo.util.ref_dirs import get_reference_directions  # noqa: E402
from pymoo.optimize import minimize  # noqa: E402

POP = int(os.environ.get("POP", 100))
NGEN = int(os.environ.get("NGEN", 300))
SEED = int(os.environ.get("SEED", 42))
N_PARTITIONS = POP - 1


def _norm(vals):
    vals = np.asarray(vals, dtype=float)
    lo, hi = vals.min(), vals.max()
    if hi - lo < 1e-12:
        return np.full_like(vals, 0.5)
    return (vals - lo) / (hi - lo)


def build_heuristic_seeds(inst: Instance, job_candidates):
    """Translate ga_common.hpp's heuristic-seed *intent* into priority-key
    genomes for the order_key/type_key SGS decoder.

    Skipped outright: fixed_wait(0.25), fixed_wait(0.5), star_wait — these
    construct deliberate mid-horizon delay, which the non-delay SGS decoder
    has no analogue for (documented limitation from the unseeded run).

    Empirically dropped after checking decode() output directly (not just
    assumed): "earliest-completion type" and "cost-ascending processing
    order" both degenerate to byte-identical genomes as plain arrival-order +
    cheapest-type on THIS dataset — jobs are already stored arrival-sorted by
    id, and most candidates share cost=0 (free on-prem types), so a stable
    sort on either key reduces to the same identity permutation. Repeating
    them would be exactly the near-duplicate-seed mistake
    make_heuristic_seeds()'s own comment warns against, so they're replaced
    below with two genomes verified to land on genuinely different (f1, f2)
    points: reverse-arrival order (processes latest-arriving jobs first) and
    a mid-cost type choice (type_key=0.5, roughly the median-cost feasible
    type per job) — a plausible knee-region seed neither extreme provides.
    """
    n = len(inst.jobs)
    arrivals = np.array([j.arrival for j in inst.jobs], dtype=float)
    order_arrival = _norm(arrivals)
    order_reverse = 1.0 - order_arrival

    cloud_idx_frac = np.zeros(n)
    n_cand = np.zeros(n, dtype=int)
    for i, cands in enumerate(job_candidates):
        n_cand[i] = len(cands)
        cloud_idxs = [k for k, c in enumerate(cands) if c.cap is None]
        cloud_idx_frac[i] = (cloud_idxs[0] + 0.5) / n_cand[i] if cloud_idxs else 0.5

    seeds = {
        "cheapest-type, arrival-order        (~ greedy-by-cost / list-schedule)":
            np.concatenate([order_arrival, np.zeros(n)]),
        "cheapest-type, reverse-arrival-order   (order-diversity variant)":
            np.concatenate([order_reverse, np.zeros(n)]),
        "cloud-first, arrival-order          (~ full-burst)":
            np.concatenate([order_arrival, cloud_idx_frac]),
        "mid-cost type, arrival-order        (knee-region variant)":
            np.concatenate([order_arrival, np.full(n, 0.5)]),
        "priciest-type, arrival-order        (opposite extreme)":
            np.concatenate([order_arrival, np.full(n, 0.999)]),
    }
    return seeds


class SeededSampling(Sampling):
    def __init__(self, seed_matrix):
        super().__init__()
        self.seed_matrix = np.asarray(seed_matrix, dtype=float)

    def _do(self, problem, n_samples, **kwargs):
        base = FloatRandomSampling()._do(problem, n_samples, **kwargs)
        n_seed = min(len(self.seed_matrix), n_samples)
        base[:n_seed] = self.seed_matrix[:n_seed]
        return base


def feasible_front(res):
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
    ideal = kpts.min(axis=0)
    nadir = kpts.max(axis=0)
    span = np.where(nadir - ideal > 1e-10, nadir - ideal, 1.0)

    prob = SGSScheduleProblem(inst)
    seeds_by_name = build_heuristic_seeds(inst, prob.job_candidates)
    seed_matrix = np.array(list(seeds_by_name.values()))

    print(f"{cmkf.INSTANCE}\n  {len(inst.jobs)} jobs, {len(inst.instance_types)} types")
    print(f"  seeded with {len(seed_matrix)} heuristic genomes:")
    for name in seeds_by_name:
        print(f"    - {name}")

    ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=N_PARTITIONS)

    algos = {
        "NSGA-II ": lambda: NSGA2(pop_size=POP, sampling=SeededSampling(seed_matrix)),
        "NSGA-III": lambda: NSGA3(pop_size=len(ref_dirs), ref_dirs=ref_dirs,
                                  sampling=SeededSampling(seed_matrix)),
        "MOEA/D  ": lambda: MOEAD(ref_dirs=ref_dirs, n_neighbors=20,
                                  sampling=SeededSampling(seed_matrix)),
    }

    print(f"\n  priority-key + non-delay-SGS, SEEDED, pop={POP} "
          f"(NSGA-III/MOEA-D use {len(ref_dirs)} ref dirs), n_gen={NGEN}, seed={SEED}")
    hdr = (f"  {'algo':8}  {'card':>4}  {'exact':>5}  {'dom_by_known':>12}  "
           f"{'ea_dom':>6}  {'IGD':>8}  {'eps+':>8}  {'(GD)':>8}  {'(HVR)':>7}  {'t(s)':>7}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for name, make in algos.items():
        ts = time.time()
        res = minimize(prob, make(), ("n_gen", NGEN), seed=SEED, verbose=False)
        dt = time.time() - ts
        ea_pts = feasible_front(res)
        m = cmkf.score(ea_pts, kpts, ideal, span)
        print(f"  {name}  {m['card']:>4}  {m['exact']:>5}  {m['dom_by_known']:>12}  "
              f"{m['ea_dominates']:>6}  {m['igd']:>8.5f}  {m['eps_plus']:>8.5f}  "
              f"{m['gd']:>8.5f}  {m['hvr']:>7.4f}  {dt:>7.1f}")

    print("\n  Reference — unseeded priority-key/SGS (same pop/n_gen, uniform-random init):")
    print("    NSGA-II     51      0            51       0   4.45900   4.84600        -        -   70.2")
    print("    NSGA-III    14      0            14       0   3.97600   4.35600        -        -   45.4")
    print("    MOEA/D      45      0            45       0   6.98900   7.38200        -        -   65.4")
    print("\n  Reference — current C++ job_slots-index scheme (pop=400, n_gen=2000):")
    print("    NSGA-II    337      0           331       0   0.07530   0.09306   0.07991   0.8771      5.0")
    print("    NSGA-III   287      0           284       0   0.07894   0.09587   0.08649   0.8722      5.6")
    print("    MOEA/D     334      0           332       0   0.06100   0.07919   0.06969   0.9034     11.5")


if __name__ == "__main__":
    main()
