"""Test the C++ priority-key/non-delay-SGS scheme (src/fedhpc/_ext/sgs_algos.hpp)
against the same pre-computed (partial, sparse) exact Pareto frontier
compare_moea_vs_pos_congestion_front.py uses, at a matched (pop, n_gen) budget
against the job_slots-index scheme's reference numbers on this instance.

data/pos_congestion_known_runtime_offline_10min.json: 3,340 jobs + 2,617
running jobs, 28 types, horizon 288 — ~3.4x the job count of the fedhpc
964-job instance everything else in this session was benchmarked on.

Usage: uv run python scripts/compare_sgs_cpp_vs_pos_congestion.py
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
import compare_moea_vs_pos_congestion_front as cmpc  # noqa: E402

from fedhpc.data import Instance  # noqa: E402
from fedhpc.formulations import configure_env  # noqa: E402
from fedhpc.moea import (  # noqa: E402
    nsga2_frontier, nsga3_frontier, moead_frontier,
    nsga2_sgs_frontier, nsga3_sgs_frontier, moead_sgs_frontier,
)

POP  = int(os.environ.get("POP", 100))
NGEN = int(os.environ.get("NGEN", 100))
SEED = int(os.environ.get("SEED", 42))


def main():
    configure_env(verbose=False)
    inst = Instance.from_file(cmpc.INSTANCE)
    known = cmpc.load_known_front()
    kpts = cmpc._pts(known)
    print(f"{cmpc.INSTANCE}\n  {len(inst.jobs)} jobs, {len(inst.instance_types)} types, "
          f"H={inst.horizon}, {len(inst.running_jobs)} running jobs")
    print(f"  known exact front: {len(kpts)} proven-optimal points (PARTIAL/sparse — "
          f"trust dom_by_known/exact/ea_dominates; IGD/eps+ are rough here)")

    ideal = kpts.min(axis=0)
    nadir = kpts.max(axis=0)
    span = (nadir - ideal)
    span[span < 1e-10] = 1.0

    algos = {
        "NSGA-II    ": lambda: nsga2_frontier(inst, pop_size=POP, n_gen=NGEN, seed=SEED),
        "NSGA-III   ": lambda: nsga3_frontier(inst, pop_size=POP, n_divisions=POP - 1,
                                               n_gen=NGEN, seed=SEED),
        "MOEA/D     ": lambda: moead_frontier(inst, n_weights=POP, n_gen=NGEN, seed=SEED),
        "NSGA-II SGS": lambda: nsga2_sgs_frontier(inst, pop_size=POP, n_gen=NGEN, seed=SEED),
        "NSGA-III SGS": lambda: nsga3_sgs_frontier(inst, pop_size=POP, n_divisions=POP - 1,
                                                    n_gen=NGEN, seed=SEED),
        "MOEA/D SGS ": lambda: moead_sgs_frontier(inst, n_weights=POP, n_gen=NGEN, seed=SEED),
    }
    print(f"\n  pop={POP}, n_gen={NGEN}, seed={SEED}  (matched budget, both schemes)")
    hdr = (f"  {'algo':12}  {'card':>4}  {'exact':>5}  {'dom_by_known':>12}  "
           f"{'ea_dom':>6}  {'IGD':>8}  {'eps+':>8}  {'t(s)':>7}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    results = {}
    for name, fn in algos.items():
        ts = time.time()
        front = fn()
        dt = time.time() - ts
        ea_pts = cmpc._pts(front)
        m = cmpc.score(ea_pts, kpts, ideal, span)
        results[name.strip()] = (front, m, dt)
        print(f"  {name}  {m['card']:>4}  {m['exact']:>5}  {m['dom_by_known']:>12}  "
              f"{m['ea_dominates']:>6}  {m['igd']:>8.5f}  {m['eps_plus']:>8.5f}  {dt:>7.1f}")

    return results, inst


if __name__ == "__main__":
    main()
