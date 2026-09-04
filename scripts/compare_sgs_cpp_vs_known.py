"""Test the C++ priority-key/non-delay-SGS scheme (src/fedhpc/_ext/sgs_algos.hpp)
against the same pre-computed exact Pareto frontier compare_moea_vs_known_front.py
uses, at the SAME (pop, n_gen) budget as the job_slots-index scheme's reference
numbers — the actually-fair comparison (matched eval budget + permutation-aware
operators, both confounds the earlier seeded-pymoo run couldn't remove).

Usage: uv run python scripts/compare_sgs_cpp_vs_known.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import compare_moea_vs_known_front as cmkf  # noqa: E402

from fedhpc.data import Instance  # noqa: E402
from fedhpc.formulations import configure_env  # noqa: E402
from fedhpc.moea import nsga2_sgs_frontier, nsga3_sgs_frontier, moead_sgs_frontier  # noqa: E402

POP  = int(os.environ.get("POP", 400))
NGEN = int(os.environ.get("NGEN", 2000))
SEED = int(os.environ.get("SEED", 42))


def main():
    configure_env(verbose=False)
    inst = Instance.from_file(cmkf.INSTANCE)
    known = cmkf.load_known_front()
    kpts = cmkf._pts(known)
    print(f"{cmkf.INSTANCE}\n  {len(inst.jobs)} jobs, {len(inst.instance_types)} types")
    print(f"  known exact front: {len(kpts)} proven-optimal points")

    ideal = kpts.min(axis=0)
    nadir = kpts.max(axis=0)
    span = (nadir - ideal)
    span[span < 1e-10] = 1.0

    algos = {
        "NSGA-II SGS ": lambda: nsga2_sgs_frontier(inst, pop_size=POP, n_gen=NGEN, seed=SEED),
        "NSGA-III SGS": lambda: nsga3_sgs_frontier(inst, pop_size=POP, n_divisions=POP - 1,
                                                   n_gen=NGEN, seed=SEED),
        "MOEA/D SGS  ": lambda: moead_sgs_frontier(inst, n_weights=POP, n_gen=NGEN, seed=SEED),
    }
    print(f"\n  C++ priority-key/SGS, pop={POP}, n_gen={NGEN}, seed={SEED}")
    hdr = (f"  {'algo':12}  {'card':>4}  {'exact':>5}  {'dom_by_known':>12}  "
           f"{'ea_dom':>6}  {'IGD':>8}  {'eps+':>8}  {'(GD)':>8}  {'(HVR)':>7}  {'t(s)':>7}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name, fn in algos.items():
        ts = time.time()
        front = fn()
        dt = time.time() - ts
        ea_pts = cmkf._pts(front)
        m = cmkf.score(ea_pts, kpts, ideal, span)
        print(f"  {name}  {m['card']:>4}  {m['exact']:>5}  {m['dom_by_known']:>12}  "
              f"{m['ea_dominates']:>6}  {m['igd']:>8.5f}  {m['eps_plus']:>8.5f}  "
              f"{m['gd']:>8.5f}  {m['hvr']:>7.4f}  {dt:>7.1f}")

    print("\n  Reference — current C++ job_slots-index scheme (pop=400, n_gen=2000):")
    print("    NSGA-II    337      0           331       0   0.07530   0.09306   0.07991   0.8771      5.0")
    print("    NSGA-III   287      0           284       0   0.07894   0.09587   0.08649   0.8722      5.6")
    print("    MOEA/D     334      0           332       0   0.06100   0.07919   0.06969   0.9034     11.5")
    print("\n  Reference — seeded Python/pymoo priority-key/SGS (pop=100, n_gen=300):")
    print("    NSGA-II    100      0            98       0   0.62144   0.54902   1.44452   0.1537    332.9")
    print("    NSGA-III    97      0            96       0   0.53902   0.51235   0.81684   0.2281    323.3")
    print("    MOEA/D      80      0            80       0   0.53835   0.59705   0.99185   0.2137    118.0")


if __name__ == "__main__":
    main()
