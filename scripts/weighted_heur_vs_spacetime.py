"""Memetic weighted-sum metaheuristic (moea.weighted_solve) vs the exact
SpaceTime MIP with dominance reductions on (SpaceTimeFormulation defaults,
objective hint passed so R3 is active), at the same lambda and the same exact
reference points.

Both minimise g = (lam/(f1_0-f1_T))*f1 + ((1-lam)/f2_T)*f2  s.t. f1 <= f1_0;
g is recomputed in Python from (f1, f2) for both sides.

Per (instance, lam): one exact solve, then the heuristic at several budgets
(pop, n_gen) and seeds. Also one "end-to-end" heuristic run that estimates
its own reference points (heuristic_weighted_reference_points), scored with
the exact g. One JSON object per line goes to OUT.
"""
from __future__ import annotations

import json
import os
import sys
import time

from fedhpc.data import Instance
from fedhpc.formulations import SpaceTimeFormulation, configure_env
from fedhpc.model import solve_weighted_sum, weighted_sum_degenerate
from fedhpc.moea import heuristic_weighted_reference_points, weighted_solve
from fedhpc.pareto import _reference_points

sys.stdout.reconfigure(line_buffering=True)

INSTANCES = sys.argv[1:] or [
    "data/fedhpc_known_runtime_offline_10min.json",
    "data/pos_congestion_known_runtime_offline_10min.json",
]
LAMS = [float(x) for x in os.environ.get("LAMS", "0.1,0.25,0.5,0.75,0.9").split(",")]
SEEDS = [int(x) for x in os.environ.get("SEEDS", "1,7,42").split(",")]
# (pop_size, n_gen): small / weighted_solve default / large
BUDGETS = [tuple(int(v) for v in b.split("x"))
           for b in os.environ.get("BUDGETS", "16x20,32x60,64x150").split(",")]
OUT = os.environ.get("OUT", "weighted_heur_vs_spacetime.jsonl")
EXACT_PARAMS = dict(OutputFlag=0, MIPGap=1e-6)


def g_of(lam, f1, f2, refs):
    f1_T, f2_T, f1_0 = refs
    if weighted_sum_degenerate(f1_T, f2_T, f1_0):
        return f2
    return (lam / (f1_0 - f1_T)) * f1 + ((1 - lam) / f2_T) * f2


def emit(fh, **rec):
    fh.write(json.dumps(rec) + "\n")
    fh.flush()


def main():
    configure_env(verbose=False)
    with open(OUT, "a") as fh:
        for path in INSTANCES:
            inst = Instance.from_file(path)
            name = os.path.basename(path).removesuffix(".json")
            print(f"\n=== {name}: {len(inst.jobs)} jobs, "
                  f"{len(inst.instance_types)} types, H={inst.horizon}")

            t = time.perf_counter()
            refs = _reference_points(inst, SpaceTimeFormulation(), **EXACT_PARAMS)
            t_ref = time.perf_counter() - t
            print(f"  exact refs f1_T={refs[0]:.2f} f2_T={refs[1]:.4f} "
                  f"f1_0={refs[2]:.2f}  ({t_ref:.1f}s)")
            emit(fh, kind="refs", inst=name, n=len(inst.jobs), refs=refs, wall=t_ref)

            t = time.perf_counter()
            hrefs = heuristic_weighted_reference_points(inst)
            t_href = time.perf_counter() - t
            print(f"  heur  refs f1_T={hrefs[0]:.2f} f2_T={hrefs[1]:.4f} "
                  f"f1_0={hrefs[2]:.2f}  ({t_href:.1f}s)")
            emit(fh, kind="hrefs", inst=name, refs=hrefs, wall=t_href)

            for lam in LAMS:
                t = time.perf_counter()
                ex = solve_weighted_sum(inst, lam, *refs, formulation=SpaceTimeFormulation(),
                                        **EXACT_PARAMS)
                t_ex = time.perf_counter() - t
                g_ex = g_of(lam, ex.f1, ex.f2, refs)
                print(f"  lam={lam:.2f} exact  {ex.status:>8} f1={ex.f1:.2f} f2={ex.f2:.4f} "
                      f"g={g_ex:.6f}  build {ex.build_time:.1f}s solve {ex.solve_time:.1f}s")
                emit(fh, kind="exact", inst=name, lam=lam, status=ex.status, f1=ex.f1,
                     f2=ex.f2, g=g_ex, build=ex.build_time, solve=ex.solve_time, wall=t_ex)

                for pop, ngen in BUDGETS:
                    for seed in SEEDS:
                        t = time.perf_counter()
                        he = weighted_solve(inst, lam, f1_T=refs[0], f2_T=refs[1],
                                            f1_0=refs[2], pop_size=pop, n_gen=ngen, seed=seed)
                        t_he = time.perf_counter() - t
                        g_he = g_of(lam, he.f1, he.f2, refs)
                        gap = 100 * (g_he - g_ex) / abs(g_ex) if abs(g_ex) > 1e-12 else float("nan")
                        print(f"    heur {pop}x{ngen} s{seed:<2} f1={he.f1:.2f} f2={he.f2:.4f} "
                              f"g={g_he:.6f} gap={gap:+.3f}%  {t_he:.1f}s")
                        emit(fh, kind="heur", inst=name, lam=lam, pop=pop, ngen=ngen, seed=seed,
                             f1=he.f1, f2=he.f2, g=g_he, gap=gap, wall=t_he)

                # end-to-end: heuristic reference points, scored with exact g
                t = time.perf_counter()
                he = weighted_solve(inst, lam, f1_T=hrefs[0], f2_T=hrefs[1], f1_0=hrefs[2])
                t_he = time.perf_counter() - t
                g_he = g_of(lam, he.f1, he.f2, refs)
                gap = 100 * (g_he - g_ex) / abs(g_ex) if abs(g_ex) > 1e-12 else float("nan")
                print(f"    heur e2e (heur refs) g={g_he:.6f} gap={gap:+.3f}%  "
                      f"{t_he:.1f}s (+{t_href:.1f}s refs)")
                emit(fh, kind="e2e", inst=name, lam=lam, f1=he.f1, f2=he.f2, g=g_he,
                     gap=gap, wall=t_he)


if __name__ == "__main__":
    main()
