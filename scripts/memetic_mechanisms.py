"""Diagnose weighted_solve_memetic against the dispatch ILS on given cells:
the memetic's initial gene-1 share, best pen_g per priority after its initial
LS (EDF vs best mu), EDF share and best per generation, crossover share of the
decodes and the final polish's gain; beside the ILS's LS-only value from the
all-auto genome for its mu start and its EDF start (no kicks), and full runs.

    uv run python scripts/memetic_mechanisms.py NAME:LAM ... [--adapt 7] [--flags 12352]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dispatch_diagnose import BENCH  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cells", nargs="+")
    ap.add_argument("--adapt", type=int, default=7)
    ap.add_argument("--flags", type=int, default=64 | 4096 | 8192)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--kw", default="{}", help="extra weighted_solve_memetic kwargs (JSON)")
    a = ap.parse_args()
    from fedhpc import instance_gen as gen
    from fedhpc.data import Instance
    from fedhpc.moea import weighted_solve_dispatch, weighted_solve_memetic

    bench = {r["name"]: r for r in map(json.loads, BENCH.read_text().splitlines())}
    for cell in a.cells:
        name, lam = cell.split(":")
        rec = bench[name]
        inst = Instance.from_file(gen.DATA / "generated" / f"{name}.json")
        ref = dict(f1_T=rec["lex_f1"]["f1"], f2_T=rec["lex_f1"]["f2"], f1_0=rec["lex_f2"]["f1"])
        L = float(lam)
        # pen_g -> gap: pen_g = w1 f1 + w2 f2 (+ penalties), g = pen_g - w1 f1_T
        w1 = L / (ref["f1_0"] - ref["f1_T"])
        gs = rec["lams"][lam]["gurobi"]["g"]
        gap = lambda pg: 100 * ((pg - w1 * ref["f1_T"]) - gs) / gs  # noqa: E731
        _, st = weighted_solve_memetic(inst, L, n_threads=a.threads, flags=a.flags, adapt=a.adapt,
                                       return_stats=True, **ref, **json.loads(a.kw))
        ib = st["init_best"]
        gens = st["gens"]
        _, mu_ls = weighted_solve_dispatch(inst, L, n_threads=a.threads, flags=64, n_kicks=0,
                                           return_stats=True, **ref)
        _, edf_ls = weighted_solve_dispatch(inst, L, n_threads=a.threads, flags=64 | 2048 | 8192,
                                            n_kicks=0, return_stats=True, **ref)
        _, ils = weighted_solve_dispatch(inst, L, n_threads=a.threads, return_stats=True, **ref)
        print(f"{cell}: memetic final {gap(st['pen_g']):.1f}% (mu {st['mu']}) | "
              f"ILS {gap(ils['pen_g']):.1f}% (start {ils['mu']}); "
              f"ILS LS-only from all-auto: mu {gap(mu_ls['pen_g']):.1f}% EDF {gap(edf_ls['pen_g']):.1f}%")
        print(f"  init: gene-1 share {st['init_cloud']:.2f}; best after init LS: "
              f"EDF {gap(ib[-1]):.1f}% vs best mu {gap(min(ib[:-1])):.1f}%")
        traj = [(g, n, gap(be) if n else None, gap(bo)) for g, n, be, bo in gens]
        pick = [traj[i] for i in sorted({0, 1, 2, 5, 10, 20, 40, len(traj) - 1}) if i < len(traj)]
        print("  gen (n_EDF, EDF best, others best): " + "  ".join(
            f"{g}:({n}, {'-' if e is None else f'{e:.1f}'}, {o:.1f})" for g, n, e, o in pick))
        print(f"  decodes: crossover {st['dec_x'] / st['decodes']:.0%} of {st['decodes']}; "
              f"polish gain {100 * st['polish_gain'] / gs:.2f} pp; time {st['total_ms'] / 1e3:.1f}s "
              f"(init {st['init_ms'] / 1e3:.1f}s) "
              f"vs ILS {ils['total_ms'] / 1e3:.1f}s", flush=True)


if __name__ == "__main__":
    main()
