"""Full ablation study on the real 964-job instance.

Multi-objective side  : IGD / eps+ / dominated-fraction vs the 173 proven-optimal
                        points, for NSGA-II / NSGA-III / MOEA/D.
Weighted-sum side      : scalar-g gap to the exact Gurobi optimum at lam in
                        {0.25, 0.5, 0.75} (proven optimal, see
                        scripts/weighted_heur_vs_gurobi.py).

Each row toggles ONE component off (ablate bitmask) or sweeps ONE parameter, so
its contribution is visible in isolation.  A row that matches the baseline within
noise is dead weight on this instance.
"""
from __future__ import annotations
import json
import sys
import time

import numpy as np

from fedhpc.data import Instance
from fedhpc.formulations import configure_env
from fedhpc.moea import (
    moead_frontier, nsga2_frontier, nsga3_frontier, weighted_solve,
)
from fedhpc.pareto import _filter_dominated, _solution_from_dict

sys.stdout.reconfigure(line_buffering=True)

INSTANCE = "data/fedhpc_known_runtime_offline_10min.json"
CHECKPOINTS = ["pareto_runs/fedhpc_10min_map_checkpoint.json",
               "pareto_runs/fedhpc_10min_gap_checkpoint.json"]
POP, NGEN, SEED = 400, 2000, 42

# proven Gurobi optima for the weighted-sum scalar (this instance)
REF = dict(f1_T=11151.0, f2_T=361.96, f1_0=16577.0)
GUR_G = {0.25: 0.7401, 0.50: 1.3895, 0.75: 1.7729}

ABL = {
    "no-heuristic-seeds": 1 << 0,
    "uniform-random-init": 1 << 1,
    "no-local-search": 1 << 2,
    "no-sched-repair": 1 << 3,
    "no-candidate-geom": 1 << 4,
    "no-crossover": 1 << 5,
    "no-ils-kick": 1 << 6,        # weighted only
    "no-elitism": 1 << 7,         # weighted only
    "no-est-shortlist": 1 << 8,   # weighted only
    "no-free-pool-balance": 1 << 9,  # weighted only
}


def load_known():
    sols = []
    for cp in CHECKPOINTS:
        sols += [_solution_from_dict(s) for s in json.load(open(cp))["solutions"]]
    front = _filter_dominated(sols)
    seen, uniq = set(), []
    for s in sorted(front, key=lambda s: s.f1):
        k = (round(s.f1, 4), round(s.f2, 4))
        if k not in seen:
            seen.add(k); uniq.append(s)
    return np.array([(s.f1, s.f2) for s in uniq])


KN = None
IDEAL = NADIR = SPAN = None


def mo_score(front):
    ea = np.array([(s.f1, s.f2) for s in front], dtype=float)
    if len(ea) == 0:
        return dict(card=0, igd=np.inf, eps=np.inf, dom=0, eadom=0)
    e = (ea - IDEAL) / SPAN
    k = (KN - IDEAL) / SPAN
    igd = float(np.mean([np.linalg.norm(e - x, axis=1).min() for x in k]))
    eps = float(max(np.min(np.maximum(e - x, 0.0).max(1)) for x in k))
    dom = sum(1 for p in ea if np.any(np.all(KN <= p, axis=1) & np.any(KN < p, axis=1)))
    eadom = sum(1 for p in ea if np.any(np.all(p <= KN, axis=1) & np.any(p < KN, axis=1)))
    return dict(card=len(ea), igd=igd, eps=eps, dom=dom, eadom=eadom)


def g_of(s, lam):
    return (REF["f1_T"] and
            lam / (REF["f1_0"] - REF["f1_T"]) * s.f1 + (1 - lam) / REF["f2_T"] * s.f2)


def wt_gap(**kw):
    """mean scalar-g gap (%) to Gurobi across the 3 lam, + per-lam."""
    gaps, gs = [], []
    for lam, gg in GUR_G.items():
        s = weighted_solve(INST, lam, f1_T=REF["f1_T"], f2_T=REF["f2_T"],
                           f1_0=REF["f1_0"], **kw)
        g = g_of(s, lam)
        gs.append(g)
        gaps.append(100 * (g - gg) / gg)
    return float(np.mean(gaps)), gaps


INST = None


def main():
    global INST, KN, IDEAL, NADIR, SPAN
    configure_env(verbose=False)
    INST = Instance.from_file(INSTANCE)
    KN = load_known()
    IDEAL = KN.min(0); NADIR = KN.max(0)
    SPAN = np.where(NADIR - IDEAL > 1e-10, NADIR - IDEAL, 1.0)
    print(f"{INSTANCE}: {len(INST.jobs)} jobs   known front {len(KN)} pts\n")

    # ── Multi-objective ablation ────────────────────────────────────────────
    algos = {
        "NSGA-II": lambda **kw: nsga2_frontier(INST, pop_size=POP, n_gen=NGEN, seed=SEED, **kw),
        "NSGA-III": lambda **kw: nsga3_frontier(INST, pop_size=POP, n_divisions=POP - 1,
                                                n_gen=NGEN, seed=SEED, **kw),
        "MOEA/D": lambda **kw: moead_frontier(INST, n_weights=POP, n_gen=NGEN, seed=SEED, **kw),
    }
    mo_variants = [
        ("baseline (sched_repair=1)", dict(sched_repair=1)),
        ("sched_repair=0", dict(sched_repair=0)),
        ("sched_repair=2 (free-pool)", dict(sched_repair=2)),
        ("no-heuristic-seeds", dict(sched_repair=1, ablate=ABL["no-heuristic-seeds"])),
        ("uniform-random-init", dict(sched_repair=1, ablate=ABL["uniform-random-init"])),
        ("no-local-search", dict(sched_repair=1, ablate=ABL["no-local-search"])),
        ("no-candidate-geom", dict(sched_repair=1, ablate=ABL["no-candidate-geom"])),
        ("no-crossover", dict(sched_repair=1, ablate=ABL["no-crossover"])),
        ("uniform-crossover", dict(sched_repair=1, crossover_kind=1)),
        ("local_search_interval=0", dict(sched_repair=1, local_search_interval=0)),
    ]
    for name, fn in algos.items():
        print(f"── {name} " + "─" * 60)
        print(f"  {'variant':32} {'card':>5} {'IGD':>9} {'eps+':>9} {'dom':>5} {'eadom':>5} {'t(s)':>6}")
        base = None
        for label, kw in mo_variants:
            if name == "MOEA/D" and kw.get("crossover_kind") is None and "tourn" in label:
                continue
            t = time.time()
            try:
                fr = fn(**kw)
            except TypeError as e:
                print(f"  {label:32} (skip: {e})"); continue
            dt = time.time() - t
            m = mo_score(fr)
            d_igd = "" if base is None else f" ({(m['igd']-base['igd'])/base['igd']*100:+.1f}%)"
            print(f"  {label:32} {m['card']:>5} {m['igd']:>9.5f} {m['eps']:>9.5f} "
                  f"{m['dom']:>5} {m['eadom']:>5} {dt:>6.1f}{d_igd}")
            if base is None:
                base = m
        # MOEA/D-specific param sweeps
        if name == "MOEA/D":
            for label, kw in [("max_replace=-1", dict(sched_repair=1, max_replace=-1)),
                              ("archive_size=0", dict(sched_repair=1, archive_size=0)),
                              ("neighborhood=10", dict(sched_repair=1, neighborhood_size=10))]:
                t = time.time(); fr = fn(**kw); dt = time.time() - t
                m = mo_score(fr)
                print(f"  {label:32} {m['card']:>5} {m['igd']:>9.5f} {m['eps']:>9.5f} "
                      f"{m['dom']:>5} {m['eadom']:>5} {dt:>6.1f}")
        else:
            for label, kw in [("tourn_k=3", dict(sched_repair=1, tourn_k=3)),
                              ("tourn_k=4", dict(sched_repair=1, tourn_k=4))]:
                t = time.time(); fr = fn(**kw); dt = time.time() - t
                m = mo_score(fr)
                print(f"  {label:32} {m['card']:>5} {m['igd']:>9.5f} {m['eps']:>9.5f} "
                      f"{m['dom']:>5} {m['eadom']:>5} {dt:>6.1f}")
        print()

    # ── Weighted-sum ablation ──────────────────────────────────────────────
    print("── weighted_solve  (mean scalar-g gap to Gurobi optimum over lam=0.25/0.5/0.75) " + "─" * 10)
    print(f"  {'variant':32} {'mean gap%':>10}   per-lam gap%")
    wt_variants = [
        ("baseline", {}),
        ("no-free-pool-balance", dict(ablate=ABL["no-free-pool-balance"])),
        ("no-heuristic-seeds", dict(ablate=ABL["no-heuristic-seeds"])),
        ("uniform-random-init", dict(ablate=ABL["uniform-random-init"])),
        ("no-local-search", dict(ablate=ABL["no-local-search"])),
        ("no-crossover (multistart-ILS)", dict(ablate=ABL["no-crossover"])),
        ("no-ils-kick", dict(ablate=ABL["no-ils-kick"])),
        ("no-elitism", dict(ablate=ABL["no-elitism"])),
        ("no-est-shortlist", dict(ablate=ABL["no-est-shortlist"])),
        ("restart_patience=0", dict(restart_patience=0)),
        ("ls_moves=3", dict(ls_moves=3)),
        ("ls_moves=12", dict(ls_moves=12)),
        ("shortlist=12", dict(shortlist=12)),
        ("shortlist=48", dict(shortlist=48)),
        ("pop=12 n_gen=80", dict(pop_size=12, n_gen=80)),
        ("pop=48 n_gen=40", dict(pop_size=48, n_gen=40)),
        ("HEAVY pop=48 gen=120 lsm=12 sl=40", dict(pop_size=48, n_gen=120, ls_moves=12, shortlist=40)),
    ]
    for label, kw in wt_variants:
        t = time.time()
        mean_gap, per = wt_gap(**kw)
        dt = time.time() - t
        print(f"  {label:32} {mean_gap:>+9.2f}%   "
              f"[{per[0]:+.2f} {per[1]:+.2f} {per[2]:+.2f}]  {dt:.0f}s")


if __name__ == "__main__":
    main()
