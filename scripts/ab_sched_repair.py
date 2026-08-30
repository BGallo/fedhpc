"""A/B: sched_repair on/off for all three EAs, scored vs the known proven front."""
from __future__ import annotations
import os, time
import numpy as np
from fedhpc.data import Instance
from fedhpc.formulations import configure_env
from fedhpc.moea import moead_frontier, nsga2_frontier, nsga3_frontier
from fedhpc.pareto import _filter_dominated, _solution_from_dict
import json

INSTANCE = "data/fedhpc_known_runtime_offline_10min.json"
CHECKPOINTS = ["pareto_runs/fedhpc_10min_map_checkpoint.json",
               "pareto_runs/fedhpc_10min_gap_checkpoint.json"]
POP = int(os.environ.get("POP", 400))
NGEN = int(os.environ.get("NGEN", 2000))
SEED = int(os.environ.get("SEED", 42))


def load_known_front():
    sols = []
    for cp in CHECKPOINTS:
        d = json.load(open(cp))
        sols += [_solution_from_dict(s) for s in d["solutions"]]
    front = _filter_dominated(sols)
    seen, uniq = set(), []
    for s in sorted(front, key=lambda s: s.f1):
        k = (round(s.f1, 4), round(s.f2, 4))
        if k not in seen:
            seen.add(k); uniq.append(s)
    return uniq


def _pts(front):
    return np.array([(s.f1, s.f2) for s in front if s.f1 is not None], dtype=float)


def _strictly_dominates(a, b):
    return np.all(a <= b) and np.any(a < b)


def score(ea_pts, known_pts, ideal, span):
    ea = (ea_pts - ideal) / span
    kn = (known_pts - ideal) / span
    d_igd = [np.linalg.norm(ea - k, axis=1).min() for k in kn] if len(ea) else [np.inf]
    eps_plus = (max(min(max(e - k) for e in ea) for k in kn) if len(ea) else np.inf)
    dom_by_known = sum(1 for e in ea_pts if any(_strictly_dominates(k, e) for k in known_pts))
    ea_dominates = sum(1 for e in ea_pts if any(_strictly_dominates(e, k) for k in known_pts))
    exact = sum(1 for e in ea_pts
                if np.any(np.all(np.abs(known_pts - e) <= 1e-4 + 1e-4 * np.abs(known_pts), axis=1)))
    return dict(card=len(ea_pts), exact=exact, dom_by_known=dom_by_known,
                ea_dominates=ea_dominates, igd=float(np.mean(d_igd)), eps_plus=float(eps_plus))


def main():
    configure_env(verbose=False)
    inst = Instance.from_file(INSTANCE)
    known = load_known_front()
    kpts = _pts(known)
    ideal = kpts.min(axis=0); nadir = kpts.max(axis=0)
    span = np.where(nadir - ideal > 1e-10, nadir - ideal, 1.0)
    print(f"pop={POP} n_gen={NGEN} seed={SEED}  known={len(kpts)} pts")
    algos = {
        "NSGA-II ": lambda sr: nsga2_frontier(inst, pop_size=POP, n_gen=NGEN, seed=SEED, sched_repair=sr),
        "NSGA-III": lambda sr: nsga3_frontier(inst, pop_size=POP, n_divisions=POP - 1, n_gen=NGEN, seed=SEED, sched_repair=sr),
        "MOEA/D  ": lambda sr: moead_frontier(inst, n_weights=POP, n_gen=NGEN, seed=SEED, sched_repair=sr),
    }
    print(f"  {'algo':8} {'sr':>3}  {'card':>4} {'exact':>5} {'dom':>4} {'eadom':>5} {'IGD':>9} {'eps+':>9}  {'t(s)':>6}")
    for name, fn in algos.items():
        for sr in (0, 1, 2):
            ts = time.time()
            try:
                front = fn(sr)
            except TypeError as e:
                print(f"  {name} sr={sr}: not wired ({e})"); continue
            dt = time.time() - ts
            m = score(_pts(front), kpts, ideal, span)
            print(f"  {name} {sr:>3}  {m['card']:>4} {m['exact']:>5} {m['dom_by_known']:>4} "
                  f"{m['ea_dominates']:>5} {m['igd']:>9.5f} {m['eps_plus']:>9.5f}  {dt:>6.1f}")


if __name__ == "__main__":
    main()
