"""Test the evolutionary MOEAs against the (partial) exact Pareto frontier for
data/pos_congestion_known_runtime_offline_10min.json (28-type config, fat
node removed — see pareto_runs/STATUS_pos_congestion.md).

Same method as scripts/compare_moea_vs_known_front.py (the fedhpc version);
see that file's docstring for what each metric does and doesn't mean when
the known front is incomplete. As of this write the known front here is
VERY sparse (5 points, 4 wide-open boxes) — treat IGD/eps+ as rough, and
dom_by_known / exact / ea_dominates as the trustworthy ones.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from fedhpc.data import Instance
from fedhpc.formulations import configure_env
from fedhpc.moea import moead_frontier, nsga2_frontier, nsga3_frontier
from fedhpc.pareto import _filter_dominated, _solution_from_dict

INSTANCE = "data/pos_congestion_known_runtime_offline_10min.json"
CHECKPOINTS = [
    "pareto_runs/pos_congestion_10min_map_checkpoint.json",
    "pareto_runs/pos_congestion_10min_gap_checkpoint.json",  # not seeded yet; skipped if missing
]

POP = int(os.environ.get("POP", 100))
NGEN = int(os.environ.get("NGEN", 100))
SEED = int(os.environ.get("SEED", 42))
HV_REF = np.array([1.1, 1.1])


def load_known_front():
    sols = []
    for cp in CHECKPOINTS:
        if not os.path.exists(cp):
            continue
        with open(cp) as f:
            d = json.load(f)
        sols += [_solution_from_dict(s) for s in d["solutions"]]
    front = _filter_dominated(sols)
    seen, uniq = set(), []
    for s in sorted(front, key=lambda s: s.f1):
        k = (round(s.f1, 4), round(s.f2, 4))
        if k not in seen:
            seen.add(k)
            uniq.append(s)
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
    exact = sum(1 for e in ea_pts
                if np.any(np.all(np.abs(known_pts - e) <= 1e-4 + 1e-4 * np.abs(known_pts), axis=1)))
    dom_by_known = sum(1 for e in ea_pts if any(_strictly_dominates(k, e) for k in known_pts))
    ea_dominates = sum(1 for e in ea_pts if any(_strictly_dominates(e, k) for k in known_pts))
    return dict(card=len(ea_pts), exact=exact, dom_by_known=dom_by_known,
                ea_dominates=ea_dominates, igd=float(np.mean(d_igd)), eps_plus=float(eps_plus))


def main():
    configure_env(verbose=False)
    inst = Instance.from_file(INSTANCE)
    known = load_known_front()
    kpts = _pts(known)
    print(f"{INSTANCE}\n  {len(inst.jobs)} jobs, {len(inst.instance_types)} types, "
          f"H={inst.horizon}, B={inst.budget}")
    print(f"  known exact front: {len(kpts)} proven-optimal points (PARTIAL — early-stage map)")
    print(f"    f1 in [{kpts[:,0].min():.2f}, {kpts[:,0].max():.2f}]   "
          f"f2 in [{kpts[:,1].min():.2f}, {kpts[:,1].max():.2f}]")

    ideal = kpts.min(axis=0)
    nadir = kpts.max(axis=0)
    span = np.where(nadir - ideal > 1e-10, nadir - ideal, 1.0)

    algos = {
        "NSGA-II ": lambda: nsga2_frontier(inst, pop_size=POP, n_gen=NGEN, seed=SEED),
        "NSGA-III": lambda: nsga3_frontier(inst, pop_size=POP, n_divisions=POP - 1,
                                           n_gen=NGEN, seed=SEED),
        "MOEA/D  ": lambda: moead_frontier(inst, n_weights=POP, n_gen=NGEN, seed=SEED),
    }
    print(f"\n  EA params: pop={POP}, n_gen={NGEN}, seed={SEED}")
    hdr = (f"  {'algo':8}  {'card':>4}  {'exact':>5}  {'dom_by_known':>12}  "
           f"{'ea_dom':>6}  {'IGD':>8}  {'eps+':>8}  {'t(s)':>7}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name, fn in algos.items():
        ts = time.time()
        front = fn()
        dt = time.time() - ts
        ea_pts = _pts(front)
        m = score(ea_pts, kpts, ideal, span)
        print(f"  {name}  {m['card']:>4}  {m['exact']:>5}  {m['dom_by_known']:>12}  "
              f"{m['ea_dominates']:>6}  {m['igd']:>8.5f}  {m['eps_plus']:>8.5f}  {dt:>7.1f}")

    print("\n  dom_by_known = EA points strictly dominated by a proven point (should be 0..few).")
    print("  ea_dominates must be 0 (else a bug: an EA point beat a proven global optimum).")


if __name__ == "__main__":
    main()
