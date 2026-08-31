"""Test the evolutionary MOEAs against the PRE-COMPUTED exact Pareto frontier
for data/fedhpc_known_runtime_offline_10min.json.

The true front here is expensive to enumerate (964 jobs, 29 types — see
pareto_runs/STATUS.md), so instead of re-running true_pareto_frontier() we
load the proven-optimal points already found and checkpointed in
    pareto_runs/fedhpc_10min_map_checkpoint.json   (main region, 26 pts)
    pareto_runs/fedhpc_10min_gap_checkpoint.json   (the big f1 gap, +7 pts)
→ 33 unique points, every one an exact MIP optimum (MIPGap 1e-9).

IMPORTANT: that combined front is EXACT-but-INCOMPLETE — ~30 boxes are still
open, so interior points are missing. That changes which indicators mean
something:

  reliable (every known point is a proven global optimum):
    dom_by_known  # EA points STRICTLY DOMINATED by a known point   → real failure, want 0
    ea_dominates  # EA points that dominate a known point            → must be 0 (else bug)
    exact         # EA points coinciding with a known point (<1e-4 rel)
    IGD           mean dist from each KNOWN point to nearest EA point (coverage of proven pts)
    eps+          additive epsilon to weakly cover all known points

  unreliable here (known front is sparse/incomplete — flagged, not trusted):
    GD, HVR       an EA point filling an unmapped gap looks "far"/"extra"
"""
from __future__ import annotations

import json
import time

import numpy as np

from fedhpc.data import Instance
from fedhpc.formulations import configure_env
from fedhpc.moea import moead_frontier, nsga2_frontier, nsga3_frontier
from fedhpc.pareto import _filter_dominated, _solution_from_dict

INSTANCE = "data/fedhpc_known_runtime_offline_10min.json"
CHECKPOINTS = [
    "pareto_runs/fedhpc_10min_map_checkpoint.json",
    "pareto_runs/fedhpc_10min_gap_checkpoint.json",
]
import os

POP = int(os.environ.get("POP", 400))
NGEN = int(os.environ.get("NGEN", 2000))
SEED = int(os.environ.get("SEED", 42))
HV_REF = np.array([1.1, 1.1])


def load_known_front():
    sols = []
    for cp in CHECKPOINTS:
        with open(cp) as f:
            d = json.load(f)
        sols += [_solution_from_dict(s) for s in d["solutions"]]
    front = _filter_dominated(sols)
    # dedupe identical (f1, f2)
    seen, uniq = set(), []
    for s in sorted(front, key=lambda s: s.f1):
        k = (round(s.f1, 4), round(s.f2, 4))
        if k not in seen:
            seen.add(k)
            uniq.append(s)
    return uniq


def _pts(front):
    return np.array([(s.f1, s.f2) for s in front if s.f1 is not None], dtype=float)


def _hv2d(pts, ref):
    if len(pts) == 0:
        return 0.0
    s = pts[np.argsort(pts[:, 0])]
    hv = 0.0
    for i in range(len(s)):
        if s[i, 0] >= ref[0] or s[i, 1] >= ref[1]:
            continue
        f1_next = ref[0] if i == len(s) - 1 else min(s[i + 1, 0], ref[0])
        hv += max(f1_next - s[i, 0], 0.0) * max(ref[1] - s[i, 1], 0.0)
    return max(hv, 0.0)


def _strictly_dominates(a, b):  # a dominates b (minimisation)
    return np.all(a <= b) and np.any(a < b)


def score(ea_pts, known_pts, ideal, span):
    ea = (ea_pts - ideal) / span
    kn = (known_pts - ideal) / span

    d_igd = [np.linalg.norm(ea - k, axis=1).min() for k in kn] if len(ea) else [np.inf]
    d_gd = [np.linalg.norm(kn - e, axis=1).min() for e in ea] if len(ea) else [np.inf]
    eps_plus = (max(min(max(e - k) for e in ea) for k in kn)
                if len(ea) else np.inf)

    exact = sum(
        1 for e in ea_pts
        if np.any(np.all(np.abs(known_pts - e) <= 1e-4 + 1e-4 * np.abs(known_pts), axis=1))
    )
    dom_by_known = sum(1 for e in ea_pts
                       if any(_strictly_dominates(k, e) for k in known_pts))
    ea_dominates = sum(1 for e in ea_pts
                       if any(_strictly_dominates(e, k) for k in known_pts))

    hv_k = _hv2d(kn, HV_REF)
    hv_e = _hv2d(ea, HV_REF)
    return dict(
        card=len(ea_pts), exact=exact,
        dom_by_known=dom_by_known, ea_dominates=ea_dominates,
        igd=float(np.mean(d_igd)), gd=float(np.mean(d_gd)),
        eps_plus=float(eps_plus),
        hvr=float(hv_e / hv_k) if hv_k > 0 else float("nan"),
    )


def recompute_f1(inst, s):
    return sum(s.completion[j.id] - j.arrival for j in inst.jobs)


def recompute_f2(inst, s):
    return sum(inst.c[jid, mid] for jid, (mid, _) in s.assignment.items())


def main():
    configure_env(verbose=False)
    inst = Instance.from_file(INSTANCE)
    known = load_known_front()
    kpts = _pts(known)
    print(f"{INSTANCE}\n  {len(inst.jobs)} jobs, {len(inst.instance_types)} types, "
          f"H={inst.horizon}, B={inst.budget}")
    print(f"  known exact front: {len(kpts)} proven-optimal points  "
          f"(EXACT-but-INCOMPLETE — see pareto_runs/STATUS.md)")
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
           f"{'ea_dom':>6}  {'IGD':>8}  {'eps+':>8}  {'(GD)':>8}  {'(HVR)':>7}  {'t(s)':>7}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name, fn in algos.items():
        ts = time.time()
        front = fn()
        dt = time.time() - ts
        # feasibility sanity: recompute objectives
        bad = sum(1 for s in front
                  if abs(recompute_f1(inst, s) - s.f1) > 1e-6
                  or abs(recompute_f2(inst, s) - s.f2) > 1e-6)
        ea_pts = _pts(front)
        m = score(ea_pts, kpts, ideal, span)
        note = f"  !! {bad} recompute-mismatch" if bad else ""
        print(f"  {name}  {m['card']:>4}  {m['exact']:>5}  {m['dom_by_known']:>12}  "
              f"{m['ea_dominates']:>6}  {m['igd']:>8.5f}  {m['eps_plus']:>8.5f}  "
              f"{m['gd']:>8.5f}  {m['hvr']:>7.4f}  {dt:>7.1f}{note}")

    print("\n  IGD / eps+ / dom_by_known / exact are trustworthy (known pts are proven optimal).")
    print("  GD and HVR are parenthesised: the known front is incomplete, so an EA point")
    print("  in an unmapped gap inflates GD and can push HVR above 1 without being wrong.")


if __name__ == "__main__":
    main()
