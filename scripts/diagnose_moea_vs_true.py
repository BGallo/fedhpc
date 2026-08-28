"""Diagnose EA points that appear to beat the exact true front on medium.json:
recompute objectives from the assignment, check budget + capacity, and report
any EA solution that strictly dominates a true Pareto point.
"""
import sys

from fedhpc.data import Instance
from fedhpc.formulations import configure_env
from fedhpc.moea import moead_frontier, nsga2_frontier, nsga3_frontier
from fedhpc.pareto import true_pareto_frontier

PATH = sys.argv[1] if len(sys.argv) > 1 else "data/medium.json"
configure_env(verbose=False)
inst = Instance.from_file(PATH)


def recompute_f1(sol):
    return sum(sol.completion[j.id] - j.arrival for j in inst.jobs)


def recompute_f2(sol):
    return sum(inst.c[jid, mid] for jid, (mid, _) in sol.assignment.items())


def capacity_violations(sol):
    occ = {}
    for jid, (mid, t) in sol.assignment.items():
        for slot in range(t, t + inst.p_occ[jid, mid]):
            occ[(mid, slot)] = occ.get((mid, slot), 0) + 1
    v = []
    for (mid, slot), cnt in occ.items():
        init = inst.occupied.get((mid, slot), 0)
        cap = next(x for x in inst.instance_types if x.id == mid).capacity
        if cap is not None and cnt + init > cap:
            v.append((mid, slot, cnt + init, cap))
    return v


def horizon_violations(sol):
    return [(jid, sol.completion[jid]) for jid in sol.completion
            if sol.completion[jid] > inst.horizon]


true_front = true_pareto_frontier(inst, verbose=False, OutputFlag=0)
true_pts = sorted((s.f1, s.f2) for s in true_front)
print(f"{PATH}: budget={inst.budget} horizon={inst.horizon}")
print(f"true front ({len(true_pts)} pts):")
for f1, f2 in true_pts:
    print(f"   f1={f1:8.2f}  f2={f2:8.2f}")

for name, fn in [
    ("NSGA-II", lambda: nsga2_frontier(inst, pop_size=200, n_gen=300, seed=42)),
    ("NSGA-III", lambda: nsga3_frontier(inst, pop_size=200, n_divisions=199, n_gen=300, seed=42)),
    ("MOEA/D", lambda: moead_frontier(inst, n_weights=200, n_gen=300, seed=42)),
]:
    print(f"\n=== {name} ===")
    front = fn()
    for s in sorted(front, key=lambda s: s.f1):
        rf1, rf2 = recompute_f1(s), recompute_f2(s)
        f1_ok = abs(rf1 - s.f1) < 1e-6
        f2_ok = abs(rf2 - s.f2) < 1e-6
        cv = capacity_violations(s)
        hv = horizon_violations(s)
        over_budget = s.f2 > inst.budget + 1e-6
        beats = [(tf1, tf2) for tf1, tf2 in true_pts
                 if s.f1 <= tf1 and s.f2 <= tf2 and (s.f1 < tf1 or s.f2 < tf2)]
        flags = []
        if not f1_ok: flags.append(f"f1!={rf1:.3f}")
        if not f2_ok: flags.append(f"f2!={rf2:.3f}")
        if cv: flags.append(f"CAP{cv[:2]}")
        if hv: flags.append(f"HORIZON{hv[:2]}")
        if over_budget: flags.append("OVERBUDGET")
        if beats: flags.append(f"DOMINATES_TRUE{beats}")
        tag = "  <-- " + " ".join(flags) if flags else ""
        print(f"   f1={s.f1:8.2f}  f2={s.f2:8.2f}{tag}")
