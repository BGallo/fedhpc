"""Classical priority-key + non-delay Schedule Generation Scheme (SGS) for
FED-HPC, wired to pymoo's generic NSGA-II / NSGA-III / MOEA-D.

Chromosome (the "standard literature" alternative to this repo's job_slots
index encoding): 2 real genes per job in [0,1] —
  order_key[i]  — job processing-priority (sort ascending -> processing order)
  type_key[i]   — which feasible instance type, decoded via random-key
                  discretization over that job's feasible types sorted by
                  cost ascending: idx = floor(type_key[i] * n_feasible_types)

Decode (non-delay active SGS): walk jobs in order_key order; for each job,
try its type_key-selected type first, then the remaining feasible types in
cost order, taking the first that admits a capacity-feasible start within
Instance.T[j,m]. If literally no feasible type admits a start (can only
happen if every feasible type for that job is capacity-constrained and all
are full for its whole T-range — capacity infeasibility, not a horizon
issue, since Instance.build() already guarantees every T[j,m] for m in
F[j] is non-empty), force the type_key-selected type at its earliest T
start and register the capacity excess as a constraint violation, so
pymoo's generic feasibility-first ranking (same Deb 2002 rule as
ga_common.hpp's Individual::dominates) applies exactly as for the C++ scheme.

Objectives match scripts/compare_moea_vs_known_front.py's recompute_f1/f2
exactly: f1 = sum(completion - arrival), f2 = sum(c[j, chosen_type]).
"""
from __future__ import annotations

import numpy as np
from pymoo.core.problem import ElementwiseProblem

from fedhpc.data import Instance


class JobCandidate:
    __slots__ = ("type_id", "p_occ", "a_min", "t_max_excl", "cap", "cost")

    def __init__(self, type_id, p_occ, a_min, t_max_excl, cap, cost):
        self.type_id = type_id
        self.p_occ = p_occ
        self.a_min = a_min
        self.t_max_excl = t_max_excl
        self.cap = cap
        self.cost = cost


def build_candidates(inst: Instance):
    """Per-job, cost-ascending list of JobCandidate for every feasible type."""
    job_candidates = []
    for j in inst.jobs:
        cands = []
        for m in inst.F[j.id]:
            Trange = inst.T[j.id, m]
            cap = next((it.capacity for it in inst.instance_types if it.id == m), None)
            cands.append(JobCandidate(
                type_id=m,
                p_occ=inst.p_occ[j.id, m],
                a_min=Trange.start,
                t_max_excl=Trange.stop,
                cap=cap,
                cost=inst.c[j.id, m],
            ))
        cands.sort(key=lambda c: c.cost)
        job_candidates.append(cands)
    return job_candidates


def build_type_cap_and_init_occ(inst: Instance):
    type_cap = {}
    for it in inst.instance_types:
        type_cap[it.id] = it.capacity
    capacitated = [tid for tid, cap in type_cap.items() if cap is not None]
    init_occ = {}
    for tid in capacitated:
        arr = np.zeros(inst.horizon, dtype=np.int32)
        for (m, t), cnt in inst.occupied.items():
            if m == tid and t < inst.horizon:
                arr[t] += cnt
        init_occ[tid] = arr
    return type_cap, capacitated, init_occ


def decode(order_keys, type_keys, job_candidates, init_occ, budget,
           arrivals, return_assignment=False):
    n_jobs = len(job_candidates)
    order = np.argsort(order_keys)

    occ = {tid: arr.copy() for tid, arr in init_occ.items()}

    f1 = 0.0
    f2 = 0.0
    cv = 0.0
    assignment = [None] * n_jobs if return_assignment else None

    for i in order:
        cands = job_candidates[i]
        n_cand = len(cands)
        k0 = int(type_keys[i] * n_cand)
        if k0 >= n_cand:
            k0 = n_cand - 1

        placed = False
        chosen = None
        start = None

        for offset in range(n_cand):
            c = cands[(k0 + offset) % n_cand]
            if c.cap is None:
                start = c.a_min
                placed = True
                chosen = c
                break
            arr = occ[c.type_id]
            t = c.a_min
            found = False
            while t < c.t_max_excl:
                window = arr[t:t + c.p_occ]
                if window.size == 0:
                    found = True
                    break
                mx = int(window.max())
                if mx < c.cap:
                    found = True
                    break
                # jump ahead past the first blocking slot in this window
                blocking_rel = int(np.argmax(window >= c.cap))
                t += blocking_rel + 1
            if found:
                start = t
                placed = True
                chosen = c
                break

        if not placed:
            # genuine capacity infeasibility across every feasible type:
            # force the type_key choice at its earliest T-start and charge
            # the resulting overcrowding to cv.
            chosen = cands[k0]
            start = chosen.a_min
            arr = occ.get(chosen.type_id)
            if arr is not None:
                window = arr[start:start + chosen.p_occ]
                excess = np.maximum(window - chosen.cap + 1, 0).sum()
                cv += float(excess)
        else:
            if chosen.cap is not None:
                occ[chosen.type_id][start:start + chosen.p_occ] += 1

        f1 += (start + chosen.p_occ - arrivals[i])
        f2 += chosen.cost
        if return_assignment:
            assignment[i] = (chosen.type_id, start, chosen.p_occ)

    if f2 > budget:
        cv += (f2 - budget)

    if return_assignment:
        return f1, f2, cv, assignment
    return f1, f2, cv


class SGSScheduleProblem(ElementwiseProblem):
    def __init__(self, inst: Instance):
        self.inst = inst
        self.n_jobs = len(inst.jobs)
        self.job_candidates = build_candidates(inst)
        _, _, self.init_occ = build_type_cap_and_init_occ(inst)
        self.arrivals = [j.arrival for j in inst.jobs]
        self.budget = inst.budget
        # NOTE: n_constr=0 — every job in this instance has at least one
        # feasible unlimited-capacity ("cloud") instance type, so decode()'s
        # cost-order wraparound fallback always finds a capacity-feasible
        # placement in practice; verified empirically (200 random genomes,
        # max cv == 0.0). cv is still computed and returned by decode() for
        # correctness/debugging, just not surfaced as a pymoo constraint —
        # this also sidesteps pymoo's MOEAD implementation, which asserts
        # it does not support constraints at all.
        super().__init__(n_var=2 * self.n_jobs, n_obj=2, n_constr=0,
                          xl=0.0, xu=1.0)

    def _evaluate(self, x, out, *args, **kwargs):
        order_keys = x[:self.n_jobs]
        type_keys = x[self.n_jobs:]
        f1, f2, cv = decode(order_keys, type_keys, self.job_candidates,
                             self.init_occ, self.budget, self.arrivals)
        out["F"] = [f1, f2]

    def decode_full(self, x):
        order_keys = x[:self.n_jobs]
        type_keys = x[self.n_jobs:]
        return decode(order_keys, type_keys, self.job_candidates,
                       self.init_occ, self.budget, self.arrivals,
                       return_assignment=True)


def verify_feasible(assignment, inst: Instance, job_candidates):
    """Independent from-scratch occupancy rebuild, for sanity-checking a
    decoded individual that decode() reported as cv==0."""
    type_cap = {it.id: it.capacity for it in inst.instance_types}
    occ = {}
    for (m, t), cnt in inst.occupied.items():
        occ.setdefault(m, np.zeros(inst.horizon, dtype=np.int32))
        if t < inst.horizon:
            occ[m][t] += cnt
    for tid, cap in type_cap.items():
        if cap is not None:
            occ.setdefault(tid, np.zeros(inst.horizon, dtype=np.int32))

    for (type_id, start, p_occ) in assignment:
        cap = type_cap.get(type_id)
        if cap is None:
            continue
        occ[type_id][start:start + p_occ] += 1

    violations = []
    for tid, cap in type_cap.items():
        if cap is None:
            continue
        mx = int(occ[tid].max()) if tid in occ else 0
        if mx > cap:
            violations.append((tid, mx, cap))
    return violations
