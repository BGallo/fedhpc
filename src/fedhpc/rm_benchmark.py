"""Adapter for the Rm||TWCT+TMUC benchmark (Chen, Chu, Sahli & Li, EJOR 2024).

Unrelated parallel machines with weighted jobs (all released at 0), minimising

    Σ_j w_j C_j  +  Σ_{k used} (U_k + L_k · Σ_{j on k} p_jk)

i.e. total weighted completion time plus machine usage cost (fixed U_k when a
machine is used at all, variable L_k per unit of processing on it).

Mapping onto fedhpc: each machine is a capacity-1 :class:`InstanceType` with
``fixed_cost = U_k``; each job gets ``weight = w_j`` and arrival 0. fedhpc
derives processing times from ``exec_time / perf`` (related machines only), so
the unrelated ``p_jk`` are written straight into the built index dicts
(``p_bill``/``p_occ``/``c``/``T``). With arrivals at 0, fedhpc's f1 is exactly
TWCT and f2 is exactly TMUC, so the benchmark objective is ``f1 + f2``.
"""
from __future__ import annotations

import time

from gurobipy import GRB

from .data import Instance, InstanceType, Job
from .formulations import Formulation, SpaceTimeFormulation
from .model import Solution, _apply_params, _extract, set_mip_start

__all__ = ["RmData", "parse", "to_instance", "evaluate", "wspt_schedule",
           "machine_horizons", "solve_total_cost"]

# (p[j][k], w[j], U[k], L[k])
RmData = tuple[list[list[int]], list[int], list[int], list[int]]


def parse(text: str) -> RmData:
    """Parse the benchmark text format (n rows of p_jk, then w, U, L)."""
    rows = [list(map(int, ln.split())) for ln in text.splitlines() if ln.strip()]
    p, w, U, L = rows[:-3], rows[-3], rows[-2], rows[-1]
    if not (len(p) == len(w) and all(len(r) == len(U) == len(L) for r in p)):
        raise ValueError("malformed Rm||TWCT+TMUC instance")
    return p, w, U, L


def to_instance(data: RmData, horizon: int | list[int]) -> Instance:
    """Build a fedhpc Instance.

    ``horizon`` is one H for every machine or a per-machine list ``H_k`` (see
    :func:`machine_horizons`); job j may start on k at ``t ∈ [0, H_k − p_jk]``
    and pairs with ``p_jk > H_k`` are dropped. The model's horizon is max H_k.
    """
    p, w, U, L = data
    Hk = [horizon] * len(U) if isinstance(horizon, int) else list(horizon)
    horizon = max(Hk)
    big = float("inf")
    types = [
        InstanceType(id=k, kind=f"M{k}", perf=1.0, io_time=0.0, deploy=0.0,
                     cost_io=0.0, cost_vm=float(L[k]), cost_stor=0.0,
                     cpu=big, mem=big, stor=big, capacity=1,
                     fixed_cost=float(U[k]))
        for k in range(len(U))
    ]
    # exec_time is a placeholder: the real p_jk are injected below.
    jobs = [Job(id=j, arrival=0.0, exec_time=1.0, io_volume=0.0, cpu=0.0,
                mem=0.0, stor=0.0, weight=float(w[j])) for j in range(len(w))]
    inst = Instance(jobs=jobs, instance_types=types, horizon=horizon, budget=big)
    inst.build()
    for j in range(len(w)):
        inst.F[j] = [k for k in range(len(U)) if p[j][k] <= Hk[k]]
        if not inst.F[j]:
            raise ValueError(f"horizon {horizon} too short for job {j}")
        for k in range(len(U)):
            del inst.p_bill[j, k], inst.p_occ[j, k], inst.c[j, k], inst.T[j, k]
        for k in inst.F[j]:
            inst.p_bill[j, k] = float(p[j][k])
            inst.p_occ[j, k] = p[j][k]
            inst.c[j, k] = float(L[k] * p[j][k])
            inst.T[j, k] = range(0, Hk[k] - p[j][k] + 1)
    return inst


def evaluate(data: RmData, machine_of: list[int]) -> float:
    """Exact objective of a job→machine assignment, each machine in WSPT order
    (Smith's rule — optimal for a fixed assignment with no release dates)."""
    p, w, U, L = data
    tot = 0.0
    for k, js in enumerate(_wspt_sequences(data, machine_of)):
        if js:
            t = 0
            for j in js:
                t += p[j][k]
                tot += w[j] * t
            tot += U[k] + L[k] * t
    return tot


def _wspt_sequences(data: RmData, machine_of: list[int]) -> list[list[int]]:
    p, w, U, _ = data
    seq: list[list[int]] = [[] for _ in U]
    for j, k in enumerate(machine_of):
        seq[k].append(j)
    for k, js in enumerate(seq):
        js.sort(key=lambda j: p[j][k] / w[j])
    return seq


def wspt_schedule(data: RmData, machine_of: list[int]) -> dict[int, tuple[int, int]]:
    """``{job: (machine, start)}`` for the idle-free WSPT sequence per machine."""
    p = data[0]
    out: dict[int, tuple[int, int]] = {}
    for k, js in enumerate(_wspt_sequences(data, machine_of)):
        t = 0
        for j in js:
            out[j] = (k, t)
            t += p[j][k]
    return out


def machine_horizons(data: RmData, ub: float) -> list[int]:
    """Per-machine load bounds ``H_k`` valid for every solution of cost ≤ ``ub``.

    An optimal schedule is idle-free, so machine k finishes at its load P_k.
    ``dp[P]`` = min over job subsets S with Σ_{j∈S} p_jk = P of
    (Σ_{j∈S} w_j C_j in WSPT order on k) + Σ_{j∉S} min_{k'≠k} (w_j + L_k') p_jk',
    a lower bound on everything except k's usage cost. Loads with
    ``dp[P] + U_k + L_k P > ub`` cannot occur in any solution of cost ≤ ub,
    so the largest remaining P is a valid horizon for machine k.
    """
    import numpy as np

    p, w, U, L = data
    n, m = len(w), len(U)
    out = []
    for k in range(m):
        if m > 1:
            elsewhere = [min((w[j] + L[q]) * p[j][q] for q in range(m) if q != k)
                         for j in range(n)]
        else:
            elsewhere = [float("inf")] * n
        total = sum(p[j][k] for j in range(n))
        P = np.arange(total + 1, dtype=np.float64)
        dp = np.full(total + 1, np.inf)
        dp[0] = sum(elsewhere)
        hi = 0
        for j in sorted(range(n), key=lambda j: p[j][k] / w[j]):   # WSPT on k
            pj = p[j][k]
            cand = dp[:hi + 1] + w[j] * P[pj:hi + pj + 1] - elsewhere[j]
            np.minimum(dp[pj:hi + pj + 1], cand, out=dp[pj:hi + pj + 1])
            hi += pj
        ok = np.nonzero(dp + U[k] + L[k] * P <= ub + 1e-6)[0]
        out.append(int(ok[-1]) if ok.size else 0)
    return out


def solve_total_cost(
    inst: Instance,
    formulation: Formulation | None = None,
    hint: Solution | None = None,
    **params,
) -> tuple[Solution, float | None]:
    """Minimise ``f1 + f2`` (the benchmark objective) with the fedhpc MIP.

    Returns the solution and Gurobi's best bound (``None`` if no incumbent).
    """
    fmt = formulation or SpaceTimeFormulation()
    t0 = time.perf_counter()
    mdl, vars_ = fmt.build(inst, objective=(1.0, 1.0))
    build_time = time.perf_counter() - t0
    _apply_params(mdl, params)
    x = vars_["x"]
    if hint is not None:
        set_mip_start(inst, x, hint)
    mdl.setObjective(fmt.f1_expr(inst, x) + fmt.f2_expr(inst, x), GRB.MINIMIZE)
    t1 = time.perf_counter()
    mdl.optimize()
    sol = _extract(inst, mdl, vars_, build_time=build_time,
                   solve_time=time.perf_counter() - t1)
    return sol, (mdl.ObjBound if mdl.SolCount else None)
