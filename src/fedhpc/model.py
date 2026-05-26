"""Solver functions for FED-HPC.

Each ``solve_*`` function accepts an optional *formulation* argument
(:class:`~fedhpc.formulations.Formulation`) that controls which MIP constraint
structure is built.  When omitted, :class:`~fedhpc.formulations.SpaceTimeFormulation`
is used (the space-time network formulation from fedhpc-1.pdf).

Decision variables exposed by every formulation
------------------------------------------------
x_jmt ∈ {0,1}  : 1 if job j starts on type m at time t
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import gurobipy as gp
from gurobipy import GRB

from .data import Instance
from .formulations import Formulation, SpaceTimeFormulation


@dataclass
class Solution:
    status: str
    objective: float | None
    f1: float | None                        # total turnaround (slots)
    f2: float | None                        # total monetary cost
    assignment: dict[int, tuple[int, int]]  # job_id -> (type_id, start_time)
    completion: dict[int, int]              # job_id -> completion time (integer)
    build_time: float | None = field(default=None)   # wall-clock seconds for MIP construction
    solve_time: float | None = field(default=None)   # wall-clock seconds inside mdl.optimize()

    def __str__(self) -> str:
        lines = [
            f"Status    : {self.status}",
            f"Objective : {self.objective}",
            f"f1 (turnaround) : {self.f1}",
            f"f2 (cost)       : {self.f2}",
            "Assignment:",
        ]
        for jid, (mid, t) in sorted(self.assignment.items()):
            c = self.completion.get(jid, "?")
            lines.append(f"  job {jid} -> type {mid}  start={t}  end={c}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _default_formulation() -> Formulation:
    return SpaceTimeFormulation()


def _apply_params(mdl: gp.Model, params: dict) -> None:
    for k, v in params.items():
        mdl.setParam(k, v)


def _extract(
    inst: Instance,
    mdl: gp.Model,
    vars: dict,
    build_time: float | None = None,
    solve_time: float | None = None,
) -> Solution:
    status_map = {
        GRB.OPTIMAL: "optimal",
        GRB.SUBOPTIMAL: "feasible",
        GRB.INFEASIBLE: "infeasible",
        GRB.UNBOUNDED: "unbounded",
    }
    status = status_map.get(mdl.Status, f"gurobi_status_{mdl.Status}")

    if mdl.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        return Solution(
            status=status, objective=None, f1=None, f2=None,
            assignment={}, completion={},
            build_time=build_time, solve_time=solve_time,
        )

    x = vars["x"]
    assignment: dict[int, tuple[int, int]] = {}
    completion: dict[int, int] = {}

    for j in inst.jobs:
        for m_id in inst.F[j.id]:
            for t in inst.T[j.id, m_id]:
                if x[j.id, m_id, t].X > 0.5:
                    assignment[j.id] = (m_id, t)
                    completion[j.id] = t + inst.p_occ[j.id, m_id]

    f1 = sum(completion[j.id] - j.arrival for j in inst.jobs if j.id in completion)
    f2 = sum(
        inst.c[j.id, m_id] * x[j.id, m_id, t].X
        for j in inst.jobs
        for m_id in inst.F[j.id]
        for t in inst.T[j.id, m_id]
    )
    return Solution(
        status=status,
        objective=mdl.ObjVal,
        f1=f1,
        f2=f2,
        assignment=assignment,
        completion=completion,
        build_time=build_time,
        solve_time=solve_time,
    )


# ---------------------------------------------------------------------------
# Public solve functions
# ---------------------------------------------------------------------------

def solve_f1(
    inst: Instance,
    formulation: Formulation | None = None,
    **params,
) -> Solution:
    """Minimise f1 (total turnaround) as a mono-objective problem."""
    fmt = formulation or _default_formulation()
    t0 = time.perf_counter()
    mdl, vars = fmt.build(inst)
    build_time = time.perf_counter() - t0
    _apply_params(mdl, params)
    mdl.setObjective(fmt.f1_expr(inst, vars["x"]), GRB.MINIMIZE)
    t1 = time.perf_counter()
    mdl.optimize()
    solve_time = time.perf_counter() - t1
    return _extract(inst, mdl, vars, build_time=build_time, solve_time=solve_time)


def solve_f2(
    inst: Instance,
    formulation: Formulation | None = None,
    **params,
) -> Solution:
    """Minimise f2 (total monetary cost) as a mono-objective problem."""
    fmt = formulation or _default_formulation()
    t0 = time.perf_counter()
    mdl, vars = fmt.build(inst)
    build_time = time.perf_counter() - t0
    _apply_params(mdl, params)
    mdl.setObjective(fmt.f2_expr(inst, vars["x"]), GRB.MINIMIZE)
    t1 = time.perf_counter()
    mdl.optimize()
    solve_time = time.perf_counter() - t1
    return _extract(inst, mdl, vars, build_time=build_time, solve_time=solve_time)


def solve_weighted_sum(
    inst: Instance,
    lam: float,
    f1_T: float,
    f2_T: float,
    f1_0: float,
    formulation: Formulation | None = None,
    **params,
) -> Solution:
    """Minimise λ·f̂1 + (1−λ)·f̂2 with f1 ≤ f1^0 (fedhpc.pdf eq. 34, 38).

    Reference points (fedhpc.pdf Section 1.10):
        f1_T  — f1^T: lexicographic minimum turnaround.
        f2_T  — f2^T: minimum cost at f1^T.
        f1_0  — f1^0: minimum turnaround at zero cost.

    Degenerate cases (Section 1.11):
        f1_0 == f1_T  → no turnaround trade-off; minimise f2 subject to f1 ≤ f1_0.
        f2_T == 0     → implied by the above; same handling.
    """
    if not (0.0 <= lam <= 1.0):
        raise ValueError("lam must be in [0, 1]")
    fmt = formulation or _default_formulation()
    t0 = time.perf_counter()
    mdl, vars = fmt.build(inst)
    build_time = time.perf_counter() - t0
    _apply_params(mdl, params)

    x = vars["x"]
    f1_expr = fmt.f1_expr(inst, x)
    f2_expr = fmt.f2_expr(inst, x)

    # Restrict to Pareto-relevant region (eq. 34)
    mdl.addConstr(f1_expr <= f1_0, name="pareto_region")

    degenerate = f1_0 <= f1_T + 1e-8
    if degenerate or f2_T <= 1e-8:
        # No turnaround/cost trade-off exists; minimise cost only
        mdl.setObjective(f2_expr, GRB.MINIMIZE)
    else:
        obj = (
            (lam / (f1_0 - f1_T)) * f1_expr
            + ((1 - lam) / f2_T) * f2_expr
        )
        mdl.setObjective(obj, GRB.MINIMIZE)

    t1 = time.perf_counter()
    mdl.optimize()
    solve_time = time.perf_counter() - t1
    return _extract(inst, mdl, vars, build_time=build_time, solve_time=solve_time)


def solve_epsilon_cost(
    inst: Instance,
    epsilon: float,
    formulation: Formulation | None = None,
    **params,
) -> Solution:
    """ε-constraint v1: min f1 subject to f2 ≤ epsilon  (eq. 26–27)."""
    fmt = formulation or _default_formulation()
    t0 = time.perf_counter()
    mdl, vars = fmt.build(inst)
    build_time = time.perf_counter() - t0
    _apply_params(mdl, params)
    mdl.addConstr(fmt.f2_expr(inst, vars["x"]) <= epsilon, name="eps_cost")
    mdl.setObjective(fmt.f1_expr(inst, vars["x"]), GRB.MINIMIZE)
    t1 = time.perf_counter()
    mdl.optimize()
    solve_time = time.perf_counter() - t1
    return _extract(inst, mdl, vars, build_time=build_time, solve_time=solve_time)


def solve_epsilon_turnaround(
    inst: Instance,
    epsilon: float,
    formulation: Formulation | None = None,
    **params,
) -> Solution:
    """ε-constraint v2: min f2 subject to f1 ≤ epsilon  (eq. 28–29)."""
    fmt = formulation or _default_formulation()
    t0 = time.perf_counter()
    mdl, vars = fmt.build(inst)
    build_time = time.perf_counter() - t0
    _apply_params(mdl, params)
    mdl.addConstr(fmt.f1_expr(inst, vars["x"]) <= epsilon, name="eps_turnaround")
    mdl.setObjective(fmt.f2_expr(inst, vars["x"]), GRB.MINIMIZE)
    t1 = time.perf_counter()
    mdl.optimize()
    solve_time = time.perf_counter() - t1
    return _extract(inst, mdl, vars, build_time=build_time, solve_time=solve_time)
