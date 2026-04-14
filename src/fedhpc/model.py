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

from dataclasses import dataclass

import gurobipy as gp
from gurobipy import GRB

from .data import Instance
from .formulations import Formulation, SpaceTimeFormulation


@dataclass
class Solution:
    status: str
    objective: float | None
    f1: float | None                        # total turnaround
    f2: float | None                        # total monetary cost
    assignment: dict[int, tuple[int, int]]  # job_id -> (type_id, start_time)
    completion: dict[int, int]              # job_id -> completion time (integer)

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


def _extract(inst: Instance, mdl: gp.Model, vars: dict) -> Solution:
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
    mdl, vars = fmt.build(inst)
    _apply_params(mdl, params)
    mdl.setObjective(fmt.f1_expr(inst, vars["x"]), GRB.MINIMIZE)
    mdl.optimize()
    return _extract(inst, mdl, vars)


def solve_f2(
    inst: Instance,
    formulation: Formulation | None = None,
    **params,
) -> Solution:
    """Minimise f2 (total monetary cost) as a mono-objective problem."""
    fmt = formulation or _default_formulation()
    mdl, vars = fmt.build(inst)
    _apply_params(mdl, params)
    mdl.setObjective(fmt.f2_expr(inst, vars["x"]), GRB.MINIMIZE)
    mdl.optimize()
    return _extract(inst, mdl, vars)


def solve_weighted_sum(
    inst: Instance,
    alpha: float,
    f1_min: float,
    f1_max: float,
    f2_max: float,
    formulation: Formulation | None = None,
    **params,
) -> Solution:
    """Minimise normalised weighted-sum λ·f̂1 + (1−λ)·f̂2  (eq. 24–25)."""
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1]")
    if f1_max <= f1_min:
        raise ValueError("f1_max must be > f1_min")
    if f2_max <= 0:
        raise ValueError("f2_max must be > 0")
    fmt = formulation or _default_formulation()
    mdl, vars = fmt.build(inst)
    _apply_params(mdl, params)
    obj = (
        (alpha / (f1_max - f1_min)) * fmt.f1_expr(inst, vars["x"])
        + ((1 - alpha) / f2_max) * fmt.f2_expr(inst, vars["x"])
    )
    mdl.setObjective(obj, GRB.MINIMIZE)
    mdl.optimize()
    return _extract(inst, mdl, vars)


def solve_epsilon_cost(
    inst: Instance,
    epsilon: float,
    formulation: Formulation | None = None,
    **params,
) -> Solution:
    """ε-constraint v1: min f1 subject to f2 ≤ epsilon  (eq. 26–27)."""
    fmt = formulation or _default_formulation()
    mdl, vars = fmt.build(inst)
    _apply_params(mdl, params)
    mdl.addConstr(fmt.f2_expr(inst, vars["x"]) <= epsilon, name="eps_cost")
    mdl.setObjective(fmt.f1_expr(inst, vars["x"]), GRB.MINIMIZE)
    mdl.optimize()
    return _extract(inst, mdl, vars)


def solve_epsilon_turnaround(
    inst: Instance,
    epsilon: float,
    formulation: Formulation | None = None,
    **params,
) -> Solution:
    """ε-constraint v2: min f2 subject to f1 ≤ epsilon  (eq. 28–29)."""
    fmt = formulation or _default_formulation()
    mdl, vars = fmt.build(inst)
    _apply_params(mdl, params)
    mdl.addConstr(fmt.f1_expr(inst, vars["x"]) <= epsilon, name="eps_turnaround")
    mdl.setObjective(fmt.f2_expr(inst, vars["x"]), GRB.MINIMIZE)
    mdl.optimize()
    return _extract(inst, mdl, vars)
