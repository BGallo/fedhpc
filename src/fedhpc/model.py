"""Core MIP model for FED-HPC — space-time network formulation.

Decision variables
------------------
x_jmt ∈ {0,1}   : 1 if job j starts on type m at time t
w_mt  ∈ Z≥0     : number of instances of type m occupied during interval [t, t+1)

Constraints
-----------
(12) Unique assignment
(13) Occupancy definition: w_mt = Σ_{j,τ: τ≤t<τ+p^occ} x_jmτ
(14) Redundant bound: w_mt ≤ |{j : m ∈ F_j}|
(15) Budget
"""
from __future__ import annotations

from dataclasses import dataclass

import gurobipy as gp
from gurobipy import GRB

from .data import Instance


@dataclass
class Solution:
    status: str
    objective: float | None
    f1: float | None                          # total turnaround
    f2: float | None                          # total monetary cost
    assignment: dict[int, tuple[int, int]]    # job_id -> (type_id, start_time)
    completion: dict[int, int]                # job_id -> completion time (integer)

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


def build_base(inst: Instance) -> tuple[gp.Model, dict]:
    H = inst.horizon
    mdl = gp.Model("fedhpc")
    mdl.setParam("OutputFlag", 0)

    # ---- Decision variables ------------------------------------------------

    # x[j,m,t] ∈ {0,1}: job j starts on type m at time t
    x = {
        (j.id, m_id, t): mdl.addVar(vtype=GRB.BINARY, name=f"x[{j.id},{m_id},{t}]")
        for j in inst.jobs
        for m_id in inst.F[j.id]
        for t in inst.T[j.id, m_id]
    }

    # w[m,t] ∈ Z≥0: occupied instances of type m during [t, t+1), t ∈ T̄ = {0..H-1}
    w = {
        (itype.id, t): mdl.addVar(vtype=GRB.INTEGER, lb=0, name=f"w[{itype.id},{t}]")
        for itype in inst.instance_types
        for t in range(H)
    }

    mdl.update()

    # ---- Constraints -------------------------------------------------------

    # (12) Unique assignment
    for j in inst.jobs:
        mdl.addConstr(
            gp.quicksum(
                x[j.id, m_id, t]
                for m_id in inst.F[j.id]
                for t in inst.T[j.id, m_id]
            ) == 1,
            name=f"assign[{j.id}]",
        )

    # (13) Occupancy definition: w_mt = number of jobs running during [t, t+1)
    # A job (j, m) starting at τ occupies interval t iff τ ≤ t < τ + p_occ
    for itype in inst.instance_types:
        mid = itype.id
        for t in range(H):
            running = gp.quicksum(
                x[j.id, mid, tau]
                for j in inst.jobs if mid in inst.F[j.id]
                for tau in inst.T[j.id, mid]
                if tau <= t < tau + inst.p_occ[j.id, mid]
            )
            mdl.addConstr(w[mid, t] == running, name=f"occupy[{mid},{t}]")

    # (14) Redundant upper bound for LP relaxation strengthening
    for itype in inst.instance_types:
        mid = itype.id
        eligible = sum(1 for j in inst.jobs if mid in inst.F[j.id])
        for t in range(H):
            mdl.addConstr(w[mid, t] <= eligible, name=f"wbound[{mid},{t}]")

    # (15) Budget
    mdl.addConstr(
        gp.quicksum(
            inst.c[j.id, m_id] * x[j.id, m_id, t]
            for j in inst.jobs
            for m_id in inst.F[j.id]
            for t in inst.T[j.id, m_id]
        ) <= inst.budget,
        name="budget",
    )

    return mdl, {"x": x, "w": w}


def _f1_expr(inst: Instance, x: dict) -> gp.LinExpr:
    # (eq. 10) turnaround uses p_occ (completion = t + p_occ)
    return gp.quicksum(
        (t + inst.p_occ[j.id, m_id] - j.arrival) * x[j.id, m_id, t]
        for j in inst.jobs
        for m_id in inst.F[j.id]
        for t in inst.T[j.id, m_id]
    )


def _f2_expr(inst: Instance, x: dict) -> gp.LinExpr:
    # (eq. 11) cost uses c^proc (which is based on p_bill)
    return gp.quicksum(
        inst.c[j.id, m_id] * x[j.id, m_id, t]
        for j in inst.jobs
        for m_id in inst.F[j.id]
        for t in inst.T[j.id, m_id]
    )


def _extract(inst: Instance, mdl: gp.Model, vars: dict) -> Solution:
    status_map = {
        GRB.OPTIMAL: "optimal",
        GRB.SUBOPTIMAL: "feasible",
        GRB.INFEASIBLE: "infeasible",
        GRB.UNBOUNDED: "unbounded",
    }
    status = status_map.get(mdl.Status, f"gurobi_status_{mdl.Status}")

    if mdl.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        return Solution(status=status, objective=None, f1=None, f2=None,
                        assignment={}, completion={})

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


def _apply_params(mdl: gp.Model, params: dict) -> None:
    for k, v in params.items():
        mdl.setParam(k, v)


def solve_f1(inst: Instance, **params) -> Solution:
    """Minimise f1 (total turnaround) as a mono-objective problem."""
    mdl, vars = build_base(inst)
    _apply_params(mdl, params)
    mdl.setObjective(_f1_expr(inst, vars["x"]), GRB.MINIMIZE)
    mdl.optimize()
    return _extract(inst, mdl, vars)


def solve_f2(inst: Instance, **params) -> Solution:
    """Minimise f2 (total cost) as a mono-objective problem."""
    mdl, vars = build_base(inst)
    _apply_params(mdl, params)
    mdl.setObjective(_f2_expr(inst, vars["x"]), GRB.MINIMIZE)
    mdl.optimize()
    return _extract(inst, mdl, vars)


def solve_weighted_sum(
    inst: Instance, alpha: float, f1_min: float, f1_max: float, f2_max: float, **params
) -> Solution:
    """Minimise normalised weighted-sum λ·f̂1 + (1-λ)·f̂2."""
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1]")
    if f1_max <= f1_min:
        raise ValueError("f1_max must be > f1_min")
    if f2_max <= 0:
        raise ValueError("f2_max must be > 0")
    mdl, vars = build_base(inst)
    _apply_params(mdl, params)
    obj = (
        (alpha / (f1_max - f1_min)) * _f1_expr(inst, vars["x"])
        + ((1 - alpha) / f2_max) * _f2_expr(inst, vars["x"])
    )
    mdl.setObjective(obj, GRB.MINIMIZE)
    mdl.optimize()
    return _extract(inst, mdl, vars)


def solve_epsilon_cost(inst: Instance, epsilon: float, **params) -> Solution:
    """ε-constraint v1: min f1 subject to f2 ≤ epsilon."""
    mdl, vars = build_base(inst)
    _apply_params(mdl, params)
    mdl.addConstr(_f2_expr(inst, vars["x"]) <= epsilon, name="eps_cost")
    mdl.setObjective(_f1_expr(inst, vars["x"]), GRB.MINIMIZE)
    mdl.optimize()
    return _extract(inst, mdl, vars)


def solve_epsilon_turnaround(inst: Instance, epsilon: float, **params) -> Solution:
    """ε-constraint v2: min f2 subject to f1 ≤ epsilon."""
    mdl, vars = build_base(inst)
    _apply_params(mdl, params)
    mdl.addConstr(_f1_expr(inst, vars["x"]) <= epsilon, name="eps_turnaround")
    mdl.setObjective(_f2_expr(inst, vars["x"]), GRB.MINIMIZE)
    mdl.optimize()
    return _extract(inst, mdl, vars)
