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


def set_mip_start(inst: Instance, x: dict, hint: Solution) -> None:
    """Seed Gurobi's MIP start (``.Start``) from a heuristic/prior ``Solution``.

    Sets every x_jmt to 1.0 for the hinted (type, start-time) pair and 0.0
    otherwise, for jobs present in ``hint.assignment``. Jobs absent from the
    hint (e.g. an infeasible heuristic result) are left unset — Gurobi fills
    those in on its own during the root relaxation.  This only biases the
    B&B search toward a known-good incumbent; it never changes the feasible
    region or the proven-optimal objective.
    """
    for j in inst.jobs:
        if j.id not in hint.assignment:
            continue
        m_hint, t_hint = hint.assignment[j.id]
        for m_id in inst.F[j.id]:
            for t in inst.T[j.id, m_id]:
                x[j.id, m_id, t].Start = (
                    1.0 if (m_id == m_hint and t == t_hint) else 0.0
                )


def _extract(
    inst: Instance,
    mdl: gp.Model,
    vars: dict,
    build_time: float | None = None,
    solve_time: float | None = None,
) -> Solution:
    status_map = {
        GRB.OPTIMAL:   "optimal",
        GRB.SUBOPTIMAL: "feasible",
        GRB.INFEASIBLE: "infeasible",
        GRB.UNBOUNDED:  "unbounded",
        GRB.TIME_LIMIT: "feasible",
    }
    status = status_map.get(mdl.Status, f"gurobi_status_{mdl.Status}")

    # A time-limited solve that found a feasible incumbent is treated as suboptimal.
    time_limit_feasible = mdl.Status == GRB.TIME_LIMIT and mdl.SolCount > 0

    if mdl.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL) and not time_limit_feasible:
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
    hint: Solution | None = None,
    **params,
) -> Solution:
    """Lex-min (f1, f2): minimise total turnaround, breaking ties by minimum cost.

    ``hint`` — an optional prior/heuristic Solution (e.g. from moea.py) used to
    seed Gurobi's MIP start. Purely a search-order hint: it cannot change the
    proven-optimal result, only how fast the solver finds it.
    """
    fmt = formulation or _default_formulation()
    t0 = time.perf_counter()
    mdl, vars = fmt.build(inst)
    build_time = time.perf_counter() - t0
    _apply_params(mdl, params)

    x = vars["x"]
    if hint is not None:
        set_mip_start(inst, x, hint)
    f1_expr = fmt.f1_expr(inst, x)
    f2_expr = fmt.f2_expr(inst, x)

    # Phase 1: minimise turnaround
    mdl.setObjective(f1_expr, GRB.MINIMIZE)
    t1 = time.perf_counter()
    mdl.optimize()

    if mdl.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        solve_time = time.perf_counter() - t1
        return _extract(inst, mdl, vars, build_time=build_time, solve_time=solve_time)

    # Phase 2: fix turnaround at optimal, minimise cost.
    # f1 = Σ(t + p_occ − arrival); arrival is float, so ObjVal is float.
    # Use a small tolerance to avoid rounding the float constraint too tight.
    f1_star = mdl.ObjVal
    mdl.addConstr(f1_expr <= f1_star + 1e-6, name="fix_f1")
    mdl.setObjective(f2_expr, GRB.MINIMIZE)
    mdl.optimize()
    solve_time = time.perf_counter() - t1

    return _extract(inst, mdl, vars, build_time=build_time, solve_time=solve_time)


def solve_f2(
    inst: Instance,
    formulation: Formulation | None = None,
    hint: Solution | None = None,
    **params,
) -> Solution:
    """Lex-min (f2, f1): minimise total cost, breaking ties by minimum turnaround.

    When minimum cost is 0, phase 2 finds the best achievable turnaround at zero cost.

    ``hint`` — an optional prior/heuristic Solution used to seed Gurobi's MIP
    start; see :func:`solve_f1`.
    """
    fmt = formulation or _default_formulation()
    t0 = time.perf_counter()
    mdl, vars = fmt.build(inst)
    build_time = time.perf_counter() - t0
    _apply_params(mdl, params)

    x = vars["x"]
    if hint is not None:
        set_mip_start(inst, x, hint)
    f1_expr = fmt.f1_expr(inst, x)
    f2_expr = fmt.f2_expr(inst, x)

    # Phase 1: minimise cost
    mdl.setObjective(f2_expr, GRB.MINIMIZE)
    t1 = time.perf_counter()
    mdl.optimize()

    if mdl.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        solve_time = time.perf_counter() - t1
        return _extract(inst, mdl, vars, build_time=build_time, solve_time=solve_time)

    # Phase 2: fix cost at optimal, minimise turnaround.
    # Use a small absolute tolerance to guard against floating-point noise.
    f2_star = mdl.ObjVal
    mdl.addConstr(f2_expr <= f2_star + 1e-6, name="fix_f2")
    mdl.setObjective(f1_expr, GRB.MINIMIZE)
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
    hint: Solution | None = None,
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

    ``hint`` — an optional prior/heuristic Solution used to seed Gurobi's MIP
    start; see :func:`solve_f1`.
    """
    if not (0.0 <= lam <= 1.0):
        raise ValueError("lam must be in [0, 1]")
    fmt = formulation or _default_formulation()
    t0 = time.perf_counter()
    mdl, vars = fmt.build(inst)
    build_time = time.perf_counter() - t0
    _apply_params(mdl, params)

    x = vars["x"]
    if hint is not None:
        set_mip_start(inst, x, hint)
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
    hint: Solution | None = None,
    **params,
) -> Solution:
    """ε-constraint v1: min f1 subject to f2 ≤ epsilon  (eq. 26–27).

    ``hint`` — an optional prior/heuristic Solution used to seed Gurobi's MIP
    start; see :func:`solve_f1`.
    """
    fmt = formulation or _default_formulation()
    t0 = time.perf_counter()
    mdl, vars = fmt.build(inst)
    build_time = time.perf_counter() - t0
    _apply_params(mdl, params)
    if hint is not None:
        set_mip_start(inst, vars["x"], hint)
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
    hint: Solution | None = None,
    **params,
) -> Solution:
    """ε-constraint v2: min f2 subject to f1 ≤ epsilon  (eq. 28–29).

    ``hint`` — an optional prior/heuristic Solution used to seed Gurobi's MIP
    start; see :func:`solve_f1`.
    """
    fmt = formulation or _default_formulation()
    t0 = time.perf_counter()
    mdl, vars = fmt.build(inst)
    build_time = time.perf_counter() - t0
    _apply_params(mdl, params)
    if hint is not None:
        set_mip_start(inst, vars["x"], hint)
    mdl.addConstr(fmt.f1_expr(inst, vars["x"]) <= epsilon, name="eps_turnaround")
    mdl.setObjective(fmt.f2_expr(inst, vars["x"]), GRB.MINIMIZE)
    t1 = time.perf_counter()
    mdl.optimize()
    solve_time = time.perf_counter() - t1
    return _extract(inst, mdl, vars, build_time=build_time, solve_time=solve_time)


def solve_true_pareto_step(
    inst: Instance,
    f1_bound: float,
    formulation: Formulation | None = None,
    hint: Solution | None = None,
    **params,
) -> Solution:
    """One step of the true-Pareto sweep: lex-min (f2, f1) s.t. f1 ≤ f1_bound.

    Phase 1 finds the minimum cost reachable within the turnaround budget.
    Phase 2 tightens to that cost and minimises turnaround, ensuring the
    returned point lies on the true Pareto front (not just the cost-efficient
    frontier within the budget).  Returns an infeasible Solution when no
    schedule exists with f1 ≤ f1_bound.
    f1_bound is float because f1 = Σ(t + p_occ − arrival) with float arrivals.

    ``hint`` — an optional prior/heuristic Solution used to seed Gurobi's MIP
    start; see :func:`solve_f1`.
    """
    fmt = formulation or _default_formulation()
    t0 = time.perf_counter()
    mdl, vars = fmt.build(inst)
    build_time = time.perf_counter() - t0
    _apply_params(mdl, params)

    x = vars["x"]
    if hint is not None:
        set_mip_start(inst, x, hint)
    f1_expr = fmt.f1_expr(inst, x)
    f2_expr = fmt.f2_expr(inst, x)

    mdl.addConstr(f1_expr <= f1_bound, name="eps_f1")

    # Phase 1: min cost within the turnaround budget
    mdl.setObjective(f2_expr, GRB.MINIMIZE)
    t1 = time.perf_counter()
    mdl.optimize()

    if mdl.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        return _extract(inst, mdl, vars, build_time=build_time,
                        solve_time=time.perf_counter() - t1)

    # Phase 2: fix cost, minimise turnaround (breaks f2 ties toward lower f1)
    f2_star = mdl.ObjVal
    mdl.addConstr(f2_expr <= f2_star + 1e-6, name="fix_f2")
    mdl.setObjective(f1_expr, GRB.MINIMIZE)
    mdl.optimize()

    return _extract(inst, mdl, vars, build_time=build_time,
                    solve_time=time.perf_counter() - t1)
