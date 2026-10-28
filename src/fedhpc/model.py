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

import math
import time
from dataclasses import dataclass, field, replace

import gurobipy as gp
from gurobipy import GRB

from .data import Instance, ScheduleError
from .formulations import LEX_F1, Formulation, SpaceTimeFormulation


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


# ---------------------------------------------------------------------------
# Exact lexicographic certificates
# ---------------------------------------------------------------------------
#
# f1 moves in whole steps of 1 (integral weights × integer completions) and f2
# in whole steps of the cost resolution δ (Instance.build rounds costs to that
# grid). A schedule is therefore optimal for f1 (or f2) once its exact value,
# recomputed from its assignment, is less than one step above a valid dual
# bound: nothing strictly better fits in between. That replaces chasing a
# 1e-9 relative MIPGap with an absolute gap just under one step, and every
# objective bound is an integer row on that grid (Formulation.bound_constr).

def objective_step(inst: Instance, which: str) -> float:
    """Smallest possible difference between two schedules' f1 (``"f1"``) or
    f2 (``"f2"``) values."""
    return 1.0 if which == "f1" else inst.cost_resolution


def grid_floor(inst: Instance, which: str, value: float) -> float:
    """Largest achievable value of ``which`` that is <= ``value``: an off-grid
    bound (e.g. a half-step box cut) snapped onto the objective's grid, the
    form :meth:`Formulation.bound_constr` writes as an integer row."""
    if which == "f1":
        offset = sum(j.weight * j.arrival for j in inst.jobs)
        return math.floor(value + offset + 1e-6) - offset
    delta = inst.cost_resolution
    return math.floor(value / delta + 1e-6) * delta


def certify_params(inst: Instance, which: str, params: dict) -> dict:
    """``params`` with defaults that make Gurobi stop only once ``which`` is
    certified optimal: no relative gap, absolute gap just under one step.
    Values the caller sets explicitly win."""
    return {"MIPGap": 0.0, "MIPGapAbs": 0.999 * objective_step(inst, which), **params}


def is_certified(inst: Instance, mdl: gp.Model, which: str, value: float) -> bool:
    """Whether ``value`` (the incumbent's exact ``which``, recomputed from its
    assignment) is within one step of ``mdl``'s dual bound."""
    if mdl.SolCount == 0 or mdl.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        return False
    step = objective_step(inst, which)
    return value - mdl.ObjBound < step * (1 - 1e-6)


def set_mip_start(inst: Instance, x: dict, hint: Solution) -> None:
    """Seed Gurobi's MIP start (``.Start``) from a heuristic/prior ``Solution``.

    Sets every x_jmt to 1.0 for the hinted (type, start-time) pair and 0.0
    otherwise, for jobs present in ``hint.assignment``. Jobs absent from the
    hint (e.g. an infeasible heuristic result) are left unset — Gurobi fills
    those in on its own during the root relaxation.  This only biases the
    B&B search toward a known-good incumbent; it never changes the feasible
    region or the proven-optimal objective. Jobs whose hinted (type, start)
    has no variable (a reduced model pruned it) are left unset as well.
    """
    for (jid, m_id, t), v in x.items():
        if jid in hint.assignment and (jid, *hint.assignment[jid]) in x:
            v.Start = 1.0 if hint.assignment[jid] == (m_id, t) else 0.0


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

    # One bulk attribute query: per-variable .X costs ~3x more on large models.
    doubled = set()
    for (jid, m_id, t), val in zip(x, mdl.getAttr("X", list(x.values()))):
        if val > 0.5:
            if jid in assignment:
                doubled.add(jid)
            assignment[jid] = (m_id, t)
            completion[jid] = t + inst.p_occ[jid, m_id]

    # The exact f1/f2 below (and every certificate built on them) describe the
    # rounded schedule, so it must itself be feasible — a check against
    # integrality/feasibility tolerances, never expected to fire.
    if doubled:
        raise ScheduleError(f"infeasible schedule: jobs {sorted(doubled)[:10]} "
                            "started more than once")
    inst.check_schedule(assignment)

    f1 = inst.f1_of(completion)
    f2 = inst.f2_of(assignment)
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


def _solve(
    inst: Instance,
    formulation: Formulation | None,
    hint: Solution | None,
    params: dict,
    primary,
    secondary: str | None = None,
    *,
    hint_objective=None,
    bounds: tuple[tuple[str, float, str], ...] = (),
) -> Solution:
    """Build a fresh model and minimise ``primary``; with ``secondary``, then
    fix primary at its optimum and minimise secondary (lex-min).

    ``primary``/``secondary`` are ``"f1"``/``"f2"``; ``primary`` may also be a
    callable ``(f1_expr, f2_expr) -> expr``. Each ``(which, value, name)`` of
    ``bounds`` adds ``which <= value`` as an integer row (Formulation.bound_constr). ``hint_objective`` is passed to
    :meth:`Formulation.build` (it must be valid for the problem solved).

    For ``"f1"``/``"f2"`` objectives each phase runs with
    :func:`certify_params` (unless ``params`` overrides them) and the result
    is ``"optimal"`` only when every phase is certified (:func:`is_certified`);
    otherwise it is ``"feasible"``. A callable primary keeps Gurobi's own
    status (its scalar objective has no step to certify against).
    """
    fmt = formulation or _default_formulation()
    t0 = time.perf_counter()
    mdl, vars = fmt.build(inst, objective=hint_objective)
    build_time = time.perf_counter() - t0
    _apply_params(mdl, params)

    x = vars["x"]
    if hint is not None:
        set_mip_start(inst, x, hint)
    exprs = {"f1": fmt.f1_expr(inst, x), "f2": fmt.f2_expr(inst, x)}
    for which, value, name in bounds:
        mdl.addConstr(fmt.bound_constr(inst, x, which, value), name=name)

    primary_expr = primary(exprs["f1"], exprs["f2"]) if callable(primary) else exprs[primary]
    mdl.setObjective(primary_expr, GRB.MINIMIZE)
    t1 = time.perf_counter()
    if callable(primary):
        mdl.optimize()
        return _extract(inst, mdl, vars, build_time=build_time,
                        solve_time=time.perf_counter() - t1)

    _apply_params(mdl, certify_params(inst, primary, params))
    mdl.optimize()
    sol = _extract(inst, mdl, vars)
    certified = sol.f1 is not None and is_certified(inst, mdl, primary, getattr(sol, primary))

    if secondary is not None and certified:
        # Fix the primary at its exact optimum, as an integer row
        # (Formulation.bound_constr): tolerances can neither cut the optimum
        # nor admit a schedule one step worse.
        mdl.addConstr(fmt.bound_constr(inst, x, primary, getattr(sol, primary)),
                      name=f"fix_{primary}")
        mdl.setObjective(exprs[secondary], GRB.MINIMIZE)
        _apply_params(mdl, certify_params(inst, secondary, params))
        mdl.optimize()
        sol = _extract(inst, mdl, vars)
        certified = sol.f1 is not None and is_certified(inst, mdl, secondary,
                                                       getattr(sol, secondary))

    sol = _extract(inst, mdl, vars, build_time=build_time,
                   solve_time=time.perf_counter() - t1)
    if sol.f1 is not None and not certified:
        sol = replace(sol, status="feasible")
    return sol


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
    return _solve(inst, formulation, hint, params, "f1", "f2", hint_objective=LEX_F1)


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
    return _solve(inst, formulation, hint, params, "f2", "f1")


def weighted_sum_degenerate(f1_T: float, f2_T: float, f1_0: float, f2_0: float = 0.0) -> bool:
    """Whether the weighted sum has no turnaround/cost trade-off (fedhpc.pdf Section 1.11)."""
    return f1_0 <= f1_T + 1e-8 or f2_T - f2_0 <= 1e-8


def weighted_sum_terms(lam: float, f1_T: float, f2_T: float, f1_0: float,
                       f2_0: float = 0.0) -> tuple[float, float, float]:
    """``(w1, w2, c)`` with w1·f1 + w2·f2 + c = λ·f̂1 + (1−λ)·f̂2, where
    f̂1 = (f1 − f1_T)/(f1_0 − f1_T) and f̂2 = (f2 − f2_0)/(f2_T − f2_0).

    The constant c matters to Gurobi, not to the optimum: MIPGap is relative
    to the objective, and without c a front far from zero turnaround
    (f1_T ≫ f1_0 − f1_T) inflates it — 34× on pos_congestion at λ = 0.75,
    where a schedule 0.009 dearer than (f1_T, f2_T) passed as optimal.
    """
    w1 = lam / (f1_0 - f1_T)
    w2 = (1 - lam) / (f2_T - f2_0)
    return w1, w2, -(w1 * f1_T + w2 * f2_0)


def weighted_sum_scale(inst: Instance, w1: float, w2: float) -> float:
    """Factor Gurobi's copy of ĝ is multiplied by: one step of the coarser
    objective (w1·1 or w2·δ) becomes 1. Unscaled, ĝ's cost coefficients
    (w2·c, ~3e-7 on pos_congestion) sit below OptimalityTol (1e-6): the LP
    reads cost differences as zero and branch and bound prunes the optimum,
    even at MIPGap 0. The relative gap is unchanged by the factor."""
    return 1.0 / max(w1 * objective_step(inst, "f1"), w2 * objective_step(inst, "f2"))


def solve_weighted_sum(
    inst: Instance,
    lam: float,
    f1_T: float,
    f2_T: float,
    f1_0: float,
    formulation: Formulation | None = None,
    hint: Solution | None = None,
    *,
    f2_0: float = 0.0,
    **params,
) -> Solution:
    """Minimise λ·f̂1 + (1−λ)·f̂2 with f1 ≤ f1^0 (fedhpc.pdf eq. 34, 38).

    Reference points (fedhpc.pdf Section 1.10):
        f1_T  — f1^T: lexicographic minimum turnaround.
        f2_T  — f2^T: minimum cost at f1^T.
        f1_0  — f1^0: minimum turnaround at zero cost.
        f2_0  — the cost at f1^0: 0 (the default), as in the paper, or the
                minimum cost when no schedule is free (see
                ``pareto.weighted_sum_reference``). f̂2 = (f2 − f2_0)/(f2_T − f2_0)
                differs from f2/(f2_T − f2_0) by a constant only.

    The objective is the normalised ĝ = λ·f̂1 + (1−λ)·f̂2 itself, constant
    included (:func:`weighted_sum_terms`), so Gurobi's relative MIPGap is on
    ĝ and the returned ``objective`` is ĝ ∈ [0, 1]; Gurobi sees it times
    :func:`weighted_sum_scale`. Besides f1 ≤ f1_0 the
    model carries f2 ≤ f2_T: every Pareto point satisfies both, so they cut
    no optimum, but a gap-tolerant incumbent dominated by (f1_T, f2_T) is
    excluded.

    Degenerate cases (Section 1.11, :func:`weighted_sum_degenerate`):
        f1_0 == f1_T  → no turnaround trade-off; minimise f2 subject to f1 ≤ f1_0.
        f2_T == f2_0  → implied by the above; same handling.

    λ = 1 gives cost no weight, so any minimum-turnaround schedule would do,
    however dear: it is solved as the lex-min (f1, f2) of :func:`solve_f1`
    instead, whose optimum is (f1_T, f2_T).

    ``hint`` — an optional prior/heuristic Solution used to seed Gurobi's MIP
    start; see :func:`solve_f1`.
    """
    if not (0.0 <= lam <= 1.0):
        raise ValueError("lam must be in [0, 1]")
    if weighted_sum_degenerate(f1_T, f2_T, f1_0, f2_0):
        # No turnaround/cost trade-off exists; minimise cost only
        return _solve(inst, formulation, hint, params, "f2",
                      bounds=[("f1", f1_0, "pareto_region")])
    w1, w2, c = weighted_sum_terms(lam, f1_T, f2_T, f1_0, f2_0)
    if lam == 1.0:
        sol = solve_f1(inst, formulation, hint, **params)
        return sol if sol.f1 is None else replace(sol, objective=w1 * sol.f1 + c)

    scale = weighted_sum_scale(inst, w1, w2)

    def primary(f1, f2):
        return scale * (w1 * f1 + w2 * f2 + c)
    # Restrict to the Pareto-relevant box (eq. 34): valid for λ < 1, whose
    # optima are Pareto optimal (λ > 0) or cost f2_0 (λ = 0)
    sol = _solve(inst, formulation, hint, params, primary, hint_objective=(w1, w2),
                 bounds=[("f1", f1_0, "pareto_region"), ("f2", f2_T, "pareto_region_f2")])
    return sol if sol.objective is None else replace(sol, objective=sol.objective / scale)


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
    return _solve(inst, formulation, hint, params, "f1", bounds=[("f2", epsilon, "eps_cost")])


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
    return _solve(inst, formulation, hint, params, "f2", bounds=[("f1", epsilon, "eps_turnaround")])


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
    return _solve(inst, formulation, hint, params, "f2", "f1", bounds=[("f1", f1_bound, "eps_f1")])
