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
    for (jid, m_id, t), val in zip(x, mdl.getAttr("X", list(x.values()))):
        if val > 0.5:
            assignment[jid] = (m_id, t)
            completion[jid] = t + inst.p_occ[jid, m_id]

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
    bound: tuple[str, float, str] | None = None,
) -> Solution:
    """Build a fresh model and minimise ``primary``; with ``secondary``, then
    fix primary at its optimum (+1e-6) and minimise secondary (lex-min).

    ``primary``/``secondary`` are ``"f1"``/``"f2"``; ``primary`` may also be a
    callable ``(f1_expr, f2_expr) -> expr``. ``bound=(expr, value, name)``
    adds ``expr <= value``. ``hint_objective`` is passed to
    :meth:`Formulation.build` (it must be valid for the problem solved).
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
    if bound is not None:
        which, value, name = bound
        mdl.addConstr(exprs[which] <= value, name=name)

    primary_expr = primary(exprs["f1"], exprs["f2"]) if callable(primary) else exprs[primary]
    mdl.setObjective(primary_expr, GRB.MINIMIZE)
    t1 = time.perf_counter()
    mdl.optimize()

    if secondary is not None and mdl.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        # f1 = Σ(t + p_occ − arrival) has float arrivals, and costs are
        # floats: a small tolerance keeps the fixing row from cutting the
        # optimum on rounding noise.
        mdl.addConstr(primary_expr <= mdl.ObjVal + 1e-6, name=f"fix_{primary}")
        mdl.setObjective(exprs[secondary], GRB.MINIMIZE)
        mdl.optimize()

    return _extract(inst, mdl, vars, build_time=build_time,
                    solve_time=time.perf_counter() - t1)


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

    Degenerate cases (Section 1.11, :func:`weighted_sum_degenerate`):
        f1_0 == f1_T  → no turnaround trade-off; minimise f2 subject to f1 ≤ f1_0.
        f2_T == f2_0  → implied by the above; same handling.

    ``hint`` — an optional prior/heuristic Solution used to seed Gurobi's MIP
    start; see :func:`solve_f1`.
    """
    if not (0.0 <= lam <= 1.0):
        raise ValueError("lam must be in [0, 1]")
    if weighted_sum_degenerate(f1_T, f2_T, f1_0, f2_0):
        # No turnaround/cost trade-off exists; minimise cost only
        weights, primary = None, "f2"
    else:
        weights = (lam / (f1_0 - f1_T), (1 - lam) / (f2_T - f2_0))
        def primary(f1, f2):
            return weights[0] * f1 + weights[1] * f2
    # Restrict to Pareto-relevant region (eq. 34)
    return _solve(inst, formulation, hint, params, primary, hint_objective=weights,
                  bound=("f1", f1_0, "pareto_region"))


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
    return _solve(inst, formulation, hint, params, "f1", bound=("f2", epsilon, "eps_cost"))


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
    return _solve(inst, formulation, hint, params, "f2", bound=("f1", epsilon, "eps_turnaround"))


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
    return _solve(inst, formulation, hint, params, "f2", "f1", bound=("f1", f1_bound, "eps_f1"))
