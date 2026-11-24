"""Pareto frontier exploration methods for FED-HPC."""
from __future__ import annotations

import json
import os
import sys
import time as _time
from collections import deque
from dataclasses import replace

from gurobipy import GRB

from .data import Instance
from .formulations import LEX_F1, Formulation, SpaceTimeFormulation, nonnegative_data
from .model import (
    Solution,
    _apply_params,
    _extract,
    certify_params,
    grid_floor,
    is_certified,
    objective_step,
    set_mip_start,
    solve_weighted_sum,
    weighted_sum_degenerate,
)


def _make_lexmin_bounded(
    inst: Instance, fmt: Formulation, params: dict,
    per_solve_time_limit: float | None = None,
    n_seed_retries: int = 1,
):
    """Build a fresh-model-per-call lex-min solver, shared by
    true_pareto_frontier, map_pareto_frontier and hybrid_frontier.

    Returns ``(lexmin_bounded, counters)`` where ``counters`` is a dict with
    live ``"n_solves"`` and ``"build_time"`` entries the caller can read at
    any point (mutated in place, not just at the end).

    See true_pareto_frontier's docstring for why every solve rebuilds the
    model from scratch instead of reusing one across the search — reuse was
    empirically unsafe here (stale internal solve state after a
    structurally different constraint change), not just a missed
    optimization.

    ``per_solve_time_limit`` caps each individual Gurobi solve attempt (not
    the overall search — see the callers' own ``max_solves``/``time_budget``
    for that). This matters independently of those: every solve must be
    proven to within one objective step (see true_pareto_frontier's
    docstring for why), and some sub-problems have a good incumbent almost
    instantly but take far longer to *prove* optimal — observed (under the
    earlier 1e-9 relative MIPGap, before certificates) running over 6 hours
    and 100M+ B&B nodes without closing the gap. Investigating that specific case rather than just
    capping it: re-solving the *exact same* sub-problem fresh converged
    cleanly in 144s with a proven 0.0000% gap — and re-solving it again
    with Gurobi's ``Seed`` bumped by 1 hit the time limit again, stuck at a
    MIPGap of 7e-6. Same problem, same true optimum, wildly different
    runtime purely from parallel B&B's search-order non-determinism — this
    reads as heavy solution symmetry (many structurally interchangeable
    jobs/slots) making the dual bound's convergence rate highly sensitive
    to exploration order, not an inherently unsolvable sub-problem.
    ``MIPFocus=3`` (bound-focused search) and ``Symmetry=2`` (aggressive
    symmetry detection) were also tried and didn't help — the former was
    outright worse (timed out with *zero* feasible solutions found).

    ``n_seed_retries`` exploits this directly: if an attempt hits the time
    limit without proving optimality, retry with ``Seed`` incremented
    (fresh B&B search order) up to this many times before giving up. Since
    we've empirically confirmed some seeds converge fast on a case where
    the default one didn't, this converts "occasionally hangs for hours"
    into "occasionally costs a few extra bounded-length attempts" — no
    reformulation (e.g. explicit symmetry-breaking constraints) required.
    Default 1 (no retry, single attempt) preserves prior behaviour; the
    callers set a larger value.

    A solve that still isn't certified after all retries returns with
    ``status != "optimal"`` (``"feasible"`` if some attempt at least found a
    solution, else no solution) — callers must treat that as
    *inconclusive*, not as a proof of anything.

    Each phase is certified exactly (``model.certify_params`` /
    ``model.is_certified``): Gurobi runs to an absolute gap just under one
    objective step (1 for f1, the cost resolution δ for f2) and the phase
    counts only if the incumbent's exact value — recomputed from its
    schedule, not Gurobi's ObjVal — is within one step of the dual bound.
    Phase 2 fixes the primary at that exact value, and every bound is an
    integer row (``Formulation.bound_constr``).
    """
    counters = {"n_solves": 0, "build_time": 0.0}

    def _optimize_with_retries(mdl) -> None:
        """Runs mdl.optimize(), retrying with a bumped Seed (fresh B&B search
        order, via mdl.reset() first) while the previous attempt hit the time
        limit without reaching GRB.OPTIMAL. Leaves the model in whichever
        attempt's final state — success or last failure — for the caller to
        read status/ObjVal from directly.
        """
        for attempt in range(n_seed_retries):
            if attempt > 0:
                mdl.reset()
                mdl.setParam("Seed", attempt)
            mdl.optimize()
            counters["n_solves"] += 1
            if mdl.Status == GRB.OPTIMAL:
                return

    _expr_of = {"f1": lambda x: fmt.f1_expr(inst, x), "f2": lambda x: fmt.f2_expr(inst, x)}

    def lexmin_bounded(
        primary: str, secondary: str,
        bound: str | None = None, bound_value: float | None = None,
        *, start: Solution | None = None,
    ) -> Solution:
        """Lex-min(primary, secondary) on a fresh model, s.t. bound <= bound_value.

        primary/secondary/bound are ``"f1"``/``"f2"`` — not callables taking
        the model's own ``x``, so the same string protocol works whichever
        engine built this closure (the BPC engine has no compact-model ``x``
        to call a callable on; see ``bpc.make_bpc_lexmin_bounded``).
        bound=None skips the bounding constraint entirely (used for
        unconstrained anchor solves).

        The returned Solution's ``.status`` is ``"optimal"`` only if *both*
        solve phases proved optimality within tolerance; check that before
        trusting the result for anything beyond "here's a decent point".
        """
        t0 = _time.perf_counter()
        # lex (f1, f2) under at most an f1 cap admits the lex reductions
        lex = primary == "f1" and secondary == "f2" and bound in (None, "f1")
        mdl, vars_ = fmt.build(inst, objective=LEX_F1 if lex else None)
        counters["build_time"] += _time.perf_counter() - t0
        _apply_params(mdl, params)
        if per_solve_time_limit is not None:
            mdl.setParam("TimeLimit", per_solve_time_limit)

        x = vars_["x"]
        primary_expr = _expr_of[primary](x)
        secondary_expr = _expr_of[secondary](x)

        if bound is not None:
            mdl.addConstr(fmt.bound_constr(inst, x, bound, bound_value), name="box_bound")
        if start is not None:
            set_mip_start(inst, x, start)
        mdl.setObjective(primary_expr, GRB.MINIMIZE)
        _apply_params(mdl, certify_params(inst, primary, params))
        t1 = _time.perf_counter()

        def done(which: str) -> Solution:
            sol = _extract(inst, mdl, vars_, build_time=counters["build_time"],
                           solve_time=_time.perf_counter() - t1)
            if sol.f1 is not None and not is_certified(inst, mdl, which, getattr(sol, which)):
                # Not proven, even after retries: inconclusive for the caller
                sol = replace(sol, status="feasible")
            return sol

        _optimize_with_retries(mdl)
        first = done(primary)
        if first.status != "optimal":
            # Includes infeasible boxes (status "infeasible", no schedule)
            # and unproven ones ("feasible"); phase 2 needs a proven optimum.
            return first
        mdl.addConstr(fmt.bound_constr(inst, x, primary, getattr(first, primary)),
                      name="fix_primary")
        mdl.setObjective(secondary_expr, GRB.MINIMIZE)
        _apply_params(mdl, certify_params(inst, secondary, params))
        mdl.setParam("Seed", 0)  # reset to the default seed for the new sub-problem
        _optimize_with_retries(mdl)
        return done(secondary)

    return lexmin_bounded, counters


def _make_engine_lexmin_bounded(
    inst: Instance,
    formulation: Formulation | None,
    params: dict,
    per_solve_time_limit: float | None,
    n_seed_retries: int,
    engine: str,
    lpfirst_options=None,
):
    """Dispatch to the requested solve engine's ``lexmin_bounded`` factory.

    Both engines' closures share the exact same call contract (primary/
    secondary/bound as ``"f1"``/``"f2"`` strings — see ``_make_lexmin_bounded``
    and ``bpc.make_bpc_lexmin_bounded``), so ``true_pareto_frontier``/
    ``map_pareto_frontier``'s box-search logic needs zero engine-awareness
    beyond this one dispatch point.

    ``engine="bpc"`` runs the exact branch-and-price engine in
    ``fedhpc.bpc``, with ``per_solve_time_limit`` capping each of its
    phases; it ignores ``formulation``/``n_seed_retries``/``params``
    (Gurobi-solve-specific knobs with no BPC equivalent). ``engine=
    "lpfirst"`` (``fedhpc.lpfirst``: column generation on the time-indexed
    LP, optimality proven by bound and reduced-cost enumeration) takes the
    same arguments; it suits large instances with tiny LP gaps and no
    fixed activation costs.
    """
    if engine == "bpc":
        from . import bpc
        return bpc.make_bpc_lexmin_bounded(inst, time_limit=per_solve_time_limit)
    if engine == "lpfirst":
        from . import lpfirst
        return lpfirst.make_lpfirst_lexmin_bounded(inst, time_limit=per_solve_time_limit,
                                                   options=lpfirst_options)
    if engine != "gurobi":
        raise ValueError(f"unknown engine {engine!r} (expected 'gurobi', 'bpc' or 'lpfirst')")
    fmt = formulation or SpaceTimeFormulation()
    return _make_lexmin_bounded(inst, fmt, params, per_solve_time_limit, n_seed_retries)


# ── Checkpointing (true_pareto_frontier / map_pareto_frontier) ────────────────
#
# Both searches can run for a very long time on a large/dense instance (see
# map_pareto_frontier's docstring), so being killed or interrupted partway
# through — deliberately or not — shouldn't waste the work already done.
# Every point/box resolution gets written to ``checkpoint_path`` as it
# happens; passing the same path back in on a later call resumes from
# exactly that state instead of restarting (skips the two anchor solves too,
# since those are already captured in the saved point/box state).

def _solution_to_dict(s: Solution) -> dict:
    return {
        "status": s.status, "objective": s.objective, "f1": s.f1, "f2": s.f2,
        "assignment": {str(k): list(v) for k, v in s.assignment.items()},
        "completion": {str(k): v for k, v in s.completion.items()},
        "build_time": s.build_time, "solve_time": s.solve_time,
    }


def _solution_from_dict(d: dict) -> Solution:
    return Solution(
        status=d["status"], objective=d["objective"], f1=d["f1"], f2=d["f2"],
        assignment={int(k): tuple(v) for k, v in d["assignment"].items()},
        completion={int(k): int(v) for k, v in d["completion"].items()},
        build_time=d.get("build_time"), solve_time=d.get("solve_time"),
    )


def _save_checkpoint(path: str, **state) -> None:
    """Atomically write ``state`` (already JSON-safe) to ``path``.

    Writes to a temp file and os.replace()'s it into place so a crash or
    kill mid-write never leaves a truncated/corrupt checkpoint behind —
    the previous checkpoint stays valid until the new one is fully written.
    """
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, path)


def _load_checkpoint(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _reference_points(
    inst: Instance, formulation: Formulation | None = None, **params
) -> tuple[float, float, float]:
    """Compute the three reference points used by the weighted-sum single solve.

    Returns
    -------
    f1_T  — f1^T: lexicographic minimum turnaround (min f1 over all feasible x).
    f2_T  — f2^T: minimum cost at f1^T (min f2 s.t. f1 = f1^T).
    f1_0  — f1^0: minimum turnaround at zero cost (min f1 s.t. f2 = 0).

    Raises RuntimeError when no schedule is free; see
    :func:`weighted_sum_reference` (``min_cost_reference=True``) for that case.
    """
    refs, _ = _weighted_sum_reference(inst, formulation, params, min_cost_reference=False, want_ideal=False)
    return refs[:3]


def weighted_sum_reference(
    inst: Instance,
    formulation: Formulation | None = None,
    min_cost_reference: bool = False,
    *,
    engine: str = "gurobi",
    lpfirst_options=None,
    time_limit: float | None = None,
    **params,
) -> tuple[tuple[float, float, float, float], Solution | None]:
    """The weighted-sum reference points (f1_T, f2_T, f1_0, f2_0), and the
    weighted sum's answer when it is degenerate.

    f1_T, f2_T, f1_0 are :func:`_reference_points`' and f2_0 = 0. With
    ``min_cost_reference``, when no schedule is free (e.g. a job has no free
    option, or the free options cannot finish every job within the horizon)
    the cheapest schedule is the reference instead: f2_0 = min f2 and
    f1_0 = min f1 s.t. f2 = f2_0 (lex-min (f2, f1), as :func:`model.solve_f2`);
    without it, that case raises RuntimeError.

    The second item is None unless the reference points are degenerate
    (:func:`model.weighted_sum_degenerate`) and the reference solves found a
    minimum-cost schedule (always, unless some data are negative). It is then a schedule with
    f1 <= f1_0 and minimum cost, i.e., an optimum of the degenerate weighted
    sum (min f2 s.t. f1 <= f1_0), found by the reference solves themselves.

    ``engine="lpfirst"`` solves the same phases with
    :func:`fedhpc.lpfirst.reference_points_lpfirst` (``lpfirst_options``,
    by default HiGHS, which needs no Gurobi licence). ``time_limit`` caps that
    call; the Gurobi engine takes Gurobi's ``TimeLimit`` in ``params``
    instead. Either engine raises RuntimeError unless every value is proven;
    ``reference_points_lpfirst(require_certified=False)`` accepts an
    unproven f1_0 (an upper bound) and reports its lower bound.
    """
    if engine == "lpfirst":
        from . import lpfirst
        if formulation is not None or params:
            raise ValueError("engine='lpfirst' takes no formulation or Gurobi params")
        r = lpfirst.reference_points_lpfirst(
            inst, min_cost_reference=min_cost_reference, time_limit=time_limit,
            options=lpfirst_options)
        return r.refs, r.ideal
    if engine != "gurobi":
        raise ValueError(f"unknown engine {engine!r} (expected 'gurobi' or 'lpfirst')")
    if time_limit is not None:
        params = {**params, "TimeLimit": time_limit}
    return _weighted_sum_reference(inst, formulation, params, min_cost_reference, want_ideal=True)


def solve_weighted_sum_with_reference(
    inst: Instance,
    lam: float,
    formulation: Formulation | None = None,
    min_cost_reference: bool = False,
    **params,
) -> tuple[Solution, tuple[float, float, float, float]]:
    """:func:`weighted_sum_reference` followed by :func:`model.solve_weighted_sum`.

    Returns the solution and the reference points (f1_T, f2_T, f1_0, f2_0).
    When :func:`weighted_sum_reference` returns the degenerate case's optimum,
    that schedule is returned instead of rebuilding the unreduced model to
    solve the same problem again.
    """
    if not (0.0 <= lam <= 1.0):
        raise ValueError("lam must be in [0, 1]")
    fmt = formulation or SpaceTimeFormulation()
    refs, ideal = _weighted_sum_reference(inst, fmt, params, min_cost_reference, want_ideal=True)
    if ideal is not None:
        return ideal, refs
    f1_T, f2_T, f1_0, f2_0 = refs
    return solve_weighted_sum(inst, lam, f1_T, f2_T, f1_0, formulation=fmt, f2_0=f2_0, **params), refs


def _zero_cost_possible(inst: Instance) -> bool:
    """Whether every job has a zero-cost option on a type without fixed cost —
    necessary for a zero-cost schedule when the data are nonnegative."""
    fixed = {m.id: m.fixed_cost for m in inst.instance_types}
    return all(
        any(inst.c[j.id, m] <= 0 and fixed[m] <= 0 for m in inst.F[j.id])
        for j in inst.jobs
    )


def _weighted_sum_reference(
    inst: Instance,
    formulation: Formulation | None,
    params: dict,
    min_cost_reference: bool,
    want_ideal: bool,
) -> tuple[tuple[float, float, float, float], Solution | None]:
    """See :func:`weighted_sum_reference`. ``want_ideal=False`` skips building
    the degenerate case's Solution.

    Phases 1–2 share one model (built with the lex hint, LEX_F1); phase 3
    shares one unhinted model — each solve also warm-starts from the previous
    one's basis/incumbent, since Gurobi keeps that state across ``optimize()``
    calls on the same model instance.
    """
    fmt = formulation or SpaceTimeFormulation()
    mdl, vars_ = fmt.build(inst, objective=LEX_F1)

    x = vars_["x"]
    f1_expr = fmt.f1_expr(inst, x)
    f2_expr = fmt.f2_expr(inst, x)

    def phase(which: str, error: str) -> float:
        """Minimise ``which`` on the current model and return its exact value
        from the schedule. Raises RuntimeError unless that value is certified
        optimal (model.certify_params / is_certified): a reference point
        every weighted sum is normalised by must be proven, not merely the
        incumbent of a time-limited or loose-gap solve."""
        mdl.setObjective(f1_expr if which == "f1" else f2_expr, GRB.MINIMIZE)
        _apply_params(mdl, certify_params(inst, which, params))
        mdl.optimize()
        if mdl.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT):
            raise RuntimeError(error)
        value = getattr(_extract(inst, mdl, vars_), which)
        if value is None or not is_certified(inst, mdl, which, value):
            raise RuntimeError(
                f"{error} No certified optimum (Gurobi status {mdl.Status}, "
                f"incumbent {value}, bound {mdl.ObjBound if mdl.SolCount else None}); "
                "raise the time limit or drop an explicit MIPGap."
            )
        return value

    def fix(which: str, value: float, name: str):
        # which <= value as an integer row (Formulation.bound_constr)
        return mdl.addConstr(fmt.bound_constr(inst, x, which, value), name=name)

    # Phase 1: f1_T = min f1
    f1_T = phase("f1", "Could not solve min-turnaround problem.")

    # Phase 2: f2_T = min f2 s.t. f1 <= f1_T (identical to solve_f1's phase 2 —
    # this *is* the min-cost-at-f1^T problem, so no separate epsilon_turnaround
    # solve is needed).
    fix("f1", f1_T, "fix_f1")
    f2_T = phase("f2", "Could not solve min-cost at f1^T problem.")

    nonnegative = nonnegative_data(inst)
    if nonnegative and f2_T <= 1e-8:
        # f2 >= 0 and the lex-min (f1, f2) schedule costs 0 (to the degeneracy
        # tolerance of model.weighted_sum_degenerate): it is free and
        # turnaround-optimal, so f1_0 = f1_T, f2_0 = 0, and it
        # is the degenerate weighted sum's optimum. Phase 3 is not needed.
        ideal = _extract(inst, mdl, vars_) if want_ideal else None
        return (f1_T, f2_T, f1_T, 0.0), ideal

    # Phase 3: f1_0 = min f1 s.t. f2 <= 0. A model pruned for the lex objective
    # is invalid under a cost bound, so rebuild without the hint; otherwise drop
    # the f1_T fix and reuse the same model.
    if vars_.get("objective_reduced"):
        mdl, vars_ = fmt.build(inst)
        x = vars_["x"]
        f1_expr = fmt.f1_expr(inst, x)
        f2_expr = fmt.f2_expr(inst, x)
    else:
        mdl.remove(mdl.getConstrByName("fix_f1"))
    f2_0 = 0.0
    # Without a free option for some job no schedule is free: skip the attempt.
    zero_cost_infeasible = min_cost_reference and nonnegative and not _zero_cost_possible(inst)
    f1_0 = None
    if not zero_cost_infeasible:
        # f2 <= 0, as an integer row
        eps_cost = fix("f2", 0.0, "eps_cost")
        try:
            f1_0 = phase("f1", "Could not solve min-turnaround at zero-cost problem.")
        except RuntimeError:
            # f1 is bounded (binary variables), so INF_OR_UNBD is infeasible.
            if not (min_cost_reference and mdl.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD)):
                raise
            mdl.remove(eps_cost)
            zero_cost_infeasible = True
    if zero_cost_infeasible:
        # Lex-min (f2, f1) on the same model, as model.solve_f2.
        f2_0 = phase("f2", "Could not solve min-cost problem.")
        fix("f2", f2_0, "fix_f2")
        f1_0 = phase("f1", "Could not solve min-turnaround at minimum cost problem.")

    ideal = None
    # Phase 3's schedule has minimum cost when it was minimised (fix_f2) or
    # when f2 >= 0 (it costs <= 0); otherwise solve_weighted_sum must be used.
    min_cost_schedule = zero_cost_infeasible or nonnegative
    if want_ideal and min_cost_schedule and weighted_sum_degenerate(f1_T, f2_T, f1_0, f2_0):
        # Phase 3's schedule has f1 = f1_0 and minimum cost: an optimum of
        # min f2 s.t. f1 <= f1_0, reported with that objective, as
        # model.solve_weighted_sum's degenerate case does.
        ideal = _extract(inst, mdl, vars_)
        ideal = replace(ideal, objective=ideal.f2)
    return (f1_T, f2_T, f1_0, f2_0), ideal


def true_pareto_frontier(
    inst: Instance,
    formulation: Formulation | None = None,
    verbose: bool = False,
    hint: Solution | None = None,
    max_solves: int | None = None,
    time_budget: float | None = None,
    per_solve_time_limit: float | None = None,
    n_seed_retries: int = 3,
    checkpoint_path: str | None = None,
    engine: str = "gurobi",
    lpfirst_options=None,
    **params,
) -> list[Solution]:
    """Enumerate the complete true Pareto front via exact box (rectangle) splitting.

    ``engine`` — ``"gurobi"`` (default, the compact ``SpaceTimeFormulation``
    MIP), ``"bpc"`` (the exact branch-and-price engine in ``fedhpc.bpc``) or
    ``"lpfirst"`` (``fedhpc.lpfirst``, for large near-integral instances;
    ``lpfirst_options`` passes its :class:`~fedhpc.lpfirst.LPFirstOptions`).
    Selecting ``"bpc"`` changes nothing about this function's algorithm —
    see ``_make_engine_lexmin_bounded`` — only which engine resolves each
    box's lex-min-bounded sub-solve. BPC currently has no equivalent of
    ``per_solve_time_limit``/``n_seed_retries``-style graceful degradation
    under a wall-clock cap (see ``bpc.make_bpc_lexmin_bounded``'s
    docstring), and ``formulation`` is ignored under it.

    Exactness rests on both objectives moving in whole steps: f1 by 1
    (integral job weights — ``Instance.build`` rejects fractional ones — and
    integer completion slots) and f2 by the cost resolution δ
    (``Instance.cost_resolution``; ``build`` rounds costs to that grid). Each
    sub-solve is certified against those steps (see ``_make_lexmin_bounded``)
    and every bound is an integer row on the grid
    (``Formulation.bound_constr``), so the search needs no tight MIPGap and
    no epsilon judgement.

    Standard exact algorithm for biobjective integer programs (Hamacher,
    Pedersen & Ruzika 2007; related to the "balanced box" method of Boland,
    Charkhgard & Savelsbergh 2015) — used here instead of a per-integer-level
    ε-constraint sweep because that sweep costs O(f1_range) solves
    regardless of how many Pareto-optimal points actually exist, which is
    intractable when f1's range spans thousands of integer levels but the
    true front is sparse. Box splitting instead costs
    roughly 2n solves for a front of n points: every solve either discovers
    a new point or proves the current gap contains nothing (closing it) —
    so cost tracks the size of the *answer*, not the size of the objective
    space.

    Algorithm
    ---------
    1. Solve lex-min(f1, f2) → point A (leftmost: minimal turnaround).
    2. Solve lex-min(f2, f1) → point B (rightmost: minimal cost).
       If A == B there is only one Pareto point; done.
    3. Maintain a deque of open *boxes* — adjacent pairs (L, R) of
       currently-known frontier points (L.f1 < R.f1, hence L.f2 > R.f2,
       since neither may dominate the other). Seed it with box (A, B).
    4. Repeatedly pop a box, alternating between the deque's two ends —
       narrowing the currently-open gap alternately from each side rather
       than always walking it in one direction:
         - popped from the LEFT  → grow from L: minimise f1 s.t.
           f2 ≤ L.f2 − δ (at least one cost step below L's value in the
           *other* objective), tie-broken by minimising f2 second.
         - popped from the RIGHT → grow from R: minimise f2 s.t.
           f1 ≤ R.f1 − 1 (one f1 step below R), tie-broken by minimising
           f1 second.
       If the result is the box's other endpoint (same grid values), the box is
       empty — discard it. Otherwise a genuinely new point P was found — and
       because P is the *global* optimum of that bounded sub-problem (not
       merely the best point inside this box), it must be L's (or R's)
       immediate neighbour on the true frontier: any point between L and P
       would have beaten P for that exact sub-problem, and any point
       between P and L's other side would dominate the already-proven-
       Pareto-optimal L or R. So only the remaining half of the box — (P, R)
       when growing from L, (L, P) when growing from R — can still be open,
       and that's the only piece pushed back. (This invariant means the
       deque never actually holds more than one box at a time under this
       always-solve-to-global-optimality strategy — the frontier is
       discovered as a single gap, closed in from alternating ends, not a
       branching search tree. The deque is kept anyway since it's the
       natural structure for this algorithm family and costs nothing here.)
    5. Terminate when the deque is empty — the accumulated points are the
       complete, provably-exact Pareto front (nothing is missed: every
       region of the front lives inside the open gap until explicitly
       proven empty, unlike hybrid_frontier's approach, which only ever
       certifies points an EA happened to propose as candidates).

    Unlike this module's other exact methods, the MIP model is rebuilt fresh
    for *every* solve rather than reused across the whole search — see the
    tolerance/correctness comment inside the function body for why reuse
    turned out to be actively unsafe here, not just a missed optimization.

    ``hint`` — an optional prior/heuristic Solution (e.g. from moea.py) used
    to seed the MIP start of the very first (anchor) solve only.

    ``max_solves`` / ``time_budget`` — optional caps (solve count / wall-
    clock seconds) on the box-search phase (the two anchor solves always run
    to completion first, regardless of budget). When either is hit, the
    search stops and returns whatever's been found so far — a **partial**
    front. Every returned point is still individually exact (each one is
    the proven global optimum of its own sub-problem), but the front is not
    guaranteed complete: some open gaps may remain unexplored. A warning is
    always printed to stderr when this happens (independent of ``verbose``),
    since silently returning a truncated "complete" front would be
    dangerous. This exists because "sparse front, cheap to enumerate fully"
    does not hold universally — on this project's largest, most heavily
    congested instance the true front turned out to have hundreds-to-
    thousands of points densely packed near the turnaround-minimal end
    (found empirically: 27 points in ~3 hours with no sign of the density
    letting up), making unbounded enumeration impractical there even though
    it completes in well under a second on every small/medium instance
    tested. Both default to ``None`` (unlimited — full exact enumeration,
    the original/default behaviour).

    ``per_solve_time_limit`` — optional cap in seconds on each solve phase
    (each Gurobi attempt, or each BPC branch-and-price call), independent
    of ``max_solves``/``time_budget``, which only check *between* solves.
    Needed because some sub-problems have a good incumbent almost instantly
    but take far longer to prove optimal — observed (under the earlier
    1e-9 relative MIPGap) running 6+ hours and 100M+ B&B nodes on one
    sub-problem, eventually killed (likely OOM from the B&B tree). A box
    whose query is still unproven after ``n_seed_retries`` attempts is
    treated as unresolved and dropped — contributing to incompleteness the
    same way a budget truncation does, reported via the same warning.
    Strongly recommended whenever running on an instance you haven't
    already characterised as tractable.

    ``checkpoint_path`` — see map_pareto_frontier's docstring; identical
    contract here (every box resolution atomically rewrites the file;
    passing the same path back in resumes from it, skipping the anchors).
    """
    # This algorithm's correctness proof (see "Algorithm" above) hinges on
    # every solve returning the *true* global optimum: a "good enough" MIP
    # solve can report a value that's off from the true one, which corrupts
    # both the box-growth exclusion cut and the "fix_primary" tie-break
    # bound. Reaching that reliably needs two things:
    #
    #   1. Exact optimality. Both objectives move in whole steps (f1 by 1,
    #      f2 by the cost resolution δ), so each phase is certified by an
    #      absolute gap under one step against the incumbent's exact value
    #      recomputed from its schedule (model.certify_params /
    #      is_certified), and every bound is an integer row on that grid
    #      (Formulation.bound_constr). This replaced a 1e-9 relative MIPGap, which
    #      occasionally ran for hours on a box. IntFeasTol stays at 1e-9:
    #      at the default 1e-5, near-integral incumbents measured a ~0.0016
    #      objective discrepancy on a real instance (large.json), more than a
    #      cost step, so their rounded schedule would fail the certificate.
    #   2. A fresh model per solve. The natural design reuses one persistent
    #      model across the whole search (only the bounding constraint and
    #      objective direction change), which is exactly what the *other*
    #      exact methods in this module do to get Gurobi's automatic
    #      basis/incumbent warm-start between solves. It's wrong here: this
    #      algorithm repeatedly changes *which* constraint is bounded and
    #      *which* expression is the objective, and reusing a solved model
    #      across a structurally different constraint set risks Gurobi
    #      treating stale internal solve state (cutting planes, presolve
    #      reductions) derived under the old constraints as still valid.
    #      Caught in testing by reproducing one specific box's sub-problem
    #      twice — once cold (fresh model, that one constraint) and once
    #      mid-chain — both reporting Status=OPTIMAL, MIPGap=0.0, but the
    #      chained one's answer was a real, verifiably suboptimal point.
    #      Neither object-reference constraint tracking nor mdl.reset()
    #      fully closed this across every instance tested, so each solve
    #      below gets its own freshly built model instead. The old
    #      per-integer-level sweep never needed to care about any of this
    #      because its f1-margin (~1.0) dwarfed all of it; this algorithm's
    #      f2 (cost) margin doesn't have that luxury, and correctness here
    #      matters more than shaving off rebuild time — this function exists
    #      specifically to be the trustworthy ground truth other methods
    #      (hybrid_frontier, the EAs) get checked against.
    #
    # This does cost real time on large instances (build alone measured
    # ~29s on the biggest one, ~13s on pos_congestion), paid once per solve:
    # about 2 builds per Pareto point. That is cheap on a sparse front and
    # dominant on a dense one — see the docstring's notes on dense fronts.
    if engine == "gurobi":
        params = {"IntFeasTol": 1e-9, **params}
    half1, half2 = objective_step(inst, "f1") / 2, objective_step(inst, "f2") / 2

    _lexmin_bounded, _counters = _make_engine_lexmin_bounded(
        inst, formulation, params, per_solve_time_limit, n_seed_retries, engine,
        lpfirst_options,
    )

    f1_of, f2_of = "f1", "f2"

    n_boxes_closed = 0
    n_boxes_unresolved = 0

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        ckpt = _load_checkpoint(checkpoint_path)
        solutions: list[Solution] = [_solution_from_dict(d) for d in ckpt["solutions"]]
        boxes: deque[tuple[Solution, Solution]] = deque(
            (_solution_from_dict(l), _solution_from_dict(r)) for l, r in ckpt["boxes"]
        )
        side = ckpt["side"]
        _counters["n_solves"] = ckpt.get("n_solves", 0)
        _counters["build_time"] = ckpt.get("build_time", 0.0)
        n_boxes_closed = ckpt.get("n_boxes_closed", 0)
        n_boxes_unresolved = ckpt.get("n_boxes_unresolved", 0)
        if verbose:
            print(f"pareto-true: resumed from checkpoint — {len(solutions)} points, "
                  f"{len(boxes)} box(es) open, {_counters['n_solves']} prior solves",
                  file=sys.stderr, flush=True)
    else:
        # ── Two extreme anchors ────────────────────────────────────────────
        point_a = _lexmin_bounded(f1_of, f2_of, start=hint)
        if point_a.status not in ("optimal", "infeasible"):
            print(f"pareto-true: WARNING — anchor A solve INCONCLUSIVE "
                  f"(status={point_a.status}); returning no points.",
                  file=sys.stderr, flush=True)
            return []
        if point_a.f1 is None:
            return []
        point_b = _lexmin_bounded(f2_of, f1_of)
        if point_b.status not in ("optimal", "infeasible"):
            print(f"pareto-true: WARNING — anchor B solve INCONCLUSIVE "
                  f"(status={point_b.status}); returning only anchor A.",
                  file=sys.stderr, flush=True)
            return _filter_dominated([point_a])
        if point_b.f1 is None:
            return _filter_dominated([point_a])

        if verbose:
            print(f"pareto-true: anchor A (min f1)  f1={point_a.f1:.4f}  f2={point_a.f2:.6g}",
                  file=sys.stderr, flush=True)
            print(f"pareto-true: anchor B (min f2)  f1={point_b.f1:.4f}  f2={point_b.f2:.6g}",
                  file=sys.stderr, flush=True)

        solutions = [point_a]
        if abs(point_a.f1 - point_b.f1) < half1 and abs(point_a.f2 - point_b.f2) < half2:
            if verbose:
                print("pareto-true: single-point front (A == B)", file=sys.stderr, flush=True)
            return solutions
        solutions.append(point_b)

        boxes = deque([(point_a, point_b)])
        side = "left"

    def _checkpoint() -> None:
        if checkpoint_path is None:
            return
        _save_checkpoint(
            checkpoint_path,
            solutions=[_solution_to_dict(s) for s in solutions],
            boxes=[[_solution_to_dict(l), _solution_to_dict(r)] for l, r in boxes],
            side=side,
            n_solves=_counters["n_solves"],
            build_time=_counters["build_time"],
            n_boxes_closed=n_boxes_closed,
            n_boxes_unresolved=n_boxes_unresolved,
        )

    _checkpoint()

    search_t0 = _time.perf_counter()
    truncated = False

    while boxes:
        if max_solves is not None and _counters["n_solves"] >= max_solves:
            truncated = True
            break
        if time_budget is not None and _time.perf_counter() - search_t0 >= time_budget:
            truncated = True
            break

        left, right = boxes.popleft() if side == "left" else boxes.pop()

        if side == "left":
            # f2 at least one cost step below L: f2 <= L.f2 - δ/2
            sol = _lexmin_bounded(f1_of, f2_of, f2_of, left.f2 - half2)
        else:
            # f1 at least one step below R. Callers pass the half-step
            # value R.f1 - 1/2; bound_constr writes it as the integer row
            # Σ w (t + p) x <= level(R) - 1. A real-valued RHS a hair below
            # an achievable value (R.f1 - 1e-4, e.g. 538.9999) once made
            # Gurobi report a suboptimal point as OPTIMAL (stess_test.json),
            # and presolve rounded f2 <= 35 - 5e-5 up to 35 on all-integer
            # cost data; integer rows leave no such ambiguity.
            sol = _lexmin_bounded(f2_of, f1_of, f1_of, right.f1 - half1)

        # A solve that hit per_solve_time_limit without proving optimality
        # (status "feasible" from a SUBOPTIMAL/TIME_LIMIT result — see
        # _make_lexmin_bounded) proves nothing: not that a point exists,
        # not that the box is empty. Drop the box rather than retry it
        # (a deterministic solve will just hang the same way again) —
        # this is a genuine, reported gap in completeness, same family as
        # a budget truncation.
        if sol.status not in ("optimal", "infeasible"):
            n_boxes_unresolved += 1
            if verbose:
                print(f"pareto-true: box [{left.f1:.4f},{right.f1:.4f}] "
                      f"({side})  → INCONCLUSIVE (status={sol.status}), dropped",
                      file=sys.stderr, flush=True)
            side = "right" if side == "left" else "left"
            _checkpoint()
            continue

        if side == "left":
            is_empty = sol.f1 is None or sol.f1 >= right.f1 - half1
        else:
            is_empty = sol.f1 is None or sol.f2 >= left.f2 - half2

        if is_empty:
            n_boxes_closed += 1
            if verbose:
                print(f"pareto-true: box [{left.f1:.4f},{right.f1:.4f}] "
                      f"({side})  → empty", file=sys.stderr, flush=True)
        else:
            solutions.append(sol)
            if verbose:
                print(f"pareto-true: box [{left.f1:.4f},{right.f1:.4f}] "
                      f"({side})  → new point f1={sol.f1:.4f} f2={sol.f2:.6g}",
                      file=sys.stderr, flush=True)
            # Only the sub-box away from the growth direction can still be
            # open — see the docstring: the other half is provably resolved
            # by sol's global optimality, so re-queuing it would only waste
            # a solve re-confirming what's already known.
            if side == "left":
                boxes.appendleft((sol, right))
            else:
                boxes.append((left, sol))

        side = "right" if side == "left" else "left"
        _checkpoint()

    _checkpoint()

    if verbose:
        print(f"pareto-true: {len(solutions)} points, {n_boxes_closed} boxes closed, "
              f"{n_boxes_unresolved} unresolved, "
              f"{_counters['n_solves']} total solves", file=sys.stderr, flush=True)

    if truncated or n_boxes_unresolved > 0:
        reason = "search budget exhausted" if truncated else "per-solve time limit hit"
        print(
            f"pareto-true: WARNING — {reason} "
            f"({_counters['n_solves']} solves, {_time.perf_counter() - search_t0:.1f}s in box search, "
            f"{n_boxes_unresolved} box(es) dropped as inconclusive); "
            f"returning a PARTIAL front ({len(solutions)} points found, "
            f"{len(boxes)} gap(s) still open and unexplored). "
            f"Every returned point is individually exact, but the front is incomplete.",
            file=sys.stderr, flush=True,
        )

    return _filter_dominated(solutions)


def map_pareto_frontier(
    inst: Instance,
    formulation: Formulation | None = None,
    verbose: bool = False,
    hint: Solution | None = None,
    max_solves: int | None = None,
    time_budget: float | None = None,
    per_solve_time_limit: float | None = None,
    n_seed_retries: int = 3,
    checkpoint_path: str | None = None,
    engine: str = "gurobi",
    **params,
) -> list[Solution]:
    """Map the Pareto front by bisection — broad coverage fast, not exhaustive proof.

    ``engine`` — see ``true_pareto_frontier``'s docstring; identical contract
    here (``"gurobi"`` default or ``"bpc"``, dispatched via
    ``_make_engine_lexmin_bounded``).

    ``true_pareto_frontier()`` always resolves a box by jumping straight to
    its known-boundary's global optimum, which is what makes it complete in
    ~2n solves for n points — but it also means it processes the front in
    strict f1-order, one immediate neighbour at a time. On a front that
    turns out to be very dense in one region (found empirically: this
    project's largest instance has hundreds-to-thousands of points packed
    near its turnaround-minimal end), that strategy can spend its *entire*
    budget resolving the first few hundred units of a 5000+-unit range
    without ever taking a single sample from the rest of the front.

    This function trades the completeness guarantee for a different one:
    under a limited solve/time budget, always spend the next solve cutting
    the *currently largest known gap* in half, alternating between bisecting
    in f1-space and f2-space each step — so coverage spreads across the
    *whole* range early, and only refines locally once the broad shape is
    already sampled. Every returned point is still individually exact (same
    solve machinery as true_pareto_frontier — global optimum of its own
    sub-problem), but unlike that function, finding a point here does *not*
    prove the neighbouring gaps are empty: a bisection query only asks "is
    there anything under this m midpoint bound", so both halves it creates
    remain open for further exploration, not just one.

    Algorithm
    ---------
    1. Same two anchors A, B as true_pareto_frontier.
    2. Maintain an unordered pool of open boxes (L, R), seeded with (A, B).
       A box is only ever discarded once L.f1 and R.f1 are adjacent integers
       (R.f1 - L.f1 <= 1) — f1 is always integer-spaced (integral job weights,
       enforced by ``Instance.build``), so that's a rigorous proof nothing
       more fits between them, regardless of which direction
       last touched the box.
    3. Each step, alternate direction and pick the open box with the largest
       gap *in that direction* (f1-span or f2-span) — always attacking
       whatever's currently the biggest unknown region:
         - f1-direction: mid = the integer midpoint of [L.f1, R.f1]. Solve
           min f2 s.t. f1 <= mid (tie-broken by min f1). If no better than L
           is found, [L.f1, mid] is proven empty — keep exploring (mid, R]
           behind a synthetic boundary point at (mid, L.f2). If a genuinely
           better point P turns up, keep *both* (L, P) and (P, R) open —
           unlike true_pareto_frontier, a midpoint query does not prove
           either half is fully resolved.
         - f2-direction: mirrors this bounding on f2's midpoint instead,
           minimising f1.
    4. Stop when the budget runs out or every box has closed (in which case
       the result is provably the complete front too, same as
       true_pareto_frontier — bisection isn't inherently incomplete, it's
       just optimized for coverage-under-budget rather than solve-count).

    Parameters mirror true_pareto_frontier's, with the same fresh-model-per-
    solve, tight-tolerance approach (see that function's docstring for why).

    ``checkpoint_path`` — optional file path. Every box resolution (found
    point / empty half / dropped-inconclusive) atomically rewrites it with
    the full search state; passing the same path back in on a later call
    (even a fresh process) resumes exactly from there instead of restarting
    — the anchor solves are skipped entirely, since a checkpoint already
    contains their result. This matters because this function's whole
    reason to exist is running on instances where individual solves can be
    slow or occasionally need seed retries (see ``n_seed_retries``), so a
    multi-hour exploration being killed or interrupted partway through
    shouldn't have to start over. If the checkpoint shows a fully-resolved
    search (no boxes left), it's returned immediately with no new solves.
    """
    solve_params = {"IntFeasTol": 1e-9, **params} if engine == "gurobi" else params
    half1, half2 = objective_step(inst, "f1") / 2, objective_step(inst, "f2") / 2
    lexmin_bounded, counters = _make_engine_lexmin_bounded(
        inst, formulation, solve_params, per_solve_time_limit, n_seed_retries, engine
    )

    def f1_steps(lo: Solution, hi: Solution) -> int:
        """Whole f1 steps between two (possibly boundary) points."""
        return round(hi.f1 - lo.f1)

    f1_of, f2_of = "f1", "f2"

    n_boxes_unresolved = 0

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        ckpt = _load_checkpoint(checkpoint_path)
        solutions = [_solution_from_dict(d) for d in ckpt["solutions"]]
        boxes = [(_solution_from_dict(l), _solution_from_dict(r)) for l, r in ckpt["boxes"]]
        direction = ckpt["direction"]
        counters["n_solves"] = ckpt.get("n_solves", 0)
        counters["build_time"] = ckpt.get("build_time", 0.0)
        n_boxes_unresolved = ckpt.get("n_boxes_unresolved", 0)
        if verbose:
            print(f"pareto-map: resumed from checkpoint — {len(solutions)} points, "
                  f"{len(boxes)} box(es) open, {counters['n_solves']} prior solves",
                  file=sys.stderr, flush=True)
    else:
        point_a = lexmin_bounded(f1_of, f2_of, start=hint)
        if point_a.status not in ("optimal", "infeasible"):
            print(f"pareto-map: WARNING — anchor A solve INCONCLUSIVE "
                  f"(status={point_a.status}); returning no points.",
                  file=sys.stderr, flush=True)
            return []
        if point_a.f1 is None:
            return []
        point_b = lexmin_bounded(f2_of, f1_of)
        if point_b.status not in ("optimal", "infeasible"):
            print(f"pareto-map: WARNING — anchor B solve INCONCLUSIVE "
                  f"(status={point_b.status}); returning only anchor A.",
                  file=sys.stderr, flush=True)
            return _filter_dominated([point_a])
        if point_b.f1 is None:
            return _filter_dominated([point_a])

        if verbose:
            print(f"pareto-map: anchor A (min f1)  f1={point_a.f1:.4f}  f2={point_a.f2:.6g}",
                  file=sys.stderr, flush=True)
            print(f"pareto-map: anchor B (min f2)  f1={point_b.f1:.4f}  f2={point_b.f2:.6g}",
                  file=sys.stderr, flush=True)

        solutions = [point_a]
        if abs(point_a.f1 - point_b.f1) < half1 and abs(point_a.f2 - point_b.f2) < half2:
            if verbose:
                print("pareto-map: single-point front (A == B)", file=sys.stderr, flush=True)
            return solutions
        solutions.append(point_b)

        boxes = [(point_a, point_b)]
        direction = "f1"

    def _checkpoint() -> None:
        if checkpoint_path is None:
            return
        _save_checkpoint(
            checkpoint_path,
            solutions=[_solution_to_dict(s) for s in solutions],
            boxes=[[_solution_to_dict(l), _solution_to_dict(r)] for l, r in boxes],
            direction=direction,
            n_solves=counters["n_solves"],
            build_time=counters["build_time"],
            n_boxes_unresolved=n_boxes_unresolved,
        )

    _checkpoint()

    search_t0 = _time.perf_counter()
    truncated = False

    def _boundary(f1: float, f2: float) -> Solution:
        return Solution(status="boundary", objective=None, f1=f1, f2=f2,
                        assignment={}, completion={})

    while boxes:
        if max_solves is not None and counters["n_solves"] >= max_solves:
            truncated = True
            break
        if time_budget is not None and _time.perf_counter() - search_t0 >= time_budget:
            truncated = True
            break

        # f2-direction needs at least two cost steps to bisect on; fall back
        # to f1 (always splittable while a box is open) otherwise.
        this_dir = direction
        if this_dir == "f2" and max(inst.f2_units(l.f2) - inst.f2_units(r.f2)
                                    for l, r in boxes) < 2:
            this_dir = "f1"

        if this_dir == "f1":
            idx = max(range(len(boxes)), key=lambda i: boxes[i][1].f1 - boxes[i][0].f1)
        else:
            idx = max(range(len(boxes)), key=lambda i: boxes[i][0].f2 - boxes[i][1].f2)
        left, right = boxes.pop(idx)

        # Midpoints sit on the objective grids (bound_constr writes the
        # half-step values as integer rows on them).
        if this_dir == "f1":
            mid = left.f1 + f1_steps(left, right) // 2
            sol = lexmin_bounded(f2_of, f1_of, f1_of, mid + half1)
            label = f"f1<={mid:.0f}"
        else:
            lo, hi = inst.f2_units(right.f2), inst.f2_units(left.f2)
            mid = (lo + (hi - lo) // 2) * inst.cost_resolution
            sol = lexmin_bounded(f1_of, f2_of, f2_of, mid + half2)
            label = f"f2<={mid:.4g}"

        # A solve that hit per_solve_time_limit without proving optimality
        # (status "feasible" from SUBOPTIMAL/TIME_LIMIT — see
        # _make_lexmin_bounded) proves nothing either way: not that a
        # better point exists, not that the half is empty. Drop the box —
        # a deterministic re-solve would just hang the same way again.
        if sol.status not in ("optimal", "infeasible"):
            n_boxes_unresolved += 1
            if verbose:
                print(f"pareto-map: box [{left.f1:.4f},{right.f1:.4f}] "
                      f"({this_dir}, {label})  → INCONCLUSIVE (status={sol.status}), dropped",
                      file=sys.stderr, flush=True)
            direction = "f2" if direction == "f1" else "f1"
            _checkpoint()
            continue

        if this_dir == "f1":
            found_better = sol.f1 is not None and sol.f2 < left.f2 - half2
        else:
            found_better = sol.f1 is not None and sol.f1 < right.f1 - half1

        if not found_better:
            # Proven empty up to mid; shrink whichever side the query was
            # anchored on and keep exploring the remainder, if any is left.
            if this_dir == "f1":
                new_box = (_boundary(mid, left.f2), right)
            else:
                new_box = (left, _boundary(right.f1, mid))
            if f1_steps(*new_box) > 1:
                boxes.append(new_box)
            if verbose:
                print(f"pareto-map: box [{left.f1:.4f},{right.f1:.4f}] "
                      f"({this_dir}, {label})  → empty half",
                      file=sys.stderr, flush=True)
        else:
            solutions.append(sol)
            if verbose:
                print(f"pareto-map: box [{left.f1:.4f},{right.f1:.4f}] "
                      f"({this_dir}, {label})  → new point f1={sol.f1:.4f} f2={sol.f2:.6g}",
                      file=sys.stderr, flush=True)
            if f1_steps(left, sol) > 1:
                boxes.append((left, sol))
            if f1_steps(sol, right) > 1:
                boxes.append((sol, right))

        direction = "f2" if direction == "f1" else "f1"
        _checkpoint()

    _checkpoint()

    if verbose:
        print(f"pareto-map: {len(solutions)} points, {counters['n_solves']} total solves, "
              f"{n_boxes_unresolved} unresolved, "
              f"{len(boxes)} box(es) still open", file=sys.stderr, flush=True)

    if truncated or n_boxes_unresolved > 0:
        reason = "search budget exhausted" if truncated else "per-solve time limit hit"
        print(
            f"pareto-map: WARNING — {reason} "
            f"({counters['n_solves']} solves, {_time.perf_counter() - search_t0:.1f}s, "
            f"{n_boxes_unresolved} box(es) dropped as inconclusive); "
            f"returning a PARTIAL map ({len(solutions)} points found, "
            f"{len(boxes)} gap(s) still open and unexplored). "
            f"Every returned point is individually exact, but the map is incomplete.",
            file=sys.stderr, flush=True,
        )

    return _filter_dominated(solutions)


def _filter_dominated(solutions: list[Solution]) -> list[Solution]:
    """The solutions no other one strictly dominates, in input order (exact
    duplicates are all kept). O(n log n): a sweep in (f1, f2) order."""
    pts = [s for s in solutions if s.f1 is not None and s.f2 is not None]
    order = sorted(range(len(pts)), key=lambda i: (pts[i].f1, pts[i].f2))
    keep = [False] * len(pts)
    best_f2 = float("inf")   # min f2 over points with strictly smaller f1
    k = 0
    while k < len(order):
        f1 = pts[order[k]].f1
        group = []
        while k < len(order) and pts[order[k]].f1 == f1:
            group.append(order[k])
            k += 1
        group_f2 = pts[group[0]].f2   # the group's min f2 (sorted by f2)
        for i in group:
            keep[i] = pts[i].f2 < best_f2 and pts[i].f2 == group_f2
        best_f2 = min(best_f2, group_f2)
    return [s for s, kept in zip(pts, keep) if kept]


def hybrid_frontier(
    inst: Instance,
    *,
    pop_size: int = 400,
    n_gen: int = 300,
    neighborhood_size: int = 20,
    seed: int = 42,
    n_threads: int = 0,
    formulation: Formulation | None = None,
    time_limit: float | None = None,
    mip_gap: float | None = None,
    verbose: bool = False,
    **mip_params,
) -> list[Solution]:
    """Hybrid Pareto frontier: EA exploration + Gurobi ε-constraint verification.

    Phase 1 — EA: run NSGA-II (seed) and MOEA/D (seed+1) to build a pool of
    non-dominated candidate solutions that jointly span the tradeoff space.

    Phase 2a — Anchors: take the EA's best-cost and best-turnaround solutions and
    solve lex-min (f1, f2) s.t. f2 ≤ f2 of the EA's cheapest, and lex-min
    (f2, f1) s.t. f1 ≤ f1 of the EA's fastest. Each is a Pareto-optimal point
    (a lex-min under a cap cannot be dominated), but not necessarily an
    extreme of the front: the EA may have missed the true cheapest or
    fastest schedule. They prune the candidate pool.

    Phase 2b — Prune: a candidate is interior only if it strictly improves on
    both anchors simultaneously (f1_c < anchor_cheap.f1 AND f2_c < anchor_fast.f2).
    All other candidates are already dominated and are discarded.

    Phase 2c — MIP interior: for each surviving interior candidate's cost
    threshold, solve lex-min (f1, f2) s.t. f2 ≤ f2_k with Gurobi, warm-started
    from the EA assignment.

    Phase 3 — Filter: final Pareto filter removes duplicates and dominances.

    Every solve goes through the same certified lex-min as
    :func:`true_pareto_frontier` (``_make_lexmin_bounded``), so a point with
    status ``"optimal"`` is proven Pareto-optimal. It is *not* a complete
    front: only the trade-offs near the EA's candidates are visited. Points
    whose solve hit ``time_limit`` (or ran under an explicit ``mip_gap``)
    keep status ``"feasible"``: a schedule, not a proven Pareto point.

    Parameters
    ----------
    inst              : built fedhpc Instance.
    pop_size          : EA population / weight-vector count.
    n_gen             : EA generations.
    neighborhood_size : MOEA/D neighbourhood size |T|.
    seed              : RNG seed for NSGA-II (MOEA/D uses seed+1).
    n_threads         : OpenMP thread count; 0 = all cores.
    formulation       : MIP formulation; defaults to SpaceTimeFormulation.
    time_limit        : Gurobi time limit per ε-constraint solve (seconds); None = no limit.
    mip_gap           : Gurobi MIPGap; None (the default) certifies each
                        phase exactly. Any value overrides the certificate
                        and can leave points ``"feasible"``.
    verbose           : print phase progress and Gurobi log.
    **mip_params      : additional Gurobi parameters forwarded to every solve.

    Returns
    -------
    Mutually non-dominated MIP solutions: ``"optimal"`` ones are proven
    Pareto-optimal, ``"feasible"`` ones are not (see above).
    """
    from .formulations import configure_env
    from .moea import moead_frontier, nsga2_frontier

    # ── Phase 1: EA exploration ───────────────────────────────────────────────

    if verbose:
        print("hybrid: phase 1 — NSGA-II …", file=sys.stderr, flush=True)
    ea1 = nsga2_frontier(
        inst, pop_size=pop_size, n_gen=n_gen, seed=seed, n_threads=n_threads,
    )

    if verbose:
        print("hybrid: phase 1 — MOEA/D …", file=sys.stderr, flush=True)
    ea2 = moead_frontier(
        inst, n_weights=pop_size, n_gen=n_gen,
        neighborhood_size=neighborhood_size, seed=seed + 1, n_threads=n_threads,
    )

    candidates = _filter_dominated(ea1 + ea2)

    if verbose:
        print(
            f"hybrid: {len(ea1)} NSGA-II + {len(ea2)} MOEA/D"
            f" → {len(candidates)} non-dominated candidates",
            file=sys.stderr, flush=True,
        )

    if not candidates:
        return []

    configure_env(verbose=verbose)
    fmt = formulation or SpaceTimeFormulation()

    # Same certified lex-min as true_pareto_frontier (IntFeasTol: see there)
    mip_kw: dict = {"IntFeasTol": 1e-9, **mip_params}
    if mip_gap is not None:
        mip_kw["MIPGap"] = mip_gap
    mip_kw["OutputFlag"] = 1 if verbose else mip_kw.get("OutputFlag", 0)
    lexmin_bounded, _ = _make_lexmin_bounded(inst, fmt, mip_kw, time_limit)

    def _mip_solve(
        hint: Solution | None,
        *,
        constrain_f2: float | None = None,
        constrain_f1: float | None = None,
        minimize_cost: bool = False,
    ) -> Solution:
        """Lex-min (f2, f1) if ``minimize_cost`` else (f1, f2), under the
        given cap, warm-started from ``hint``."""
        primary, secondary = ("f2", "f1") if minimize_cost else ("f1", "f2")
        bound, value = (("f2", constrain_f2) if constrain_f2 is not None
                        else ("f1", constrain_f1) if constrain_f1 is not None
                        else (None, None))
        return lexmin_bounded(primary, secondary, bound, value, start=hint)

    # ── Phase 2a: Anchor solves — pin both extremes of the Pareto front ───────
    # ea_fast  has the lowest turnaround among EA candidates.
    # ea_cheap has the lowest cost among EA candidates.
    # We solve the complementary objective for each to get exact Pareto corners.

    ea_fast  = min(candidates, key=lambda s: s.f1 or float('inf'))
    ea_cheap = min(candidates, key=lambda s: s.f2 or float('inf'))

    if verbose:
        print(
            f"hybrid: phase 2a — anchor solves"
            f"  (EA f1_min={ea_fast.f1:.1f}  f2_min={ea_cheap.f2:.4f}) …",
            file=sys.stderr, flush=True,
        )

    # lex-min (f1, f2) within the EA's best cost: a Pareto point at or below it
    anchor_fast  = _mip_solve(ea_cheap, constrain_f2=ea_cheap.f2)
    # lex-min (f2, f1) within the EA's best turnaround
    anchor_cheap = _mip_solve(ea_fast,  constrain_f1=ea_fast.f1, minimize_cost=True)

    exact: list[Solution] = [s for s in (anchor_fast, anchor_cheap) if s.f1 is not None]
    exact = _filter_dominated(exact)

    if verbose:
        for sol, label in ((anchor_fast, "anchor_fast"), (anchor_cheap, "anchor_cheap")):
            if sol.f1 is not None:
                print(
                    f"  {label}: f1={sol.f1:.1f}  f2={sol.f2:.4f}  [{sol.status}]",
                    file=sys.stderr, flush=True,
                )

    # ── Phase 2b: Prune candidates dominated by the anchors ───────────────────
    # An interior Pareto point must strictly improve on both anchors:
    #   f1_c < anchor_cheap.f1  — better turnaround than the min-cost corner
    #   f2_c < anchor_fast.f2   — better cost than the min-turnaround corner
    f1_cut = anchor_cheap.f1 if anchor_cheap.f1 is not None else float('inf')
    f2_cut = anchor_fast.f2  if anchor_fast.f2  is not None else float('inf')

    interior = [c for c in candidates
                if (c.f1 or float('inf')) < f1_cut
                and (c.f2 or float('inf')) < f2_cut]

    # Sort by f2 ascending; deduplicate equal cost levels (keep best-f1 hint).
    sorted_interior = sorted(interior, key=lambda s: (s.f2 or 0.0, s.f1 or 0.0))
    deduped: list[Solution] = []
    seen_f2: list[float] = []
    for cand in sorted_interior:
        f2 = cand.f2 or 0.0
        if not any(abs(f2 - prev) < 1e-9 for prev in seen_f2):
            seen_f2.append(f2)
            deduped.append(cand)

    if verbose:
        print(
            f"hybrid: {len(candidates)} candidates"
            f" → {len(deduped)} interior after anchor pruning",
            file=sys.stderr, flush=True,
        )

    # ── Phase 2c: MIP verification for interior candidates ────────────────────

    n = len(deduped)
    for i, cand in enumerate(deduped):
        if verbose:
            print(
                f"hybrid: [{i + 1}/{n}]  min f1  s.t. f2 ≤ {cand.f2:.4f}"
                f"  (EA hint: f1={cand.f1:.1f})",
                file=sys.stderr, flush=True,
            )

        sol = _mip_solve(cand, constrain_f2=cand.f2)
        if sol.f1 is not None:
            prev_len = len(exact)
            exact.append(sol)
            exact = _filter_dominated(exact)
            dropped = prev_len + 1 - len(exact)
            if verbose:
                drop_str = f"  (dropped {dropped} dominated)" if dropped else ""
                print(
                    f"  → f1={sol.f1:.1f}  f2={sol.f2:.4f}  [{sol.status}]"
                    f"  front={len(exact)}{drop_str}",
                    file=sys.stderr, flush=True,
                )

    # ── Phase 3: deduplicate + final Pareto filter ───────────────────────────
    # Multiple EA ε-values can collapse to identical (f1, f2) points; remove
    # duplicates before filtering so the reported front has unique tradeoff points.

    seen_pts: list[tuple[float, float]] = []
    unique_exact: list[Solution] = []
    for sol in exact:
        if sol.f1 is None or sol.f2 is None:
            continue
        if not any(abs(sol.f1 - f1) < 1e-9 and abs(sol.f2 - f2) < 1e-9
                   for f1, f2 in seen_pts):
            seen_pts.append((sol.f1, sol.f2))
            unique_exact.append(sol)

    result = _filter_dominated(unique_exact)
    if verbose:
        print(
            f"hybrid: {len(exact)} exact → {len(unique_exact)} unique"
            f" → {len(result)} non-dominated",
            file=sys.stderr, flush=True,
        )
    return result


# ── Parallel exact front (box splitting on independent boxes) ────────────────

_WORKER_ENGINE = None


def _parallel_worker_init(inst: Instance, engine: str, per_solve_time_limit, lpfirst_options):
    global _WORKER_ENGINE
    if engine == "lpfirst":
        from . import lpfirst
        _WORKER_ENGINE, _ = lpfirst.make_lpfirst_lexmin_bounded(
            inst, per_solve_time_limit, options=lpfirst_options)
    else:
        _WORKER_ENGINE, _ = _make_engine_lexmin_bounded(
            inst, None, {"IntFeasTol": 1e-9} if engine == "gurobi" else {},
            per_solve_time_limit, 3, engine)


def _parallel_worker_solve(args):
    primary, secondary, bound, bound_value, start = args
    t0 = _time.perf_counter()
    sol = _WORKER_ENGINE(primary, secondary, bound, bound_value, start=start)
    return sol, _time.perf_counter() - t0


def _parallel_anchors(pool: "_SolvePool", st: dict):
    """Both lexicographic anchors, solved concurrently. A killed anchor comes
    back as an inconclusive Solution."""
    pool.submit(("f1", "f2", None, None, None), "A")
    pool.submit(("f2", "f1", None, None, None), "B")
    got: dict = {}
    while len(got) < 2:
        for tag, res in pool.wait():
            st["n_solves"] += 1
            if res is None:
                got[tag] = Solution(status="time_limit", objective=None, f1=None, f2=None,
                                    assignment={}, completion={})
            else:
                got[tag] = res[0]
                st["solve_time"] += res[1]
    return got["A"], got["B"]


class _SolvePool:
    """A spawn process pool for box solves with a *hard* time limit.

    Solver time limits are not always honoured: on the 964-job trace some
    HiGHS-backed lpfirst solves ran for hours past a 300 s cap, which left a
    parallel search waiting on one box forever. A solve that has run longer
    than ``hard_limit`` seconds gets every worker process killed, and the
    pool is rebuilt. Its tag comes back with result ``None`` (the caller marks
    the box unresolved), and the other in-flight solves are resubmitted."""

    def __init__(self, n_workers: int, initargs: tuple, hard_limit: float | None):
        self.n, self.initargs, self.hard_limit = n_workers, initargs, hard_limit
        self.jobs: dict = {}          # future -> (args, tag, start time)
        self.n_restarts = 0
        self.on_restart = None
        self._start()

    def _start(self) -> None:
        import concurrent.futures as cf
        import multiprocessing as mp
        self.pool = cf.ProcessPoolExecutor(self.n, mp_context=mp.get_context("spawn"),
                                           initializer=_parallel_worker_init,
                                           initargs=self.initargs)

    def __len__(self) -> int:
        return len(self.jobs)

    def submit(self, args: tuple, tag) -> None:
        self.jobs[self.pool.submit(_parallel_worker_solve, args)] = (args, tag, _time.perf_counter())

    def wait(self) -> list[tuple]:
        """Block until a solve finishes or one overruns. Returns
        ``[(tag, (sol, dt) | None), ...]`` with ``None`` for killed solves."""
        import concurrent.futures as cf
        while True:
            timeout = None
            if self.hard_limit is not None:
                oldest = min(t for _a, _g, t in self.jobs.values())
                timeout = max(min(1.0, self.hard_limit),
                              oldest + self.hard_limit - _time.perf_counter())
            done, _ = cf.wait(self.jobs, timeout=timeout, return_when=cf.FIRST_COMPLETED)
            if done:
                return [(self.jobs.pop(f)[1], f.result()) for f in done]
            now = _time.perf_counter()
            if any(now - t > self.hard_limit for _a, _g, t in self.jobs.values()):
                return self._restart(now)

    def _restart(self, now: float) -> list[tuple]:
        out, keep = [], []
        for f, (args, tag, t) in self.jobs.items():
            if f.done() and f.exception() is None:
                out.append((tag, f.result()))
            elif now - t > self.hard_limit:
                out.append((tag, None))
            else:
                keep.append((args, tag))
        for p in list(getattr(self.pool, "_processes", {}).values()):
            p.kill()
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.jobs = {}
        self.n_restarts += 1
        if self.on_restart is not None:
            self.on_restart(self.n_restarts)
        self._start()
        for args, tag in keep:
            self.submit(args, tag)
        return out

    def shutdown(self) -> None:
        if self.jobs:   # abandoned solves: do not wait for them
            for p in list(getattr(self.pool, "_processes", {}).values()):
                p.kill()
        self.pool.shutdown(wait=not self.jobs, cancel_futures=True)


def parallel_pareto_frontier(
    inst: Instance,
    n_workers: int | None = None,
    *,
    engine: str = "lpfirst",
    lpfirst_options=None,
    per_solve_time_limit: float | None = None,
    time_budget: float | None = None,
    hard_time_limit: float | None = None,
    min_split_width: float = 2.0,
    verbose: bool = False,
    stats: dict | None = None,
    on_point=None,
) -> list[Solution]:
    """The complete exact Pareto front, with box solves run in parallel.

    Same sub-solves and certificates as :func:`true_pareto_frontier`, but the
    open boxes are worked on concurrently by ``n_workers`` processes (each
    with its own engine, built once). The serial search is a single chain of
    boxes, so this function also *splits* boxes to make parallel work, as the
    balanced box method does (Boland, Charkhgard & Savelsbergh 2015):

    - **split** a box (L, R) at an f1 level ``mid`` strictly between L.f1 and
      R.f1: ``M = lexmin(f2, f1 | f1 <= mid)``. M is Pareto-optimal and the
      rightmost point with f1 <= mid, so the box becomes (L, M) and (M, R),
      both open (or only (L, R), now known to be clear up to ``mid``, if
      M is L);
    - **grow** a box from one end, exactly as the serial search: from L,
      ``lexmin(f1, f2 | f2 <= L.f2 - δ)``; from R,
      ``lexmin(f2, f1 | f1 <= R.f1 - 1)``. The result is the end's
      neighbour on the front, or proves the box empty.

    A box is split while fewer boxes are open than workers and it is at
    least ``min_split_width`` f1 levels wide (and not already known clear
    past its midpoint); otherwise it is grown, alternating ends. Each box
    has at most one solve in flight. Every solve is seeded with a box end
    that is feasible for it (R for a left growth, L otherwise).

    ``engine`` — ``"lpfirst"`` (default; ``lpfirst_options`` passes an
    :class:`fedhpc.lpfirst.LPFirstOptions`, e.g. ``backend="highs"``),
    ``"bpc"`` or ``"gurobi"``. ``time_budget`` stops submitting new solves
    once that many seconds have passed; open boxes and dropped
    (inconclusive) boxes make the front partial, with the same warning as
    the serial search. ``stats`` receives counters. ``on_point`` — called
    with each proven Pareto-optimal point as soon as it is found (to stream
    a long search's results). ``hard_time_limit`` — wall seconds after which
    a solve is killed (its worker processes restarted) and its box counted
    unresolved; default ``2 * per_solve_time_limit + 60`` (two phases), none
    without a per-solve limit.
    """
    n_workers = n_workers or max(1, (os.cpu_count() or 2) - 1)
    half1, half2 = objective_step(inst, "f1") / 2, objective_step(inst, "f2") / 2
    st = stats if stats is not None else {}
    st.update(n_solves=0, n_split=0, n_grow=0, n_empty=0, n_unresolved=0, solve_time=0.0)
    t0 = _time.perf_counter()
    if hard_time_limit is None and per_solve_time_limit is not None:
        hard_time_limit = 2 * per_solve_time_limit + 60

    pool = _SolvePool(n_workers, (inst, engine, per_solve_time_limit, lpfirst_options),
                      hard_time_limit)
    st["n_restarts"] = 0
    pool.on_restart = lambda n: st.__setitem__("n_restarts", n)
    boxes: list[list] = []
    truncated = False
    try:
        a, b = _parallel_anchors(pool, st)
        if a.status not in ("optimal", "infeasible") or a.f1 is None:
            print(f"parallel-pareto: WARNING — anchor A inconclusive ({a.status})",
                  file=sys.stderr, flush=True)
            return []
        if b.status not in ("optimal", "infeasible") or b.f1 is None:
            print(f"parallel-pareto: WARNING — anchor B inconclusive ({b.status})",
                  file=sys.stderr, flush=True)
            return [a]
        def found(sol) -> None:
            solutions.append(sol)
            if on_point is not None:
                on_point(sol)

        solutions: list[Solution] = []
        found(a)
        if abs(a.f1 - b.f1) < half1 and abs(a.f2 - b.f2) < half2:
            return solutions
        found(b)

        # box: [L, R, clear_to, side]; clear_to — no Pareto point has f1 in
        # (L.f1, clear_to]; side — the end the next growth starts from
        boxes = [[a, b, a.f1, "left"]]

        def width(bx) -> float:
            return bx[1].f1 - max(bx[0].f1, bx[2])

        def submit(bx) -> None:
            L, R, clear_to, side = bx
            mid = grid_floor(inst, "f1", (max(L.f1, clear_to) + R.f1) / 2)
            if (len(boxes) + len(pool) < n_workers and width(bx) >= min_split_width
                    and max(L.f1, clear_to) + half1 < mid < R.f1 - half1):
                op = ("split", mid)
                args = ("f2", "f1", "f1", mid, L)
                st["n_split"] += 1
            elif side == "left":
                op = ("left",)
                args = ("f1", "f2", "f2", L.f2 - half2, R)
                st["n_grow"] += 1
            else:
                op = ("right",)
                args = ("f2", "f1", "f1", R.f1 - half1, L)
                st["n_grow"] += 1
            pool.submit(args, (bx, op))

        while boxes or len(pool):
            if time_budget is not None and _time.perf_counter() - t0 >= time_budget:
                truncated = True
            # no f1 grid level (or f2 one) strictly inside: empty without a solve
            for bx in [bx for bx in boxes if width(bx) < 1 + half1
                       or bx[0].f2 - bx[1].f2 < 3 * half2]:
                boxes.remove(bx)
                st["n_empty"] += 1
            while boxes and not truncated and len(pool) < n_workers:
                boxes.sort(key=width)
                submit(boxes.pop())
            if not len(pool):
                break
            for ((L, R, clear_to, side), op), res in pool.wait():
                st["n_solves"] += 1
                sol, dt = res if res is not None else (None, hard_time_limit)
                st["solve_time"] += dt
                if sol is None or sol.status not in ("optimal", "infeasible"):
                    st["n_unresolved"] += 1
                    if verbose:
                        print(f"parallel-pareto: box [{L.f1:.2f},{R.f1:.2f}] {op[0]} "
                              f"INCONCLUSIVE ({sol.status if sol else 'killed'}), dropped",
                              file=sys.stderr, flush=True)
                    continue
                if op[0] == "split":
                    if sol.f1 is None or sol.f1 < L.f1 + half1:     # M is L: clear up to mid
                        boxes.append([L, R, op[1], side])
                    else:
                        found(sol)
                        boxes.append([L, sol, L.f1, "left"])
                        boxes.append([sol, R, sol.f1, "right"])
                    continue
                if op[0] == "left":
                    empty = sol.f1 is None or sol.f1 >= R.f1 - half1
                else:
                    empty = sol.f1 is None or sol.f2 >= L.f2 - half2
                if empty:
                    st["n_empty"] += 1
                    continue
                found(sol)
                if op[0] == "left":
                    boxes.append([sol, R, sol.f1, "right"])
                else:
                    boxes.append([L, sol, clear_to, "left"])
                if verbose:
                    print(f"parallel-pareto: {op[0]} → f1={sol.f1:.2f} f2={sol.f2:.6g} "
                          f"({dt:.1f}s; {len(solutions)} pts, {len(boxes)} open)",
                          file=sys.stderr, flush=True)
    finally:
        pool.shutdown()

    st["wall"] = _time.perf_counter() - t0
    st["open_boxes"] = len(boxes)
    if truncated or st["n_unresolved"]:
        print(f"parallel-pareto: WARNING — "
              f"{'time budget exhausted' if truncated else 'per-solve time limit hit'}; "
              f"returning a PARTIAL front ({len(solutions)} points, {len(boxes)} box(es) open, "
              f"{st['n_unresolved']} dropped).", file=sys.stderr, flush=True)
    return _filter_dominated(solutions)



class _Pt:
    """A known point of :func:`parallel_epsilon_frontier`: a schedule ``sol``
    and what is proven about it. ``ex1``: some Pareto point has f1 = sol.f1
    and f2 <= sol.f2 (from ``Q``). ``ex2``: some Pareto point has
    f2 = sol.f2 and f1 <= sol.f1 (from ``M``). Both: sol is that Pareto
    point."""
    __slots__ = ("sol", "ex1", "ex2", "reported")

    def __init__(self, sol: Solution, ex1: bool, ex2: bool):
        self.sol, self.ex1, self.ex2, self.reported = sol, ex1, ex2, False

    @property
    def f1(self) -> float:
        return self.sol.f1

    @property
    def f2(self) -> float:
        return self.sol.f2


def parallel_epsilon_frontier(
    inst: Instance,
    n_workers: int | None = None,
    *,
    direction: str = "adaptive",
    lpfirst_options=None,
    per_solve_time_limit: float | None = None,
    time_budget: float | None = None,
    hard_time_limit: float | None = None,
    verbose: bool = False,
    stats: dict | None = None,
    on_point=None,
) -> list[Solution]:
    """The complete exact Pareto front from *single-phase* ε-constraint
    solves, run in parallel (``engine="lpfirst"``):

    - ``Q(c) = min f1 s.t. f2 <= c``, proven, with any schedule attaining it.
      Some Pareto point then has f1 = Q(c) and f2 <= that schedule's cost;
    - ``M(b) = min f2 s.t. f1 <= b``, the mirror image.

    The box searches (:func:`true_pareto_frontier`,
    :func:`parallel_pareto_frontier`) pay one solve of *each* kind per point,
    and which kind is expensive depends on where the point sits. On the
    964-job trace, M costs 2-4 s against 6-8 s for Q at the high-cost end,
    while at the cost-minimal end Q costs ~20 s against 50-165 s for M. Here
    each point costs about one solve, of the kind that is cheap in its
    region. A point known from one kind of solve only is kept as a
    candidate (:class:`_Pt`). It is confirmed lazily by the next solve from
    it (``Q(f2 - δ) > f1``, or ``M(f1 - 1) > f2``) or replaced by that
    solve's better schedule.

    Box ``[P, N, lo, hi]`` between consecutive known points (P.f1 < N.f1):
    no schedule with f2 <= ``lo`` has f1 < N.f1, and no schedule with
    f1 <= ``hi`` has f2 < P.f2. So every unknown Pareto point in it has
    f1 in (max(P.f1, hi), N.f1) and f2 in (max(N.f2, lo), P.f2). The box
    also has to settle P if P is a Q-candidate and N if N is an M-candidate.
    While fewer boxes are open than workers, a box is split with a Q or M
    solve at the midpoint of its f2 or f1 range; otherwise it is grown from
    P (``Q(P.f2 - δ)``) or from N (``M(N.f1 - 1)``).

    ``direction`` — ``"adaptive"`` (default) picks Q or M per box by the
    median time of the three nearest (in f1) solves of each kind, trying
    each kind at least three times. It also decides, per candidate, whether
    to grow lazily from it or to settle it first with one solve of the
    other kind (``M(P.f1)`` for a Q-candidate, seeded with its schedule).
    A lazy growth is a replacement with the observed probability ``p``
    (the front's own schedule was not cost-minimal at its f1), so it costs
    ``t_grow / (1 - p)`` per point against ``t_settle + t_grow``. ``"f1"`` always uses Q, ``"f2"``
    always M, ``"random"`` picks at random (to test the bookkeeping). Other arguments as for :func:`parallel_pareto_frontier`
    (``hard_time_limit`` defaults to ``2 * per_solve_time_limit + 60``,
    since the two anchors are two-phase solves).
    """
    if direction not in ("adaptive", "f1", "f2", "random"):
        raise ValueError(f"direction must be 'adaptive', 'f1', 'f2' or 'random', "
                         f"not {direction!r}")
    import random as _random
    rng = _random.Random(0)
    n_workers = n_workers or max(1, (os.cpu_count() or 2) - 1)
    step1, step2 = objective_step(inst, "f1"), objective_step(inst, "f2")
    half1, half2 = step1 / 2, step2 / 2
    st = stats if stats is not None else {}
    st.update(n_solves=0, n_q=0, n_m=0, n_split=0, n_grow=0, n_settle=0, n_replaced=0,
              n_wasted=0, n_unresolved=0, solve_time=0.0)
    t0 = _time.perf_counter()
    if hard_time_limit is None and per_solve_time_limit is not None:
        hard_time_limit = 2 * per_solve_time_limit + 60

    pool = _SolvePool(n_workers, (inst, "lpfirst", per_solve_time_limit, lpfirst_options),
                      hard_time_limit)
    st["n_restarts"] = 0
    pool.on_restart = lambda n: st.__setitem__("n_restarts", n)
    solutions: list[Solution] = []
    # solve times, (f1 at the box centre, seconds), per label: "Q"/"M" for
    # splits and growths, "SM"/"SQ" for M(P.f1)/Q(N.f2) settling a candidate
    obs: dict[str, list] = {"Q": [], "M": [], "SM": [], "SQ": []}
    # lazy growths from a candidate of the same kind: [replaced, total]
    rep = {"Q": [0, 0], "M": [0, 0]}
    boxes: list[list] = []
    truncated = False

    def confirm(pt: _Pt) -> None:
        if pt.ex1 and pt.ex2 and not pt.reported:
            pt.reported = True
            solutions.append(pt.sol)
            if on_point is not None:
                on_point(pt.sol)

    def est(label: str, x: float):
        o = obs[label]
        if len(o) < 3:
            return None
        return sorted(r[1] for r in sorted(o, key=lambda r: abs(r[0] - x))[:3])[1]

    def kind_for(bx) -> str:
        if direction == "random":
            return rng.choice("QM")
        if direction != "adaptive":
            return "Q" if direction == "f1" else "M"
        x = (bx[0].f1 + bx[1].f1) / 2
        eq, em = est("Q", x), est("M", x)
        if eq is None or em is None:
            return "Q" if eq is None else "M"      # explore
        return "Q" if eq <= em else "M"

    def settle_first(bx, k: str) -> bool:
        """Settle the candidate end with the other kind of solve (M(P.f1)
        for a Q-candidate P), instead of growing lazily from it: a lazy
        growth is replaced with probability p, so it costs t/(1-p) per point
        against t_settle + t."""
        if direction == "random":
            return rng.random() < 0.5
        if direction != "adaptive":
            return False
        x = (bx[0].f1 + bx[1].f1) / 2
        e_grow, e_settle = est(k, x), est("SM" if k == "Q" else "SQ", x)
        if e_settle is None:
            return len(obs["SM" if k == "Q" else "SQ"]) < 3   # explore
        if e_grow is None:
            return False
        r, n = rep[k]
        p = (r + 1) / (n + 2)
        return e_settle < e_grow * p / (1 - p)

    def interior_empty(bx) -> bool:
        P, N, lo, hi = bx
        return (N.f1 - max(P.f1, hi) < 3 * half1) or (P.f2 - max(N.f2, lo) < 3 * half2)

    def settle_by_grid(bx) -> None:
        P, N, lo, hi = bx
        if P.ex1 and not P.ex2 and P.f2 - lo < 3 * half2:
            P.ex2 = True        # a cheaper schedule at f1 = P.f1 would have f2 <= lo
        if N.ex2 and not N.ex1 and N.f1 - hi < 3 * half1:
            N.ex1 = True        # a faster schedule at f2 = N.f2 would have f1 <= hi
        confirm(P)
        confirm(N)

    def submit(bx) -> None:
        P, N, lo, hi = bx
        k = kind_for(bx)
        need_p = P.ex1 and not P.ex2
        need_n = N.ex2 and not N.ex1
        # op: (kind, level, timing label, lazy growth from a same-kind candidate)
        grow_q = ("Q", grid_floor(inst, "f2", P.f2 - half2), "Q", need_p)
        grow_m = ("M", grid_floor(inst, "f1", N.f1 - half1), "M", need_n)
        settle_p = ("M", P.f1, "SM", False)
        settle_n = ("Q", N.f2, "SQ", False)
        if interior_empty(bx):
            st["n_settle"] += 1
            if need_p:
                op = grow_q if k == "Q" else settle_p
            else:
                op = grow_m if k == "M" else settle_n
        else:
            op = None
            if len(boxes) + len(pool) < n_workers:
                if k == "Q":
                    c = grid_floor(inst, "f2", (max(N.f2, lo) + P.f2) / 2)
                    if max(N.f2, lo) + half2 < c < P.f2 - 3 * half2:
                        op = ("Q", c, "Q", False)
                else:
                    b = grid_floor(inst, "f1", (max(P.f1, hi) + N.f1) / 2)
                    if max(P.f1, hi) + half1 < b < N.f1 - 3 * half1:
                        op = ("M", b, "M", False)
            if op is not None:
                st["n_split"] += 1
            elif k == "Q" and need_p and settle_first(bx, "Q"):
                st["n_settle"] += 1
                op = settle_p
            elif k == "M" and need_n and settle_first(bx, "M"):
                st["n_settle"] += 1
                op = settle_n
            else:
                st["n_grow"] += 1
                op = grow_q if k == "Q" else grow_m
        # both ends as seeds: one is feasible (an incumbent), the other is
        # next to the answer of a growth (the right columns)
        if op[0] == "Q":
            args = ("f1", None, "f2", op[1], [P.sol, N.sol])
        else:
            args = ("f2", None, "f1", op[1], [N.sol, P.sol])
        pool.submit(args, (bx, op))

    def on_q(bx, c, sol) -> None:
        P, N, lo, hi = bx
        a, s = sol.f1, sol.f2
        grow = c >= P.f2 - 3 * half2          # c == P.f2 - δ
        if a >= N.f1 - half1:                  # nothing with f2 <= c left of N
            if N.ex2 and not N.ex1:
                N.ex1 = True
            elif N.ex1 and not N.ex2 and s < N.f2 - half2:
                N.sol = sol
            if grow and P.ex1:
                P.ex2 = True
            boxes.append([P, N, max(lo, c), hi])
        elif a <= P.f1 + half1:                # same f1 as P, cheaper: P was not optimal
            if not (P.ex1 and not P.ex2) or s >= P.f2 - half2:
                raise RuntimeError(f"Q({c}) = ({a}, {s}) contradicts P = ({P.f1}, {P.f2})")
            st["n_replaced"] += 1
            st["n_wasted"] += 1
            P.sol = sol
            boxes.append([P, N, lo, hi])
        else:
            if grow and P.ex1:
                P.ex2 = True
            if s <= N.f2 + half2:              # N's Pareto point, found at its true f1
                if not (N.ex2 and not N.ex1):
                    raise RuntimeError(f"Q({c}) = ({a}, {s}) contradicts N = ({N.f1}, {N.f2})")
                st["n_replaced"] += 1
                N.sol, N.ex1 = sol, True
                boxes.append([P, N, max(lo, c), hi])
            else:
                X = _Pt(sol, True, False)
                boxes.append([P, X, c, hi])
                boxes.append([X, N, lo, hi])
        confirm(P)
        confirm(N)

    def on_m(bx, b, sol) -> None:
        P, N, lo, hi = bx
        a, s = sol.f1, sol.f2
        grow = b >= N.f1 - 3 * half1           # b == N.f1 - 1
        if s >= P.f2 - half2:                  # nothing with f1 <= b cheaper than P
            if P.ex1 and not P.ex2:
                P.ex2 = True
            elif P.ex2 and not P.ex1 and a < P.f1 - half1:
                P.sol = sol
            if grow and N.ex2:
                N.ex1 = True
            boxes.append([P, N, lo, max(hi, b)])
        elif s <= N.f2 + half2:                # same f2 as N, faster: N was not optimal
            if not (N.ex2 and not N.ex1) or a >= N.f1 - half1:
                raise RuntimeError(f"M({b}) = ({a}, {s}) contradicts N = ({N.f1}, {N.f2})")
            st["n_replaced"] += 1
            st["n_wasted"] += 1
            N.sol = sol
            boxes.append([P, N, lo, hi])
        else:
            if grow and N.ex2:
                N.ex1 = True
            if a <= P.f1 + half1:              # P's Pareto point, found at its true f2
                if not (P.ex1 and not P.ex2):
                    raise RuntimeError(f"M({b}) = ({a}, {s}) contradicts P = ({P.f1}, {P.f2})")
                st["n_replaced"] += 1
                P.sol, P.ex2 = sol, True
                boxes.append([P, N, lo, max(hi, b)])
            else:
                Y = _Pt(sol, False, True)
                boxes.append([P, Y, lo, hi])
                boxes.append([Y, N, lo, b])
        confirm(P)
        confirm(N)

    try:
        a, b = _parallel_anchors(pool, st)
        for name, x in (("A", a), ("B", b)):
            if x.status != "optimal" or x.f1 is None:
                print(f"parallel-epsilon: WARNING — anchor {name} inconclusive ({x.status})",
                      file=sys.stderr, flush=True)
                return [y for y in (a, b) if y.status == "optimal"]
        A, B = _Pt(a, True, True), _Pt(b, True, True)
        confirm(A)
        if abs(a.f1 - b.f1) < half1 and abs(a.f2 - b.f2) < half2:
            return solutions
        confirm(B)
        boxes = [[A, B, b.f2, a.f1]]

        while boxes or len(pool):
            if time_budget is not None and _time.perf_counter() - t0 >= time_budget:
                truncated = True
            for bx in list(boxes):
                settle_by_grid(bx)
                P, N = bx[0], bx[1]
                if interior_empty(bx) and not (P.ex1 and not P.ex2) and not (N.ex2 and not N.ex1):
                    boxes.remove(bx)
            while boxes and not truncated and len(pool) < n_workers:
                boxes.sort(key=lambda bx: bx[1].f1 - bx[0].f1)
                submit(boxes.pop())
            if not len(pool):
                break
            for (bx, (kind, level, label, lazy)), res in pool.wait():
                st["n_solves"] += 1
                st["n_q" if kind == "Q" else "n_m"] += 1
                sol, dt = res if res is not None else (None, hard_time_limit)
                st["solve_time"] += dt
                obs[label].append(((bx[0].f1 + bx[1].f1) / 2, dt))
                if sol is None or sol.status != "optimal" or sol.f1 is None:
                    st["n_unresolved"] += 1
                    print(f"parallel-epsilon: box [{bx[0].f1:.2f},{bx[1].f1:.2f}] "
                          f"{label}({level:.6g}) INCONCLUSIVE "
                          f"({sol.status if sol else 'killed'}), dropped",
                          file=sys.stderr, flush=True)
                    continue
                n_rep = st["n_wasted"]
                (on_q if kind == "Q" else on_m)(bx, level, sol)
                if lazy:
                    rep[kind][1] += 1
                    rep[kind][0] += st["n_wasted"] > n_rep
                if verbose:
                    print(f"parallel-epsilon: {kind}({level:.6g}) → f1={sol.f1:.2f} "
                          f"f2={sol.f2:.6g} ({dt:.1f}s; {len(solutions)} pts, "
                          f"{len(boxes)} open)", file=sys.stderr, flush=True)
    finally:
        pool.shutdown()

    st["wall"] = _time.perf_counter() - t0
    st["open_boxes"] = len(boxes)
    if truncated or st["n_unresolved"]:
        print(f"parallel-epsilon: WARNING — "
              f"{'time budget exhausted' if truncated else 'per-solve time limit hit'}; "
              f"returning a PARTIAL front ({len(solutions)} points, {len(boxes)} box(es) open, "
              f"{st['n_unresolved']} dropped).", file=sys.stderr, flush=True)
    return _filter_dominated(solutions)
