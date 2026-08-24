"""Pareto frontier exploration methods for FED-HPC."""
from __future__ import annotations

import json
import os
import sys
import time as _time
from collections import deque

from gurobipy import GRB

from .data import Instance
from .formulations import Formulation, SpaceTimeFormulation
from .model import Solution, _apply_params, _extract, set_mip_start


def _make_lexmin_bounded(
    inst: Instance, fmt: Formulation, params: dict, eps: float,
    per_solve_time_limit: float | None = None,
    n_seed_retries: int = 1,
):
    """Build a fresh-model-per-call lex-min solver, shared by true_pareto_frontier
    and map_pareto_frontier.

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
    for that). This matters independently of those: correctness here needs
    a very tight MIPGap/IntFeasTol (see true_pareto_frontier's docstring for
    why), and that combination occasionally hits a sub-problem where a good
    incumbent is found almost instantly but *proving* it optimal takes far
    longer — observed running over 6 hours and 100M+ B&B nodes without
    closing the gap. Investigating that specific case rather than just
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

    A solve that still hasn't reached ``GRB.OPTIMAL`` after all retries
    returns with ``status != "optimal"`` (``"feasible"`` if some attempt at
    least found a solution, else no solution) — callers must treat that as
    *inconclusive*, not as a proof of anything, since MIPGap was never
    reached.
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

    def lexmin_bounded(
        primary_expr_fn, secondary_expr_fn,
        bound_expr_fn=None, bound_value: float | None = None,
        *, start: Solution | None = None,
    ) -> Solution:
        """Lex-min(primary, secondary) on a fresh model, s.t. bound_expr <= bound_value.

        primary_expr_fn/secondary_expr_fn/bound_expr_fn each take the fresh
        model's own x and return the relevant expression — they can't be
        precomputed outside since every call gets a brand-new model.
        bound_expr_fn=None skips the bounding constraint entirely (used for
        unconstrained anchor solves).

        The returned Solution's ``.status`` is ``"optimal"`` only if *both*
        solve phases proved optimality within tolerance; check that before
        trusting the result for anything beyond "here's a decent point".
        """
        t0 = _time.perf_counter()
        mdl, vars_ = fmt.build(inst)
        counters["build_time"] += _time.perf_counter() - t0
        _apply_params(mdl, params)
        if per_solve_time_limit is not None:
            mdl.setParam("TimeLimit", per_solve_time_limit)

        x = vars_["x"]
        primary_expr = primary_expr_fn(x)
        secondary_expr = secondary_expr_fn(x)

        if bound_expr_fn is not None:
            mdl.addConstr(bound_expr_fn(x) <= bound_value, name="box_bound")
        if start is not None:
            set_mip_start(inst, x, start)
        mdl.setObjective(primary_expr, GRB.MINIMIZE)
        t1 = _time.perf_counter()
        _optimize_with_retries(mdl)
        if mdl.Status != GRB.OPTIMAL:
            # Includes SUBOPTIMAL/TIME_LIMIT: MIPGap wasn't proven even
            # after retries, so there's no primary_star to trust for the
            # phase-2 tie-break — return as-is (status "feasible"/
            # "infeasible"/etc, never "optimal") and let the caller treat
            # it as inconclusive.
            return _extract(inst, mdl, vars_, build_time=counters["build_time"],
                            solve_time=_time.perf_counter() - t1)
        primary_star = mdl.ObjVal
        mdl.addConstr(primary_expr <= primary_star + eps, name="fix_primary")
        mdl.setObjective(secondary_expr, GRB.MINIMIZE)
        mdl.setParam("Seed", 0)  # reset to the default seed for the new sub-problem
        _optimize_with_retries(mdl)
        return _extract(inst, mdl, vars_, build_time=counters["build_time"],
                        solve_time=_time.perf_counter() - t1)

    return lexmin_bounded, counters


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

    All three share the same feasible region (only the objective/bound differs),
    so a single model is built once and reused across all three solves — each
    solve also warm-starts from the previous one's basis/incumbent, since Gurobi
    keeps that state across ``optimize()`` calls on the same model instance.
    """
    fmt = formulation or SpaceTimeFormulation()
    mdl, vars_ = fmt.build(inst)
    _apply_params(mdl, params)

    x = vars_["x"]
    f1_expr = fmt.f1_expr(inst, x)
    f2_expr = fmt.f2_expr(inst, x)

    # Phase 1: f1_T = min f1
    mdl.setObjective(f1_expr, GRB.MINIMIZE)
    mdl.optimize()
    if mdl.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        raise RuntimeError("Could not solve min-turnaround problem.")
    f1_T = mdl.ObjVal

    # Phase 2: f2_T = min f2 s.t. f1 <= f1_T (identical to solve_f1's phase 2 —
    # this *is* the min-cost-at-f1^T problem, so no separate epsilon_turnaround
    # solve is needed).
    mdl.addConstr(f1_expr <= f1_T + 1e-6, name="fix_f1")
    mdl.setObjective(f2_expr, GRB.MINIMIZE)
    mdl.optimize()
    if mdl.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        raise RuntimeError("Could not solve min-cost at f1^T problem.")
    f2_T = mdl.ObjVal

    # Phase 3: f1_0 = min f1 s.t. f2 <= 0, reusing the same model — drop the
    # f1_T fix and impose the zero-cost bound instead.
    mdl.remove(mdl.getConstrByName("fix_f1"))
    mdl.addConstr(f2_expr <= 0.0, name="eps_cost")
    mdl.setObjective(f1_expr, GRB.MINIMIZE)
    mdl.optimize()
    if mdl.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        raise RuntimeError("Could not solve min-turnaround at zero-cost problem.")
    f1_0 = mdl.ObjVal

    return f1_T, f2_T, f1_0


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
    **params,
) -> list[Solution]:
    """Enumerate the complete true Pareto front via exact box (rectangle) splitting.

    Standard exact algorithm for biobjective integer programs (Hamacher,
    Pedersen & Ruzika 2007; related to the "balanced box" method of Boland,
    Charkhgard & Savelsbergh 2015) — used here instead of a per-integer-level
    ε-constraint sweep because that sweep costs O(f1_range) solves
    regardless of how many Pareto-optimal points actually exist, which is
    intractable when f1's range spans thousands of integer levels but the
    true front turns out to be sparse (as it does on this project's larger
    instances — see hybrid_frontier's findings). Box splitting instead costs
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
           f2 ≤ L.f2 − ε ("bounded by L's optimal value in the *other*
           objective"), tie-broken by minimising f2 second.
         - popped from the RIGHT → grow from R: minimise f2 s.t.
           f1 ≤ R.f1 − ε, tie-broken by minimising f1 second.
       If the result is (within ε of) the box's other endpoint, the box is
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

    ``per_solve_time_limit`` — optional per-Gurobi-solve cap in seconds
    (independent of ``max_solves``/``time_budget``, which only check
    *between* solves). Needed because this function's tight MIPGap/
    IntFeasTol can occasionally hit a sub-problem where a good incumbent
    is found almost instantly but proving it optimal takes far longer —
    observed in practice running 6+ hours and 100M+ B&B nodes without
    closing the gap on one sub-problem, eventually killed (likely OOM
    from the B&B tree). A box whose query hits this limit is treated as
    unresolved and dropped (not retried — it would just hang again) —
    contributing to incompleteness the same way a budget truncation does,
    reported via the same warning. Strongly recommended whenever running
    on an instance you haven't already characterised as tractable.

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
    #   1. Tight solver tolerances. Testing against real instances turned up
    #      two distinct sources of solve noise: Gurobi's default MIPGap
    #      (1e-4) lets a solve stop ~0.01% early, and its default IntFeasTol
    #      (1e-5) — individually negligible per variable, but summed across
    #      every job's assignment variable and multiplied by each one's cost
    #      coefficient — measured a ~0.0016 *objective-value* discrepancy on
    #      a real instance (large.json), two orders of magnitude past what
    #      the raw tolerance suggests.
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
    # ~29s on the biggest one) — but the whole point of this algorithm is
    # that solve *count* tracks the front's size, not the objective range,
    # and this project's largest instances turned out to have very sparse
    # fronts (2-3 points — see hybrid_frontier's findings), so the total
    # rebuild cost stays a handful of builds, not thousands.
    params = {"MIPGap": 1e-9, "IntFeasTol": 1e-9, **params}
    _EPS = 1e-4

    fmt = formulation or SpaceTimeFormulation()
    _lexmin_bounded, _counters = _make_lexmin_bounded(
        inst, fmt, params, _EPS, per_solve_time_limit, n_seed_retries
    )

    f1_of = lambda x: fmt.f1_expr(inst, x)
    f2_of = lambda x: fmt.f2_expr(inst, x)

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
        if abs(point_a.f1 - point_b.f1) < _EPS and abs(point_a.f2 - point_b.f2) < _EPS:
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
            sol = _lexmin_bounded(f1_of, f2_of, f2_of, left.f2 - _EPS)
        else:
            # f1's bound must be an exact integer decrement (right.f1 - 1),
            # not right.f1 - a small epsilon: f1 is always integer-spaced
            # (see the docstring), so both should be mathematically
            # equivalent as a cut — but empirically they are not. Testing
            # found a case (stess_test.json) where bounding at right.f1 -
            # 1e-4 (e.g. 538.9999) made Gurobi report a genuinely
            # suboptimal point as OPTIMAL/MIPGap=0.0, while the exact
            # integer bound (538) correctly found the true optimum on an
            # otherwise-identical, freshly-built model — most likely a
            # presolve/cut-generation quirk specific to bounding an
            # integer-valued expression with a non-integer RHS. Use the
            # clean integer cut here, same as the old sweep's "_STEP = 1 -
            # ε" convention, to avoid the pathological RHS entirely.
            sol = _lexmin_bounded(f2_of, f1_of, f1_of, right.f1 - 1.0 + _EPS)

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
            is_empty = sol.f1 is None or sol.f1 >= right.f1 - _EPS
        else:
            is_empty = sol.f1 is None or sol.f2 >= left.f2 - _EPS

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
    **params,
) -> list[Solution]:
    """Map the Pareto front by bisection — broad coverage fast, not exhaustive proof.

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
       (R.f1 - L.f1 <= 1) — f1 is always integer-spaced, so that's a rigorous
       proof nothing more fits between them, regardless of which direction
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
    _EPS = 1e-4
    fmt = formulation or SpaceTimeFormulation()
    solve_params = {"MIPGap": 1e-9, "IntFeasTol": 1e-9, **params}
    lexmin_bounded, counters = _make_lexmin_bounded(
        inst, fmt, solve_params, _EPS, per_solve_time_limit, n_seed_retries
    )

    f1_of = lambda x: fmt.f1_expr(inst, x)
    f2_of = lambda x: fmt.f2_expr(inst, x)

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
        if abs(point_a.f1 - point_b.f1) < _EPS and abs(point_a.f2 - point_b.f2) < _EPS:
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

        # f2-direction needs a numerically meaningful gap to bisect on; fall
        # back to f1 (always meaningful — integer-spaced) if the largest
        # available f2 gap is too fine to usefully split.
        this_dir = direction
        if this_dir == "f2" and max(l.f2 - r.f2 for l, r in boxes) <= 10 * _EPS:
            this_dir = "f1"

        if this_dir == "f1":
            idx = max(range(len(boxes)), key=lambda i: boxes[i][1].f1 - boxes[i][0].f1)
        else:
            idx = max(range(len(boxes)), key=lambda i: boxes[i][0].f2 - boxes[i][1].f2)
        left, right = boxes.pop(idx)

        if this_dir == "f1":
            mid = left.f1 + (right.f1 - left.f1) // 2  # integer midpoint
            sol = lexmin_bounded(f2_of, f1_of, f1_of, mid)
            label = f"f1<={mid:.0f}"
        else:
            mid = (left.f2 + right.f2) / 2.0
            sol = lexmin_bounded(f1_of, f2_of, f2_of, mid)
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
            found_better = sol.f1 is not None and sol.f2 < left.f2 - _EPS
        else:
            found_better = sol.f1 is not None and sol.f1 < right.f1 - _EPS

        if not found_better:
            # Proven empty up to mid; shrink whichever side the query was
            # anchored on and keep exploring the remainder, if any is left.
            if this_dir == "f1":
                new_box = (_boundary(mid, left.f2), right)
            else:
                new_box = (left, _boundary(right.f1, mid))
            if new_box[1].f1 - new_box[0].f1 > 1:
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
            if sol.f1 - left.f1 > 1:
                boxes.append((left, sol))
            if right.f1 - sol.f1 > 1:
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
    pts = [(s.f1, s.f2, s) for s in solutions if s.f1 is not None and s.f2 is not None]
    return [
        s for i, (f1i, f2i, s) in enumerate(pts)
        if not any(
            f1j <= f1i and f2j <= f2i and (f1j < f1i or f2j < f2i)
            for j, (f1j, f2j, _) in enumerate(pts) if j != i
        )
    ]


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
    mip_gap: float = 1e-4,
    verbose: bool = False,
    **mip_params,
) -> list[Solution]:
    """Hybrid Pareto frontier: EA exploration + Gurobi ε-constraint verification.

    Phase 1 — EA: run NSGA-II (seed) and MOEA/D (seed+1) to build a pool of
    non-dominated candidate solutions that jointly span the tradeoff space.

    Phase 2a — Anchors: take the EA's best-cost and best-turnaround solutions and
    solve the complementary objective exactly (min f1 s.t. f2 ≤ f2_ea_best and
    min f2 s.t. f1 ≤ f1_ea_best).  These two exact Pareto-frontier corners are
    used to prune the remaining candidate pool aggressively.

    Phase 2b — Prune: a candidate is interior only if it strictly improves on
    both anchors simultaneously (f1_c < anchor_cheap.f1 AND f2_c < anchor_fast.f2).
    All other candidates are already dominated and are discarded.

    Phase 2c — MIP interior: for each surviving interior candidate's cost threshold
    solve  min f1  s.t.  f2 ≤ f2_k  exactly with Gurobi, warm-started from the EA
    assignment.

    Phase 3 — Filter: final Pareto filter removes duplicates and dominances.

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
    mip_gap           : Gurobi MIPGap.
    verbose           : print phase progress and Gurobi log.
    **mip_params      : additional Gurobi parameters forwarded to every solve.

    Returns
    -------
    Exact (status="optimal"/"feasible") non-dominated solutions.
    """
    from gurobipy import GRB
    from .moea import moead_frontier, nsga2_frontier
    from .formulations import configure_env

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

    mip_kw: dict = {"MIPGap": mip_gap, **mip_params}
    if time_limit is not None:
        mip_kw["TimeLimit"] = time_limit
    mip_kw["OutputFlag"] = 1 if verbose else mip_kw.get("OutputFlag", 0)

    def _mip_solve(
        hint: Solution | None,
        *,
        constrain_f2: float | None = None,
        constrain_f1: float | None = None,
        minimize_cost: bool = False,
    ) -> Solution:
        t0 = _time.perf_counter()
        mdl, vars_ = fmt.build(inst)
        build_time = _time.perf_counter() - t0
        _apply_params(mdl, mip_kw)
        x = vars_["x"]
        if hint is not None:
            set_mip_start(inst, x, hint)
        if constrain_f2 is not None:
            mdl.addConstr(fmt.f2_expr(inst, x) <= constrain_f2, name="eps_cost")
        if constrain_f1 is not None:
            mdl.addConstr(fmt.f1_expr(inst, x) <= constrain_f1, name="eps_turnaround")
        obj = fmt.f2_expr(inst, x) if minimize_cost else fmt.f1_expr(inst, x)
        mdl.setObjective(obj, GRB.MINIMIZE)
        t1 = _time.perf_counter()
        mdl.optimize()
        return _extract(inst, mdl, vars_, build_time=build_time,
                        solve_time=_time.perf_counter() - t1)

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

    # min turnaround at EA's best cost level → exact right-end of front
    anchor_fast  = _mip_solve(ea_cheap, constrain_f2=ea_cheap.f2)
    # min cost at EA's best turnaround level → exact left-end of front
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
