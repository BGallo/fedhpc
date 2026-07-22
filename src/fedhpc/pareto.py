"""Pareto frontier exploration methods for FED-HPC."""
from __future__ import annotations

import sys
import time as _time

from gurobipy import GRB

from .data import Instance
from .formulations import Formulation, SpaceTimeFormulation
from .model import Solution, _apply_params, _extract, set_mip_start


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
    **params,
) -> list[Solution]:
    """Enumerate the complete true Pareto front via sequential ε-constraint sweep.

    f1 = Σ_j (t_j + p_occ_j − arrival_j) is a float because job arrivals are
    continuous.  However, since t and p_occ are integers, consecutive
    Pareto-optimal f1 values always differ by at least 1 (an integer change in
    the K = Σ(t+p_occ) component).  The sweep therefore steps by (1 − ε) in
    f1-space to correctly bracket each integer K level.

    Algorithm:
      1. Solve lex-min(f2, f1) → cheapest anchor (right extreme).
      2. Solve lex-min(f1, f2) → fastest anchor (left extreme).
      3. Sweep interior: f1_bound = f1_cheap − _STEP, solve lex-min(f2,f1)
         s.t. f1 ≤ f1_bound, update f1_bound = f1_result − _STEP.
      4. Stop when infeasible or f1_bound < f1_min.

    Both anchors are added upfront so the extremes are always in the output.

    Every step above shares the same feasible region (only the ε-bound on f1
    and the lex-min order differ), so the MIP model is built exactly once and
    reused for the whole sweep instead of once per step — the dominant cost on
    large instances. Reusing the model also means Gurobi automatically carries
    the basis/incumbent from one step's solve into the next step's re-optimize.

    ``hint`` — an optional prior/heuristic Solution (e.g. from moea.py) used to
    seed the MIP start of the very first solve only. It is not worth re-applying
    on every sweep step: setting ``.Start`` over every job/type/time combination
    costs O(|x|), comparable to a full model build on large instances, while
    Gurobi's own cross-solve warm start (above) is free.
    """
    _EPS  = 1e-6          # numerical tolerance for float comparisons
    _STEP = 1.0 - _EPS    # sweep decrement: crosses exactly one integer K level

    fmt = formulation or SpaceTimeFormulation()
    t0 = _time.perf_counter()
    mdl, vars_ = fmt.build(inst)
    build_time = _time.perf_counter() - t0
    _apply_params(mdl, params)
    mdl.update()  # index constraint names before the first getConstrByName lookup

    x = vars_["x"]
    f1_expr = fmt.f1_expr(inst, x)
    f2_expr = fmt.f2_expr(inst, x)

    def _lexmin(
        primary_expr, secondary_expr, *, f1_bound: float | None = None,
        start: Solution | None = None,
    ) -> Solution:
        """Lex-min(primary, secondary) on the shared model, s.t. f1 <= f1_bound.

        Clears constraints left over from a previous call on this model before
        adding the ones for this call, so the model is safe to reuse repeatedly.
        """
        for name in ("eps_f1", "fix_primary"):
            c = mdl.getConstrByName(name)
            if c is not None:
                mdl.remove(c)
        if f1_bound is not None:
            mdl.addConstr(f1_expr <= f1_bound, name="eps_f1")
        if start is not None:
            set_mip_start(inst, x, start)
        mdl.setObjective(primary_expr, GRB.MINIMIZE)
        t1 = _time.perf_counter()
        mdl.optimize()
        if mdl.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            return _extract(inst, mdl, vars_, build_time=build_time,
                            solve_time=_time.perf_counter() - t1)
        primary_star = mdl.ObjVal
        mdl.addConstr(primary_expr <= primary_star + _EPS, name="fix_primary")
        mdl.setObjective(secondary_expr, GRB.MINIMIZE)
        mdl.optimize()
        return _extract(inst, mdl, vars_, build_time=build_time,
                        solve_time=_time.perf_counter() - t1)

    # ── Anchor 1: cheapest-fast (right extreme) ───────────────────────────────
    sol_cheap = _lexmin(f2_expr, f1_expr, start=hint)
    if sol_cheap.f1 is None:
        return []

    # ── Anchor 2: fastest-cheap (left extreme) ───────────────────────────────
    sol_fast = _lexmin(f1_expr, f2_expr)

    f1_cheap: float = sol_cheap.f1
    solutions: list[Solution] = [sol_cheap]

    if sol_fast.f1 is not None:
        f1_min: float = sol_fast.f1
        solutions.append(sol_fast)
    else:
        f1_min = -float("inf")  # anchor failed; sweep until infeasible

    if verbose:
        print(
            f"pareto-true: anchor_cheap  f1={f1_cheap:.4f}  f2={sol_cheap.f2:.6g}",
            file=sys.stderr, flush=True,
        )
        if sol_fast.f1 is not None:
            print(
                f"pareto-true: anchor_fast   f1={f1_min:.4f}  f2={sol_fast.f2:.6g}",
                file=sys.stderr, flush=True,
            )
        else:
            print(
                "pareto-true: anchor_fast   — failed, sweeping without lower bound",
                file=sys.stderr, flush=True,
            )

    # ── Interior sweep between the two anchors ────────────────────────────────
    f1_bound: float = f1_cheap - _STEP
    n_iter = 0

    while f1_bound > f1_min - _EPS:
        sol = _lexmin(f2_expr, f1_expr, f1_bound=f1_bound)
        n_iter += 1
        if sol.f1 is None:
            if verbose:
                print(
                    f"pareto-true: [{n_iter:3d}]  f1 ≤ {f1_bound:.4f}  → infeasible",
                    file=sys.stderr, flush=True,
                )
            break
        f1_actual: float = sol.f1
        solutions.append(sol)
        if verbose:
            print(
                f"pareto-true: [{n_iter:3d}]  f1 ≤ {f1_bound:.4f}"
                f"  → f1={f1_actual:.4f}  f2={sol.f2:.6g}  [{sol.status}]",
                file=sys.stderr, flush=True,
            )
        if f1_actual <= f1_min + _EPS:
            break  # reached the fast anchor; done
        f1_bound = f1_actual - _STEP

    if verbose:
        print(
            f"pareto-true: {len(solutions)} solutions in {n_iter + 2} solves",
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
    pop_size: int = 200,
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
