"""Pareto frontier exploration methods for FED-HPC."""
from __future__ import annotations

import sys
import time as _time

from .data import Instance
from .formulations import Formulation, SpaceTimeFormulation
from .model import Solution, _apply_params, _extract, solve_epsilon_cost, solve_epsilon_turnaround, solve_f1, solve_f2, solve_true_pareto_step, solve_weighted_sum


def _reference_points(
    inst: Instance, formulation: Formulation | None = None, **params
) -> tuple[float, float, float]:
    """Compute the three reference points used by the weighted-sum single solve.

    Returns
    -------
    f1_T  — f1^T: lexicographic minimum turnaround (min f1 over all feasible x).
    f2_T  — f2^T: minimum cost at f1^T (min f2 s.t. f1 = f1^T).
    f1_0  — f1^0: minimum turnaround at zero cost (min f1 s.t. f2 = 0).
    """
    sol_f1T = solve_f1(inst, formulation=formulation, **params)
    if sol_f1T.f1 is None:
        raise RuntimeError("Could not solve min-turnaround problem.")
    f1_T = sol_f1T.f1

    sol_f2T = solve_epsilon_turnaround(inst, epsilon=f1_T, formulation=formulation, **params)
    if sol_f2T.f2 is None:
        raise RuntimeError("Could not solve min-cost at f1^T problem.")
    f2_T = sol_f2T.f2

    sol_f10 = solve_epsilon_cost(inst, epsilon=0.0, formulation=formulation, **params)
    if sol_f10.f1 is None:
        raise RuntimeError("Could not solve min-turnaround at zero-cost problem.")
    f1_0 = sol_f10.f1

    return f1_T, f2_T, f1_0


def true_pareto_frontier(
    inst: Instance,
    formulation: Formulation | None = None,
    verbose: bool = False,
    **params,
) -> list[Solution]:
    """Enumerate the complete true Pareto front via sequential ε-constraint sweep.

    Since f1 (total turnaround) is integer-valued, the front has a finite
    number of points and can be found exactly.  The algorithm:

      1. Solve lex-min(f2, f1) to get the rightmost extreme point
         (minimum cost, then best turnaround at that cost).
      2. Set f1_bound = f1_result - 1.
      3. Solve lex-min(f2, f1) s.t. f1 ≤ f1_bound.
         Each solve either returns a new Pareto point or proves infeasibility.
      4. Update f1_bound = f1_result - 1 and repeat until infeasible.

    This guarantees that every non-dominated integer f1 level is visited
    exactly once — no sampling, no heuristic, no missed points.
    The total number of MIP solves equals the size of the Pareto front.
    """
    sol = solve_f2(inst, formulation=formulation, **params)
    if sol.f1 is None:
        return []

    solutions: list[Solution] = [sol]
    f1_bound = int(round(sol.f1)) - 1
    n_iter = 0

    if verbose:
        print(
            f"pareto-true: anchor  f1={int(round(sol.f1))}  f2={sol.f2:.6g}",
            file=sys.stderr, flush=True,
        )

    while f1_bound >= 0:
        sol = solve_true_pareto_step(
            inst, f1_bound=f1_bound, formulation=formulation, **params,
        )
        n_iter += 1
        if sol.f1 is None:
            if verbose:
                print(
                    f"pareto-true: [{n_iter:3d}]  f1 ≤ {f1_bound}  → infeasible",
                    file=sys.stderr, flush=True,
                )
            break
        solutions.append(sol)
        f1_actual = int(round(sol.f1))
        if verbose:
            print(
                f"pareto-true: [{n_iter:3d}]  f1 ≤ {f1_bound}"
                f"  → f1={f1_actual}  f2={sol.f2:.6g}  [{sol.status}]",
                file=sys.stderr, flush=True,
            )
        f1_bound = f1_actual - 1

    if verbose:
        print(
            f"pareto-true: {len(solutions)} Pareto points in {n_iter + 1} solves",
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
    time_limit: float = 60.0,
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
    time_limit        : Gurobi time limit per ε-constraint solve (seconds).
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

    mip_kw: dict = {"TimeLimit": time_limit, "MIPGap": mip_gap, **mip_params}
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
            for j in inst.jobs:
                if j.id not in hint.assignment:
                    continue
                m_hint, t_hint = hint.assignment[j.id]
                for m_id in inst.F[j.id]:
                    for t in inst.T[j.id, m_id]:
                        x[j.id, m_id, t].Start = (
                            1.0 if (m_id == m_hint and t == t_hint) else 0.0
                        )
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
