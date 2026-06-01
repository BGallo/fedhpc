"""Pareto frontier approximation for FED-HPC."""
from __future__ import annotations

import sys
import time as _time

import numpy as np

from .data import Instance
from .formulations import Formulation, SpaceTimeFormulation
from .model import Solution, _apply_params, _extract, solve_epsilon_cost, solve_epsilon_turnaround, solve_f1, solve_weighted_sum


def _reference_points(
    inst: Instance, formulation: Formulation | None = None, **params
) -> tuple[float, float, float]:
    """Compute the three reference points from fedhpc.pdf Section 1.10.

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


def weighted_sum_frontier(
    inst: Instance,
    n_points: int = 11,
    formulation: Formulation | None = None,
    **params,
) -> list[Solution]:
    f1_T, f2_T, f1_0 = _reference_points(inst, formulation=formulation, **params)
    solutions = []
    for lam in np.linspace(0.0, 1.0, n_points):
        sol = solve_weighted_sum(
            inst, float(lam),
            f1_T=f1_T, f2_T=f2_T, f1_0=f1_0,
            formulation=formulation,
            **params,
        )
        if sol.f1 is not None:
            solutions.append(sol)
    return _filter_dominated(solutions)


def epsilon_constraint_frontier(
    inst: Instance,
    n_points: int = 20,
    formulation: Formulation | None = None,
    **params,
) -> list[Solution]:
    _, f2_T, _ = _reference_points(inst, formulation=formulation, **params)
    epsilons = list(np.linspace(f2_T, 0.0, n_points, endpoint=False)) + [f2_T]
    solutions = []
    for eps in sorted(epsilons, reverse=True):
        sol = solve_epsilon_cost(
            inst, epsilon=float(eps),
            formulation=formulation,
            **params,
        )
        if sol.f1 is not None:
            solutions.append(sol)
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

    Phase 2 — MIP: for each candidate cost threshold f2_k (sorted ascending),
    solve  min f1  s.t.  f2 ≤ f2_k  exactly with Gurobi, warm-started from
    the EA assignment.  The EA candidates act as guided ε-values; any candidate
    that was sub-optimal at its cost level is promoted to the true Pareto point.

    Phase 3 — Filter: final Pareto filter on exact solutions removes duplicates
    and dominances that arise when multiple EA ε-values collapse to the same
    exact optimal point.

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

    # ── Phase 2: MIP verification at each EA cost level ──────────────────────

    configure_env(verbose=verbose)
    fmt = formulation or SpaceTimeFormulation()

    mip_kw: dict = {"TimeLimit": time_limit, "MIPGap": mip_gap, **mip_params}
    # OutputFlag: 1 when verbose so individual solve logs are visible.
    mip_kw["OutputFlag"] = 1 if verbose else mip_kw.get("OutputFlag", 0)

    # Sort by f2 ascending and deduplicate by cost level: two EA candidates at the
    # same f2 threshold produce identical MIP instances, so keep only the one with
    # the better (lower) f1 hint as the warm-start source.
    sorted_cands = sorted(candidates, key=lambda s: (s.f2 or 0.0, s.f1 or 0.0))
    deduped_cands: list[Solution] = []
    seen_f2: list[float] = []
    for cand in sorted_cands:
        f2 = cand.f2 or 0.0
        if not any(abs(f2 - prev) < 1e-9 for prev in seen_f2):
            seen_f2.append(f2)
            deduped_cands.append(cand)
    sorted_cands = deduped_cands

    n = len(sorted_cands)
    exact: list[Solution] = []

    for i, cand in enumerate(sorted_cands):
        if verbose:
            print(
                f"hybrid: [{i + 1}/{n}]  min f1  s.t. f2 ≤ {cand.f2:.4f}"
                f"  (EA hint: f1={cand.f1:.1f})",
                file=sys.stderr, flush=True,
            )

        t0 = _time.perf_counter()
        mdl, vars_ = fmt.build(inst)
        build_time = _time.perf_counter() - t0

        _apply_params(mdl, mip_kw)

        # Warm-start Gurobi from the EA assignment so it begins with a feasible
        # incumbent that is at most as bad as the heuristic solution.
        x = vars_["x"]
        for j in inst.jobs:
            if j.id not in cand.assignment:
                continue
            m_hint, t_hint = cand.assignment[j.id]
            for m_id in inst.F[j.id]:
                for t in inst.T[j.id, m_id]:
                    x[j.id, m_id, t].Start = (
                        1.0 if (m_id == m_hint and t == t_hint) else 0.0
                    )

        mdl.addConstr(fmt.f2_expr(inst, x) <= cand.f2, name="eps_cost")
        mdl.setObjective(fmt.f1_expr(inst, x), GRB.MINIMIZE)

        t1 = _time.perf_counter()
        mdl.optimize()
        solve_time = _time.perf_counter() - t1

        sol = _extract(inst, mdl, vars_, build_time=build_time, solve_time=solve_time)
        if sol.f1 is not None:
            prev_len = len(exact)
            exact.append(sol)
            exact = _filter_dominated(exact)
            dropped = prev_len + 1 - len(exact)
            if verbose:
                drop_str = f"  (dropped {dropped} dominated)" if dropped else ""
                print(
                    f"  → f1={sol.f1:.1f}  f2={sol.f2:.4f}  [{sol.status}]"
                    f"  build={build_time:.2f}s  solve={solve_time:.2f}s"
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
