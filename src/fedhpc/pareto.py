"""Pareto frontier approximation for FED-HPC."""
from __future__ import annotations

import numpy as np

from .data import Instance
from .formulations import Formulation
from .model import Solution, solve_epsilon_cost, solve_epsilon_turnaround, solve_f1, solve_weighted_sum


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
