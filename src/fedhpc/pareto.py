"""Pareto frontier approximation for FED-HPC."""
from __future__ import annotations
import numpy as np
from .data import Instance
from .model import Solution, solve_epsilon_cost, solve_f1, solve_f2, solve_weighted_sum


def _anchor_points(inst, **params):
    sol_f1 = solve_f1(inst, **params)
    sol_f2 = solve_f2(inst, **params)
    if sol_f1.f1 is None:
        raise RuntimeError("Could not solve mono-objective f1.")
    if sol_f2.f2 is None:
        raise RuntimeError("Could not solve mono-objective f2.")
    f1_min = sol_f1.f1
    f1_max = max(sol_f2.f1, f1_min + 1e-6)
    f2_max = max(sol_f1.f2, 1e-6)
    return f1_min, f1_max, 0.0, f2_max


def weighted_sum_frontier(inst, n_points=11, **params):
    f1_min, f1_max, _, f2_max = _anchor_points(inst, **params)
    solutions = []
    for alpha in np.linspace(0.0, 1.0, n_points):
        sol = solve_weighted_sum(inst, float(alpha),
                                  f1_min=f1_min, f1_max=f1_max, f2_max=f2_max, **params)
        if sol.f1 is not None:
            solutions.append(sol)
    return _filter_dominated(solutions)


def epsilon_constraint_frontier(inst, n_points=20, **params):
    _, _, _, f2_max = _anchor_points(inst, **params)
    epsilons = list(np.linspace(f2_max, 0.0, n_points, endpoint=False)) + [f2_max]
    solutions = []
    for eps in sorted(epsilons, reverse=True):
        sol = solve_epsilon_cost(inst, epsilon=float(eps), **params)
        if sol.f1 is not None:
            solutions.append(sol)
    return _filter_dominated(solutions)


def _filter_dominated(solutions):
    pts = [(s.f1, s.f2, s) for s in solutions if s.f1 is not None and s.f2 is not None]
    return [
        s for i, (f1i, f2i, s) in enumerate(pts)
        if not any(
            f1j <= f1i and f2j <= f2i and (f1j < f1i or f2j < f2i)
            for j, (f1j, f2j, _) in enumerate(pts) if j != i
        )
    ]
