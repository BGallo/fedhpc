"""Tests for the exact Pareto-frontier functions (pareto.py).

Coverage
────────
ReferencePoints    Correctness of f1_T, f2_T, f1_0 on the known-answer instance.
WeightedSum        A λ sweep of solve_weighted_sum_with_reference recovers the
                   supported points of the known front.
TrueFrontier       true_pareto_frontier: exactly the known front; non-dominance
                   and feasibility on a bundled instance.
MapFrontier        map_pareto_frontier: exact points, and the known front when
                   the budget is not binding.
Consistency        The two enumerations agree on a bundled instance.
"""
import pytest
from conftest import capacity_violations, is_non_dominated_set, recompute_f1, recompute_f2, unique_front_points

from fedhpc.data import Instance
from fedhpc.pareto import (
    _reference_points,
    map_pareto_frontier,
    solve_weighted_sum_with_reference,
    true_pareto_frontier,
)

_GUROBI = {"OutputFlag": 0}

KNOWN_FRONT = {(10.0, 100.0), (15.0, 50.0), (30.0, 0.0)}


@pytest.fixture(scope="module")
def bundled_inst() -> Instance:
    return Instance.from_file("data/tiny_small.json")


# ---------------------------------------------------------------------------
# Reference points
# ---------------------------------------------------------------------------

class TestReferencePoints:
    def test_exact_on_known_instance(self, known_pareto_inst):
        f1_T, f2_T, f1_0 = _reference_points(known_pareto_inst, **_GUROBI)
        assert (f1_T, f2_T, f1_0) == pytest.approx((10.0, 100.0, 30.0))

    def test_f1_T_equals_solve_f1(self, known_pareto_inst):
        from fedhpc.model import solve_f1
        f1_T, _, _ = _reference_points(known_pareto_inst, **_GUROBI)
        assert f1_T == pytest.approx(solve_f1(known_pareto_inst, **_GUROBI).f1, abs=1e-4)

    def test_f1_0_equals_epsilon_cost_at_zero(self, known_pareto_inst):
        from fedhpc.model import solve_epsilon_cost
        _, _, f1_0 = _reference_points(known_pareto_inst, **_GUROBI)
        sol = solve_epsilon_cost(known_pareto_inst, 0.0, **_GUROBI)
        assert f1_0 == pytest.approx(sol.f1, abs=1e-4)

    def test_f2_T_equals_epsilon_turnaround_at_f1_T(self, known_pareto_inst):
        from fedhpc.model import solve_epsilon_turnaround
        f1_T, f2_T, _ = _reference_points(known_pareto_inst, **_GUROBI)
        sol = solve_epsilon_turnaround(known_pareto_inst, f1_T, **_GUROBI)
        assert f2_T == pytest.approx(sol.f2, abs=1e-4)


# ---------------------------------------------------------------------------
# Weighted sum
# ---------------------------------------------------------------------------

class TestWeightedSum:
    def test_lambda_sweep_finds_supported_points(self, known_pareto_inst):
        """(15, 50) lies on the segment between the extremes, so it is
        supported: some λ reaches each of the three points."""
        sols = [solve_weighted_sum_with_reference(known_pareto_inst, lam, **_GUROBI)[0]
                for lam in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0)]
        assert all(s.status == "optimal" for s in sols)
        assert unique_front_points(sols) <= KNOWN_FRONT
        assert {(10.0, 100.0), (30.0, 0.0)} <= unique_front_points(sols)

    def test_invalid_lambda(self, known_pareto_inst):
        with pytest.raises(ValueError):
            solve_weighted_sum_with_reference(known_pareto_inst, 1.5, **_GUROBI)


# ---------------------------------------------------------------------------
# Exact enumerations
# ---------------------------------------------------------------------------

def _check_feasible(inst, front):
    job_ids = {j.id for j in inst.jobs}
    for s in front:
        assert set(s.assignment) == job_ids
        assert capacity_violations(inst, s) == []
        assert recompute_f1(inst, s) == pytest.approx(s.f1, abs=1e-6)
        assert recompute_f2(inst, s) == pytest.approx(s.f2, abs=1e-6)


class TestTrueFrontier:
    def test_finds_exactly_the_known_front(self, known_pareto_inst):
        front = true_pareto_frontier(known_pareto_inst, **_GUROBI)
        assert unique_front_points(front) == KNOWN_FRONT

    def test_bundled_instance(self, bundled_inst):
        front = true_pareto_frontier(bundled_inst, **_GUROBI)
        assert len(front) >= 1
        assert is_non_dominated_set(front)
        _check_feasible(bundled_inst, front)
        f1 = sorted(front, key=lambda s: s.f1)
        assert [s.f2 for s in f1] == sorted((s.f2 for s in f1), reverse=True)


class TestMapFrontier:
    def test_finds_exactly_the_known_front(self, known_pareto_inst):
        front = map_pareto_frontier(known_pareto_inst, **_GUROBI)
        assert unique_front_points(front) == KNOWN_FRONT

    def test_budgeted_points_are_on_the_true_front(self, bundled_inst):
        true = unique_front_points(true_pareto_frontier(bundled_inst, **_GUROBI))
        mapped = map_pareto_frontier(bundled_inst, max_solves=4, **_GUROBI)
        assert is_non_dominated_set(mapped)
        _check_feasible(bundled_inst, mapped)
        assert unique_front_points(mapped) <= true

    def test_unbudgeted_matches_true_front(self, bundled_inst):
        true = unique_front_points(true_pareto_frontier(bundled_inst, **_GUROBI))
        assert unique_front_points(map_pareto_frontier(bundled_inst, **_GUROBI)) == true
