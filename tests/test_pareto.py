"""Tests for Pareto-frontier approximation functions (pareto.py).

Coverage
────────
ReferencePoints    Correctness of f1_T, f2_T, f1_0 on the known-answer instance.
WeightedSum        Non-dominance, extreme-point coverage, monotonicity, known answer.
EpsilonConstraint  Non-dominance, constraint satisfaction, monotonicity, known answer.
Consistency        Both methods find the same extreme points; neither front
                   dominates the other.
"""
import pytest
from conftest import is_non_dominated_set, unique_front_points

from fedhpc.data import Instance
from fedhpc.pareto import (
    _reference_points,
    epsilon_constraint_frontier,
    weighted_sum_frontier,
)

_GUROBI = {"OutputFlag": 0}

KNOWN_FRONT = {(10.0, 100.0), (15.0, 50.0), (30.0, 0.0)}


# ---------------------------------------------------------------------------
# Reference points
# ---------------------------------------------------------------------------

class TestReferencePoints:
    def test_f1_T_exact_on_known_instance(self, known_pareto_inst):
        f1_T, _, _ = _reference_points(known_pareto_inst, **_GUROBI)
        assert f1_T == pytest.approx(10.0)

    def test_f2_T_exact_on_known_instance(self, known_pareto_inst):
        _, f2_T, _ = _reference_points(known_pareto_inst, **_GUROBI)
        assert f2_T == pytest.approx(100.0)

    def test_f1_0_exact_on_known_instance(self, known_pareto_inst):
        _, _, f1_0 = _reference_points(known_pareto_inst, **_GUROBI)
        assert f1_0 == pytest.approx(30.0)

    def test_f1_T_leq_f1_0_always(self, known_pareto_inst):
        """Minimum turnaround is ≤ minimum free-schedule turnaround."""
        f1_T, _, f1_0 = _reference_points(known_pareto_inst, **_GUROBI)
        assert f1_T <= f1_0 + 1e-6

    def test_f2_T_non_negative(self, known_pareto_inst):
        _, f2_T, _ = _reference_points(known_pareto_inst, **_GUROBI)
        assert f2_T >= -1e-9

    def test_f1_T_equals_solve_f1(self, known_pareto_inst):
        from fedhpc.model import solve_f1
        f1_T, _, _ = _reference_points(known_pareto_inst, **_GUROBI)
        sol_f1 = solve_f1(known_pareto_inst, **_GUROBI)
        assert f1_T == pytest.approx(sol_f1.f1, abs=1e-4)

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
# Weighted-sum frontier
# ---------------------------------------------------------------------------

class TestWeightedSumFrontier:
    def _front(self, inst, n=11):
        return weighted_sum_frontier(inst, n_points=n, **_GUROBI)

    def test_non_dominated_set(self, known_pareto_inst):
        front = self._front(known_pareto_inst)
        assert is_non_dominated_set(front), "weighted_sum_frontier returned dominated solutions"

    def test_all_solutions_have_status_optimal(self, known_pareto_inst):
        for s in self._front(known_pareto_inst):
            assert s.status == "optimal"

    def test_all_solutions_respect_budget(self, known_pareto_inst):
        inst = known_pareto_inst
        for s in self._front(inst):
            assert s.f2 <= inst.budget + 1e-6

    def test_all_solutions_all_jobs_assigned(self, known_pareto_inst):
        inst = known_pareto_inst
        job_ids = {j.id for j in inst.jobs}
        for s in self._front(inst):
            assert set(s.assignment.keys()) == job_ids

    def test_finds_minimum_f1_point(self, known_pareto_inst):
        front = self._front(known_pareto_inst)
        assert min(s.f1 for s in front) == pytest.approx(10.0)

    def test_finds_minimum_f2_point(self, known_pareto_inst):
        front = self._front(known_pareto_inst)
        assert min(s.f2 for s in front) == pytest.approx(0.0, abs=1e-9)

    def test_finds_all_three_known_pareto_points(self, known_pareto_inst):
        front = self._front(known_pareto_inst, n=11)
        found = unique_front_points(front)
        assert found == KNOWN_FRONT, f"Expected {KNOWN_FRONT}, got {found}"

    def test_monotone_tradeoff_sorted_by_f1(self, known_pareto_inst):
        """Sorting by f1 ascending, f2 must be non-increasing."""
        front = sorted(self._front(known_pareto_inst), key=lambda s: s.f1)
        f2_vals = [s.f2 for s in front]
        assert f2_vals == sorted(f2_vals, reverse=True), (
            f"f2 not monotone-decreasing when sorted by f1: {f2_vals}"
        )

    def test_more_points_not_fewer_than_fewer_points(self, known_pareto_inst):
        """Increasing n_points should produce at least as many unique Pareto points."""
        front_5  = unique_front_points(self._front(known_pareto_inst, n=5))
        front_21 = unique_front_points(self._front(known_pareto_inst, n=21))
        assert len(front_5) <= len(front_21)

    def test_f1_f2_non_negative(self, known_pareto_inst):
        for s in self._front(known_pareto_inst):
            assert s.f1 >= -1e-9
            assert s.f2 >= -1e-9

    def test_on_small_json_non_dominated(self):
        inst = Instance.from_file("data/small.json")
        front = weighted_sum_frontier(inst, n_points=11, **_GUROBI)
        assert is_non_dominated_set(front)

    def test_on_small_json_at_least_two_distinct_points(self):
        """A non-trivial instance must yield at least 2 distinct Pareto points."""
        inst = Instance.from_file("data/small.json")
        front = weighted_sum_frontier(inst, n_points=11, **_GUROBI)
        assert len(unique_front_points(front)) >= 2


# ---------------------------------------------------------------------------
# Epsilon-constraint frontier
# ---------------------------------------------------------------------------

class TestEpsilonConstraintFrontier:
    def _front(self, inst, n=10):
        return epsilon_constraint_frontier(inst, n_points=n, **_GUROBI)

    def test_non_dominated_set(self, known_pareto_inst):
        front = self._front(known_pareto_inst)
        assert is_non_dominated_set(front), "epsilon_constraint_frontier returned dominated solutions"

    def test_all_solutions_have_status_optimal(self, known_pareto_inst):
        for s in self._front(known_pareto_inst):
            assert s.status == "optimal"

    def test_all_solutions_respect_budget(self, known_pareto_inst):
        inst = known_pareto_inst
        for s in self._front(inst):
            assert s.f2 <= inst.budget + 1e-6

    def test_all_solutions_all_jobs_assigned(self, known_pareto_inst):
        inst = known_pareto_inst
        job_ids = {j.id for j in inst.jobs}
        for s in self._front(inst):
            assert set(s.assignment.keys()) == job_ids

    def test_finds_minimum_f1_point(self, known_pareto_inst):
        front = self._front(known_pareto_inst)
        assert min(s.f1 for s in front) == pytest.approx(10.0)

    def test_finds_minimum_f2_point(self, known_pareto_inst):
        front = self._front(known_pareto_inst)
        assert min(s.f2 for s in front) == pytest.approx(0.0, abs=1e-9)

    def test_finds_all_three_known_pareto_points(self, known_pareto_inst):
        front = self._front(known_pareto_inst, n=10)
        found = unique_front_points(front)
        assert found == KNOWN_FRONT, f"Expected {KNOWN_FRONT}, got {found}"

    def test_monotone_tradeoff_sorted_by_f1(self, known_pareto_inst):
        front = sorted(self._front(known_pareto_inst), key=lambda s: s.f1)
        f2_vals = [s.f2 for s in front]
        assert f2_vals == sorted(f2_vals, reverse=True), (
            f"f2 not monotone-decreasing when sorted by f1: {f2_vals}"
        )

    def test_f1_f2_non_negative(self, known_pareto_inst):
        for s in self._front(known_pareto_inst):
            assert s.f1 >= -1e-9
            assert s.f2 >= -1e-9

    def test_on_small_json_non_dominated(self):
        inst = Instance.from_file("data/small.json")
        front = epsilon_constraint_frontier(inst, n_points=10, **_GUROBI)
        assert is_non_dominated_set(front)

    def test_on_small_json_at_least_two_distinct_points(self):
        inst = Instance.from_file("data/small.json")
        front = epsilon_constraint_frontier(inst, n_points=10, **_GUROBI)
        assert len(unique_front_points(front)) >= 2


# ---------------------------------------------------------------------------
# Consistency between weighted-sum and epsilon-constraint
# ---------------------------------------------------------------------------

class TestFrontierConsistency:
    """Both methods should find the same extreme points and be mutually consistent."""

    def test_both_methods_find_min_f1(self, known_pareto_inst):
        ws  = weighted_sum_frontier(known_pareto_inst, n_points=11, **_GUROBI)
        eps = epsilon_constraint_frontier(known_pareto_inst, n_points=10, **_GUROBI)
        assert min(s.f1 for s in ws)  == pytest.approx(10.0)
        assert min(s.f1 for s in eps) == pytest.approx(10.0)

    def test_both_methods_find_min_f2(self, known_pareto_inst):
        ws  = weighted_sum_frontier(known_pareto_inst, n_points=11, **_GUROBI)
        eps = epsilon_constraint_frontier(known_pareto_inst, n_points=10, **_GUROBI)
        assert min(s.f2 for s in ws)  == pytest.approx(0.0, abs=1e-9)
        assert min(s.f2 for s in eps) == pytest.approx(0.0, abs=1e-9)

    def test_both_methods_find_all_known_pareto_points(self, known_pareto_inst):
        ws  = unique_front_points(weighted_sum_frontier(known_pareto_inst, n_points=11, **_GUROBI))
        eps = unique_front_points(epsilon_constraint_frontier(known_pareto_inst, n_points=10, **_GUROBI))
        assert ws  == KNOWN_FRONT
        assert eps == KNOWN_FRONT

    def test_weighted_sum_points_not_dominated_by_epsilon_constraint(self, known_pareto_inst):
        """No ws solution should be strictly dominated by any eps solution."""
        ws  = weighted_sum_frontier(known_pareto_inst, n_points=11, **_GUROBI)
        eps = epsilon_constraint_frontier(known_pareto_inst, n_points=10, **_GUROBI)
        for s_ws in ws:
            for s_eps in eps:
                dominated = (
                    s_eps.f1 <= s_ws.f1 and s_eps.f2 <= s_ws.f2
                    and (s_eps.f1 < s_ws.f1 or s_eps.f2 < s_ws.f2)
                )
                assert not dominated, (
                    f"WS point ({s_ws.f1},{s_ws.f2}) dominated by EPS ({s_eps.f1},{s_eps.f2})"
                )

    def test_epsilon_constraint_points_not_dominated_by_weighted_sum(self, known_pareto_inst):
        """No eps solution should be strictly dominated by any ws solution."""
        ws  = weighted_sum_frontier(known_pareto_inst, n_points=11, **_GUROBI)
        eps = epsilon_constraint_frontier(known_pareto_inst, n_points=10, **_GUROBI)
        for s_eps in eps:
            for s_ws in ws:
                dominated = (
                    s_ws.f1 <= s_eps.f1 and s_ws.f2 <= s_eps.f2
                    and (s_ws.f1 < s_eps.f1 or s_ws.f2 < s_eps.f2)
                )
                assert not dominated, (
                    f"EPS point ({s_eps.f1},{s_eps.f2}) dominated by WS ({s_ws.f1},{s_ws.f2})"
                )

    def test_on_small_json_both_non_dominated(self):
        inst = Instance.from_file("data/small.json")
        ws  = weighted_sum_frontier(inst, n_points=11, **_GUROBI)
        eps = epsilon_constraint_frontier(inst, n_points=10, **_GUROBI)
        assert is_non_dominated_set(ws)
        assert is_non_dominated_set(eps)
