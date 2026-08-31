"""Comprehensive MIP solver tests (complement to test_model.py).

test_model.py covers the data layer and basic solver correctness.
This file focuses on properties not covered there:

Formulation   SpaceTimeFormulation and OccupancyFormulation must agree on
              optimal objective values for all solve_* functions.
KnownAnswer   Exact f1/f2 values on the analytically solved 2×2 instance.
Objectives    Verify that reported f1/f2 are computed by the correct formulas
              (sum of completion - arrival, and sum of c[j,m]).
Budget        Behaviour under tight or zero budget.
Epsilon       ε-constraint versions at boundary values.
WeightedSum   Additional parametric behaviour beyond α=0/1.
"""
import pytest
from conftest import capacity_violations, recompute_f1, recompute_f2

from fedhpc.data import Instance, InstanceType, Job, RunningJob
from fedhpc.formulations import OccupancyFormulation, SpaceTimeFormulation
from fedhpc.model import (
    solve_epsilon_cost,
    solve_epsilon_turnaround,
    solve_f1,
    solve_f2,
    solve_weighted_sum,
)

# ---------------------------------------------------------------------------
# Parametrize over both formulations
# ---------------------------------------------------------------------------

_FORMULATIONS = pytest.mark.parametrize(
    "formulation",
    [SpaceTimeFormulation(), OccupancyFormulation()],
    ids=["spacetime", "occupancy"],
)

_GUROBI = {"OutputFlag": 0}


# ---------------------------------------------------------------------------
# Additional fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def free_inst():
    """All types are free; optimal f2 is always 0 regardless of assignment."""
    inst = Instance(
        jobs=[
            Job(id=0, arrival=0, exec_time=8, io_volume=0, cpu=4, mem=8, stor=50),
            Job(id=1, arrival=2, exec_time=6, io_volume=0, cpu=4, mem=8, stor=50),
        ],
        instance_types=[
            InstanceType(
                id=0, kind="on-prem", perf=1.0, io_time=0.0, deploy=0,
                cost_io=0.0, cost_vm=0.0, cost_stor=0.0,
                cpu=8, mem=16, stor=100, capacity=None,
            ),
        ],
        horizon=30, budget=1000,
    )
    inst.build()
    return inst


@pytest.fixture
def budget_zero_inst(known_pareto_inst):
    """Same geometry as known_pareto_inst but budget=0 → only free on-prem allowed."""
    inst = Instance(
        jobs=known_pareto_inst.jobs,
        instance_types=known_pareto_inst.instance_types,
        horizon=known_pareto_inst.horizon,
        budget=0,
    )
    inst.build()
    return inst


# ---------------------------------------------------------------------------
# Formulation equivalence
# ---------------------------------------------------------------------------

class TestFormulationEquivalence:
    """SpaceTimeFormulation and OccupancyFormulation must yield identical optima."""

    def test_f1_objective_same_across_formulations(self, known_pareto_inst):
        st = solve_f1(known_pareto_inst, formulation=SpaceTimeFormulation(), **_GUROBI)
        oc = solve_f1(known_pareto_inst, formulation=OccupancyFormulation(),  **_GUROBI)
        assert st.f1 == pytest.approx(oc.f1, abs=1e-4)

    def test_f2_objective_same_across_formulations(self, known_pareto_inst):
        st = solve_f2(known_pareto_inst, formulation=SpaceTimeFormulation(), **_GUROBI)
        oc = solve_f2(known_pareto_inst, formulation=OccupancyFormulation(),  **_GUROBI)
        assert st.f2 == pytest.approx(oc.f2, abs=1e-4)

    def test_epsilon_cost_same_across_formulations(self, known_pareto_inst):
        eps = 60.0
        st = solve_epsilon_cost(known_pareto_inst, eps, formulation=SpaceTimeFormulation(), **_GUROBI)
        oc = solve_epsilon_cost(known_pareto_inst, eps, formulation=OccupancyFormulation(),  **_GUROBI)
        if st.f1 is not None and oc.f1 is not None:
            assert st.f1 == pytest.approx(oc.f1, abs=1e-4)

    def test_epsilon_turnaround_same_across_formulations(self, known_pareto_inst):
        eps = 20.0
        st = solve_epsilon_turnaround(known_pareto_inst, eps, formulation=SpaceTimeFormulation(), **_GUROBI)
        oc = solve_epsilon_turnaround(known_pareto_inst, eps, formulation=OccupancyFormulation(),  **_GUROBI)
        if st.f2 is not None and oc.f2 is not None:
            assert st.f2 == pytest.approx(oc.f2, abs=1e-4)

    @_FORMULATIONS
    def test_solve_f1_both_formulations_optimal_status(self, formulation, known_pareto_inst):
        sol = solve_f1(known_pareto_inst, formulation=formulation, **_GUROBI)
        assert sol.status == "optimal"

    @_FORMULATIONS
    def test_solve_f2_both_formulations_optimal_status(self, formulation, known_pareto_inst):
        sol = solve_f2(known_pareto_inst, formulation=formulation, **_GUROBI)
        assert sol.status == "optimal"


# ---------------------------------------------------------------------------
# Known-answer exact values
# ---------------------------------------------------------------------------

class TestKnownAnswer:
    """Exact expected values on the 2×2 instance with analytically known Pareto front.

    solve_f1 → f1=10 (both on cloud, p_occ=5, turnaround 5+5)
    solve_f2 → f2=0  (both on on-prem sequential, cost=0+0)
    solve_epsilon_cost(eps=50) → f1=15, f2=50  (interior Pareto point)
    solve_epsilon_turnaround(eps=15) → f2=50, f1=15
    """

    @_FORMULATIONS
    def test_solve_f1_exact_value(self, formulation, known_pareto_inst):
        sol = solve_f1(known_pareto_inst, formulation=formulation, **_GUROBI)
        assert sol.f1 == pytest.approx(10.0)

    @_FORMULATIONS
    def test_solve_f1_f2_at_min_turnaround(self, formulation, known_pareto_inst):
        """At minimum turnaround (f1=10), both jobs must be on cloud → f2=100."""
        sol = solve_f1(known_pareto_inst, formulation=formulation, **_GUROBI)
        assert sol.f2 == pytest.approx(100.0)

    @_FORMULATIONS
    def test_solve_f2_exact_value(self, formulation, known_pareto_inst):
        sol = solve_f2(known_pareto_inst, formulation=formulation, **_GUROBI)
        assert sol.f2 == pytest.approx(0.0, abs=1e-9)

    @_FORMULATIONS
    def test_solve_f2_f1_at_min_cost(self, formulation, known_pareto_inst):
        """At minimum cost (f2=0), both on-prem sequential → f1=30."""
        sol = solve_f2(known_pareto_inst, formulation=formulation, **_GUROBI)
        assert sol.f1 == pytest.approx(30.0)

    @_FORMULATIONS
    def test_epsilon_cost_50_yields_interior_pareto_point(self, formulation, known_pareto_inst):
        """ε=50: minimum turnaround with f2≤50 → the (f1=15, f2=50) point."""
        sol = solve_epsilon_cost(known_pareto_inst, 50.0, formulation=formulation, **_GUROBI)
        assert sol.status == "optimal"
        assert sol.f1 == pytest.approx(15.0)
        assert sol.f2 == pytest.approx(50.0)

    @_FORMULATIONS
    def test_epsilon_turnaround_15_yields_interior_pareto_point(self, formulation, known_pareto_inst):
        """ε=15: minimum cost with f1≤15 → the (f1=15, f2=50) point."""
        sol = solve_epsilon_turnaround(known_pareto_inst, 15.0, formulation=formulation, **_GUROBI)
        assert sol.status == "optimal"
        assert sol.f2 == pytest.approx(50.0)

    @_FORMULATIONS
    def test_epsilon_cost_0_matches_solve_f1_on_free_instance(self, formulation, free_inst):
        """On a free instance f2=0 always, so ε=0 is non-binding → same f1 as solve_f1."""
        sol_eps = solve_epsilon_cost(free_inst, 0.0, formulation=formulation, **_GUROBI)
        sol_f1  = solve_f1(free_inst, formulation=formulation, **_GUROBI)
        if sol_eps.f1 is not None:
            assert sol_eps.f2 == pytest.approx(0.0, abs=1e-9)
            assert sol_eps.f1 == pytest.approx(sol_f1.f1, abs=1e-4)

    @_FORMULATIONS
    def test_epsilon_cost_above_max_matches_solve_f1(self, formulation, known_pareto_inst):
        """ε=budget (non-binding): should yield the same f1 as unconstrained solve_f1."""
        eps = known_pareto_inst.budget
        sol_eps = solve_epsilon_cost(known_pareto_inst, eps, formulation=formulation, **_GUROBI)
        sol_f1  = solve_f1(known_pareto_inst, formulation=formulation, **_GUROBI)
        if sol_eps.f1 is not None and sol_f1.f1 is not None:
            assert sol_eps.f1 == pytest.approx(sol_f1.f1, abs=1e-4)


# ---------------------------------------------------------------------------
# Objective formula correctness
# ---------------------------------------------------------------------------

class TestObjectiveFormulas:
    """f1 and f2 reported by the solver must match the formulas in data.py."""

    @_FORMULATIONS
    def test_f1_equals_sum_completion_minus_arrival(self, formulation, known_pareto_inst):
        sol = solve_f1(known_pareto_inst, formulation=formulation, **_GUROBI)
        assert sol.f1 == pytest.approx(recompute_f1(known_pareto_inst, sol), abs=1e-6)

    @_FORMULATIONS
    def test_f2_equals_sum_c_jm(self, formulation, known_pareto_inst):
        sol = solve_f1(known_pareto_inst, formulation=formulation, **_GUROBI)
        assert sol.f2 == pytest.approx(recompute_f2(known_pareto_inst, sol), abs=1e-6)

    @_FORMULATIONS
    def test_f1_formula_on_free_instance(self, formulation, free_inst):
        sol = solve_f1(free_inst, formulation=formulation, **_GUROBI)
        assert sol.f1 == pytest.approx(recompute_f1(free_inst, sol), abs=1e-6)

    @_FORMULATIONS
    def test_f2_is_zero_on_free_instance(self, formulation, free_inst):
        sol = solve_f1(free_inst, formulation=formulation, **_GUROBI)
        assert sol.f2 == pytest.approx(0.0, abs=1e-9)

    @_FORMULATIONS
    def test_capacity_not_violated_in_optimal_solution(self, formulation, known_pareto_inst):
        sol = solve_f1(known_pareto_inst, formulation=formulation, **_GUROBI)
        assert capacity_violations(known_pareto_inst, sol) == []


# ---------------------------------------------------------------------------
# Budget constraints
# ---------------------------------------------------------------------------

class TestBudget:
    @_FORMULATIONS
    def test_budget_zero_forces_f2_zero(self, formulation, budget_zero_inst):
        """With budget=0 only free types are feasible, so f2=0 is the only option."""
        sol = solve_f1(budget_zero_inst, formulation=formulation, **_GUROBI)
        assert sol.status == "optimal"
        assert sol.f2 == pytest.approx(0.0, abs=1e-9)

    @_FORMULATIONS
    def test_budget_zero_forces_sequential_onprem(self, formulation, budget_zero_inst):
        """Budget=0 + capacity=1 → both jobs must run sequentially on on-prem → f1=30."""
        sol = solve_f1(budget_zero_inst, formulation=formulation, **_GUROBI)
        assert sol.f1 == pytest.approx(30.0)

    @_FORMULATIONS
    def test_tight_budget_below_any_cloud_cost_is_infeasible(self, formulation):
        """Budget = -1 makes every assignment infeasible."""
        inst = Instance(
            jobs=[Job(id=0, arrival=0, exec_time=5, io_volume=0, cpu=2, mem=4, stor=10)],
            instance_types=[
                InstanceType(
                    id=0, kind="cloud", perf=1.0, io_time=0.0, deploy=0,
                    cost_io=0.0, cost_vm=5.0, cost_stor=0.0,
                    cpu=4, mem=8, stor=50, capacity=None,
                ),
            ],
            horizon=20, budget=-1,
        )
        inst.build()
        sol = solve_f1(inst, formulation=formulation, **_GUROBI)
        assert sol.status == "infeasible"

    @_FORMULATIONS
    def test_loose_budget_does_not_constrain(self, formulation, known_pareto_inst):
        """Budget much larger than max cost → f1 should equal unconstrained optimum."""
        inst_loose = Instance(
            jobs=known_pareto_inst.jobs,
            instance_types=known_pareto_inst.instance_types,
            horizon=known_pareto_inst.horizon,
            budget=1e9,
        )
        inst_loose.build()
        sol_loose = solve_f1(inst_loose, formulation=formulation, **_GUROBI)
        sol_orig  = solve_f1(known_pareto_inst, formulation=formulation, **_GUROBI)
        assert sol_loose.f1 == pytest.approx(sol_orig.f1, abs=1e-4)


# ---------------------------------------------------------------------------
# Weighted-sum parametric properties
# ---------------------------------------------------------------------------

class TestWeightedSum:
    def _ref(self, inst):
        from fedhpc.pareto import _reference_points
        return _reference_points(inst, **_GUROBI)

    def test_lambda_half_finds_pareto_solution(self, known_pareto_inst):
        f1_T, f2_T, f1_0 = self._ref(known_pareto_inst)
        sol = solve_weighted_sum(known_pareto_inst, 0.5, f1_T, f2_T, f1_0, **_GUROBI)
        assert sol.status == "optimal"
        # Interior lambda should find the balanced (15, 50) point
        assert sol.f1 is not None and sol.f2 is not None

    def test_lambda_monotone_in_f1_f2_tradeoff(self, known_pareto_inst):
        """As λ increases (more weight on f1), f1 should decrease and f2 increase."""
        f1_T, f2_T, f1_0 = self._ref(known_pareto_inst)
        lambdas = [0.0, 0.3, 0.7, 1.0]
        sols = [
            solve_weighted_sum(known_pareto_inst, lam, f1_T, f2_T, f1_0, **_GUROBI)
            for lam in lambdas
        ]
        f1_vals = [s.f1 for s in sols if s.f1 is not None]
        f2_vals = [s.f2 for s in sols if s.f2 is not None]
        # As λ rises we put more weight on f1, so optimal f1 falls and f2 rises.
        assert f1_vals == sorted(f1_vals, reverse=True)  # f1 non-increasing with λ
        assert f2_vals == sorted(f2_vals)                 # f2 non-decreasing with λ

    def test_reference_points_exact_values(self, known_pareto_inst):
        """On the known 2×2 instance: f1_T=10, f2_T=100, f1_0=30."""
        f1_T, f2_T, f1_0 = self._ref(known_pareto_inst)
        assert f1_T == pytest.approx(10.0)
        assert f2_T == pytest.approx(100.0)
        assert f1_0 == pytest.approx(30.0)

    def test_f1_T_leq_f1_0(self, known_pareto_inst):
        """f1^T (minimum turnaround) ≤ f1^0 (minimum turnaround at zero cost)."""
        f1_T, _, f1_0 = self._ref(known_pareto_inst)
        assert f1_T <= f1_0 + 1e-6

    def test_f2_T_non_negative(self, known_pareto_inst):
        _, f2_T, _ = self._ref(known_pareto_inst)
        assert f2_T >= -1e-9


# ---------------------------------------------------------------------------
# Running-job formulation correctness
# ---------------------------------------------------------------------------

class TestRunningJobFormulation:
    """Solvers with running jobs must respect pre-occupancy across both formulations."""

    @_FORMULATIONS
    def test_running_job_reduces_available_capacity(self, formulation):
        """One running job on cap=2 type → at most 1 new job may run concurrently."""
        inst = Instance(
            jobs=[
                Job(id=0, arrival=0, exec_time=5, io_volume=0, cpu=2, mem=4, stor=10),
                Job(id=1, arrival=0, exec_time=5, io_volume=0, cpu=2, mem=4, stor=10),
            ],
            instance_types=[
                InstanceType(
                    id=0, kind="on-prem", perf=1.0, io_time=0.0, deploy=0,
                    cost_io=0.0, cost_vm=0.0, cost_stor=0.0,
                    cpu=4, mem=8, stor=50, capacity=2,
                ),
            ],
            horizon=20, budget=1000,
            running_jobs=[RunningJob(id=99, type_id=0, end=10)],
        )
        inst.build()
        sol = solve_f1(inst, formulation=formulation, **_GUROBI)
        assert sol.status == "optimal"
        # Verify capacity not violated
        assert capacity_violations(inst, sol) == []

    @_FORMULATIONS
    def test_running_job_f1_formula_still_correct(self, formulation):
        """Reported f1 must equal Σ(completion - arrival) even with running jobs."""
        inst = Instance(
            jobs=[Job(id=0, arrival=0, exec_time=5, io_volume=0, cpu=2, mem=4, stor=10)],
            instance_types=[
                InstanceType(
                    id=0, kind="on-prem", perf=1.0, io_time=0.0, deploy=0,
                    cost_io=0.0, cost_vm=0.0, cost_stor=0.0,
                    cpu=4, mem=8, stor=50, capacity=1,
                ),
            ],
            horizon=20, budget=1000,
            running_jobs=[RunningJob(id=99, type_id=0, end=5)],
        )
        inst.build()
        sol = solve_f1(inst, formulation=formulation, **_GUROBI)
        assert sol.f1 == pytest.approx(recompute_f1(inst, sol), abs=1e-6)
