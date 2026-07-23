"""Tests for the NSGA-II and MOEA/D heuristic solvers.

Structure
─────────
Unit        Python-layer encoding helpers (_job_slots, _type_cap, _init_occ).
Validity    Structural invariants every returned Solution must satisfy.
KnownAnswer Exact Pareto-front recovery on the analytically solved 2×2 instance.
Reproducibility  Same seed → identical output.
EdgeCases   Infeasible instances, running jobs, tight budget.
MipComparison  MOEA output not dominated by MIP optima; agrees on extreme points.
"""
import math
import pytest

from fedhpc.data import Instance, InstanceType, Job, RunningJob
from fedhpc.moea import _init_occ, _job_slots, _type_cap, moead_frontier, nsga2_frontier

# Pull shared helpers from conftest (importable because conftest is a plain module)
from conftest import (
    capacity_violations,
    is_non_dominated_set,
    recompute_f1,
    recompute_f2,
    unique_front_points,
)

# ---------------------------------------------------------------------------
# Parametrization helpers
# ---------------------------------------------------------------------------

_BOTH = pytest.mark.parametrize(
    "frontier_fn,kwargs",
    [
        (nsga2_frontier, {"pop_size": 100, "n_gen": 200, "seed": 42}),
        (moead_frontier,  {"n_weights": 100, "n_gen": 200, "seed": 42}),
    ],
    ids=["nsga2", "moead"],
)

_BOTH_LIGHT = pytest.mark.parametrize(
    "frontier_fn,kwargs",
    [
        (nsga2_frontier, {"pop_size": 50, "n_gen": 100, "seed": 42}),
        (moead_frontier,  {"n_weights": 50, "n_gen": 100, "seed": 42}),
    ],
    ids=["nsga2", "moead"],
)


# ---------------------------------------------------------------------------
# Additional fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def single_job_inst():
    """One job, one free type with unlimited capacity → exactly one feasible assignment."""
    inst = Instance(
        jobs=[Job(id=0, arrival=0, exec_time=6, io_volume=0, cpu=2, mem=4, stor=10)],
        instance_types=[
            InstanceType(
                id=0, kind="on-prem", perf=1.0, io_time=0.0, deploy=0,
                cost_io=0.0, cost_vm=0.0, cost_stor=0.0,
                cpu=4, mem=8, stor=50, capacity=None,
            ),
        ],
        horizon=20, budget=1000,
    )
    inst.build()
    return inst


@pytest.fixture
def blocked_onprem_inst():
    """On-prem slot filled for entire horizon by a running job; new job must use cloud."""
    inst = Instance(
        jobs=[Job(id=0, arrival=0, exec_time=5, io_volume=0, cpu=2, mem=4, stor=10)],
        instance_types=[
            InstanceType(
                id=0, kind="on-prem", perf=1.0, io_time=0.0, deploy=0,
                cost_io=0.0, cost_vm=0.0, cost_stor=0.0,
                cpu=4, mem=8, stor=50, capacity=1,
            ),
            InstanceType(
                id=1, kind="cloud", perf=1.0, io_time=0.0, deploy=0,
                cost_io=0.0, cost_vm=5.0, cost_stor=0.0,
                cpu=4, mem=8, stor=50, capacity=None,
            ),
        ],
        horizon=20, budget=1000,
        running_jobs=[RunningJob(id=99, type_id=0, end=20)],
    )
    inst.build()
    return inst


@pytest.fixture
def partial_block_inst():
    """On-prem blocked until t=5; new job can run on on-prem from t=5 onward."""
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
    return inst


# ---------------------------------------------------------------------------
# Encoding unit tests
# ---------------------------------------------------------------------------

class TestEncoding:
    """Pure-Python layer: verify that Instance data is correctly flattened for C++."""

    def test_job_slots_count_equals_sum_of_T_jm(self, known_pareto_inst):
        inst = known_pareto_inst
        slots = _job_slots(inst)
        for i, j in enumerate(inst.jobs):
            expected = sum(len(inst.T[j.id, m_id]) for m_id in inst.F[j.id])
            assert len(slots[i]) == expected

    def test_job_slots_type_id_in_feasible_set(self, known_pareto_inst):
        inst = known_pareto_inst
        for i, j in enumerate(inst.jobs):
            for m_id, t, p_occ, f1c, cost in _job_slots(inst)[i]:
                assert m_id in inst.F[j.id]

    def test_job_slots_start_time_in_T_jm(self, known_pareto_inst):
        inst = known_pareto_inst
        for i, j in enumerate(inst.jobs):
            for m_id, t, p_occ, f1c, cost in _job_slots(inst)[i]:
                assert t in inst.T[j.id, m_id]

    def test_job_slots_p_occ_matches_instance(self, known_pareto_inst):
        inst = known_pareto_inst
        for i, j in enumerate(inst.jobs):
            for m_id, t, p_occ, f1c, cost in _job_slots(inst)[i]:
                assert p_occ == inst.p_occ[j.id, m_id]

    def test_job_slots_f1_contrib_is_t_plus_pocc_minus_arrival(self, known_pareto_inst):
        inst = known_pareto_inst
        for i, j in enumerate(inst.jobs):
            for m_id, t, p_occ, f1c, cost in _job_slots(inst)[i]:
                assert f1c == pytest.approx(t + p_occ - j.arrival)

    def test_job_slots_cost_matches_c_dict(self, known_pareto_inst):
        inst = known_pareto_inst
        for i, j in enumerate(inst.jobs):
            for m_id, t, p_occ, f1c, cost in _job_slots(inst)[i]:
                assert cost == pytest.approx(inst.c[j.id, m_id])

    def test_job_slots_arrival_constraint_encoded(self, known_pareto_inst):
        """All start times must satisfy t >= ceil(arrival)."""
        inst = known_pareto_inst
        for i, j in enumerate(inst.jobs):
            a_min = math.ceil(j.arrival)
            for m_id, t, p_occ, f1c, cost in _job_slots(inst)[i]:
                assert t >= a_min

    def test_job_slots_horizon_constraint_encoded(self, known_pareto_inst):
        """All start times must satisfy t + p_occ <= horizon."""
        inst = known_pareto_inst
        for i, j in enumerate(inst.jobs):
            for m_id, t, p_occ, f1c, cost in _job_slots(inst)[i]:
                assert t + p_occ <= inst.horizon

    def test_type_cap_unlimited_becomes_minus_one(self, known_pareto_inst):
        inst = known_pareto_inst
        cap = _type_cap(inst)
        for m in inst.instance_types:
            if m.capacity is None:
                assert cap[m.id] == -1

    def test_type_cap_finite_preserved(self, known_pareto_inst):
        inst = known_pareto_inst
        cap = _type_cap(inst)
        for m in inst.instance_types:
            if m.capacity is not None:
                assert cap[m.id] == m.capacity

    def test_type_cap_list_covers_all_type_ids(self, known_pareto_inst):
        inst = known_pareto_inst
        cap = _type_cap(inst)
        for m in inst.instance_types:
            assert m.id < len(cap)

    def test_init_occ_empty_when_no_running_jobs(self, known_pareto_inst):
        assert _init_occ(known_pareto_inst) == []

    def test_init_occ_matches_inst_occupied(self, partial_block_inst):
        inst = partial_block_inst
        occ_list = _init_occ(inst)
        occ_dict = {(m, t): cnt for m, t, cnt in occ_list}
        assert occ_dict == {k: v for k, v in inst.occupied.items()}

    def test_init_occ_covers_all_blocked_slots(self, partial_block_inst):
        inst = partial_block_inst
        occ_list = _init_occ(inst)
        blocked_slots = {(0, t) for t in range(5)}  # running job ends at 5
        returned_slots = {(m, t) for m, t, cnt in occ_list}
        assert blocked_slots <= returned_slots


# ---------------------------------------------------------------------------
# Output validity — structural invariants
# ---------------------------------------------------------------------------

class TestOutputValidity:
    """Every Solution in the output must pass all structural correctness checks."""

    @_BOTH
    def test_returns_non_empty_list(self, frontier_fn, kwargs, known_pareto_inst):
        assert len(frontier_fn(known_pareto_inst, **kwargs)) > 0

    @_BOTH
    def test_all_jobs_assigned(self, frontier_fn, kwargs, known_pareto_inst):
        inst = known_pareto_inst
        expected_ids = {j.id for j in inst.jobs}
        for s in frontier_fn(inst, **kwargs):
            assert set(s.assignment.keys()) == expected_ids

    @_BOTH
    def test_assignment_completion_keys_match_jobs(self, frontier_fn, kwargs, known_pareto_inst):
        inst = known_pareto_inst
        expected_ids = {j.id for j in inst.jobs}
        for s in frontier_fn(inst, **kwargs):
            assert set(s.completion.keys()) == expected_ids

    @_BOTH
    def test_assigned_type_is_resource_feasible(self, frontier_fn, kwargs, known_pareto_inst):
        inst = known_pareto_inst
        for s in frontier_fn(inst, **kwargs):
            for jid, (mid, t) in s.assignment.items():
                assert mid in inst.F[jid], f"job {jid} → infeasible type {mid}"

    @_BOTH
    def test_start_time_in_T_jm(self, frontier_fn, kwargs, known_pareto_inst):
        inst = known_pareto_inst
        for s in frontier_fn(inst, **kwargs):
            for jid, (mid, t) in s.assignment.items():
                assert t in inst.T[jid, mid], f"job {jid}: t={t} not in T[{jid},{mid}]"

    @_BOTH
    def test_arrival_constraint_respected(self, frontier_fn, kwargs, known_pareto_inst):
        inst = known_pareto_inst
        for s in frontier_fn(inst, **kwargs):
            for j in inst.jobs:
                _, t = s.assignment[j.id]
                assert t >= math.ceil(j.arrival), (
                    f"job {j.id} starts at {t} before arrival {j.arrival}"
                )

    @_BOTH
    def test_completion_within_horizon(self, frontier_fn, kwargs, known_pareto_inst):
        inst = known_pareto_inst
        for s in frontier_fn(inst, **kwargs):
            for jid, c in s.completion.items():
                assert c <= inst.horizon, f"job {jid} completes at {c} > H={inst.horizon}"

    @_BOTH
    def test_completion_equals_start_plus_p_occ(self, frontier_fn, kwargs, known_pareto_inst):
        inst = known_pareto_inst
        for s in frontier_fn(inst, **kwargs):
            for jid, (mid, t) in s.assignment.items():
                assert s.completion[jid] == t + inst.p_occ[jid, mid]

    @_BOTH
    def test_f1_matches_recomputed_from_assignment(self, frontier_fn, kwargs, known_pareto_inst):
        inst = known_pareto_inst
        for s in frontier_fn(inst, **kwargs):
            assert s.f1 == pytest.approx(recompute_f1(inst, s), abs=1e-6)

    @_BOTH
    def test_f2_matches_recomputed_from_assignment(self, frontier_fn, kwargs, known_pareto_inst):
        inst = known_pareto_inst
        for s in frontier_fn(inst, **kwargs):
            assert s.f2 == pytest.approx(recompute_f2(inst, s), abs=1e-6)

    @_BOTH
    def test_f1_non_negative(self, frontier_fn, kwargs, known_pareto_inst):
        for s in frontier_fn(known_pareto_inst, **kwargs):
            assert s.f1 >= -1e-9

    @_BOTH
    def test_f2_non_negative(self, frontier_fn, kwargs, known_pareto_inst):
        for s in frontier_fn(known_pareto_inst, **kwargs):
            assert s.f2 >= -1e-9

    @_BOTH
    def test_budget_respected(self, frontier_fn, kwargs, known_pareto_inst):
        inst = known_pareto_inst
        for s in frontier_fn(inst, **kwargs):
            assert s.f2 <= inst.budget + 1e-6, f"f2={s.f2} exceeds budget={inst.budget}"

    @_BOTH
    def test_no_capacity_violations(self, frontier_fn, kwargs, known_pareto_inst):
        inst = known_pareto_inst
        for s in frontier_fn(inst, **kwargs):
            viols = capacity_violations(inst, s)
            assert viols == [], f"Capacity violated at {viols}"

    @_BOTH
    def test_output_is_non_dominated_set(self, frontier_fn, kwargs, known_pareto_inst):
        sols = frontier_fn(known_pareto_inst, **kwargs)
        assert is_non_dominated_set(sols), "Output contains a dominated solution"

    @_BOTH
    def test_status_field_is_heuristic(self, frontier_fn, kwargs, known_pareto_inst):
        for s in frontier_fn(known_pareto_inst, **kwargs):
            assert s.status == "heuristic"

    @_BOTH
    def test_objective_field_is_none(self, frontier_fn, kwargs, known_pareto_inst):
        """The scalarised MIP objective has no meaning for MOEA solutions."""
        for s in frontier_fn(known_pareto_inst, **kwargs):
            assert s.objective is None


# ---------------------------------------------------------------------------
# Known-answer tests
# ---------------------------------------------------------------------------

class TestKnownAnswer:
    """Exact verification against the analytically solved 2×2 instance.

    Known Pareto front:
        (f1=10, f2=100)  — both on cloud
        (f1=15, f2= 50)  — one cloud + one on-prem
        (f1=30, f2=  0)  — both on-prem (sequential)
    """

    @_BOTH
    def test_finds_all_three_pareto_points(self, frontier_fn, kwargs,
                                            known_pareto_inst, known_pareto_front):
        sols = frontier_fn(known_pareto_inst, **kwargs)
        found = unique_front_points(sols)
        assert found == known_pareto_front, f"Expected {known_pareto_front}, got {found}"

    @_BOTH
    def test_minimum_turnaround_value_is_10(self, frontier_fn, kwargs, known_pareto_inst):
        sols = frontier_fn(known_pareto_inst, **kwargs)
        assert min(s.f1 for s in sols) == pytest.approx(10.0)

    @_BOTH
    def test_minimum_cost_value_is_0(self, frontier_fn, kwargs, known_pareto_inst):
        sols = frontier_fn(known_pareto_inst, **kwargs)
        assert min(s.f2 for s in sols) == pytest.approx(0.0, abs=1e-9)

    @_BOTH
    def test_interior_pareto_point_f1_15_f2_50(self, frontier_fn, kwargs, known_pareto_inst):
        sols = frontier_fn(known_pareto_inst, **kwargs)
        assert any(round(s.f1) == 15 and round(s.f2) == 50 for s in sols)

    @_BOTH
    def test_min_turnaround_assignment_uses_only_cloud(self, frontier_fn, kwargs,
                                                        known_pareto_inst):
        """f1=10 requires both jobs on type 1 (cloud, perf=2, p_occ=5)."""
        inst = known_pareto_inst
        sols = frontier_fn(inst, **kwargs)
        min_sol = min(sols, key=lambda s: s.f1)
        assert round(min_sol.f1) == 10
        for jid, (mid, _) in min_sol.assignment.items():
            assert mid == 1, f"Expected cloud (type 1), got type {mid} for job {jid}"

    @_BOTH
    def test_zero_cost_assignment_uses_only_onprem(self, frontier_fn, kwargs,
                                                    known_pareto_inst):
        """f2=0 requires all jobs on type 0 (on-prem, free)."""
        inst = known_pareto_inst
        sols = frontier_fn(inst, **kwargs)
        zero_cost = [s for s in sols if s.f2 == pytest.approx(0.0, abs=1e-9)]
        assert len(zero_cost) > 0, "No zero-cost solution found"
        for s in zero_cost:
            for jid, (mid, _) in s.assignment.items():
                assert mid == 0, f"Expected on-prem (type 0), got type {mid}"

    @_BOTH
    def test_no_solution_dominated_by_known_optimal(self, frontier_fn, kwargs,
                                                      known_pareto_inst, known_pareto_front):
        """No MOEA solution should be strictly dominated by any known Pareto point."""
        sols = frontier_fn(known_pareto_inst, **kwargs)
        for s in sols:
            for (kf1, kf2) in known_pareto_front:
                dominated = kf1 <= s.f1 and kf2 <= s.f2 and (kf1 < s.f1 or kf2 < s.f2)
                assert not dominated, (
                    f"MOEA solution ({s.f1},{s.f2}) dominated by known point ({kf1},{kf2})"
                )


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    @_BOTH_LIGHT
    def test_same_seed_gives_identical_front(self, frontier_fn, kwargs, known_pareto_inst):
        r1 = frontier_fn(known_pareto_inst, **kwargs)
        r2 = frontier_fn(known_pareto_inst, **kwargs)
        assert unique_front_points(r1) == unique_front_points(r2)

    def test_nsga2_different_seeds_both_valid(self, known_pareto_inst):
        r0 = nsga2_frontier(known_pareto_inst, pop_size=20, n_gen=10, seed=7)
        r1 = nsga2_frontier(known_pareto_inst, pop_size=20, n_gen=10, seed=13)
        assert is_non_dominated_set(r0)
        assert is_non_dominated_set(r1)

    def test_moead_different_seeds_both_valid(self, known_pareto_inst):
        r0 = moead_frontier(known_pareto_inst, n_weights=20, n_gen=10, seed=7)
        r1 = moead_frontier(known_pareto_inst, n_weights=20, n_gen=10, seed=13)
        assert is_non_dominated_set(r0)
        assert is_non_dominated_set(r1)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    @_BOTH_LIGHT
    def test_single_job_single_type_one_pareto_point(self, frontier_fn, kwargs,
                                                      single_job_inst):
        sols = frontier_fn(single_job_inst, **kwargs)
        assert len(sols) == 1
        s = sols[0]
        assert set(s.assignment.keys()) == {0}

    def test_nsga2_globally_infeasible_returns_empty(self):
        """3 jobs, capacity=2, all forced to t=0 → infeasible → []."""
        inst = Instance(
            jobs=[
                Job(id=i, arrival=0, exec_time=10, io_volume=0, cpu=1, mem=1, stor=1)
                for i in range(3)
            ],
            instance_types=[
                InstanceType(
                    id=0, kind="on-prem", perf=1.0, io_time=0.0, deploy=0,
                    cost_io=0.0, cost_vm=0.0, cost_stor=0.0,
                    cpu=4, mem=8, stor=50, capacity=2,
                ),
            ],
            horizon=10, budget=1e6,
        )
        inst.build()
        sols = nsga2_frontier(inst, pop_size=50, n_gen=100, seed=0)
        assert sols == [], f"Expected empty list for infeasible instance, got {len(sols)} solutions"

    def test_moead_globally_infeasible_returns_empty(self):
        inst = Instance(
            jobs=[
                Job(id=i, arrival=0, exec_time=10, io_volume=0, cpu=1, mem=1, stor=1)
                for i in range(3)
            ],
            instance_types=[
                InstanceType(
                    id=0, kind="on-prem", perf=1.0, io_time=0.0, deploy=0,
                    cost_io=0.0, cost_vm=0.0, cost_stor=0.0,
                    cpu=4, mem=8, stor=50, capacity=2,
                ),
            ],
            horizon=10, budget=1e6,
        )
        inst.build()
        sols = moead_frontier(inst, n_weights=50, n_gen=100, seed=0)
        assert sols == []

    @_BOTH
    def test_running_job_fills_horizon_forces_cloud(self, frontier_fn, kwargs,
                                                     blocked_onprem_inst):
        """On-prem occupied for entire horizon; every solution must use cloud."""
        inst = blocked_onprem_inst
        sols = frontier_fn(inst, **kwargs)
        assert len(sols) > 0
        for s in sols:
            mid, _ = s.assignment[0]
            assert mid == 1, f"Expected cloud (type 1); got type {mid}"

    @_BOTH
    def test_running_job_partial_block_no_capacity_violation(self, frontier_fn, kwargs,
                                                              partial_block_inst):
        """Running job frees at t=5; new job may start on-prem at t≥5."""
        inst = partial_block_inst
        sols = frontier_fn(inst, **kwargs)
        assert len(sols) > 0
        for s in sols:
            viols = capacity_violations(inst, s)
            assert viols == [], f"Capacity violated: {viols}"

    @_BOTH
    def test_tight_budget_eliminates_costly_solutions(self, frontier_fn, kwargs,
                                                       known_pareto_inst):
        """Budget=40 < cheapest cloud option (50) → only f2=0 solutions allowed."""
        inst = Instance(
            jobs=known_pareto_inst.jobs,
            instance_types=known_pareto_inst.instance_types,
            horizon=known_pareto_inst.horizon,
            budget=40,
        )
        inst.build()
        sols = frontier_fn(inst, **kwargs)
        assert len(sols) > 0
        for s in sols:
            assert s.f2 == pytest.approx(0.0, abs=1e-6), (
                f"Budget=40 but got f2={s.f2}"
            )


# ---------------------------------------------------------------------------
# End-to-end comparison with MIP
# ---------------------------------------------------------------------------

class TestMipComparison:
    """MOEA solutions must be feasible and agree with exact MIP solutions."""

    @_BOTH
    def test_extreme_turnaround_matches_mip(self, frontier_fn, kwargs, known_pareto_inst):
        """MOEA best f1 must equal MIP's optimal f1 on the known-answer instance."""
        from fedhpc.model import solve_f1
        mip_f1 = solve_f1(known_pareto_inst, OutputFlag=0).f1
        sols = frontier_fn(known_pareto_inst, **kwargs)
        assert min(s.f1 for s in sols) == pytest.approx(mip_f1, abs=1e-4)

    @_BOTH
    def test_extreme_cost_matches_mip(self, frontier_fn, kwargs, known_pareto_inst):
        """MOEA best f2 must equal MIP's optimal f2 on the known-answer instance."""
        from fedhpc.model import solve_f2
        mip_f2 = solve_f2(known_pareto_inst, OutputFlag=0).f2
        sols = frontier_fn(known_pareto_inst, **kwargs)
        assert min(s.f2 for s in sols) == pytest.approx(mip_f2, abs=1e-4)

    def test_small_json_nsga2_all_constraints_satisfied(self):
        """All NSGA-II solutions on small.json must be feasible."""
        inst = Instance.from_file("data/small.json")
        sols = nsga2_frontier(inst, pop_size=100, n_gen=200, seed=42)
        assert len(sols) > 0
        for s in sols:
            assert s.f2 <= inst.budget + 1e-6
            assert capacity_violations(inst, s) == []
            assert recompute_f1(inst, s) == pytest.approx(s.f1, abs=1e-6)
            assert recompute_f2(inst, s) == pytest.approx(s.f2, abs=1e-6)

    def test_small_json_moead_all_constraints_satisfied(self):
        """All MOEA/D solutions on small.json must be feasible."""
        inst = Instance.from_file("data/small.json")
        sols = moead_frontier(inst, n_weights=100, n_gen=200, seed=42)
        assert len(sols) > 0
        for s in sols:
            assert s.f2 <= inst.budget + 1e-6
            assert capacity_violations(inst, s) == []
            assert recompute_f1(inst, s) == pytest.approx(s.f1, abs=1e-6)
            assert recompute_f2(inst, s) == pytest.approx(s.f2, abs=1e-6)

    def test_nsga2_output_non_dominated_on_small_json(self):
        inst = Instance.from_file("data/small.json")
        sols = nsga2_frontier(inst, pop_size=100, n_gen=200, seed=42)
        assert is_non_dominated_set(sols)

    def test_moead_output_non_dominated_on_small_json(self):
        inst = Instance.from_file("data/small.json")
        sols = moead_frontier(inst, n_weights=100, n_gen=200, seed=42)
        assert is_non_dominated_set(sols)


# ---------------------------------------------------------------------------
# MOEA/D bounded replacement (max_replace / nr)
# ---------------------------------------------------------------------------

class TestMoeadMaxReplace:
    """Correctness invariants must hold for every max_replace setting.

    max_replace=-1 must reproduce the original unbounded-replacement output
    exactly (same seed → same RNG draws, since capping is skipped entirely).
    Bounded settings (< neighborhood_size) introduce an extra shuffle draw
    per parent, so they are checked for validity, not bit-for-bit equality.
    """

    _MAX_REPLACE_VALUES = [-1, 1, 5, 10, 20]  # 20 == neighborhood_size (no-op cap)

    @pytest.mark.parametrize("max_replace", _MAX_REPLACE_VALUES)
    def test_valid_and_non_dominated_on_known_instance(self, max_replace,
                                                        known_pareto_inst):
        inst = known_pareto_inst
        sols = moead_frontier(inst, n_weights=50, n_gen=100,
                              neighborhood_size=20, max_replace=max_replace, seed=42)
        assert len(sols) > 0
        assert is_non_dominated_set(sols)
        for s in sols:
            assert set(s.assignment.keys()) == {j.id for j in inst.jobs}
            assert s.f2 <= inst.budget + 1e-6
            assert capacity_violations(inst, s) == []
            assert recompute_f1(inst, s) == pytest.approx(s.f1, abs=1e-6)
            assert recompute_f2(inst, s) == pytest.approx(s.f2, abs=1e-6)

    @pytest.mark.parametrize("max_replace", _MAX_REPLACE_VALUES)
    def test_no_solution_dominated_by_known_optimal(self, max_replace,
                                                     known_pareto_inst,
                                                     known_pareto_front):
        sols = moead_frontier(known_pareto_inst, n_weights=50, n_gen=100,
                              neighborhood_size=20, max_replace=max_replace, seed=42)
        for s in sols:
            for (kf1, kf2) in known_pareto_front:
                dominated = kf1 <= s.f1 and kf2 <= s.f2 and (kf1 < s.f1 or kf2 < s.f2)
                assert not dominated, (
                    f"max_replace={max_replace}: solution ({s.f1},{s.f2}) "
                    f"dominated by known point ({kf1},{kf2})"
                )

    def test_default_matches_explicit_max_replace_5(self, known_pareto_inst):
        """moead_frontier's default (5) must be bit-for-bit identical to
        passing max_replace=5 explicitly — chosen via A/B benchmark as the
        new default; -1 (original unbounded replacement) remains available
        explicitly."""
        explicit = moead_frontier(known_pareto_inst, n_weights=50, n_gen=100,
                                  neighborhood_size=20, max_replace=5, seed=42)
        default = moead_frontier(known_pareto_inst, n_weights=50, n_gen=100,
                                 neighborhood_size=20, seed=42)
        assert unique_front_points(explicit) == unique_front_points(default)

    def test_unbounded_still_reachable_via_max_replace_negative_one(
            self, known_pareto_inst):
        """max_replace=-1 must still reproduce the original unbounded
        replacement loop (no shuffle draws), independent of the new default."""
        sols = moead_frontier(known_pareto_inst, n_weights=50, n_gen=100,
                              neighborhood_size=20, max_replace=-1, seed=42)
        assert len(sols) > 0
        assert is_non_dominated_set(sols)

    @pytest.mark.parametrize("max_replace", [1, 5, 10])
    def test_same_seed_gives_identical_front(self, max_replace, known_pareto_inst):
        r1 = moead_frontier(known_pareto_inst, n_weights=50, n_gen=100,
                            neighborhood_size=20, max_replace=max_replace, seed=42)
        r2 = moead_frontier(known_pareto_inst, n_weights=50, n_gen=100,
                            neighborhood_size=20, max_replace=max_replace, seed=42)
        assert unique_front_points(r1) == unique_front_points(r2)

    def test_valid_on_small_json(self):
        inst = Instance.from_file("data/small.json")
        sols = moead_frontier(inst, n_weights=100, n_gen=200,
                              neighborhood_size=20, max_replace=5, seed=42)
        assert len(sols) > 0
        assert is_non_dominated_set(sols)
        for s in sols:
            assert s.f2 <= inst.budget + 1e-6
            assert capacity_violations(inst, s) == []
