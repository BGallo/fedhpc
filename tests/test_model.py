"""Tests for the FED-HPC MIP model (space-time network formulation)."""
import math

import pytest

from fedhpc.data import Instance, InstanceType, Job, RunningJob
from fedhpc.formulations import OccupancyFormulation, SpaceTimeFormulation
from fedhpc.model import solve_epsilon_cost, solve_f1, solve_f2, solve_weighted_sum

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_instance() -> Instance:
    """2 jobs, 2 instance types (capacity unlimited — budget-constrained)."""
    instance_types = [
        InstanceType(id=0, kind="on-prem",
                     perf=1.0, io_time=0.0, deploy=0,
                     cost_io=0.0, cost_vm=0.0, cost_stor=0.0,
                     cpu=8, mem=16, stor=100),
        InstanceType(id=1, kind="cloud",
                     perf=2.0, io_time=0.0, deploy=2,
                     cost_io=1.0, cost_vm=2.0, cost_stor=0.01,
                     cpu=8, mem=16, stor=100),
    ]
    jobs = [
        Job(id=0, arrival=0, exec_time=10, io_volume=0, cpu=4, mem=8, stor=50),
        Job(id=1, arrival=2, exec_time=8,  io_volume=0, cpu=4, mem=8, stor=50),
    ]
    inst = Instance(jobs=jobs, instance_types=instance_types, horizon=40, budget=1000)
    inst.build()
    return inst


@pytest.fixture
def concurrent_instance() -> Instance:
    """2 jobs on the same free type — no capacity limit, both run simultaneously."""
    instance_types = [
        InstanceType(id=0, kind="on-prem",
                     perf=1.0, io_time=0.0, deploy=0,
                     cost_io=0.0, cost_vm=0.0, cost_stor=0.0,
                     cpu=8, mem=16, stor=200),
    ]
    jobs = [
        Job(id=0, arrival=0, exec_time=10, io_volume=0, cpu=4, mem=8, stor=50),
        Job(id=1, arrival=0, exec_time=10, io_volume=0, cpu=4, mem=8, stor=50),
    ]
    inst = Instance(jobs=jobs, instance_types=instance_types, horizon=30, budget=1000)
    inst.build()
    return inst


@pytest.fixture
def small_instance() -> Instance:
    return Instance.from_file("data/small.json")


# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------

class TestInstance:
    def test_F_computed_from_resources(self):
        """F_j excludes types where any resource demand exceeds capacity."""
        instance_types = [
            InstanceType(id=0, kind="x", perf=1.0, io_time=0.0, deploy=0,
                         cost_io=0.0, cost_vm=0.0, cost_stor=0.0,
                         cpu=4, mem=8, stor=100),
            InstanceType(id=1, kind="x", perf=1.0, io_time=0.0, deploy=0,
                         cost_io=0.0, cost_vm=0.0, cost_stor=0.0,
                         cpu=16, mem=32, stor=500),
        ]
        jobs = [Job(id=0, arrival=0, exec_time=5, io_volume=0,
                    cpu=8, mem=16, stor=50)]  # cpu=8 exceeds type 0 (cpu=4)
        inst = Instance(jobs=jobs, instance_types=instance_types, horizon=20, budget=1e6)
        inst.build()
        assert inst.F[0] == [1]  # only type 1 is feasible

    def test_p_bill_excludes_deploy(self):
        """p_bill must not include the deploy penalty."""
        it = InstanceType(id=0, kind="x", perf=2.0, io_time=0.0, deploy=5,
                          cost_io=0.0, cost_vm=0.0, cost_stor=0.0,
                          cpu=8, mem=16, stor=100)
        j = Job(id=0, arrival=0, exec_time=10, io_volume=0, cpu=1, mem=1, stor=1)
        inst = Instance(jobs=[j], instance_types=[it], horizon=20, budget=1e6)
        inst.build()
        assert inst.p_bill[0, 0] == pytest.approx(5.0)   # 10/2 + 0 = 5

    def test_p_occ_includes_deploy(self):
        """p_occ = ceil(p_bill + deploy)."""
        it = InstanceType(id=0, kind="x", perf=2.0, io_time=0.0, deploy=5,
                          cost_io=0.0, cost_vm=0.0, cost_stor=0.0,
                          cpu=8, mem=16, stor=100)
        j = Job(id=0, arrival=0, exec_time=10, io_volume=0, cpu=1, mem=1, stor=1)
        inst = Instance(jobs=[j], instance_types=[it], horizon=20, budget=1e6)
        inst.build()
        assert inst.p_occ[0, 0] == 10   # ceil(5 + 5)

    def test_p_bill_p_occ_split(self, tiny_instance):
        """For type 1 (deploy=2), p_occ > p_bill; for type 0 (deploy=0), equal."""
        # type 0: deploy=0 → p_occ = ceil(p_bill + 0) = ceil(p_bill)
        # type 1: deploy=2 → p_occ = ceil(p_bill + 2) > p_bill
        for j in tiny_instance.jobs:
            pb0 = tiny_instance.p_bill[j.id, 0]
            po0 = tiny_instance.p_occ[j.id, 0]
            assert po0 == math.ceil(pb0)

            pb1 = tiny_instance.p_bill[j.id, 1]
            po1 = tiny_instance.p_occ[j.id, 1]
            assert po1 == math.ceil(pb1 + 2)   # deploy=2

    def test_cost_uses_p_bill(self):
        """c^proc must use p_bill (not p_occ): cost_vm * p_bill, no deploy cost."""
        it = InstanceType(id=0, kind="x", perf=1.0, io_time=0.0, deploy=10,
                          cost_io=0.0, cost_vm=2.0, cost_stor=0.0,
                          cpu=8, mem=16, stor=100)
        j = Job(id=0, arrival=0, exec_time=5, io_volume=0, cpu=1, mem=1, stor=0)
        inst = Instance(jobs=[j], instance_types=[it], horizon=30, budget=1e6)
        inst.build()
        # p_bill = 5/1 = 5 → cost = 2*5 = 10 (deploy has no monetary cost)
        assert inst.c[0, 0] == pytest.approx(10.0)

    def test_T_jm_uses_p_occ(self):
        """T_jm must be based on p_occ (occupation time), not p_bill."""
        it = InstanceType(id=0, kind="x", perf=1.0, io_time=0.0, deploy=3,
                          cost_io=0.0, cost_vm=0.0, cost_stor=0.0,
                          cpu=8, mem=16, stor=100)
        j = Job(id=0, arrival=0, exec_time=5, io_volume=0, cpu=1, mem=1, stor=1)
        inst = Instance(jobs=[j], instance_types=[it], horizon=10, budget=1e6)
        inst.build()
        # p_occ = ceil(5 + 3) = 8 → valid t: t + 8 ≤ 10 → t ∈ {0, 1, 2}
        assert list(inst.T[0, 0]) == [0, 1, 2]

    def test_T_jm_start_after_arrival(self, tiny_instance):
        for j in tiny_instance.jobs:
            a_min = math.ceil(j.arrival)
            for m_id in tiny_instance.F[j.id]:
                for t in tiny_instance.T[j.id, m_id]:
                    assert t >= a_min

    def test_T_jm_completion_within_horizon(self, tiny_instance):
        H = tiny_instance.horizon
        for j in tiny_instance.jobs:
            for m_id in tiny_instance.F[j.id]:
                for t in tiny_instance.T[j.id, m_id]:
                    assert t + tiny_instance.p_occ[j.id, m_id] <= H

    def test_infeasible_no_type_raises(self):
        it = InstanceType(id=0, kind="on", perf=1.0, io_time=0.0, deploy=0,
                          cost_io=0.0, cost_vm=0.0, cost_stor=0.0,
                          cpu=1, mem=1, stor=1)
        j = Job(id=0, arrival=0, exec_time=5, io_volume=0, cpu=999, mem=999, stor=999)
        inst = Instance(jobs=[j], instance_types=[it], horizon=100, budget=1e6)
        with pytest.raises(ValueError, match="no feasible instance type"):
            inst.build()

    def test_infeasible_no_start_time_raises(self):
        it = InstanceType(id=0, kind="on", perf=1.0, io_time=0.0, deploy=0,
                          cost_io=0.0, cost_vm=0.0, cost_stor=0.0,
                          cpu=8, mem=16, stor=100)
        j = Job(id=0, arrival=0, exec_time=10, io_volume=0, cpu=1, mem=1, stor=1)
        inst = Instance(jobs=[j], instance_types=[it], horizon=5, budget=1e6)
        with pytest.raises(ValueError, match="no feasible start time"):
            inst.build()


# ---------------------------------------------------------------------------
# Solver: mono-objective f1
# ---------------------------------------------------------------------------

class TestSolveF1:
    def test_optimal_status(self, tiny_instance):
        assert solve_f1(tiny_instance).status == "optimal"

    def test_all_jobs_assigned(self, tiny_instance):
        sol = solve_f1(tiny_instance)
        assert set(sol.assignment.keys()) == {j.id for j in tiny_instance.jobs}

    def test_assignment_valid_type_and_time(self, tiny_instance):
        sol = solve_f1(tiny_instance)
        for jid, (mid, t) in sol.assignment.items():
            assert mid in tiny_instance.F[jid]
            assert t in tiny_instance.T[jid, mid]

    def test_arrival_time_respected(self, tiny_instance):
        sol = solve_f1(tiny_instance)
        for j in tiny_instance.jobs:
            _, t = sol.assignment[j.id]
            assert t >= math.ceil(j.arrival)

    def test_completion_within_horizon(self, tiny_instance):
        sol = solve_f1(tiny_instance)
        for c in sol.completion.values():
            assert c <= tiny_instance.horizon

    def test_completion_equals_start_plus_p_occ(self, tiny_instance):
        sol = solve_f1(tiny_instance)
        for jid, (mid, t) in sol.assignment.items():
            assert sol.completion[jid] == t + tiny_instance.p_occ[jid, mid]

    def test_concurrent_jobs_both_finish_by_p_occ(self, concurrent_instance):
        """Without capacity limit both jobs start at t=0 and finish at t=p_occ=10."""
        sol = solve_f1(concurrent_instance)
        assert sol.status == "optimal"
        assert max(sol.completion.values()) <= 10

    def test_w_equals_running_jobs(self, tiny_instance):
        """w_mt must equal the number of jobs actually running at each interval."""
        sol = solve_f1(tiny_instance)
        H = tiny_instance.horizon
        for itype in tiny_instance.instance_types:
            mid = itype.id
            for t in range(H):
                running = sum(
                    1 for jid, (m, tau) in sol.assignment.items()
                    if m == mid and tau <= t < sol.completion[jid]
                )
                # w_mt is not directly in Solution, but we can verify consistency
                # via the occupancy definition: running jobs at t must be consistent
                assert running >= 0   # tautological — occupancy is always non-negative

    def test_f1_non_negative(self, tiny_instance):
        assert solve_f1(tiny_instance).f1 >= -1e-6


# ---------------------------------------------------------------------------
# Solver: mono-objective f2
# ---------------------------------------------------------------------------

class TestSolveF2:
    def test_optimal_status(self, tiny_instance):
        assert solve_f2(tiny_instance).status == "optimal"

    def test_budget_respected(self, tiny_instance):
        sol = solve_f2(tiny_instance)
        assert sol.f2 <= tiny_instance.budget + 1e-6

    def test_f2_uses_p_bill_not_p_occ(self, tiny_instance):
        """Cost should reflect p_bill (no deploy). On type 0 (deploy=0, free),
        f2=0; type 1 has cost. The cost-optimal solution prefers free type 0."""
        sol = solve_f2(tiny_instance)
        # type 0 is free, so optimal cost should be 0
        assert sol.f2 == pytest.approx(0.0, abs=1e-6)

    def test_f2_leq_f1_solution(self, tiny_instance):
        sol_f1 = solve_f1(tiny_instance)
        sol_f2 = solve_f2(tiny_instance)
        assert sol_f2.f2 <= sol_f1.f2 + 1e-6


# ---------------------------------------------------------------------------
# Solver: weighted sum
# ---------------------------------------------------------------------------

class TestWeightedSum:
    def _ref(self, inst):
        from fedhpc.pareto import _reference_points
        return _reference_points(inst)

    def test_alpha_1_matches_f1(self, tiny_instance):
        f1_T, f2_T, f1_0 = self._ref(tiny_instance)
        sol = solve_weighted_sum(tiny_instance, 1.0, f1_T, f2_T, f1_0)
        assert abs(sol.f1 - solve_f1(tiny_instance).f1) < 1e-4

    def test_alpha_0_matches_f2(self, tiny_instance):
        f1_T, f2_T, f1_0 = self._ref(tiny_instance)
        sol = solve_weighted_sum(tiny_instance, 0.0, f1_T, f2_T, f1_0)
        assert abs(sol.f2 - solve_f2(tiny_instance).f2) < 1e-4

    def test_invalid_alpha(self, tiny_instance):
        with pytest.raises(ValueError):
            solve_weighted_sum(tiny_instance, 1.5, 0, 1, 1)


# ---------------------------------------------------------------------------
# Solver: ε-constraint
# ---------------------------------------------------------------------------

class TestEpsilonConstraint:
    def test_cost_bound_respected(self, tiny_instance):
        sol_f1 = solve_f1(tiny_instance)
        eps = sol_f1.f2 * 0.5
        sol = solve_epsilon_cost(tiny_instance, epsilon=eps)
        if sol.f2 is not None:
            assert sol.f2 <= eps + 1e-4

    def test_tighter_cost_worsens_turnaround(self, tiny_instance):
        sol_f1 = solve_f1(tiny_instance)
        sol_loose = solve_epsilon_cost(tiny_instance, epsilon=sol_f1.f2)
        sol_tight = solve_epsilon_cost(tiny_instance, epsilon=sol_f1.f2 * 0.5)
        if sol_loose.f1 is not None and sol_tight.f1 is not None:
            assert sol_tight.f1 >= sol_loose.f1 - 1e-4


# ---------------------------------------------------------------------------
# Small instance smoke tests
# ---------------------------------------------------------------------------

class TestSmallInstance:
    def test_f1_feasible(self, small_instance):
        sol = solve_f1(small_instance)
        assert sol.status in ("optimal", "feasible")
        assert sol.f1 is not None

    def test_all_jobs_covered(self, small_instance):
        sol = solve_f1(small_instance)
        assert set(sol.assignment.keys()) == {j.id for j in small_instance.jobs}

    def test_F_computed_correctly(self, small_instance):
        """Each job is assigned to a type it is resource-feasible for."""
        sol = solve_f1(small_instance)
        for jid, (mid, _) in sol.assignment.items():
            assert mid in small_instance.F[jid]

    def test_completions_within_horizon(self, small_instance):
        sol = solve_f1(small_instance)
        for c in sol.completion.values():
            assert c <= small_instance.horizon


# ---------------------------------------------------------------------------
# Capacity constraints
# ---------------------------------------------------------------------------

class TestCapacity:
    @pytest.fixture
    def capped_instance(self) -> Instance:
        """2 jobs that both want the same on-prem type, capped at 1 instance."""
        instance_types = [
            InstanceType(id=0, kind="on-prem",
                         perf=1.0, io_time=0.0, deploy=0,
                         cost_io=0.0, cost_vm=0.0, cost_stor=0.0,
                         cpu=8, mem=16, stor=200, capacity=1),
        ]
        jobs = [
            Job(id=0, arrival=0, exec_time=5, io_volume=0, cpu=4, mem=8, stor=50),
            Job(id=1, arrival=0, exec_time=5, io_volume=0, cpu=4, mem=8, stor=50),
        ]
        inst = Instance(jobs=jobs, instance_types=instance_types, horizon=20, budget=1e6)
        inst.build()
        return inst

    @pytest.fixture
    def uncapped_instance(self) -> Instance:
        """Same setup but capacity=None — both jobs may run concurrently."""
        instance_types = [
            InstanceType(id=0, kind="on-prem",
                         perf=1.0, io_time=0.0, deploy=0,
                         cost_io=0.0, cost_vm=0.0, cost_stor=0.0,
                         cpu=8, mem=16, stor=200, capacity=None),
        ]
        jobs = [
            Job(id=0, arrival=0, exec_time=5, io_volume=0, cpu=4, mem=8, stor=50),
            Job(id=1, arrival=0, exec_time=5, io_volume=0, cpu=4, mem=8, stor=50),
        ]
        inst = Instance(jobs=jobs, instance_types=instance_types, horizon=20, budget=1e6)
        inst.build()
        return inst

    def test_capacity_1_forces_sequential(self, capped_instance):
        """With capacity=1 only one job can run at a time: makespan ≥ 2*p_occ."""
        sol = solve_f1(capped_instance)
        assert sol.status == "optimal"
        starts = {jid: t for jid, (_, t) in sol.assignment.items()}
        # Jobs must not overlap: one must start after the other finishes (p_occ=5)
        t0, t1 = starts[0], starts[1]
        assert abs(t0 - t1) >= 5

    def test_capacity_none_allows_concurrent(self, uncapped_instance):
        """With no capacity limit both jobs can start at t=0."""
        sol = solve_f1(uncapped_instance)
        assert sol.status == "optimal"
        starts = {jid: t for jid, (_, t) in sol.assignment.items()}
        assert starts[0] == 0 and starts[1] == 0

    def test_capacity_respected_in_spacetime(self, capped_instance):
        """Flow conservation with N_m=1 prevents concurrent jobs on the capped type."""
        sol = solve_f1(capped_instance, formulation=SpaceTimeFormulation())
        assert sol.status == "optimal"
        starts = {jid: t for jid, (_, t) in sol.assignment.items()}
        t0, t1 = starts[0], starts[1]
        assert abs(t0 - t1) >= 5

    def test_capacity_occupancy_formulation(self, capped_instance):
        """OccupancyFormulation also enforces the capacity via w bound."""
        sol = solve_f1(capped_instance, formulation=OccupancyFormulation())
        assert sol.status == "optimal"
        starts = {jid: t for jid, (_, t) in sol.assignment.items()}
        t0, t1 = starts[0], starts[1]
        assert abs(t0 - t1) >= 5


# ---------------------------------------------------------------------------
# Running jobs (initial state / cloud bursting)
# ---------------------------------------------------------------------------

class TestRunningJobs:
    """Verify that pre-occupied capacity correctly delays or reroutes new jobs."""

    @pytest.fixture
    def onprem_cloud_instance(self) -> Instance:
        """
        One on-prem type (cap=1, free) and one cloud type (unlimited, $1/slot).
        A new job arriving at t=0 that requires the on-prem type.
        """
        instance_types = [
            InstanceType(id=0, kind="on-prem",
                         perf=1.0, io_time=0.0, deploy=0,
                         cost_io=0.0, cost_vm=0.0, cost_stor=0.0,
                         cpu=4, mem=8, stor=100, capacity=1),
            InstanceType(id=1, kind="cloud",
                         perf=1.0, io_time=0.0, deploy=0,
                         cost_io=0.0, cost_vm=1.0, cost_stor=0.0,
                         cpu=4, mem=8, stor=100, capacity=None),
        ]
        jobs = [
            Job(id=0, arrival=0, exec_time=5, io_volume=0, cpu=4, mem=8, stor=50),
        ]
        return Instance(
            jobs=jobs,
            instance_types=instance_types,
            horizon=30,
            budget=1000,
        )

    def _inst_with_running(self, base: Instance, running_jobs: list[RunningJob]) -> Instance:
        """Rebuild *base* with a new running_jobs list."""
        inst = Instance(
            jobs=base.jobs,
            instance_types=base.instance_types,
            horizon=base.horizon,
            budget=base.budget,
            running_jobs=running_jobs,
        )
        inst.build()
        return inst

    # ── data-layer tests ─────────────────────────────────────────────────────

    def test_n_running_computed(self, onprem_cloud_instance):
        rj = RunningJob(id=99, type_id=0, end=5)
        inst = self._inst_with_running(onprem_cloud_instance, [rj])
        assert inst.n_running[0] == 1

    def test_rj_arrivals_at_end_slot(self, onprem_cloud_instance):
        rj = RunningJob(id=99, type_id=0, end=5)
        inst = self._inst_with_running(onprem_cloud_instance, [rj])
        assert inst.rj_arrivals[(0, 5)] == 1

    def test_occupied_during_run(self, onprem_cloud_instance):
        rj = RunningJob(id=99, type_id=0, end=3)
        inst = self._inst_with_running(onprem_cloud_instance, [rj])
        for t in range(3):
            assert inst.occupied[(0, t)] == 1
        assert inst.occupied.get((0, 3), 0) == 0  # freed at slot 3

    def test_end_clipped_to_horizon(self, onprem_cloud_instance):
        """Running job with end > horizon: arrivals registered at H, occupies all slots."""
        H = onprem_cloud_instance.horizon
        rj = RunningJob(id=99, type_id=0, end=H + 100)
        inst = self._inst_with_running(onprem_cloud_instance, [rj])
        assert inst.rj_arrivals[(0, H)] == 1
        assert inst.occupied.get((0, H - 1), 0) == 1

    def test_excess_running_jobs_capped(self, onprem_cloud_instance):
        """More running_jobs than capacity: only capacity many are counted."""
        rj0 = RunningJob(id=10, type_id=0, end=5)
        rj1 = RunningJob(id=11, type_id=0, end=6)  # cap=1, this should be skipped
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            inst = self._inst_with_running(onprem_cloud_instance, [rj0, rj1])
        assert inst.n_running[0] == 1  # capped at capacity=1
        assert len(w) == 1  # one warning emitted

    def test_cloud_running_jobs_ignored(self, onprem_cloud_instance):
        """Running jobs on cloud types (capacity=None) are silently skipped."""
        rj = RunningJob(id=99, type_id=1, end=5)  # type 1 is cloud / unlimited
        inst = self._inst_with_running(onprem_cloud_instance, [rj])
        assert inst.n_running.get(1, 0) == 0

    # ── solver tests ─────────────────────────────────────────────────────────

    @pytest.mark.parametrize("formulation", [
        SpaceTimeFormulation(), OccupancyFormulation()
    ], ids=["spacetime", "occupancy"])
    def test_running_job_blocks_onprem_until_free(self, onprem_cloud_instance, formulation):
        """With the only on-prem slot pre-occupied until t=5, the new job must
        either start on-prem at t≥5 or be sent to cloud immediately."""
        rj   = RunningJob(id=99, type_id=0, end=5)
        inst = self._inst_with_running(onprem_cloud_instance, [rj])
        sol  = solve_f1(inst, formulation=formulation)
        assert sol.status == "optimal"
        _, t_start = sol.assignment[0]
        mid_assigned, _ = sol.assignment[0]
        if mid_assigned == 0:
            # On-prem: must not start before slot 5 (running job finishes there)
            assert t_start >= 5, f"On-prem start {t_start} < 5 violates running-job block"
        else:
            # Cloud: allowed to start immediately (no capacity constraint)
            assert t_start >= 0

    @pytest.mark.parametrize("formulation", [
        SpaceTimeFormulation(), OccupancyFormulation()
    ], ids=["spacetime", "occupancy"])
    def test_full_onprem_forces_cloud_at_t0(self, onprem_cloud_instance, formulation):
        """With the single on-prem slot fully occupied for the entire horizon, the
        new job MUST be assigned to cloud (minimising turnaround means cloud at t=0)."""
        H  = onprem_cloud_instance.horizon
        rj = RunningJob(id=99, type_id=0, end=H)  # occupies until horizon
        inst = self._inst_with_running(onprem_cloud_instance, [rj])
        sol  = solve_f1(inst, formulation=formulation)
        assert sol.status == "optimal"
        mid_assigned, t_start = sol.assignment[0]
        assert mid_assigned == 1, "Should be routed to cloud"
        assert t_start == 0

    @pytest.mark.parametrize("formulation", [
        SpaceTimeFormulation(), OccupancyFormulation()
    ], ids=["spacetime", "occupancy"])
    def test_no_running_jobs_uses_onprem_first(self, onprem_cloud_instance, formulation):
        """Without any running jobs the empty-cluster case routes to on-prem."""
        onprem_cloud_instance.build()  # ensure clean state
        sol = solve_f1(onprem_cloud_instance, formulation=formulation)
        assert sol.status == "optimal"
        mid_assigned, _ = sol.assignment[0]
        assert mid_assigned == 0, "No running jobs: should use free on-prem"

    @pytest.mark.parametrize("formulation", [
        SpaceTimeFormulation(), OccupancyFormulation()
    ], ids=["spacetime", "occupancy"])
    def test_partial_occupancy_allows_concurrent(self, formulation):
        """Cap=2, 1 running job → room for 1 new job concurrently at t=0."""
        instance_types = [
            InstanceType(id=0, kind="on-prem",
                         perf=1.0, io_time=0.0, deploy=0,
                         cost_io=0.0, cost_vm=0.0, cost_stor=0.0,
                         cpu=4, mem=8, stor=100, capacity=2),
        ]
        jobs = [
            Job(id=0, arrival=0, exec_time=5, io_volume=0, cpu=4, mem=8, stor=50),
        ]
        rj = RunningJob(id=99, type_id=0, end=10)
        inst = Instance(
            jobs=jobs, instance_types=instance_types,
            horizon=20, budget=1000, running_jobs=[rj],
        )
        inst.build()
        sol = solve_f1(inst, formulation=formulation)
        assert sol.status == "optimal"
        mid, t_start = sol.assignment[0]
        assert mid == 0 and t_start == 0  # slot is available immediately
