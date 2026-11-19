"""Multi-node jobs (Job.nodes, RunningJob.units) and GPUs (Job.gpu, InstanceType.gpu)."""
import pytest

from fedhpc.baselines import onprem_only_spt
from fedhpc.data import Instance, InstanceType, Job, RunningJob
from fedhpc.formulations import OccupancyFormulation, SpaceTimeFormulation, unbounded_types
from fedhpc.model import solve_f1
from fedhpc.moea import _type_cap


def _type(tid, capacity, gpu=0.0, cost_vm=0.0):
    return InstanceType(id=tid, kind=f"t{tid}", perf=1.0, io_time=0.0, deploy=0.0, cost_io=0.0, cost_vm=cost_vm,
                        cost_stor=0.0, cpu=40.0, mem=1e9, stor=1e9, capacity=capacity, gpu=gpu)


def _job(jid, exec_time, nodes=1, gpu=0.0, arrival=0.0):
    return Job(id=jid, arrival=arrival, exec_time=exec_time, io_volume=0.0, cpu=40.0, mem=1.0, stor=1.0,
               nodes=nodes, gpu=gpu)


def _inst(jobs, types, horizon=12, running=()):
    inst = Instance(jobs=jobs, instance_types=types, horizon=horizon, running_jobs=list(running))
    inst.build()
    return inst


FORMULATIONS = {
    "spacetime": SpaceTimeFormulation(),
    "patterns": SpaceTimeFormulation(normal_patterns=True),
    "compressed": SpaceTimeFormulation(compressed_flow=True),
    "occupancy": OccupancyFormulation(),
}


class TestFeasibility:
    def test_gpu_jobs_fit_only_types_with_enough_gpus(self):
        inst = _inst([_job(0, 1, gpu=1), _job(1, 1)], [_type(0, 4), _type(1, 4, gpu=2)])
        assert inst.F[0] == [1]
        assert inst.F[1] == [0, 1]

    def test_a_job_with_more_nodes_than_the_capacity_is_infeasible_there(self):
        inst = _inst([_job(0, 1, nodes=3)], [_type(0, 2), _type(1, None)])
        assert inst.F[0] == [1]

    def test_every_node_is_billed(self):
        inst = _inst([_job(0, 2, nodes=1), _job(1, 2, nodes=3)], [_type(0, None, cost_vm=1.5)])
        assert inst.c[1, 0] == pytest.approx(3 * inst.c[0, 0])

    def test_defaults_keep_the_unit_model(self):
        inst = _inst([_job(0, 1)], [_type(0, 1)])
        assert inst.unit_demand
        assert _type_cap(inst) == [1]

    def test_nodes_must_be_positive_integers(self):
        with pytest.raises(ValueError):
            _inst([_job(0, 1, nodes=0)], [_type(0, 2)])


class TestCapacity:
    """Capacity 3: two 2-node jobs never overlap; a 1-node job fits beside either."""

    JOBS = [(0, 2, 2), (1, 2, 2), (2, 1, 1)]  # (id, runtime, nodes)

    def _instance(self):
        return _inst([_job(j, p, nodes=q) for j, p, q in self.JOBS], [_type(0, 3)])

    @pytest.mark.parametrize("name", FORMULATIONS)
    def test_the_mip_never_overlaps_jobs_beyond_the_capacity(self, name):
        inst = self._instance()
        sol = solve_f1(inst, formulation=FORMULATIONS[name])
        inst.check_schedule(sol.assignment)
        # One 2-node job and the 1-node job at 0, the other 2-node job at 2: completions 2 + 1 + 4.
        assert sol.f1 == pytest.approx(7)

    @pytest.mark.parametrize("name", FORMULATIONS)
    def test_running_units_hold_capacity(self, name):
        # A 2-unit running job holds 2 of 3 units until slot 2: a 2-node job starts at 2.
        inst = _inst([_job(0, 1, nodes=2)], [_type(0, 3)], running=[RunningJob(id=9, type_id=0, end=2, units=2)])
        sol = solve_f1(inst, formulation=FORMULATIONS[name])
        assert sol.assignment[0] == (0, 2)

    def test_overlapping_multi_node_jobs_are_a_violation(self):
        inst = self._instance()
        assert any("capacity" in v for v in inst.schedule_violations({0: (0, 0), 1: (0, 0), 2: (0, 2)}))
        assert inst.schedule_violations({0: (0, 0), 2: (0, 0), 1: (0, 2)}) == []

    def test_unbounded_types_count_units(self):
        assert unbounded_types(self._instance()) == set()
        assert unbounded_types(_inst([_job(0, 1, nodes=2), _job(1, 1)], [_type(0, 3)])) == {0}

    def test_the_list_schedule_respects_units(self):
        inst = self._instance()
        sol = onprem_only_spt(inst)
        inst.check_schedule(sol.assignment)

    def test_heuristics_without_multi_node_support_refuse(self):
        with pytest.raises(NotImplementedError):
            _type_cap(self._instance())


def _random_instance(seed, n_jobs=14, capacity=4, horizon=40, cloud=True):
    import random
    rng = random.Random(seed)
    jobs = [_job(j, rng.randint(1, 5), nodes=rng.choice([1, 1, 2, 3, 4]), arrival=float(rng.randint(0, 8)))
            for j in range(n_jobs)]
    types = [_type(0, capacity)] + ([_type(1, None, cost_vm=1.0)] if cloud else [])
    running = [RunningJob(id=100, type_id=0, end=3, units=2)]
    return _inst(jobs, types, horizon=horizon, running=running)


class TestLpFirst:
    @pytest.mark.parametrize("seed", range(6))
    def test_min_f1_matches_the_mip(self, seed):
        from fedhpc.lpfirst import solve_f1_lpfirst
        inst = _random_instance(seed, cloud=False)
        ref = solve_f1(inst, formulation=SpaceTimeFormulation())
        sol = solve_f1_lpfirst(inst)
        inst.check_schedule(sol.assignment)
        assert sol.status == "optimal"
        assert sol.f1 == pytest.approx(ref.f1)

    @pytest.mark.parametrize("seed", range(6))
    def test_weighted_sum_matches_the_mip(self, seed):
        from fedhpc.lpfirst import solve_lpfirst
        from fedhpc.model import _solve
        inst = _random_instance(seed)
        ref = _solve(inst, SpaceTimeFormulation(), None, {"MIPGap": 0.0}, lambda f1, f2: f1 + 3.0 * f2)
        res = solve_lpfirst(inst, 1.0, 3.0)
        assert res.status == "optimal"
        inst.check_schedule(res.assignment)
        assert res.objective == pytest.approx(ref.f1 + 3.0 * ref.f2)

    def test_the_capacity_instance(self):
        from fedhpc.lpfirst import solve_f1_lpfirst
        inst = TestCapacity()._instance()
        sol = solve_f1_lpfirst(inst)
        inst.check_schedule(sol.assignment)
        assert sol.f1 == pytest.approx(7)


class TestDispatchHeuristics:
    """The dispatch ILS and the memetic GA (Overseer's ga_mid) on multi-node jobs."""

    GA = {"flags": 12352, "adapt": 215, "shortlist": 48, "pop_size": 8, "n_gen": 20}

    def test_the_capacity_instance(self):
        from fedhpc import moea
        inst = TestCapacity()._instance()
        for sol in (moea.weighted_solve_dispatch(inst, 1.0, n_threads=1),
                    moea.weighted_solve_memetic(inst, 1.0, n_threads=1, **self.GA)):
            inst.check_schedule(sol.assignment)
            assert sol.f1 == pytest.approx(7)

    @pytest.mark.parametrize("decoder", [0, 256])   # dispatch, serial SGS
    @pytest.mark.parametrize("seed", range(4))
    def test_each_decoder_respects_units(self, seed, decoder):
        from fedhpc import moea
        inst = _random_instance(seed, n_jobs=20, cloud=False, horizon=80)   # room for list schedules
        moea._require_ext()
        asgn, _f1, _f2, _pen, _why, cv = moea._ext.dispatch_decode(
            H=inst.horizon, jobs=moea._dispatch_jobs(inst), type_cap=moea._type_cap(inst, require_unit=False),
            init_occ=moea._init_occ(inst), w1=1.0, w2=0.0, f1_cap=1e18, gen=[0] * len(inst.jobs),
            flags=decoder, nodes=moea._dispatch_nodes(inst))
        assert cv == 0
        inst.check_schedule({j.id: tuple(a) for j, a in zip(inst.jobs, asgn)})

    @pytest.mark.parametrize("cloud", [False, True])
    @pytest.mark.parametrize("seed", range(4))
    def test_feasible_and_near_the_mip(self, seed, cloud):
        from fedhpc import moea
        inst = _random_instance(seed, cloud=cloud)
        for sol in (moea.weighted_solve_dispatch(inst, 0.5, n_threads=1),
                    moea.weighted_solve_memetic(inst, 0.5, n_threads=1, **self.GA)):
            inst.check_schedule(sol.assignment)
        if not cloud:   # lam = 1 is min f1: within 10 % of the MIP optimum
            sol = moea.weighted_solve_memetic(inst, 1.0, n_threads=1, **self.GA)
            assert sol.f1 <= 1.10 * solve_f1(inst).f1

    def test_reference_points_come_from_the_dispatch_ils(self):
        from fedhpc import moea
        inst = _random_instance(1)
        f1_T, f2_T, f1_0, f2_0 = moea.heuristic_weighted_sum_reference(inst, n_threads=1)
        assert f1_T <= f1_0 and f2_0 <= f2_T

    def test_the_sgs_heuristics_refuse(self):
        from fedhpc import moea
        with pytest.raises(NotImplementedError):
            moea.nsga2_sgs_frontier(TestCapacity()._instance(), pop_size=8, n_gen=2)


class TestUrgentFirstPriority:
    """DF_URGENT_ADD (1 << 22): WSPT starves a wide job whose latest start the horizon forces early."""

    @staticmethod
    def _starving_instance():
        # 3 nodes, H = 14: the 3-node 10-slot job must start by slot 4, while a 1-slot 1-node job arrives
        # at every slot 0..10 and WSPT always prefers it, so 3 nodes are never free together. Feasible only
        # with the wide job at 0..4 (30 + 11 node-slots of 42).
        jobs = [_job(0, 10, nodes=3)] + [_job(1 + t, 1, arrival=float(t)) for t in range(11)]
        return _inst(jobs, [_type(0, 3)], horizon=14)

    def _decode(self, inst, flags):
        from fedhpc import moea
        moea._require_ext()
        _a, f1, _f2, _g, st = moea._ext.weighted_dispatch(
            H=inst.horizon, jobs=moea._dispatch_jobs(inst), type_cap=moea._type_cap(inst, require_unit=False),
            init_occ=moea._init_occ(inst), nodes=moea._dispatch_nodes(inst), w1=1.0, w2=0.0, flags=flags,
            mu=0.0, n_kicks=0, ls=0, n_threads=1)
        return st

    def test_wspt_starves_the_wide_job(self):
        assert self._decode(self._starving_instance(), 0)["cv"] == 1

    def test_the_urgent_start_places_it(self):
        st = self._decode(self._starving_instance(), 1 << 22)
        assert st["cv"] == 0 and st["mu"] == -3.0

    def test_the_memetic_ga_uses_it(self):
        from fedhpc import moea
        inst = self._starving_instance()
        # ga_mid's population (16): the initial spread covers every priority index (10 mu + EDF + urgent)
        sol = moea.weighted_solve_memetic(inst, 1.0, n_threads=1, flags=12352 | (1 << 22), adapt=215,
                                          shortlist=48, pop_size=16, n_gen=20)
        inst.check_schedule(sol.assignment)
        assert sol.assignment[0][1] <= 4
