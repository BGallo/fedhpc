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
