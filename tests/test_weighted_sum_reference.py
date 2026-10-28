"""pareto.weighted_sum_reference / solve_weighted_sum_with_reference: the
reference points against independent optima of the plain formulation, the
cheapest-schedule reference, and the degenerate case answered without a
rebuild."""
import random

import pytest
from gurobipy import GRB

from fedhpc import formulations
from fedhpc.data import Instance, InstanceType, Job, RunningJob
from fedhpc.formulations import SpaceTimeFormulation
from fedhpc.model import solve_weighted_sum
from fedhpc.pareto import (
    _reference_points,
    solve_weighted_sum_with_reference,
    weighted_sum_reference,
)

formulations.configure_env()

PLAIN = dict(collapse_unbounded=False, pareto_unbounded=False,
             rejection_deadlines=False, lex_deadlines=False)
BIG = 1e9


def _type(id, kind, perf, deploy, cost_vm, capacity, fixed_cost=0.0, cpu=BIG):
    return InstanceType(id=id, kind=kind, perf=perf, io_time=0.0, deploy=deploy,
                        cost_io=0.0, cost_vm=cost_vm, cost_stor=0.0, cpu=cpu,
                        mem=BIG, stor=BIG, capacity=capacity, fixed_cost=fixed_cost)


def _instance(seed: int, cloud_only_jobs: int = 0) -> Instance:
    """Capacity-limited on-prem, paid cloud, a fixed-cost type, a running job.
    The last ``cloud_only_jobs`` jobs do not fit on-prem: no schedule is free."""
    rng = random.Random(seed)
    types = [
        _type(0, "on-prem", 1.0, 0.0, 0.0, 2, cpu=8.0),
        _type(1, "cloud", 1.5, 1.0, 1.0, None),
        _type(2, "cloud-reserved", 1.2, 0.0, 0.4, 1, fixed_cost=3.0),
    ]
    jobs = [Job(id=j, arrival=float(rng.randint(0, 12)), exec_time=float(rng.randint(1, 7)),
                io_volume=0.0, cpu=16.0 if j >= 9 - cloud_only_jobs else 0.0, mem=0.0, stor=0.0,
                weight=float(rng.randint(1, 5)))
            for j in range(9)]
    inst = Instance(jobs=jobs, instance_types=types, horizon=40,
                    running_jobs=[RunningJob(id=100, type_id=0, end=rng.randint(1, 6))])
    inst.build()
    return inst


def _short_horizon_instance() -> Instance:
    """Every job has a free (on-prem) option, but 2 on-prem instances cannot
    finish 9 jobs of 6 slots within 12 slots: 4 run on-prem, 5 on the cloud."""
    types = [_type(0, "on-prem", 1.0, 0.0, 0.0, 2), _type(1, "cloud", 1.0, 0.0, 1.0, None)]
    inst = Instance(jobs=[Job(id=j, arrival=0.0, exec_time=6.0, io_volume=0.0, cpu=0.0, mem=0.0, stor=0.0)
                          for j in range(9)], instance_types=types, horizon=12)
    inst.build()
    return inst


def _cloud_only_instance(cost_vm: float = 1.0) -> Instance:
    """One unlimited cloud type: every job starts at its arrival at a fixed
    cost, so the ideal point is attained at a positive (or negative) cost."""
    rng = random.Random(7)
    types = [_type(0, "cloud", 1.0, 1.0, cost_vm, None)]
    inst = Instance(jobs=[Job(id=j, arrival=float(rng.randint(0, 5)), exec_time=float(rng.randint(1, 4)),
                              io_volume=0.0, cpu=0.0, mem=0.0, stor=0.0) for j in range(5)],
                    instance_types=types, horizon=20)
    inst.build()
    return inst


def _free_instance() -> Instance:
    """Enough on-prem capacity for every job at its arrival: f2_T = 0."""
    rng = random.Random(3)
    types = [_type(0, "on-prem", 1.0, 0.0, 0.0, 20), _type(1, "cloud", 0.5, 0.0, 1.0, None)]
    inst = Instance(jobs=[Job(id=j, arrival=float(rng.randint(0, 5)), exec_time=float(rng.randint(1, 4)),
                              io_volume=0.0, cpu=0.0, mem=0.0, stor=0.0) for j in range(6)],
                    instance_types=types, horizon=20)
    inst.build()
    return inst


def _optimum(inst, objective, f1_max=None, f2_max=None) -> float:
    """The optimum of the plain (unreduced) formulation."""
    fmt = SpaceTimeFormulation(**PLAIN)
    mdl, v = fmt.build(inst)
    mdl.setParam("MIPGap", 0)
    f1, f2 = fmt.f1_expr(inst, v["x"]), fmt.f2_expr(inst, v["x"])
    if f1_max is not None:
        mdl.addConstr(f1 <= f1_max + 1e-9)
    if f2_max is not None:
        mdl.addConstr(f2 <= f2_max)
    if objective == "f1":
        mdl.setObjective(f1, GRB.MINIMIZE)
    elif objective == "f2":
        mdl.setObjective(f2, GRB.MINIMIZE)
    else:
        _, lam, f1_T, f2_T, f1_0, f2_0 = objective
        mdl.setObjective(lam * (f1 - f1_T) / (f1_0 - f1_T) + (1 - lam) * (f2 - f2_0) / (f2_T - f2_0),
                         GRB.MINIMIZE)
    mdl.optimize()
    value = mdl.ObjVal
    mdl.dispose()
    return value


class _CountingFormulation(SpaceTimeFormulation):
    def __init__(self):
        super().__init__()
        self.builds = []

    def build(self, inst, objective=None):
        self.builds.append(objective)
        return super().build(inst, objective=objective)


@pytest.mark.parametrize("seed", range(6))
def test_with_a_free_schedule_they_are_the_papers(seed):
    inst = _instance(seed)
    (f1_T, f2_T, f1_0, f2_0), _ = weighted_sum_reference(inst, MIPGap=0)
    assert f2_0 == 0.0
    assert (f1_T, f2_T, f1_0) == _reference_points(inst, MIPGap=0)
    assert f1_T == pytest.approx(_optimum(inst, "f1"))
    assert f2_T == pytest.approx(_optimum(inst, "f2", f1_max=f1_T + 1e-6))
    assert f1_0 == pytest.approx(_optimum(inst, "f1", f2_max=0.0))
    # min_cost_reference changes nothing when a schedule is free.
    assert weighted_sum_reference(inst, min_cost_reference=True, MIPGap=0)[0] == (f1_T, f2_T, f1_0, f2_0)


@pytest.mark.parametrize("seed", range(6))
def test_without_a_free_option_the_cheapest_schedule_is_the_reference(seed):
    inst = _instance(seed, cloud_only_jobs=2)
    with pytest.raises(RuntimeError):
        _reference_points(inst, MIPGap=0)
    (f1_T, f2_T, f1_0, f2_0), _ = weighted_sum_reference(inst, min_cost_reference=True, MIPGap=0)
    assert f2_0 > 0.0
    assert f2_0 == pytest.approx(_optimum(inst, "f2"))
    assert f1_0 == pytest.approx(_optimum(inst, "f1", f2_max=f2_0 + 1e-6))
    assert f1_T == pytest.approx(_optimum(inst, "f1"))
    assert f2_T == pytest.approx(_optimum(inst, "f2", f1_max=f1_T + 1e-6))


def test_when_free_options_cannot_finish_every_job_the_cheapest_schedule_is_the_reference():
    inst = _short_horizon_instance()
    with pytest.raises(RuntimeError):
        _reference_points(inst, MIPGap=0)
    fmt = _CountingFormulation()
    (f1_T, f2_T, f1_0, f2_0), _ = weighted_sum_reference(inst, fmt, min_cost_reference=True, MIPGap=0)
    assert f2_0 == pytest.approx(30.0)
    assert f1_0 == pytest.approx(_optimum(inst, "f1", f2_max=f2_0 + 1e-6))
    assert f1_T == pytest.approx(_optimum(inst, "f1"))
    # Phases 1-2 ran once: the lex-hinted model and one unhinted model.
    assert len(fmt.builds) == 2


@pytest.mark.parametrize("lam", [0.25, 0.5, 0.9])
@pytest.mark.parametrize("cloud_only_jobs", [0, 2])
@pytest.mark.parametrize("seed", range(4))
def test_the_solve_minimizes_the_normalized_objective(seed, cloud_only_jobs, lam):
    inst = _instance(seed, cloud_only_jobs)
    sol, (f1_T, f2_T, f1_0, f2_0) = solve_weighted_sum_with_reference(
        inst, lam, min_cost_reference=True, MIPGap=0)
    assert f1_0 - f1_T > 1e-8 and f2_T - f2_0 > 1e-8
    expected = _optimum(inst, ("ws", lam, f1_T, f2_T, f1_0, f2_0), f1_max=f1_0)
    achieved = lam * (sol.f1 - f1_T) / (f1_0 - f1_T) + (1 - lam) * (sol.f2 - f2_0) / (f2_T - f2_0)
    assert achieved == pytest.approx(expected, abs=1e-6)
    # Same as model.solve_weighted_sum with those reference points.
    direct = solve_weighted_sum(inst, lam, f1_T, f2_T, f1_0, f2_0=f2_0, MIPGap=0)
    assert direct.objective == pytest.approx(sol.objective)


@pytest.mark.parametrize("make, builds", [(_free_instance, 1), (_cloud_only_instance, 2)])
def test_degenerate_weighted_sum_is_answered_without_a_rebuild(make, builds):
    inst = make()
    fmt = _CountingFormulation()
    sol, (f1_T, f2_T, f1_0, f2_0) = solve_weighted_sum_with_reference(
        inst, 0.5, fmt, min_cost_reference=True, MIPGap=0)
    assert len(fmt.builds) == builds
    assert f1_0 == pytest.approx(f1_T) and f2_T == pytest.approx(f2_0)
    # The degenerate problem: min f2 s.t. f1 <= f1_0.
    direct = solve_weighted_sum(inst, 0.5, f1_T, f2_T, f1_0, f2_0=f2_0, MIPGap=0)
    assert sol.status == direct.status == "optimal"
    assert sol.f2 == pytest.approx(direct.f2) == pytest.approx(_optimum(inst, "f2", f1_max=f1_0))
    assert sol.objective == pytest.approx(direct.objective)
    assert sol.f1 <= f1_0 + 1e-6
    assert set(sol.assignment) == {j.id for j in inst.jobs}


def test_negative_costs_fall_back_to_the_solve():
    """With negative data a cost-<=0 schedule need not be the cheapest."""
    inst = _cloud_only_instance(cost_vm=-1.0)
    fmt = _CountingFormulation()
    sol, (f1_T, f2_T, f1_0, f2_0) = solve_weighted_sum_with_reference(inst, 0.5, fmt, MIPGap=0)
    assert f2_0 == 0.0 and f2_T < 0.0
    direct = solve_weighted_sum(inst, 0.5, f1_T, f2_T, f1_0, MIPGap=0)
    assert sol.f1 == pytest.approx(direct.f1) and sol.f2 == pytest.approx(direct.f2)


def test_uncertified_reference_points_raise(monkeypatch):
    """A reference point every weighted sum is normalised by must be proven:
    an incumbent without a certificate raises instead of being used."""
    from fedhpc import pareto
    monkeypatch.setattr(pareto, "is_certified", lambda *a: False)
    with pytest.raises(RuntimeError, match="No certified optimum"):
        weighted_sum_reference(_instance(0))
