"""SpaceTimeFormulation reduction options
must leave every optimum unchanged."""
import itertools
import random

import pytest

from fedhpc import formulations
from fedhpc.data import Instance, InstanceType, Job, RunningJob
from fedhpc.formulations import SpaceTimeFormulation, normal_pattern_starts
from fedhpc.model import solve_f1, solve_weighted_sum

formulations.configure_env()


def _instance(seed: int) -> Instance:
    rng = random.Random(seed)
    big = 1e9
    types = [
        InstanceType(id=0, kind="on-prem", perf=1.0, io_time=0.0, deploy=0.0,
                     cost_io=0.0, cost_vm=0.0, cost_stor=0.0, cpu=big, mem=big,
                     stor=big, capacity=2),
        InstanceType(id=1, kind="cloud", perf=1.5, io_time=0.0, deploy=1.0,
                     cost_io=0.0, cost_vm=1.0, cost_stor=0.0, cpu=big, mem=big,
                     stor=big, capacity=None),
    ]
    jobs = [Job(id=j, arrival=float(rng.randint(0, 12)), exec_time=float(rng.randint(1, 7)),
                io_volume=0.0, cpu=0.0, mem=0.0, stor=0.0,
                weight=float(rng.randint(1, 5)))
            for j in range(9)]
    running = [RunningJob(id=100, type_id=0, end=rng.randint(1, 6))]
    inst = Instance(jobs=jobs, instance_types=types, horizon=40, budget=12.0,
                    running_jobs=running)
    inst.build()
    return inst


COMBOS = [dict(zip(("continuous_idle", "cover_ge", "normal_patterns"), c))
          for c in itertools.product([False, True], repeat=3)]
COMBOS.append(dict(compressed_flow=True))
COMBOS.append(dict(compressed_flow=True, cover_ge=True))


@pytest.mark.parametrize("seed", range(4))
def test_options_preserve_lexicographic_optimum(seed):
    inst = _instance(seed)
    base = solve_f1(inst, formulation=SpaceTimeFormulation(), MIPGap=0)
    assert base.status == "optimal"
    for opts in COMBOS[1:]:
        sol = solve_f1(inst, formulation=SpaceTimeFormulation(**opts), MIPGap=0)
        assert sol.status == "optimal", opts
        assert sol.f1 == pytest.approx(base.f1), opts
        assert sol.f2 == pytest.approx(base.f2), opts


@pytest.mark.parametrize("seed", range(3))
def test_options_preserve_weighted_optimum(seed):
    inst = _instance(seed)
    # reference points on the instance's own scale: a huge f1_0 shrinks the f1
    # coefficient below Gurobi's optimality tolerance and it stops a unit early
    kw = dict(lam=0.5, f1_T=150.0, f2_T=5.0, f1_0=600.0, MIPGap=0)
    base = solve_weighted_sum(inst, formulation=SpaceTimeFormulation(), **kw)
    for opts in COMBOS[1:]:
        sol = solve_weighted_sum(inst, formulation=SpaceTimeFormulation(**opts), **kw)
        assert sol.objective == pytest.approx(base.objective), opts


def test_normal_patterns_contain_release_and_running_slots():
    inst = _instance(0)
    masks = normal_pattern_starts(inst)
    for j in inst.jobs:
        for m in inst.F[j.id]:
            assert masks[m][inst.T[j.id, m].start]
    for (m, v) in inst.rj_arrivals:
        assert masks[m][v]


@pytest.mark.parametrize("seed", range(4))
def test_cpp_graph_builder_matches_python_fallback(seed, monkeypatch):
    inst = _instance(seed)
    compiled = formulations._normal_pattern_graph_data(inst)
    ext = formulations._graph_ext
    assert ext is not None
    monkeypatch.setattr(formulations, "_graph_ext", None)
    fallback = formulations._normal_pattern_graph_data(inst)
    monkeypatch.setattr(formulations, "_graph_ext", ext)
    assert compiled == fallback


def test_python_fallback_running_completion_off_pattern(monkeypatch):
    """A running job completing before every release is an event that no
    allowed start or completion touches; the fallback used to drop it from
    the node set and crash wiring the loss arc that ends there."""
    big = 1e9
    itype = InstanceType(id=0, kind="on-prem", perf=1.0, io_time=0.0, deploy=0.0,
                         cost_io=0.0, cost_vm=0.0, cost_stor=0.0, cpu=big, mem=big,
                         stor=big, capacity=2)
    jobs = [Job(id=j, arrival=5.0, exec_time=2.0, io_volume=0.0, cpu=0.0, mem=0.0,
                stor=0.0) for j in range(2)]
    inst = Instance(jobs=jobs, instance_types=[itype], horizon=10, budget=1e9,
                    running_jobs=[RunningJob(id=9, type_id=0, end=2)])
    inst.build()
    compiled = formulations._normal_pattern_graph_data(inst)
    monkeypatch.setattr(formulations, "_graph_ext", None)
    assert formulations._normal_pattern_graph_data(inst) == compiled
    sol = solve_f1(inst, formulation=SpaceTimeFormulation(compressed_flow=True), MIPGap=0)
    assert sol.status == "optimal"
