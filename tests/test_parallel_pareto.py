"""pareto.parallel_pareto_frontier: the same exact front as the serial box
search (true_pareto_frontier), on the HiGHS-backed lpfirst engine, so no
Gurobi licence is needed."""
import random

import pytest

from conftest import trap_small
from fedhpc import lp_backend, lpfirst, pareto
from fedhpc.data import Instance

pytestmark = pytest.mark.skipif(not lp_backend.available("highs"), reason="HiGHS not installed")

OPTS = lpfirst.LPFirstOptions(backend="highs", lns_sweeps=0)


def _random_instance(seed: int, n: int = 9) -> Instance:
    """Two scarce on-prem types (running jobs at t=0) plus a cheap slow cloud
    and a dear fast one: a front with several nonsupported points."""
    rng = random.Random(seed)

    def itype(id, kind, perf, deploy, cost_vm, capacity):
        return dict(id=id, kind=kind, perf=perf, io_time=0.0, deploy=deploy, cost_io=0.0,
                    cost_vm=cost_vm, cost_stor=0.0, cpu=64, mem=512, stor=1000,
                    capacity=capacity)
    return Instance.from_dict(dict(
        horizon=40,
        instance_types=[itype(0, "on-prem", 1.0, 0.0, 0.0, 2),
                        itype(1, "on-prem", 1.25, 0.0, 0.0, 1),
                        itype(2, "cloud", 1.0, 1.0, 0.2 + 0.1 * rng.random(), None),
                        itype(3, "cloud", 2.0, 2.0, 0.9 + 0.2 * rng.random(), None)],
        jobs=[dict(id=j, arrival=round(rng.uniform(0, 8), 2),
                   exec_time=round(rng.uniform(1, 9), 2), io_volume=0, cpu=16, mem=64,
                   stor=10, weight=rng.randint(1, 3)) for j in range(n)],
        running_jobs=[dict(id=100, type_id=0, end=3), dict(id=101, type_id=1, end=5)],
    ))


def _points(front):
    return sorted((round(s.f1, 6), round(s.f2, 6)) for s in front)


@pytest.mark.parametrize("make", [trap_small, *(lambda s=s: _random_instance(s) for s in range(3))])
def test_parallel_front_equals_serial(make):
    inst = make()
    serial = pareto.true_pareto_frontier(inst, engine="lpfirst", lpfirst_options=OPTS)
    st = {}
    par = pareto.parallel_pareto_frontier(inst, 3, lpfirst_options=OPTS, stats=st)
    assert _points(par) == _points(serial)
    assert st["n_unresolved"] == 0 and st["open_boxes"] == 0
    assert all(s.status == "optimal" for s in par)
    for s in par:
        assert not inst.schedule_violations(s.assignment)


@pytest.mark.parametrize("make", [trap_small, *(lambda s=s: _random_instance(s) for s in range(3))])
def test_parallel_epsilon_front_equals_serial(make):
    """Single-phase min-f1 ε-constraint search with lazy confirmation."""
    inst = make()
    serial = pareto.true_pareto_frontier(inst, engine="lpfirst", lpfirst_options=OPTS)
    st = {}
    par = pareto.parallel_epsilon_frontier(inst, 3, lpfirst_options=OPTS, stats=st)
    assert _points(par) == _points(serial)
    assert st["n_unresolved"] == 0 and st["open_boxes"] == 0
    assert all(s.status == "optimal" for s in par)
    for s in par:
        assert not inst.schedule_violations(s.assignment)


@pytest.mark.parametrize("search", [pareto.parallel_pareto_frontier, pareto.parallel_epsilon_frontier])
def test_hard_time_limit_kills_overrunning_solves(search):
    """A solve past hard_time_limit is killed and the pool rebuilt; the search
    returns (a partial front) instead of waiting for it."""
    st = {}
    front = search(trap_small(), 2, lpfirst_options=OPTS, hard_time_limit=0.01, stats=st)
    assert st["n_restarts"] >= 1
    assert front == [] or all(s.status == "optimal" for s in front)



@pytest.mark.parametrize("direction", ["f1", "f2"])
@pytest.mark.parametrize("make", [trap_small, *(lambda s=s: _random_instance(s) for s in range(3))])
def test_parallel_epsilon_fixed_direction(make, direction):
    """Q-only and M-only searches (all candidates of one kind) give the
    exact front too."""
    inst = make()
    serial = pareto.true_pareto_frontier(inst, engine="lpfirst", lpfirst_options=OPTS)
    st = {}
    par = pareto.parallel_epsilon_frontier(inst, 3, lpfirst_options=OPTS, stats=st,
                                           direction=direction)
    assert _points(par) == _points(serial)
    assert st["n_unresolved"] == 0 and st["open_boxes"] == 0
    assert st["n_q" if direction == "f2" else "n_m"] == 0


@pytest.mark.parametrize("seed", range(3, 9))
def test_parallel_epsilon_random_direction(seed):
    """Q and M solves mixed at random: candidates of both kinds meet in the
    same boxes, and the front is still exact."""
    inst = _random_instance(seed, n=10)
    serial = pareto.true_pareto_frontier(inst, engine="lpfirst", lpfirst_options=OPTS)
    st = {}
    par = pareto.parallel_epsilon_frontier(inst, 3, lpfirst_options=OPTS, stats=st,
                                           direction="random")
    assert _points(par) == _points(serial)
    assert st["n_unresolved"] == 0 and st["open_boxes"] == 0
