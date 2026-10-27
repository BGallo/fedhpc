"""The trap instances (tests/conftest.py) are hard because their optimum is
non-obvious, not because they are large: every list-scheduling baseline
lands well above the exact turnaround anchors, and the exact front is dense.
These checks pin that property so the fixtures cannot silently degrade into
instances a greedy rule solves.
"""
import pytest

from conftest import TRAP_INSTANCES
from fedhpc import baselines
from fedhpc.model import solve_f1, solve_f2
from fedhpc.pareto import true_pareto_frontier

_GUROBI = {"OutputFlag": 0}

# name: (min relative gap of any baseline above f1_T,
#        min relative gap of any free baseline above f1_0, front size)
_EXPECTED = {
    "trap_small": (0.09, 0.24, 10),
    "trap_medium": (0.14, 0.81, 27),
}


def _baselines(inst):
    sols = [baselines.onprem_only_spt(inst)]
    sols += [baselines.greedy_earliest_completion(inst, order=o) for o in ("spt", "arrival")]
    sols += [baselines.threshold_bursting(inst, theta) for theta in range(8)]
    return [s for s in sols if len(s.assignment) == len(inst.jobs)]


@pytest.mark.parametrize("name", sorted(TRAP_INSTANCES))
def test_baselines_miss_the_exact_anchors(name):
    inst = TRAP_INSTANCES[name]()
    gap_T, gap_0, _ = _EXPECTED[name]
    f1_T = solve_f1(inst, MIPGap=0, **_GUROBI).f1
    cheapest = solve_f2(inst, MIPGap=0, **_GUROBI)
    assert cheapest.f2 == pytest.approx(0.0, abs=1e-9)  # a free schedule exists
    sols = _baselines(inst)
    assert min(s.f1 for s in sols) >= f1_T * (1 + gap_T) - 1e-6
    free = [s.f1 for s in sols if s.f2 <= 1e-9]
    assert free and min(free) >= cheapest.f1 * (1 + gap_0) - 1e-6


@pytest.mark.parametrize("name", sorted(TRAP_INSTANCES))
def test_front_is_dense(name):
    inst = TRAP_INSTANCES[name]()
    assert len(true_pareto_frontier(inst, **_GUROBI)) == _EXPECTED[name][2]


@pytest.mark.parametrize("name", sorted(TRAP_INSTANCES))
def test_running_jobs_shape_the_optimum(name):
    """Dropping the running jobs lowers both turnaround anchors, so solvers
    that ignore them fail on these instances."""
    inst = TRAP_INSTANCES[name]()
    assert inst.running_jobs
    free = TRAP_INSTANCES[name]()
    free.running_jobs = []
    free.n_running, free.rj_arrivals, free.occupied = {}, {}, {}
    for solve in (solve_f1, solve_f2):
        assert solve(free, **_GUROBI).f1 < solve(inst, **_GUROBI).f1 - 1e-6
