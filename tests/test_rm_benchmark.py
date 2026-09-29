"""Weighted turnaround + fixed usage costs, checked on Rm||TWCT+TMUC instances
whose optima are published (Chen et al., EJOR 2024; values brute-force verified)."""
import itertools

import pytest

from fedhpc import rm_benchmark as rm
from fedhpc.model import Solution

# Benchmarks/V1_F1/M2_N10_V1_F1_1.txt — published optimum 145220
M2N10_1 = """66 82
5 44
22 69
72 14
18 90
74 1
93 75
15 44
48 66
55 11
52 43 96 2 58 28 83 89 87 5
58625 35867
56 60
"""


def _brute(data):
    n, m = len(data[1]), len(data[2])
    return min(rm.evaluate(data, list(a)) for a in itertools.product(range(m), repeat=n))


def test_evaluate_matches_published_optimum():
    assert _brute(rm.parse(M2N10_1)) == 145220


@pytest.mark.parametrize("use_hint", [False, True])
def test_mip_reaches_published_optimum(use_hint):
    data = rm.parse(M2N10_1)
    p = data[0]
    inst = rm.to_instance(data, horizon=sum(max(r) for r in p))
    hint = None
    if use_hint:
        asg = rm.wspt_schedule(data, [0] * len(p))   # everything on machine 0
        hint = Solution("hint", None, None, None, asg, {})
    sol, bound = rm.solve_total_cost(inst, hint=hint, MIPGap=0)
    assert sol.status == "optimal"
    assert sol.f1 + sol.f2 == pytest.approx(145220)
    assert bound == pytest.approx(145220)
    # the MIP's own schedule, re-scored independently
    machine_of = [sol.assignment[j][0] for j in range(len(p))]
    assert rm.evaluate(data, machine_of) == pytest.approx(145220)


def test_fixed_cost_only_charged_for_used_types():
    data = ([[3, 50], [4, 60]], [1, 1], [1000, 5], [1, 1])
    inst = rm.to_instance(data, horizon=200)
    # machine 1 is slow but its fixed cost is tiny; machine 0's fee dominates
    sol, _ = rm.solve_total_cost(inst, MIPGap=0)
    assert {m for m, _t in sol.assignment.values()} == {1}
    assert sol.f2 == pytest.approx(5 + 110)
    assert sol.f1 == pytest.approx(50 + 110)


def test_unit_weights_and_no_fixed_cost_keep_old_objective():
    data = ([[3, 50], [4, 60]], [1, 1], [0, 0], [1, 1])
    inst = rm.to_instance(data, horizon=200)
    assert not inst.has_fixed_costs
    sol, _ = rm.solve_total_cost(inst, MIPGap=0)
    assert sol.f1 == pytest.approx(3 + 7) and sol.f2 == pytest.approx(7)


def test_cpp_heuristics_refuse_fixed_costs():
    from fedhpc import moea
    inst = rm.to_instance(rm.parse(M2N10_1), horizon=500)
    with pytest.raises(NotImplementedError):
        moea._job_slots(inst)


def test_machine_horizons_are_valid_and_keep_optimum():
    data = rm.parse(M2N10_1)
    p = data[0]
    ub = 145220 * 1.2
    Hk = rm.machine_horizons(data, ub)
    # every assignment with cost <= ub has per-machine load within H_k
    for a in itertools.product(range(2), repeat=len(p)):
        if rm.evaluate(data, list(a)) <= ub:
            for k in range(2):
                assert sum(p[j][k] for j in range(len(p)) if a[j] == k) <= Hk[k]
    assert Hk[0] < sum(r[0] for r in p) or Hk[1] < sum(r[1] for r in p)
    sol, _ = rm.solve_total_cost(rm.to_instance(data, Hk), MIPGap=0)
    assert sol.f1 + sol.f2 == pytest.approx(145220)
