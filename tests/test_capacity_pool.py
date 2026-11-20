"""CapacityPool: a capacity shared by several capacity=None types (a cap on all cloud VMs).

Every feasible schedule of a tiny instance is enumerated directly (pool checked slot by slot, options
crossing the horizon counted up to H) and the SpaceTimeFormulation with its reductions must hit the same
lex optima and weighted optima.
"""
import math
import random

import pytest
from gurobipy import GRB

from fedhpc import formulations
from fedhpc.data import CapacityPool, Instance, InstanceType, Job, RunningJob, ScheduleError
from fedhpc.formulations import SpaceTimeFormulation, binding_pools, unbounded_types
from fedhpc.model import solve_f1, solve_f2, solve_weighted_sum

formulations.configure_env()


def _type(id, perf, cost_vm, capacity):
    big = 1e9
    return InstanceType(id=id, kind="on-prem" if cost_vm == 0 else "cloud", perf=perf, io_time=0.0, deploy=0.0,
                        cost_io=0.0, cost_vm=cost_vm, cost_stor=0.0, cpu=big, mem=big, stor=big, capacity=capacity)


def _instance(seed: int, cap: int, n_jobs: int = 4, horizon: int = 12, window=None) -> Instance:
    rng = random.Random(seed)
    types = [
        _type(0, 1.0, 0.0, 2),
        _type(1, 2.0, 1.0, None),
        _type(2, 1.5, 0.5, None),
        _type(3, 1.5, 0.6, None),   # dominated by type 2 in the pool (same p_occ, dearer)
    ]
    jobs = [Job(id=j, arrival=round(rng.uniform(0, 2), 2), exec_time=float(rng.randint(1, 5)), io_volume=0.0,
                cpu=0.0, mem=0.0, stor=0.0, weight=float(rng.choice((0, 1, 1, 2))),
                nodes=rng.choice((1, 1, 2)))
            for j in range(n_jobs)]
    occupied = {t: 1 for t in range(rng.randint(0, 3))}
    inst = Instance(jobs=jobs, instance_types=types, horizon=horizon,
                    running_jobs=[RunningJob(id=100, type_id=0, end=rng.randint(1, 3))],
                    pools=[CapacityPool(type_ids=frozenset({1, 2, 3}), capacity=cap, occupied=occupied,
                                        start_window=window)])
    inst.build()
    return inst


def _enumerate(inst: Instance) -> list[tuple[float, float]]:
    """(f1, f2) of every feasible schedule (brute force, no reductions)."""
    H = inst.horizon
    pool = inst.pools[0]
    onprem = [inst.occupied.get((0, t), 0) for t in range(H)]
    use = [pool.occupied.get(t, 0) for t in range(H)]
    opts = [[(m, t) for m in inst.F[j.id] for t in inst.T[j.id, m]] for j in inst.jobs]
    out = []

    def dfs(k, f1, f2):
        if k == len(opts):
            out.append((f1, f2))
            return
        j = inst.jobs[k]
        for m, t in opts[k]:
            end = t + inst.p_occ[j.id, m]
            span = range(t, min(H, end))
            if m == 0:
                if end > H or any(onprem[s] + j.nodes > 2 for s in span):
                    continue
                arr = onprem
            else:
                if any(use[s] + j.nodes > pool.capacity for s in span):
                    continue
                arr = use
            for s in span:
                arr[s] += j.nodes
            dfs(k + 1, f1 + j.weight * (end - j.arrival), f2 + inst.c[j.id, m])
            for s in span:
                arr[s] -= j.nodes

    dfs(0, 0.0, 0.0)
    return out


def _lex(points, first):
    a = min(p[first] for p in points)
    return a, min(p[1 - first] for p in points if math.isclose(p[first], a, abs_tol=1e-6))


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("cap", [1, 2, 3])
def test_pool_matches_brute_force(seed, cap):
    inst = _instance(seed, cap)
    points = _enumerate(inst)
    assert points, "the generator always leaves a feasible schedule"
    s1 = solve_f1(inst)
    assert (s1.f1, s1.f2) == pytest.approx(_lex(points, 0), abs=1e-6)
    s2 = solve_f2(inst)
    assert (s2.f2, s2.f1) == pytest.approx(_lex(points, 1), abs=1e-6)
    inst.check_schedule(s1.assignment)
    inst.check_schedule(s2.assignment)
    f1_T, f2_T = _lex(points, 0)
    f2_0, f1_0 = _lex(points, 1)
    for lam in (0.25, 0.5, 0.75):
        sw = solve_weighted_sum(inst, lam, f1_T, f2_T, f1_0, f2_0=f2_0)
        scale = lambda p: (lam * (p[0] - f1_T) / max(1e-9, f1_0 - f1_T)
                           + (1 - lam) * (p[1] - f2_0) / max(1e-9, f2_T - f2_0))
        if f1_0 - f1_T > 1e-9 and f2_T - f2_0 > 1e-9:
            # Gurobi's default MIPGap (1e-4) bounds the scalarised optimum.
            assert scale((sw.f1, sw.f2)) == pytest.approx(min(scale(p) for p in points), rel=2e-4, abs=2e-4)
        inst.check_schedule(sw.assignment)


def test_pool_binds_and_options_cross_horizon():
    inst = _instance(3, cap=1)
    assert binding_pools(inst)
    assert not unbounded_types(inst) & {1, 2, 3}
    # A pooled option may end after H: its starts run up to the window, not H - p_occ.
    j = max(inst.jobs, key=lambda j: inst.p_occ[j.id, 1])
    assert inst.T[j.id, 1].stop == inst.horizon


def test_non_binding_pool_is_the_unbounded_model():
    """A pool with room for every job changes nothing: same optima as without it."""
    inst = _instance(5, cap=100)
    assert not binding_pools(inst)
    free = Instance(jobs=inst.jobs, instance_types=inst.instance_types, horizon=inst.horizon,
                    running_jobs=inst.running_jobs)
    free.build()
    for solve in (solve_f1, solve_f2):
        a, b = solve(inst), solve(free)
        assert (a.f1, a.f2) == pytest.approx((b.f1, b.f2))


def test_dominated_pool_type_dropped():
    inst = _instance(1, cap=1)
    mdl, v = SpaceTimeFormulation().build(inst)
    assert not any(m == 3 for (_j, m, _t) in v["x"])
    assert any(m == 2 for (_j, m, _t) in v["x"])


def test_pool_row_count_and_schedule_check():
    inst = _instance(2, cap=1)
    mdl, _ = SpaceTimeFormulation().build(inst)
    mdl.update()
    assert any(c.ConstrName.startswith("pool[") for c in mdl.getConstrs())
    # Two cloud jobs at once break a capacity-1 pool.
    a, b = inst.jobs[:2]
    with pytest.raises(ScheduleError):
        inst.check_schedule({a.id: (2, inst.T[a.id, 2].start + 3), b.id: (2, inst.T[b.id, 2].start + 3)} |
                            {j.id: (1, inst.T[j.id, 1].start) for j in inst.jobs[2:]}, complete=False)


def test_pool_member_must_be_unlimited():
    inst = _instance(0, cap=1)
    bad = Instance(jobs=inst.jobs, instance_types=inst.instance_types, horizon=12,
                   pools=[CapacityPool(type_ids=frozenset({0, 1}), capacity=2)])
    with pytest.raises(ValueError):
        bad.build()
