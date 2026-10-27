"""SpaceTimeFormulation dominance reductions (R1/R2/R3/lex) must leave every
optimum unchanged, and must switch off on negative data."""
import random

import pytest

from fedhpc import formulations
from fedhpc.data import Instance, InstanceType, Job, RunningJob
from fedhpc.formulations import LEX_F1, SpaceTimeFormulation, unbounded_types
from fedhpc.model import (
    solve_epsilon_cost,
    solve_epsilon_turnaround,
    solve_f1,
    solve_f2,
    solve_weighted_sum,
)
from fedhpc.pareto import _reference_points

formulations.configure_env()

PLAIN = dict(collapse_unbounded=False, pareto_unbounded=False,
             rejection_deadlines=False, lex_deadlines=False)


def _seeds(n):
    """Seeds 0..n-1; seeds 1 and 2 build unreduced models beyond Gurobi's
    size-limited licence, so they are marked needs_full_license."""
    return [pytest.param(s, marks=pytest.mark.needs_full_license) if s in (1, 2) else s
            for s in range(n)]


def _type(id, kind, perf, deploy, cost_vm, capacity, fixed_cost=0.0):
    big = 1e9
    return InstanceType(id=id, kind=kind, perf=perf, io_time=0.0, deploy=deploy,
                        cost_io=0.0, cost_vm=cost_vm, cost_stor=0.0, cpu=big,
                        mem=big, stor=big, capacity=capacity, fixed_cost=fixed_cost)


def _instance(seed: int, fixed: bool = False) -> Instance:
    """Two congested on-prem types (one with a running job), a large on-prem
    type that never binds, and three cloud types trading speed for cost."""
    rng = random.Random(seed)
    types = [
        _type(0, "on-prem", 1.0, 0.0, 0.0, 1),
        _type(1, "on-prem", 0.8, 0.0, 0.0, 1),
        _type(2, "on-prem", 0.5, 0.0, 0.0, 50),          # unbounded in effect
        _type(3, "cloud", 1.5, 1.0, 1.0, None),
        _type(4, "cloud", 1.0, 1.0, 0.4, None, fixed_cost=2.0 if fixed else 0.0),
        _type(5, "cloud", 0.8, 1.0, 0.5, None),            # dominated by 4 w/o U
    ]
    jobs = [Job(id=j, arrival=float(rng.randint(0, 8)), exec_time=float(rng.randint(1, 6)),
                io_volume=0.0, cpu=0.0, mem=0.0, stor=0.0,
                weight=float(rng.randint(1, 4)))
            for j in range(8)]
    running = [RunningJob(id=100, type_id=0, end=rng.randint(1, 5))]
    inst = Instance(jobs=jobs, instance_types=types, horizon=45,
                    running_jobs=running)
    inst.build()
    return inst


def _same(a, b):
    assert a.status == b.status == "optimal"
    assert a.f1 == pytest.approx(b.f1)
    assert a.f2 == pytest.approx(b.f2)


def test_unbounded_types():
    assert unbounded_types(_instance(0)) == {2, 3, 4, 5}


def test_reductions_shrink_model():
    inst = _instance(0)
    sizes = []
    for objective in (None, (1.0, 1.0), LEX_F1):
        mdl, _ = SpaceTimeFormulation().build(inst, objective=objective)
        mdl.update()
        sizes.append(mdl.NumVars)
        assert mdl.getConstrByName("budget") is None
    plain, _ = SpaceTimeFormulation(**PLAIN).build(inst)
    plain.update()
    assert plain.getConstrByName("budget") is None
    assert plain.NumVars > sizes[0] > max(sizes[1:])


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("seed", _seeds(4))
def test_lexicographic_optima_unchanged(seed, fixed):
    inst = _instance(seed, fixed=fixed)
    _same(solve_f1(inst, formulation=SpaceTimeFormulation(), MIPGap=0),
          solve_f1(inst, formulation=SpaceTimeFormulation(**PLAIN), MIPGap=0))
    _same(solve_f2(inst, formulation=SpaceTimeFormulation(), MIPGap=0),
          solve_f2(inst, formulation=SpaceTimeFormulation(**PLAIN), MIPGap=0))


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("seed", _seeds(4))
def test_weighted_optima_unchanged(seed, fixed):
    inst = _instance(seed, fixed=fixed)
    for lam in (0.0, 0.2, 0.5, 0.8, 1.0):
        # f1_0 deliberately tight: R3 must stay exact under a binding f1 cap
        kw = dict(lam=lam, f1_T=40.0, f2_T=8.0, f1_0=170.0, MIPGap=0)
        red = solve_weighted_sum(inst, formulation=SpaceTimeFormulation(), **kw)
        pln = solve_weighted_sum(inst, formulation=SpaceTimeFormulation(**PLAIN), **kw)
        assert red.status == pln.status, lam
        if pln.status == "optimal":
            assert red.objective == pytest.approx(pln.objective, rel=1e-9, abs=1e-9), lam


@pytest.mark.parametrize("seed", _seeds(3))
def test_epsilon_optima_unchanged(seed):
    inst = _instance(seed, fixed=True)
    for eps in (0.0, 3.0, 10.0):
        # min f1 s.t. f2 <= eps has no f2 tie-break: equally optimal schedules
        # may differ in f2, so only f1 and the cap are comparable.
        red = solve_epsilon_cost(inst, eps, formulation=SpaceTimeFormulation(), MIPGap=0)
        pln = solve_epsilon_cost(inst, eps, formulation=SpaceTimeFormulation(**PLAIN), MIPGap=0)
        assert red.status == pln.status == "optimal"
        assert red.f1 == pytest.approx(pln.f1)
        assert red.f2 <= eps + 1e-9 and pln.f2 <= eps + 1e-9
    f1_star = solve_f1(inst, formulation=SpaceTimeFormulation(**PLAIN), MIPGap=0).f1
    for slack in (0.0, 5.0, 30.0):
        red = solve_epsilon_turnaround(inst, f1_star + slack,
                                       formulation=SpaceTimeFormulation(), MIPGap=0)
        pln = solve_epsilon_turnaround(inst, f1_star + slack,
                                       formulation=SpaceTimeFormulation(**PLAIN), MIPGap=0)
        assert red.f2 == pytest.approx(pln.f2)


@pytest.mark.parametrize("seed", _seeds(3))
def test_reference_points_unchanged(seed):
    inst = _instance(seed)
    red = _reference_points(inst, formulation=SpaceTimeFormulation(), MIPGap=0)
    pln = _reference_points(inst, formulation=SpaceTimeFormulation(**PLAIN), MIPGap=0)
    assert red == pytest.approx(pln)


@pytest.mark.parametrize("seed", _seeds(3))
def test_zero_cost_cap_optimum_unchanged(seed):
    inst = _instance(seed, fixed=True)
    _same(solve_epsilon_cost(inst, 0.0, formulation=SpaceTimeFormulation(), MIPGap=0),
          solve_epsilon_cost(inst, 0.0, formulation=SpaceTimeFormulation(**PLAIN), MIPGap=0))


@pytest.mark.parametrize("opts", [dict(normal_patterns=True),
                                  dict(compressed_flow=True),
                                  dict(compressed_flow=True, cover_ge=True)])
@pytest.mark.parametrize("seed", _seeds(2))
def test_reductions_compose_with_flow_options(seed, opts):
    inst = _instance(seed, fixed=True)
    _same(solve_f1(inst, formulation=SpaceTimeFormulation(**opts), MIPGap=0),
          solve_f1(inst, formulation=SpaceTimeFormulation(**PLAIN), MIPGap=0))
    kw = dict(lam=0.5, f1_T=40.0, f2_T=8.0, f1_0=400.0, MIPGap=0)
    red = solve_weighted_sum(inst, formulation=SpaceTimeFormulation(**opts), **kw)
    pln = solve_weighted_sum(inst, formulation=SpaceTimeFormulation(**PLAIN), **kw)
    assert red.objective == pytest.approx(pln.objective, rel=1e-9, abs=1e-9)


def test_lex_rule_exempts_zero_weight_jobs():
    """A weight-0 job may finish late on free on-prem capacity instead of
    paying for cloud; the lex deadline must not cut that option."""
    types = [_type(0, "on-prem", 1.0, 0.0, 0.0, 1), _type(1, "cloud", 1.0, 0.0, 1.0, None)]
    jobs = [Job(id=j, arrival=0.0, exec_time=4.0, io_volume=0.0, cpu=0.0, mem=0.0,
                stor=0.0, weight=w) for j, w in enumerate((1.0, 0.0))]
    inst = Instance(jobs=jobs, instance_types=types, horizon=20)
    inst.build()
    sol = solve_f1(inst, formulation=SpaceTimeFormulation(), MIPGap=0)
    assert (sol.f1, sol.f2) == pytest.approx((4.0, 0.0))
    _same(sol, solve_f1(inst, formulation=SpaceTimeFormulation(**PLAIN), MIPGap=0))


def test_negative_data_builds_plain_model():
    inst = _instance(0)
    inst.jobs[0].weight = -1.0
    red, v = SpaceTimeFormulation().build(inst, objective=LEX_F1)
    pln, _ = SpaceTimeFormulation(**PLAIN).build(inst, objective=LEX_F1)
    red.update(), pln.update()
    assert red.NumVars == pln.NumVars and not v["objective_reduced"]
