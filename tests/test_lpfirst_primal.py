"""lpfirst primal heuristics (BPC plan Phase 7), job aggregation in its MIPs
(Phase 8) and objective scaling. The checks:

- α-point schedules and fix-and-solve / LNS results are feasible and never
  change the proven optimum;
- the aggregated MIP (identical jobs share a coverage row) gives the same
  optimum as the per-job MIP, with and without fixed jobs and bound rows;
- tiny normalised weights (the weighted sums' ~1e-5) still give the exact
  optimum, which an unscaled MIP misses.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

from conftest import TRAP_INSTANCES
from fedhpc import bpc, lpfirst
from fedhpc.data import Instance, InstanceType, Job
from fedhpc.formulations import configure_env

configure_env()

HEUR_ONLY = lpfirst.LPFirstOptions(rmp_mip_time=0.0, max_eligible=0)
NO_HEUR = lpfirst.LPFirstOptions(heuristics=False)


def _dup_instance(seed: int, n_kinds: int = 4, copies: int = 3, cap: int = 2) -> Instance:
    """``n_kinds`` job kinds, ``copies`` of each with arrivals in the same
    slot but different fractions (so their f1 coefficients differ by a
    constant), on a small on-prem pool plus a priced cloud."""
    rng = random.Random(seed)
    jobs = []
    for k in range(n_kinds):
        a, e, w = rng.randint(0, 3), rng.randint(2, 6), float(rng.randint(1, 3))
        for c in range(copies):
            jobs.append(Job(id=len(jobs), arrival=a - 0.1 * c, exec_time=e, io_volume=0,
                            cpu=2, mem=4, stor=20, weight=w))
    types = [InstanceType(id=0, kind="on-prem", perf=1.0, io_time=0.0, deploy=0, cost_io=0.0,
                          cost_vm=0.0, cost_stor=0.0, cpu=4, mem=8, stor=50, capacity=cap),
             InstanceType(id=1, kind="cloud", perf=1.0, io_time=0.0, deploy=0, cost_io=0.0,
                          cost_vm=2.5, cost_stor=0.0, cpu=4, mem=8, stor=50, capacity=None)]
    inst = Instance(jobs=jobs, instance_types=types, horizon=60)
    inst.build()
    return inst


def _cases():
    return {**{k: f() for k, f in TRAP_INSTANCES.items()},
            **{f"dup{s}": _dup_instance(s) for s in (0, 1, 2)}}


def test_classes_group_identical_jobs():
    inst = _dup_instance(0)
    ti = lpfirst._TI(inst, bpc._prepare(inst))
    cls, canon = ti.classes()
    assert len(set(cls.tolist())) == 4
    # a placement and its canonical twin: same type, start, duration and cost
    for a in range(ti.n_arcs):
        c = canon[a]
        assert (ti.MID[a], ti.T[a], ti.P[a], ti.C2[a]) == (ti.MID[c], ti.T[c], ti.P[c], ti.C2[c])
        assert ti.C1[a] == pytest.approx(ti.C1[c] + ti.class_offset[ti.J[a]])


@pytest.mark.parametrize("name", ["dup0", "dup1", "dup2", "trap_medium"])
@pytest.mark.parametrize("w", [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
def test_aggregated_mip_matches_per_job_mip(name, w):
    inst = _cases()[name]
    ti = lpfirst._TI(inst, bpc._prepare(inst))
    allarcs = list(range(ti.n_arcs))
    step = bpc._objective_step(inst, *w)
    s1, o1, a1 = lpfirst._mip(ti, allarcs, *w, None, None, float("inf"), 60, step, 1,
                              aggregate=False)
    s2, o2, a2 = lpfirst._mip(ti, allarcs, *w, None, None, float("inf"), 60, step, 1,
                              aggregate=True)
    assert s1 == s2 == "optimal"
    assert o1 == pytest.approx(o2, abs=1e-6)
    inst.check_schedule(a2)
    assert lpfirst._objective_of(inst, a2, *w) == pytest.approx(o2, abs=1e-6)
    # with half the jobs fixed at the optimum and bound rows at its values
    rng = random.Random(1)
    opt_arcs = ti.arcs_of(a1)
    fixed = [a for a in opt_arcs if rng.random() < 0.5]
    f1, f2 = (inst.f1_of({j: t + inst.p_occ[j, m] for j, (m, t) in a1.items()}), inst.f2_of(a1))
    rest = [a for a in allarcs if ti.J[a] not in set(ti.J[fixed])]
    for agg in (False, True):
        s, o, a = lpfirst._mip(ti, rest, *w, f1, f2, float("inf"), 60, step, 1, fixed=fixed,
                               aggregate=agg)
        assert s == "optimal" and o == pytest.approx(o1, abs=1e-6)
        inst.check_schedule(a)


@pytest.mark.parametrize("name", ["dup0", "dup1", "dup2", "trap_small", "trap_medium"])
@pytest.mark.parametrize("w", [(1.0, 0.0), (1.0, 1.0)])
def test_heuristics_feasible_and_exact(name, w):
    inst = _cases()[name]
    ref, _ = bpc._branch_and_price(inst, *w)
    seeds = bpc.heuristic_schedules(inst)
    worst = max(seeds, key=lambda a: lpfirst._objective_of(inst, a, *w))
    st: dict = {}
    res = lpfirst.solve_lpfirst(inst, *w, seed_schedules=[worst], stats=st)
    assert res.status == "optimal"
    assert res.objective == pytest.approx(ref, abs=1e-6)
    inst.check_schedule(res.assignment)
    assert st["lns_obj"] >= ref - 1e-6
    # without any proof step, whatever the heuristics return is feasible and
    # no worse than the seed
    res2 = lpfirst.solve_lpfirst(inst, *w, seed_schedules=[worst], options=HEUR_ONLY)
    inst.check_schedule(res2.assignment)
    assert res2.objective <= lpfirst._objective_of(inst, worst, *w) + 1e-6
    assert res2.objective >= ref - 1e-6


@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 0.75])
@pytest.mark.parametrize("name", ["dup0", "dup1", "trap_medium"])
def test_alpha_schedule_is_feasible(name, alpha):
    inst = _cases()[name]
    ti = lpfirst._TI(inst, bpc._prepare(inst))
    st: dict = {}
    lpfirst.solve_lpfirst(inst, 1.0, 0.0, ti=ti, seed_schedules=[], stats=st,
                          options=lpfirst.LPFirstOptions(heuristics=False, rmp_mip_time=0.0,
                                                         max_eligible=0))
    rng = np.random.default_rng(3)
    x = rng.random(ti.n_arcs) * (rng.random(ti.n_arcs) < 0.2)
    sums = np.add.reduceat(x, ti.first)
    x = np.where(sums[ti.J] > 0, x / np.where(sums[ti.J] > 0, sums[ti.J], 1), 0)
    a = lpfirst._alpha_schedule(ti, x, alpha)
    if a is not None:
        assert not inst.schedule_violations(a)
        assert len(a) == len(inst.jobs)


@pytest.mark.parametrize("name", ["dup0", "dup2", "trap_medium"])
def test_tiny_weights_still_exact(name):
    """Weights of the normalised weighted sum's magnitude: the optimum must
    match the one at unit weights of the same ratio."""
    inst = _cases()[name]
    w1, w2 = 1.3e-5, 2.1e-5
    ref = lpfirst.solve_lpfirst(inst, 1.3, 2.1, options=NO_HEUR)
    res = lpfirst.solve_lpfirst(inst, w1, w2, options=NO_HEUR)
    assert res.status == ref.status == "optimal"
    assert res.objective == pytest.approx(ref.objective * 1e-5, rel=1e-9, abs=1e-12)


def _gap(seed):
    from test_lpfirst import _gap_instance
    return _gap_instance(seed)


@pytest.mark.parametrize("node_eligible", [0, 8])
@pytest.mark.parametrize("name", ["gap0", "gap5", "gap24", "dup0", "dup1", "dup2", "trap_medium"])
@pytest.mark.parametrize("w", [(1.0, 1.0), (1.0, 0.0), (0.3, 0.7)])
def test_tree_is_exact(name, w, node_eligible):
    """Phase 9: with the heuristics and the RMP MIP off and a poor incumbent,
    the placement tree alone (pure branching, or branching down to small
    node MIPs) must reach the branch-and-price optimum."""
    inst = _gap(int(name[3:])) if name.startswith("gap") else _cases()[name]
    ref, _ = bpc._branch_and_price(inst, *w)
    seeds = bpc.heuristic_schedules(inst)
    worst = max(seeds, key=lambda a: lpfirst._objective_of(inst, a, *w))
    opts = lpfirst.LPFirstOptions(heuristics=False, rmp_mip_time=0.0, tree=True,
                                  max_eligible=0, node_eligible=node_eligible)
    st: dict = {}
    res = lpfirst.solve_lpfirst(inst, *w, seed_schedules=[worst], options=opts, stats=st)
    assert res.status == "optimal"
    assert res.objective == pytest.approx(ref, abs=1e-6)
    inst.check_schedule(res.assignment)
    assert res.bound <= res.objective + 1e-6


@pytest.mark.parametrize("name", ["gap0", "gap5"])
def test_tree_with_bound_rows(name):
    """Bound rows active (a Pareto box): f2 capped below its free optimum."""
    inst = _gap(int(name[3:]))
    free = lpfirst.solve_lpfirst(inst, 1.0, 0.0, options=NO_HEUR)
    cap = free.assignment and inst.f2_of(free.assignment) * 0.5
    ref, _ = bpc._branch_and_price(inst, 1.0, 0.0, bound2_value=cap)
    opts = lpfirst.LPFirstOptions(heuristics=False, rmp_mip_time=0.0, max_eligible=0,
                                  node_eligible=0)
    res = lpfirst.solve_lpfirst(inst, 1.0, 0.0, None, cap, options=opts, seed_schedules=[])
    if ref == float("inf") or ref is None:
        assert res.status != "optimal" or res.assignment is None
    else:
        assert res.status == "optimal" and res.objective == pytest.approx(ref, abs=1e-6)
        assert inst.f2_of(res.assignment) <= cap + 1e-6
