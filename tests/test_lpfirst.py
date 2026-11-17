"""LP-first engine (``fedhpc.lpfirst``, BPC plan Phase 5). The checks:

- the time-indexed LP equals the compact model's LP relaxation;
- the Lagrangian bound is valid;
- each proof step (bound, RMP MIP, eligible-placement MIP) gives the
  branch-and-price optimum, with and without bound rows;
- Pareto fronts match the Gurobi engine's.
"""
from __future__ import annotations

import random

import numpy as np
import pytest
from gurobipy import GRB

from conftest import TRAP_INSTANCES
from fedhpc import bpc, lpfirst
from fedhpc.data import Instance, InstanceType, Job
from fedhpc.formulations import SpaceTimeFormulation, configure_env
from fedhpc.pareto import map_pareto_frontier, true_pareto_frontier

configure_env()

_W = [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.3, 0.7)]
NO_MIP = lpfirst.LPFirstOptions(rmp_mip_time=0.0)


def _cases(known_pareto_inst):
    return {**{k: f() for k, f in TRAP_INSTANCES.items()}, "known_pareto": known_pareto_inst}


def _compact_lp(inst, w1, w2) -> float:
    fmt = SpaceTimeFormulation()
    mdl, vars_ = fmt.build(inst)
    for v in mdl.getVars():
        v.vtype = GRB.CONTINUOUS
    x = vars_["x"]
    mdl.setObjective(w1 * fmt.f1_expr(inst, x) + w2 * fmt.f2_expr(inst, x), GRB.MINIMIZE)
    mdl.optimize()
    assert mdl.Status == GRB.OPTIMAL
    return mdl.ObjVal


@pytest.mark.parametrize("name", sorted(TRAP_INSTANCES) + ["known_pareto"])
@pytest.mark.parametrize("w", _W)
def test_lp_equals_compact_lp(name, w, known_pareto_inst):
    """Without start-set reductions the master is the compact LP in
    time-indexed form, so the converged column generation gives its value."""
    inst = _cases(known_pareto_inst)[name]
    ti = lpfirst._TI(inst, bpc._prepare(inst, reduce=False))
    st: dict = {}
    lpfirst.solve_lpfirst(inst, *w, ti=ti, seed_schedules=[], stats=st, options=NO_MIP)
    assert st["lp"] == pytest.approx(_compact_lp(inst, *w), rel=1e-7, abs=1e-6)


@pytest.mark.parametrize("name", sorted(TRAP_INSTANCES) + ["known_pareto"])
@pytest.mark.parametrize("w", _W)
def test_optimum_matches_branch_and_price(name, w, known_pareto_inst):
    inst = _cases(known_pareto_inst)[name]
    ref, _ = bpc._branch_and_price(inst, *w)
    res = lpfirst.solve_lpfirst(inst, *w)
    assert res.status == "optimal"
    assert res.objective == pytest.approx(ref, abs=1e-6)
    assert res.bound <= res.objective + 1e-6
    inst.check_schedule(res.assignment)


@pytest.mark.parametrize("name", sorted(TRAP_INSTANCES))
def test_boxes_match_branch_and_price(name):
    """Bound rows binding at several levels, both directions."""
    inst = TRAP_INSTANCES[name]()
    f1_min = bpc.solve_f1_bpc(inst).f1
    f2_min = bpc.solve_f2_bpc(inst).f2
    for b2 in (f2_min, f2_min * 1.5 + 1, f2_min * 3 + 5):
        b2 = bpc.grid_floor(inst, "f2", b2)
        ref, _ = bpc._branch_and_price(inst, 1.0, 0.0, bound2_value=b2)
        res = lpfirst.solve_lpfirst(inst, 1.0, 0.0, None, b2)
        assert res.objective == pytest.approx(ref, abs=1e-6)
        assert inst.f2_of(res.assignment) <= b2 + 1e-6
    for b1 in (f1_min + 1, f1_min * 1.2, f1_min * 1.5):
        b1 = bpc.grid_floor(inst, "f1", b1)
        ref, _ = bpc._branch_and_price(inst, 0.0, 1.0, bound1_value=b1)
        res = lpfirst.solve_lpfirst(inst, 0.0, 1.0, b1, None)
        assert res.objective == pytest.approx(ref, abs=1e-6)


@pytest.mark.parametrize("name", sorted(TRAP_INSTANCES) + ["known_pareto"])
@pytest.mark.parametrize("w", [(1.0, 0.0), (1.0, 1.0)])
def test_eligible_mip_alone_is_exact(name, w, known_pareto_inst):
    """With a poor incumbent and no RMP MIP, the MIP over the eligible
    placements must still reach the optimum."""
    inst = _cases(known_pareto_inst)[name]
    ref, _ = bpc._branch_and_price(inst, *w)
    seeds = bpc.heuristic_schedules(inst)
    worst = max(seeds, key=lambda a: lpfirst._objective_of(inst, a, *w))
    res = lpfirst.solve_lpfirst(inst, *w, seed_schedules=[worst], options=NO_MIP)
    assert res.status == "optimal"
    assert res.objective == pytest.approx(ref, abs=1e-6)
    if lpfirst._objective_of(inst, worst, *w) > ref + 1e-6:
        assert res.how == "eligible_mip"


@pytest.mark.parametrize("name", sorted(TRAP_INSTANCES))
@pytest.mark.parametrize("w", [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
def test_lagrangian_bound_is_valid(name, w):
    inst = TRAP_INSTANCES[name]()
    ti = lpfirst._TI(inst, bpc._prepare(inst))
    st: dict = {}
    lpfirst.solve_lpfirst(inst, *w, ti=ti, seed_schedules=[], stats=st, options=NO_MIP)
    lp = st["lp"]
    rng = random.Random(5)
    scale = 3.0 * max(1.0, w[0] * inst.horizon + w[1] * max(inst.c.values()))
    for _ in range(20):
        d = lpfirst._D(sigma=np.array([rng.uniform(-scale, scale) for _ in ti.jobs]),
                       pi=-np.array([[rng.uniform(0, scale) * (rng.random() < 0.3)
                                      for _ in range(ti.H)] for _ in ti.types]).reshape(len(ti.types), ti.H),
                       nu1=0.0, nu2=0.0)
        L, _m = lpfirst._lagrangian(ti, d, lpfirst._price(ti, d, *w), None, None)
        assert L <= lp + 1e-6


def _gap_instance(seed: int, n: int = 7) -> Instance:
    """Two on-prem instances plus a priced cloud: for these seeds the
    f1 + f2 LP lies 0.5-1 below the optimum, so the optimum uses
    placements of positive reduced cost (the trap and trace instances
    tested above are LP-integral)."""
    rng = random.Random(seed)
    jobs = [Job(id=i, arrival=rng.randint(0, 3), exec_time=rng.randint(2, 7), io_volume=0,
                cpu=2, mem=4, stor=20, weight=float(rng.randint(1, 4))) for i in range(n)]
    types = [InstanceType(id=0, kind="on-prem", perf=1.0, io_time=0.0, deploy=0, cost_io=0.0,
                          cost_vm=0.0, cost_stor=0.0, cpu=4, mem=8, stor=50, capacity=2),
             InstanceType(id=1, kind="cloud", perf=1.0, io_time=0.0, deploy=0, cost_io=0.0,
                          cost_vm=3.0, cost_stor=0.0, cpu=4, mem=8, stor=50, capacity=None)]
    inst = Instance(jobs=jobs, instance_types=types, horizon=40)
    inst.build()
    return inst


@pytest.mark.parametrize("seed", [0, 5, 24])
def test_eligible_mip_is_exact_with_an_lp_gap(seed):
    inst = _gap_instance(seed)
    ref, _ = bpc._branch_and_price(inst, 1.0, 1.0)
    st: dict = {}
    seeds = bpc.heuristic_schedules(inst)
    worst = max(seeds, key=lambda a: lpfirst._objective_of(inst, a, 1.0, 1.0))
    res = lpfirst.solve_lpfirst(inst, 1.0, 1.0, seed_schedules=[worst], options=NO_MIP, stats=st)
    assert st["lp"] < ref - 0.25          # the gap is real
    assert res.status == "optimal" and res.how == "eligible_mip"
    assert res.objective == pytest.approx(ref, abs=1e-6)
    assert lpfirst.solve_lpfirst(inst, 1.0, 1.0).objective == pytest.approx(ref, abs=1e-6)


def _random_duals(ti, rng, scale):
    pi = np.zeros((len(ti.types), ti.H))
    mask = np.array([[rng.random() < 0.3 for _ in range(ti.H)] for _ in ti.types])
    pi[mask] = -np.array([rng.uniform(0, scale) for _ in range(int(mask.sum()))])
    return lpfirst._D(sigma=np.array([rng.uniform(-scale, scale) for _ in ti.jobs]), pi=pi,
                      nu1=-rng.uniform(0, 1), nu2=-rng.uniform(0, 1))


@pytest.mark.parametrize("case", ["gap0", "gap5", "gap24", "trap_small", "trap_medium"])
def test_excess_inequality_behind_the_eligible_mip(case):
    """Step 3 rests on obj(x) >= L + sum_j (rc_j(x) - min_j) for every
    feasible x within the bound rows, at any sign-feasible duals. Checked
    on the optimum and the heuristic schedules, at random duals with both
    bound rows, and, non-vacuously, at the converged duals of an instance
    with an LP gap (where the optimum's excess must be positive)."""
    inst = _gap_instance(int(case[3:])) if case.startswith("gap") else TRAP_INSTANCES[case]()
    w = (1.0, 1.0)
    ti = lpfirst._TI(inst, bpc._prepare(inst))
    opt, a_opt = bpc._branch_and_price(inst, *w)
    schedules = [a_opt, *bpc.heuristic_schedules(inst)]
    f = [(lpfirst._objective_of(inst, a, 1, 0), lpfirst._objective_of(inst, a, 0, 1)) for a in schedules]
    b1, b2 = max(x for x, _ in f), max(y for _, y in f)   # every schedule satisfies both rows
    rng = random.Random(9)

    def excess(d, rc, mins, a):
        idx = ti.arcs_of(a)
        assert len(idx) == len(ti.jobs)
        return float(sum(rc[i] + d.sigma[ti.J[i]] - mins[ti.J[i]] for i in idx))

    for _ in range(15):
        d = _random_duals(ti, rng, 5.0)
        rc = lpfirst._price(ti, d, *w)
        L, mins = lpfirst._lagrangian(ti, d, rc, b1, b2)
        for a in schedules:
            assert lpfirst._objective_of(inst, a, *w) >= L + excess(d, rc, mins, a) - 1e-6
    if case.startswith("gap"):
        m = lpfirst._Master(ti, *w, None, None, bpc._big_m_for(inst))
        m.add(range(ti.n_arcs))
        m.mdl.optimize()
        d = m.duals()
        rc = lpfirst._price(ti, d, *w)
        L, mins = lpfirst._lagrangian(ti, d, rc, None, None)
        assert L == pytest.approx(m.mdl.ObjVal, abs=1e-6) and L < opt - 0.25
        ex = excess(d, rc, mins, a_opt)
        assert 0.25 < ex <= opt - L + 1e-6


def test_infeasible_box(trap_small):
    inst = trap_small
    f1_min = bpc.solve_f1_bpc(inst).f1
    res = lpfirst.solve_lpfirst(inst, 0.0, 1.0, f1_min - 1, None)
    assert res.status == "infeasible" and res.assignment is None


def test_fixed_costs_are_rejected(weighted_fixed_cost_inst):
    with pytest.raises(NotImplementedError):
        lpfirst.solve_lpfirst(weighted_fixed_cost_inst, 1.0, 0.0)


def test_lexmin_wrappers_match_branch_and_price(trap_medium):
    for ours, ref in ((lpfirst.solve_f1_lpfirst, bpc.solve_f1_bpc),
                      (lpfirst.solve_f2_lpfirst, bpc.solve_f2_bpc)):
        a, b = ours(trap_medium), ref(trap_medium)
        assert a.status == "optimal"
        assert (a.f1, a.f2) == pytest.approx((b.f1, b.f2), abs=1e-6)


def _points(solutions) -> set[tuple[float, float]]:
    return {(round(s.f1, 3), round(s.f2, 4)) for s in solutions}


@pytest.mark.parametrize("engine_fn", [true_pareto_frontier, map_pareto_frontier])
@pytest.mark.parametrize("name", ["trap_small", "trap_medium", "known_pareto"])
def test_pareto_front_matches_gurobi(engine_fn, name, known_pareto_inst):
    inst = _cases(known_pareto_inst)[name]
    assert _points(engine_fn(inst, engine="lpfirst")) == _points(engine_fn(inst, engine="gurobi"))


def test_zero_time_limit_is_inconclusive(trap_small, capsys):
    assert true_pareto_frontier(trap_small, engine="lpfirst", per_solve_time_limit=0.0) == []
    assert "INCONCLUSIVE" in capsys.readouterr().err
