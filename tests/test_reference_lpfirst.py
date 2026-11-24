"""lpfirst.reference_points_lpfirst (pareto.weighted_sum_reference(engine=
"lpfirst")): the same reference points as the Gurobi engine, on both LP
backends, through the restricted placement sets of each phase; and the
HiGHS path builds no Gurobi model."""
import random

import gurobipy
import pytest

from conftest import TRAP_INSTANCES
from fedhpc import bpc, formulations, lp_backend, lpfirst
from fedhpc.data import Instance, InstanceType, Job, RunningJob
from fedhpc.pareto import weighted_sum_reference

formulations.configure_env()

BIG = 1e9
BACKENDS = ["gurobi"] + (["highs"] if lp_backend.available("highs") else [])


def _type(id, kind, perf, deploy, cost_vm, capacity, cpu=BIG):
    return InstanceType(id=id, kind=kind, perf=perf, io_time=0.0, deploy=deploy,
                        cost_io=0.0, cost_vm=cost_vm, cost_stor=0.0, cpu=cpu,
                        mem=BIG, stor=BIG, capacity=capacity)


def _instance(seed: int, cloud_only_jobs: int = 0, n: int = 10) -> Instance:
    """Two capacity-limited on-prem types (one slow), a cheap slow cloud and a
    dear fast one, a running job. The last ``cloud_only_jobs`` jobs do not
    fit on-prem: no schedule is free."""
    rng = random.Random(seed)
    types = [
        _type(0, "on-prem", 1.0, 0.0, 0.0, 2, cpu=8.0),
        _type(1, "on-prem", 0.7, 0.0, 0.0, 1, cpu=8.0),
        _type(2, "cloud", 1.5, 1.0, 1.0, None),
        _type(3, "cloud", 0.8, 2.0, 0.5, None),
    ]
    jobs = [Job(id=j, arrival=float(rng.randint(0, 10)), exec_time=float(rng.randint(1, 7)),
                io_volume=0.0, cpu=16.0 if j >= n - cloud_only_jobs else 0.0, mem=0.0, stor=0.0,
                weight=float(rng.randint(1, 5)))
            for j in range(n)]
    inst = Instance(jobs=jobs, instance_types=types, horizon=60,
                    running_jobs=[RunningJob(id=100, type_id=0, end=rng.randint(1, 6))])
    inst.build()
    return inst


def _free_instance() -> Instance:
    """On-prem room for every job at its arrival: f2_T = 0 (degenerate)."""
    rng = random.Random(3)
    types = [_type(0, "on-prem", 1.0, 0.0, 0.0, 20), _type(1, "cloud", 0.5, 0.0, 1.0, None)]
    inst = Instance(jobs=[Job(id=j, arrival=float(rng.randint(0, 5)), exec_time=float(rng.randint(1, 4)),
                              io_volume=0.0, cpu=0.0, mem=0.0, stor=0.0) for j in range(6)],
                    instance_types=types, horizon=20)
    inst.build()
    return inst


def _short_horizon_instance() -> Instance:
    """Free options for every job, but no free schedule fits the horizon."""
    types = [_type(0, "on-prem", 1.0, 0.0, 0.0, 2), _type(1, "cloud", 1.0, 0.0, 1.0, None)]
    inst = Instance(jobs=[Job(id=j, arrival=0.0, exec_time=6.0, io_volume=0.0, cpu=0.0, mem=0.0, stor=0.0)
                          for j in range(9)], instance_types=types, horizon=12)
    inst.build()
    return inst


def _cases():
    out = {f"rand{s}": (lambda s=s: _instance(s)) for s in range(5)}
    out.update({f"nofree{s}": (lambda s=s: _instance(s, cloud_only_jobs=2)) for s in range(3)})
    out.update(TRAP_INSTANCES)
    out["short_horizon"] = _short_horizon_instance
    out["free"] = _free_instance
    return out


CASES = _cases()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", sorted(CASES))
def test_matches_the_gurobi_engine(name, backend):
    inst = CASES[name]()
    expected, ideal_g = weighted_sum_reference(inst, min_cost_reference=True, MIPGap=0)
    r = lpfirst.reference_points_lpfirst(inst, options=lpfirst.LPFirstOptions(backend=backend))
    assert r.certified
    assert r.refs == pytest.approx(expected, abs=1e-6)
    assert r.f1_0_bound == pytest.approx(r.refs[2])
    assert (r.ideal is None) == (ideal_g is None)
    if r.ideal is not None:
        assert r.ideal.f2 == pytest.approx(ideal_g.f2)
        assert r.ideal.f1 <= r.refs[2] + 1e-6


@pytest.mark.parametrize("name", sorted(CASES))
def test_restricted_sets_keep_the_optima(name):
    """The phase masks only drop placements no optimum of that phase uses:
    the restricted and full LP-first solves agree."""
    inst = CASES[name]()
    ti = lpfirst._TI(inst, bpc._prepare(inst))
    full = lpfirst.solve_f1_lpfirst(inst, ti=ti)
    sub = ti.restrict(ti.lex_f1_mask())
    lex = lpfirst.solve_f1_lpfirst(inst, ti=sub, seed_schedules=[])
    assert (lex.f1, lex.f2) == pytest.approx((full.f1, full.f2))
    try:
        free = ti.restrict(ti.free_mask())
    except ValueError:   # some job has no free placement
        return
    a = lpfirst.solve_lpfirst(inst, 1.0, 0.0, None, 0.0, ti=ti)
    b = lpfirst.solve_lpfirst(inst, 1.0, 0.0, ti=free, seed_schedules=[])
    assert a.status == b.status
    if a.status == "optimal":
        assert a.objective == pytest.approx(b.objective)


needs_highs = pytest.mark.skipif("highs" not in BACKENDS, reason="highspy not installed")


@needs_highs
def test_engine_dispatch():
    inst = _instance(1)
    g, _ = weighted_sum_reference(inst, min_cost_reference=True, MIPGap=0)
    h, _ = weighted_sum_reference(inst, min_cost_reference=True, engine="lpfirst")
    assert h == pytest.approx(g)
    with pytest.raises(ValueError):
        weighted_sum_reference(inst, engine="nope")


@pytest.mark.skipif("highs" not in BACKENDS, reason="highspy not installed")
def test_highs_path_builds_no_gurobi_model(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("a Gurobi model was built")
    monkeypatch.setattr(gurobipy, "Model", refuse)
    for name in ("rand0", "nofree0", "trap_medium"):
        r = lpfirst.reference_points_lpfirst(
            CASES[name](), options=lpfirst.LPFirstOptions(backend="highs"))
        assert r.certified



def _clustered_instance(seed: int, n: int = 24) -> Instance:
    """Few distinct jobs, many copies (arrivals within one slot differ, so
    class members' f1 offsets are nonzero): the aggregated master applies."""
    rng = random.Random(seed)
    types = [
        _type(0, "on-prem", 1.0, 0.0, 0.0, 2),
        _type(1, "on-prem", 0.7, 0.0, 0.0, 1),
        _type(2, "cloud", 1.5, 1.0, 1.0, None),
    ]
    kinds = [(rng.randint(0, 6), float(rng.randint(1, 4)), float(rng.randint(1, 3))) for _ in range(4)]
    jobs = []
    for j in range(n):
        a, p, w = kinds[rng.randrange(len(kinds))]
        jobs.append(Job(id=j, arrival=a + rng.choice([0.0, 0.25, 0.5]), exec_time=p, io_volume=0.0,
                        cpu=0.0, mem=0.0, stor=0.0, weight=w))
    inst = Instance(jobs=jobs, instance_types=types, horizon=80)
    inst.build()
    return inst


AGG = lpfirst.LPFirstOptions(aggregate_lp=True)
AGG_BARE = lpfirst.LPFirstOptions(aggregate_lp=True, heuristics=False, rmp_mip_time=0.0)


@pytest.mark.parametrize("opts", [AGG, AGG_BARE], ids=["full", "proof_only"])
@pytest.mark.parametrize("bounds", [(None, None), ("f1", None), (None, "f2")])
@pytest.mark.parametrize("w", [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
@pytest.mark.parametrize("seed", range(4))
def test_aggregated_master_matches(seed, w, bounds, opts):
    inst = _clustered_instance(seed)
    ti = lpfirst._TI(inst, bpc._prepare(inst))
    assert len(ti.aggregated().jobs) <= 0.8 * len(ti.jobs)
    ref = lpfirst.solve_lpfirst(inst, 1.0, 0.0, ti=ti)
    ref2 = lpfirst.solve_lpfirst(inst, 0.0, 1.0, ti=ti)
    # bound values strictly inside the front: between the two anchors
    b1 = (ref.objective + 0.6 * (lpfirst._solution(inst, ref2).f1 - ref.objective)) \
        if bounds[0] else None
    b2 = 0.5 * lpfirst._solution(inst, ref).f2 if bounds[1] else None
    st_p: dict = {}
    plain = lpfirst.solve_lpfirst(inst, *w, b1, b2, ti=ti, stats=st_p,
                                  options=lpfirst.LPFirstOptions(heuristics=False,
                                                                 rmp_mip_time=0.0))
    st_a: dict = {}
    agg = lpfirst.solve_lpfirst(inst, *w, b1, b2, ti=ti, stats=st_a, options=opts)
    assert agg.status == plain.status
    if plain.status == "optimal":
        assert agg.objective == pytest.approx(plain.objective, abs=1e-6)
        assert agg.bound <= agg.objective + 1e-6
        inst.check_schedule(agg.assignment)
    if "lp" in st_p and "lp" in st_a:
        assert st_a["lp"] == pytest.approx(st_p["lp"], rel=1e-7, abs=1e-6)


@needs_highs
@pytest.mark.parametrize("seed", range(4))
def test_reference_points_on_clustered_instances(seed):
    inst = _clustered_instance(seed)
    expected, _ = weighted_sum_reference(inst, min_cost_reference=True, MIPGap=0)
    r = lpfirst.reference_points_lpfirst(inst)   # REFERENCE_OPTIONS: aggregated, HiGHS
    assert r.certified and r.refs == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("seed", range(4))
def test_disaggregation_is_a_full_lp_solution(seed):
    """The disaggregated LP solution covers every job once, uses the same
    capacity per slot and has the full LP's value."""
    import numpy as np
    inst = _clustered_instance(seed)
    ti = lpfirst._TI(inst, bpc._prepare(inst))
    agg = ti.aggregated()
    m = lpfirst._Master(agg, 1.0, 0.0, None, None, 1e6, cover_rhs=agg.mult)
    m.add(np.arange(agg.n_arcs))
    assert m.solve_lp() == "optimal"
    x = agg.disaggregate(m.x())
    cover = np.bincount(ti.J, weights=x, minlength=len(ti.jobs))
    assert cover == pytest.approx(np.ones(len(ti.jobs)), abs=1e-7)
    occ = np.zeros((len(ti.types), ti.H + 1))
    occ_a = np.zeros_like(occ)
    for arr, tt, out in ((x, ti, occ), (m.x(), agg, occ_a)):
        c = tt.capped & (arr > 0)
        np.add.at(out, (tt.K[c], tt.T[c]), arr[c] * tt.Q[c])
        np.add.at(out, (tt.K[c], tt.T[c] + tt.P[c]), -arr[c] * tt.Q[c])
    assert np.cumsum(occ, axis=1) == pytest.approx(np.cumsum(occ_a, axis=1), abs=1e-7)
    full = lpfirst._Master(ti, 1.0, 0.0, None, None, 1e6)
    full.add(np.arange(ti.n_arcs))
    assert full.solve_lp() == "optimal"
    assert float(x @ ti.C1) == pytest.approx(full.objective(), abs=1e-6)
