"""Site-aware data transfer (docs/geo_distributed_transfer.md, Level 1) and the
two-centre generator fedhpc.geo_gen."""
import math
import random

import pytest

from fedhpc import baselines, formulations
from fedhpc import geo_gen as gg
from fedhpc.data import Instance, InstanceType, Job, Link, RunningJob
from fedhpc.formulations import SpaceTimeFormulation
from fedhpc.model import solve_f1, solve_weighted_sum
from fedhpc.moea import weighted_solve_dispatch
from fedhpc.pareto import weighted_sum_reference

formulations.configure_env()

PLAIN = dict(collapse_unbounded=False, pareto_unbounded=False,
             rejection_deadlines=False, lex_deadlines=False)
BIG = 1e9


def _type(id, site, capacity, cost_vm=0.0, kind=None):
    return InstanceType(id=id, kind=kind or ("on-prem" if capacity else "cloud"), perf=1.0,
                        io_time=0.0, deploy=0.0, cost_io=0.0, cost_vm=cost_vm, cost_stor=0.0,
                        cpu=BIG, mem=BIG, stor=BIG, capacity=capacity, site=site)


def _links(wan=1.0, up=0.5, down=0.5, egress=0.1):
    return {("A", "B"): Link("A", "B", wan), ("B", "A"): Link("B", "A", wan),
            ("A", "C"): Link("A", "C", up), ("B", "C"): Link("B", "C", up),
            ("C", "A"): Link("C", "A", down, egress), ("C", "B"): Link("C", "B", down, egress)}


def _job(id, origin, arrival=0.0, e=2.0, io_in=0.0, io_out=0.0, **kw):
    return Job(id=id, arrival=arrival, exec_time=e, io_volume=io_out, cpu=0.0, mem=0.0,
               stor=0.0, io_in=io_in, origin=origin, **kw)


# ─── data layer ──────────────────────────────────────────────────────────────

def test_link_sums_enter_p_and_c():
    types = [_type(0, "A", 1), _type(1, "B", 1), _type(2, "C", None, cost_vm=1.0)]
    inst = Instance(jobs=[_job(0, "A", e=2.0, io_in=2.0, io_out=3.0)], instance_types=types,
                    horizon=40, links=_links(wan=1.0, up=0.5, down=0.25, egress=0.1))
    inst.build()
    # local: nothing moves
    assert inst.io_t[0, 0] == 0 and inst.p_occ[0, 0] == 2 and inst.c[0, 0] == 0
    # remote centre: 2 GB in + 3 GB out at 1 slot/GB, free
    assert inst.io_t[0, 1] == 5 and inst.p_occ[0, 1] == 7 and inst.c[0, 1] == 0
    # cloud: 2·0.5 in + 3·0.25 out = 1.75 slots; egress 3·0.1; VM billed for e + τ
    assert inst.p_bill[0, 2] == pytest.approx(3.75)
    assert inst.p_occ[0, 2] == 4
    assert inst.c[0, 2] == pytest.approx(0.3 + 3.75, abs=1e-4)


def test_dest_differs_from_origin():
    types = [_type(0, "A", 1), _type(1, "B", 1)]
    inst = Instance(jobs=[_job(0, "A", io_in=1.0, io_out=2.0, dest="B")], instance_types=types,
                    horizon=20, links=_links(wan=1.0))
    inst.build()
    assert inst.io_t[0, 0] == 2.0      # runs at A, ships 2 GB of results to B
    assert inst.io_t[0, 1] == 1.0      # runs at B, pulls 1 GB of inputs from A


def test_sites_restrict_F():
    types = [_type(0, "A", 1), _type(1, "B", 1), _type(2, "C", None, cost_vm=1.0)]
    inst = Instance(jobs=[_job(0, "A", sites=["A", "C"]), _job(1, "B")], instance_types=types,
                    horizon=20, links=_links())
    inst.build()
    assert inst.F[0] == [0, 2] and inst.F[1] == [0, 1, 2]


def test_per_type_mode_unchanged():
    """Without links, sites/origins are ignored and transfer is τ^IO_m · ι_j."""
    t = InstanceType(id=0, kind="cloud", perf=1.0, io_time=0.5, deploy=0.0, cost_io=0.2,
                     cost_vm=1.0, cost_stor=0.0, cpu=BIG, mem=BIG, stor=BIG, site="X")
    inst = Instance(jobs=[_job(0, "nowhere", e=1.0, io_in=99.0, io_out=4.0, sites=["Y"])],
                    instance_types=[t], horizon=10)
    inst.build()
    assert inst.F[0] == [0]
    assert inst.p_bill[0, 0] == pytest.approx(3.0)
    assert inst.c[0, 0] == pytest.approx(0.8 + 3.0, abs=1e-4)


@pytest.mark.parametrize("break_it, msg", [
    (lambda ts, js, ln: setattr(ts[1], "site", None), "without a site"),
    (lambda ts, js, ln: setattr(ts[1], "io_time", 0.1), "must be 0"),
    (lambda ts, js, ln: setattr(js[0], "origin", None), "without an origin"),
    (lambda ts, js, ln: ln.pop(("A", "B")), "No link"),
    (lambda ts, js, ln: ln.__setitem__(("A", "B"), Link("A", "B", -1.0)), "negative"),
])
def test_site_aware_validation(break_it, msg):
    types = [_type(0, "A", 1), _type(1, "B", 1)]
    jobs = [_job(0, "A", io_in=1.0)]
    links = _links()
    break_it(types, jobs, links)
    with pytest.raises(ValueError, match=msg):
        Instance(jobs=jobs, instance_types=types, horizon=20, links=links).build()


def test_from_dict_reads_links_and_sites():
    d = dict(horizon=20,
             instance_types=[dict(id=0, kind="on-prem", perf=1, io_time=0, deploy=0, cost_io=0,
                                  cost_vm=0, cost_stor=0, cpu=BIG, mem=BIG, stor=BIG,
                                  capacity=1, site="A"),
                             dict(id=1, kind="on-prem", perf=1, io_time=0, deploy=0, cost_io=0,
                                  cost_vm=0, cost_stor=0, cpu=BIG, mem=BIG, stor=BIG,
                                  capacity=1, site="B")],
             jobs=[dict(id=0, arrival=0, exec_time=1, io_volume=0, io_in=3, cpu=0, mem=0,
                        stor=0, origin="A")],
             links=[dict(src="A", dst="B", time=1.0), dict(src="B", dst="A", time=1.0)])
    inst = Instance.from_dict(d)
    assert inst.site_aware and inst.p_occ[0, 1] == 4


# ─── scheduling consequences ─────────────────────────────────────────────────

def _wait_or_move(wan: float) -> Instance:
    """One busy node at the job's home A (free at t = 5), an idle one at B."""
    types = [_type(0, "A", 1), _type(1, "B", 1)]
    inst = Instance(jobs=[_job(0, "A", e=2.0, io_in=1.0, io_out=0.0)], instance_types=types,
                    horizon=30, running_jobs=[RunningJob(id=9, type_id=0, end=5)],
                    links=_links(wan=wan))
    inst.build()
    return inst


@pytest.mark.parametrize("wan, where, completion", [(1.0, 1, 3), (6.0, 0, 7)])
def test_exact_solver_weighs_staging_against_waiting(wan, where, completion):
    inst = _wait_or_move(wan)
    sol = solve_f1(inst, OutputFlag=0)
    assert sol.assignment[0][0] == where and sol.completion[0] == completion


def test_federation_baseline_ignores_staging():
    inst = _wait_or_move(6.0)          # moving finishes at 8, waiting at 7
    sol = baselines.earliest_start_federation(inst)
    assert sol.assignment[0] == (1, 0) and sol.f1 == 8


def test_threshold_compares_completions():
    inst = _wait_or_move(6.0)
    sol = baselines.threshold_bursting(inst, theta=10)
    assert sol.assignment[0] == (0, 5)


def _random_geo(seed: int) -> Instance:
    rng = random.Random(seed)
    types = [_type(0, "A", 1), _type(1, "A", 1), _type(2, "B", 2),
             _type(3, "C", None, cost_vm=1.0), _type(4, "C", None, cost_vm=0.6, kind="cloud")]
    types[4].perf = 0.7
    jobs = [_job(j, rng.choice("AB"), arrival=float(rng.randint(0, 6)),
                 e=float(rng.randint(1, 5)), io_in=rng.uniform(0, 2), io_out=rng.uniform(0, 3),
                 weight=float(rng.randint(1, 3)),
                 sites=None if rng.random() < 0.8 else [rng.choice("AB")])
            for j in range(9)]
    for j in jobs:
        if j.sites:
            j.origin = j.sites[0]
    inst = Instance(jobs=jobs, instance_types=types, horizon=40,
                    running_jobs=[RunningJob(id=100, type_id=0, end=rng.randint(1, 6))],
                    links=_links(wan=rng.uniform(0.3, 2.0), up=0.4, down=0.6, egress=0.15))
    inst.build()
    return inst


@pytest.mark.parametrize("seed", range(4))
def test_reductions_stay_exact_on_site_aware_data(seed):
    """The proposal's claim: R1/R2/R3/lex need only (j, m)-tabled p and c."""
    inst = _random_geo(seed)
    red, plain = SpaceTimeFormulation(), SpaceTimeFormulation(**PLAIN)
    a, b = solve_f1(inst, formulation=red, OutputFlag=0), solve_f1(inst, formulation=plain, OutputFlag=0)
    assert (a.f1, a.f2) == pytest.approx((b.f1, b.f2))
    (f1_T, f2_T, f1_0, f2_0), _ = weighted_sum_reference(inst, red, OutputFlag=0)
    for lam in (0.25, 0.75):
        x = solve_weighted_sum(inst, lam, f1_T, f2_T, f1_0, formulation=red, f2_0=f2_0, OutputFlag=0)
        y = solve_weighted_sum(inst, lam, f1_T, f2_T, f1_0, formulation=plain, f2_0=f2_0, OutputFlag=0)
        assert x.objective == pytest.approx(y.objective, abs=1e-6)
        inst.check_schedule(x.assignment)


@pytest.mark.parametrize("seed", range(3))
def test_dispatch_feasible_and_no_better_than_exact(seed):
    inst = _random_geo(seed)
    (f1_T, f2_T, f1_0, f2_0), _ = weighted_sum_reference(inst, OutputFlag=0)
    ex = solve_weighted_sum(inst, 0.5, f1_T, f2_T, f1_0, f2_0=f2_0, OutputFlag=0)
    h = weighted_solve_dispatch(inst, 0.5, f1_T=f1_T, f2_T=f2_T, f1_0=f1_0, f2_0=f2_0, n_threads=1)
    inst.check_schedule(h.assignment)
    g = lambda s: 0.5 * (s.f1 - f1_T) / max(f1_0 - f1_T, 1e-9) + 0.5 * (s.f2 - f2_0) / max(f2_T - f2_0, 1e-9)
    assert g(h) >= g(ex) - 1e-6


# ─── generator ───────────────────────────────────────────────────────────────

NAME = "G_fedhpc_n50_lH_iM_w1_fedC"


def test_name_round_trip_and_validation():
    assert gg.GeoSpec.parse(NAME).name == NAME
    with pytest.raises(ValueError):
        gg.GeoSpec.parse("G_fedhpc_n50_lH_iM_w1_fooC")
    with pytest.raises(ValueError):
        gg.GeoSpec("fedhpc", 10, "H", "M", 1, "fed", "C")


def test_generate_is_reproducible():
    spec = gg.GeoSpec.parse(NAME)
    assert gg.generate(spec) == gg.generate(spec)


@pytest.fixture(scope="module")
def geo_pair():
    return {n: gg.build(n) for n in (NAME, NAME.replace("_w1_", "_w0_"),
                                     NAME.replace("_fedC", "_isoC"),
                                     NAME.replace("_fedC", "_fedN"))}


def test_generated_structure(geo_pair):
    inst = geo_pair[NAME]
    site = {m.id: m.site for m in inst.instance_types}
    assert sorted(set(site.values())) == ["S1", "S2", "aws"]
    assert sum(m.capacity or 0 for m in inst.instance_types) == 2 * 2 * 50
    assert all(j.io_in >= 0 and j.io_volume >= 0 and j.origin in gg.SITES for j in inst.jobs)
    for j in inst.jobs:                 # local placement is never slower than remote
        loc = [inst.p_occ[j.id, m] for m in inst.F[j.id] if site[m] == j.origin]
        rem = [inst.p_occ[j.id, m] for m in inst.F[j.id] if site[m] not in (j.origin, "aws")]
        assert loc and (not rem or min(rem) >= min(loc))
        assert all(inst.c[j.id, m] == 0 for m in inst.F[j.id] if site[m] != "aws")


def test_colocated_control_has_no_remote_penalty(geo_pair):
    ctrl = geo_pair[NAME.replace("_w1_", "_w0_")]
    site = {m.id: m.site for m in ctrl.instance_types}
    for j in ctrl.jobs:
        on = {inst_p for m, inst_p in ((m, ctrl.p_occ[j.id, m]) for m in ctrl.F[j.id])
              if site[m] != "aws"}
        assert len(on) == 1


def test_iso_and_no_cloud_modes(geo_pair):
    iso = geo_pair[NAME.replace("_fedC", "_isoC")]
    site = {m.id: m.site for m in iso.instance_types}
    assert all({site[m] for m in iso.F[j.id]} <= {j.origin, "aws"} for j in iso.jobs)
    noc = geo_pair[NAME.replace("_fedC", "_fedN")]
    assert all(m.capacity is not None for m in noc.instance_types)


def test_single_runs_pinned_to_origin(geo_pair):
    inst = geo_pair[NAME]
    site = {m.id: m.site for m in inst.instance_types}
    pinned = [j for j in inst.jobs if j.sites is not None]
    assert pinned and all({site[m] for m in inst.F[j.id]} == {j.origin} for j in pinned)


def test_link_rates_follow_b_eff():
    spec = gg.GeoSpec.parse(NAME)
    ln = {(d["src"], d["dst"]): d for d in gg.links(spec, 600.0)}
    assert ln["S1", "S2"]["time"] == pytest.approx(1000 / 125 / 600)      # 1 Gbps binds
    assert ln["aws", "S1"]["cost"] == 0.09 and ln["S1", "aws"]["cost"] == 0.0
    assert math.isclose(ln["S1", "aws"]["time"], ln["aws", "S1"]["time"])  # gp3 disk binds both ways
