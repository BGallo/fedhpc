"""Compiled pricing of the BPC engine (``_bpcprice`` via ``bpc._NodePricing``):
agreement with the Python reference DP (``bpc._price_type``) under random
duals and branching decisions, ng-route relaxation order, multiplicity
coefficients, validity of the Lagrangian bound, and end-to-end agreement of
the pricing engines.
"""
from __future__ import annotations

import math
import random
import zlib

import pytest

from conftest import TRAP_INSTANCES
from fedhpc import bpc
from fedhpc.formulations import configure_env

configure_env()

pytestmark = pytest.mark.skipif(bpc._bpcprice is None, reason="_bpcprice extension not built")

_TRAPS = sorted(TRAP_INSTANCES)
ELEM = bpc.PricingOptions(ng_size=None, cols_per_type=50)


def _random_duals(prep, rng, w1, w2, links=()):
    inst = prep.inst
    scale = 3.0 * max(1.0, w1 * inst.horizon + w2 * max(inst.c.values()))
    return bpc._Duals(
        sigma={j.id: rng.uniform(0, scale) for j in inst.jobs},
        rho={(m, r): rng.uniform(-scale, scale) for m in prep.path_types for r in prep.releases[m]},
        nu1=None, nu2=None, theta={k: -rng.uniform(0, scale) for k in links},
    )


def _random_branching(prep, rng):
    inst = prep.inst
    jobs = [j.id for j in inst.jobs]
    forbidden = set()
    for (jid, mid), ts in prep.starts.items():
        for t in ts:
            if rng.random() < 0.15:
                forbidden.add((jid, mid, t))
    pinned = {}
    jid = rng.choice(jobs)
    options = [(m, t) for (j, m), ts in sorted(prep.starts.items())
               if j == jid and m in prep.path_types for t in ts]
    if options:
        pinned[jid] = rng.choice(options)
    a, b, c, d = rng.sample(jobs, 4)
    return frozenset(forbidden), pinned, frozenset({(a, b)}), frozenset({(c, d)})


@pytest.mark.parametrize("name", _TRAPS)
@pytest.mark.parametrize("w", [(1.0, 0.0), (0.0, 1.0), (0.5, 0.5)])
@pytest.mark.parametrize("branching", [False, True])
def test_elementary_pricer_matches_python_dp(name, w, branching):
    inst = TRAP_INSTANCES[name]()
    prep = bpc._prepare(inst)
    rng = random.Random(zlib.crc32(repr((name, w, branching)).encode()))
    for _ in range(6):
        d = _random_duals(prep, rng, *w)
        fb, pin, dif, same = _random_branching(prep, rng) if branching else (frozenset(), {}, frozenset(), frozenset())
        pricing = bpc._NodePricing(prep, ELEM, fb, pin, dif, same)
        for mid in prep.path_types:
            mn, exact, _n, cols = pricing.price(mid, d, *w, 1e-7)
            ref = bpc._price_type(
                inst, mid, prep.arcs_by_type[mid], *w, d.sigma, None, None, 0.0, {},
                prep.arrival_of, prep.weight_of, 1e-7, fb, pin, dif, same, None,
                {r: d.rho[mid, r] for r in prep.releases[mid]}, bucket_cap=None)
            assert exact
            if ref is None:
                assert not cols and mn >= -1e-7 - 1e-9
                continue
            best, col = ref
            assert cols, (mid, best)
            assert mn == pytest.approx(best, abs=1e-6)
            assert bpc._column_rc(cols[0], d, *w) == pytest.approx(best, abs=1e-6)
            for c in cols:
                assert bpc._is_elementary(c)
                assert bpc._column_allowed(c, fb, pin, dif, same)
                assert all(not (j in {x for x, _ in c.jobs}) != (k in {x for x, _ in c.jobs})
                           for j, k in same)
                assert c.release in prep.releases[mid]


@pytest.mark.parametrize("name", _TRAPS)
def test_ng_relaxation_order(name):
    """min over ng-paths <= elementary min, with equality once the
    neighbourhoods hold every job."""
    inst = TRAP_INSTANCES[name]()
    rng = random.Random(7)
    for size in (1, 3, len(inst.jobs) + 1):
        prep = bpc._prepare(inst)
        opts = bpc.PricingOptions(ng_size=size, cols_per_type=50)
        for _ in range(4):
            d = _random_duals(prep, rng, 1.0, 0.0)
            el = bpc._NodePricing(prep, ELEM, frozenset(), {}, frozenset(), frozenset())
            ng = bpc._NodePricing(prep, opts, frozenset(), {}, frozenset(), frozenset())
            for mid in prep.path_types:
                e_min = el.price(mid, d, 1.0, 0.0, 1e-7)[0]
                n_min, _ex, _n, cols = ng.price(mid, d, 1.0, 0.0, 1e-7)
                assert n_min <= e_min + 1e-7
                if size > len(inst.jobs):
                    assert n_min == pytest.approx(e_min, abs=1e-7)
                    assert all(bpc._is_elementary(c) for c in cols)
                for c in cols:   # every reported reduced cost is the column's own
                    assert bpc._column_rc(c, d, 1.0, 0.0) < 0


def test_repeat_column_gets_multiplicity_coefficients(trap_small):
    inst = trap_small
    prep = bpc._prepare(inst)
    mid = prep.path_types[0]
    arc = max(prep.arcs_by_type[mid], key=lambda a: len(a.starts))
    ts = sorted(arc.starts)
    t1, t2 = ts[0], next(t for t in ts if t >= ts[0] + arc.p_occ)
    jobs = ((arc.job_id, t1), (arc.job_id, t2))
    col = bpc.Column(type_id=mid, jobs=jobs,
                     cost1=sum(arc.weight * (t + arc.p_occ - prep.arrival_of[arc.job_id]) for _j, t in jobs),
                     cost2=2 * arc.raw_cost2, release=min(prep.releases[mid]))
    mdl, state = bpc._build_rmp(inst, 1.0, 0.0, None, None, 1e6, extra_columns=[col], prep=prep)
    var = state.vars[-1]
    row = mdl.getCol(var)
    coef = {row.getConstr(i).ConstrName: row.getCoeff(i) for i in range(row.size())}
    assert coef[f"cover[{arc.job_id}]"] == 2.0
    assert not bpc._is_elementary(col)


@pytest.mark.parametrize("name", _TRAPS)
@pytest.mark.parametrize("w", [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
def test_lagrangian_bound_is_valid(name, w):
    """L(d) <= LP optimum for random sign-feasible duals, and L == LP at the
    converged duals (up to the pricing tolerance)."""
    inst = TRAP_INSTANCES[name]()
    prep = bpc._prepare(inst)
    big_m = bpc._big_m_for(inst)
    root = bpc.solve_root_relaxation(inst, *w, pricing=bpc.PricingOptions(ng_size=None))
    lp = root.objective
    state = root.state
    statics: dict = {}
    for c in prep.static_columns:
        statics.setdefault(c.jobs[0][0], []).append(c)
    fixed = {m.id: m.fixed_cost for m in inst.instance_types}
    pricing = bpc._NodePricing(prep, ELEM, frozenset(), {}, frozenset(), frozenset())

    def L(d):
        mins = {m: pricing.price(m, d, *w, 1e-9)[0] for m in prep.path_types}
        return bpc._lagrangian_bound(prep, state, d, mins, *w, big_m, statics, fixed)

    # The compiled loop may stop once the bound proves the RMP optimal; its
    # value must equal the fully converged Python column generation's. The
    # last RMP's duals are then one of several optimal dual solutions of a
    # degenerate RMP, so they only give *a* valid bound, not the LP value.
    ref = bpc.solve_root_relaxation(inst, *w, pricing=bpc.PricingOptions(engine="python"))
    assert lp == pytest.approx(ref.objective, abs=1e-5 * max(1.0, abs(lp)))
    d_opt = bpc._read_duals(state)
    assert L(d_opt) <= lp + 1e-6
    rng = random.Random(3)
    for _ in range(10):
        d = _random_duals(prep, rng, *w, links=state.fixed_link)
        assert L(d) <= lp + 1e-6
        assert L(d_opt.mix(d, rng.random())) <= lp + 1e-6


ENGINES = {
    "python": bpc.PricingOptions(engine="python"),
    "elementary": bpc.PricingOptions(ng_size=None),
    "ng1": bpc.PricingOptions(ng_size=1, ng_max=2),
    "ng8_smooth": bpc.PricingOptions(smoothing=0.5),
}


@pytest.mark.parametrize("engine", sorted(ENGINES))
@pytest.mark.parametrize("name", _TRAPS)
@pytest.mark.parametrize("w", [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
def test_engines_agree_end_to_end(engine, name, w):
    inst = TRAP_INSTANCES[name]()
    ref, _ = bpc._branch_and_price(inst, *w, pricing=ENGINES["python"])
    obj, a = bpc._branch_and_price(inst, *w, pricing=ENGINES[engine])
    assert obj == pytest.approx(ref, abs=1e-6)
    inst.check_schedule(a)


def test_root_bound_between_compact_lp_and_elementary(trap_medium):
    """ng with no room to grow sits between the compact LP and the
    elementary master's bound."""
    inst = trap_medium
    el = bpc.solve_root_relaxation(inst, 1.0, 0.0, pricing=bpc.PricingOptions(ng_size=None))
    ng = bpc.solve_root_relaxation(inst, 1.0, 0.0, pricing=bpc.PricingOptions(ng_size=1, ng_max=1))
    assert ng.objective <= el.objective + 1e-6
    assert ng.artificial_total < 1e-6 and math.isfinite(ng.objective)


def test_objective_step(trap_small):
    inst = trap_small
    assert bpc._objective_step(inst, 1.0, 0.0) >= 1.0
    s2 = bpc._objective_step(inst, 0.0, 1.0)
    assert s2 > 0 and abs(s2 / inst.cost_resolution - round(s2 / inst.cost_resolution)) < 1e-9
    assert bpc._objective_step(inst, 1.0, math.pi) == 0.0


@pytest.mark.parametrize("name", _TRAPS)
@pytest.mark.parametrize("ng_size", [None, 3])
def test_activation_dual_is_priced_once_per_column(name, ng_size):
    """With an activation dual eta_m < 0 every non-empty column pays -eta_m
    once. The pricer's minimum must equal the best over the eta = 0 columns
    re-scored with eta (a dominating eta = 0 label stays dominating, since
    both pay it) and the null paths — an empty label may not dominate a
    non-empty one on cost alone."""
    from dataclasses import replace
    inst = TRAP_INSTANCES[name]()
    rng = random.Random(11)
    opts = bpc.PricingOptions(ng_size=ng_size, cols_per_type=100_000)
    for _ in range(6):
        prep = bpc._prepare(inst)
        d0 = _random_duals(prep, rng, 1.0, 0.0)
        scale = max(d0.sigma.values())
        d = replace(d0, eta={m: -rng.uniform(0, 2 * scale) for m in prep.path_types})
        pricing = bpc._NodePricing(prep, opts, frozenset(), {}, frozenset(), frozenset())
        for mid in prep.path_types:
            mn = pricing.price(mid, d, 1.0, 0.0, 1e-9)[0]
            cols0 = pricing.price(mid, d0, 1.0, 0.0, 1e-9)[3]
            brute = min([bpc._column_rc(c, d, 1.0, 0.0) for c in cols0]
                        + [-d.rho[mid, r] for r in prep.releases[mid]])
            if brute < -1e-9:   # an improving column exists: found exactly
                assert mn == pytest.approx(brute, abs=1e-6)
            else:               # none: a valid bound no lower than -tol
                assert -1e-9 - 1e-6 <= mn <= brute + 1e-6


def _rm_m2n10():
    from fedhpc import rm_benchmark as rm
    from test_rm_benchmark import M2N10_1
    data = rm.parse(M2N10_1)
    return rm.to_instance(data, horizon=sum(max(r) for r in data[0]))


def _kernels():
    ks = [1, 2]
    probe = bpc._bpcprice.Pricer(4, [1], [[0]])
    try:
        probe.set_kernel(3)
        ks.append(3)
    except ValueError:   # no AVX2 on this CPU
        pass
    return ks


@pytest.mark.parametrize("name", [*_TRAPS, "rm_M2N10_1"])
@pytest.mark.parametrize("w", [(1.0, 0.0), (1.0, 1.0)])
def test_dp_kernels_agree(name, w):
    """The DP kernels (AoS, SoA scalar, SoA AVX2) give the same minimum,
    exactness and reduced costs on random duals."""
    inst = _rm_m2n10() if name == "rm_M2N10_1" else TRAP_INSTANCES[name]()
    prep = bpc._prepare(inst)
    rng = random.Random(zlib.crc32(name.encode()))
    opts = bpc.PricingOptions()
    for _draw in range(6):
        d = _random_duals(prep, rng, *w)
        out = {}
        for k in _kernels():
            pricing = bpc._NodePricing(prep, opts, frozenset(), {}, frozenset(), frozenset())
            for pricer, _arcs, _local in pricing.types.values():
                pricer.set_kernel(k)
            res = []
            for mid in prep.path_types:
                mn, exact, _n, cols = pricing.price(mid, d, *w, 1e-7)
                rcs = sorted(round(bpc._column_rc(c, d, *w), 6) for c in cols)
                res.append((round(mn, 9) if math.isfinite(mn) else mn, exact, rcs))
            out[k] = res
        first = out[1]
        for k, res in out.items():
            assert res == first, f"kernel {k} differs from kernel 1"


def test_dp_kernels_same_optimum(monkeypatch):
    inst = _rm_m2n10()
    objs = {}
    for k in _kernels():
        monkeypatch.setattr(bpc, "_PRICER_KERNEL", k)
        objs[k] = bpc._branch_and_price(inst, 1.0, 1.0)[0]
    assert len(set(objs.values())) == 1, objs
    assert objs[1] == 145220
