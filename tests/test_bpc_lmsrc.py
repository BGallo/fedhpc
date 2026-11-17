"""Limited-memory subset-row cuts in the compiled BPC engine (Phase 4):
the coefficient rule, the pricer's cut charging and soft dominance (against
the Python DP for full-memory cuts, and by self-consistency and a
bound check for limited memory), separation, and end-to-end agreement."""
from __future__ import annotations

import random
import zlib
from dataclasses import replace

import pytest

from conftest import TRAP_INSTANCES
from fedhpc import bpc
from fedhpc.bpc import Column, _LmCut, _lm_coef
from fedhpc.formulations import configure_env

configure_env()

pytestmark = pytest.mark.skipif(bpc._bpcprice is None, reason="_bpcprice extension not built")

_TRAPS = sorted(TRAP_INSTANCES)
ELEM = bpc.PricingOptions(ng_size=None, cols_per_type=50)


def test_coefficient_rule():
    full = _LmCut(S=frozenset({1, 2, 3}), M=frozenset(range(10)))
    seq = lambda *js: [(j, 0) for j in js]   # noqa: E731
    assert _lm_coef(full, seq(1, 5, 2)) == 1
    assert _lm_coef(full, seq(1, 2, 3)) == 1
    assert _lm_coef(full, seq(1, 5, 6)) == 0
    assert _lm_coef(full, seq(1, 2, 3, 1)) == 2          # repeats count (ng-routes)
    lm = _LmCut(S=frozenset({1, 2, 3}), M=frozenset({1, 2, 3, 5}))
    assert _lm_coef(lm, seq(1, 5, 2)) == 1               # 5 is remembered
    assert _lm_coef(lm, seq(1, 6, 2)) == 0               # 6 resets the state
    assert _lm_coef(lm, seq(1, 6, 2, 3)) == 1


def _random_duals(prep, rng, w1, w2):
    inst = prep.inst
    scale = 3.0 * max(1.0, w1 * inst.horizon + w2 * max(inst.c.values()))
    return bpc._Duals(
        sigma={j.id: rng.uniform(0, scale) for j in inst.jobs},
        rho={(m, r): rng.uniform(-scale, scale) for m in prep.path_types for r in prep.releases[m]},
        nu1=None, nu2=None, theta={},
    ), scale


def _random_cuts(inst, rng, k, full):
    jobs = [j.id for j in inst.jobs]
    out = []
    for _ in range(k):
        S = frozenset(rng.sample(jobs, 3))
        M = frozenset(jobs) if full else S | frozenset(rng.sample(jobs, rng.randint(0, 3)))
        out.append(_LmCut(S=S, M=M))
    return out


@pytest.mark.parametrize("name", _TRAPS)
@pytest.mark.parametrize("w", [(1.0, 0.0), (1.0, 1.0)])
def test_full_memory_cuts_match_python_dp(name, w):
    """With M = every job the cut is Jepsen's SR cut, which the Python DP
    prices exactly (one dominance bit per cut, no soft dominance)."""
    inst = TRAP_INSTANCES[name]()
    prep = bpc._prepare(inst)
    rng = random.Random(zlib.crc32(repr((name, w)).encode()))
    for _ in range(6):
        d0, scale = _random_duals(prep, rng, *w)
        cuts = _random_cuts(inst, rng, 4, full=True)
        d = replace(d0, cuts=tuple(cuts), beta=tuple(-rng.uniform(0, scale) for _ in cuts))
        pricing = bpc._NodePricing(prep, ELEM, frozenset(), {}, frozenset(), frozenset())
        pricing.set_cuts(cuts)
        for mid in prep.path_types:
            mn, exact, _n, cols = pricing.price(mid, d, *w, 1e-7)
            ref = bpc._price_type(
                inst, mid, prep.arcs_by_type[mid], *w, d.sigma, None, None, 0.0, {},
                prep.arrival_of, prep.weight_of, 1e-7,
                cut_duals={c.S: b for c, b in zip(cuts, d.beta)},
                sources={r: d.rho[mid, r] for r in prep.releases[mid]}, bucket_cap=None)
            assert exact
            if ref is None:
                assert not cols and mn >= -1e-7 - 1e-9
                continue
            assert mn == pytest.approx(ref[0], abs=1e-6)
            rcs = [bpc._column_rc(c, d, *w) for c in cols]
            assert rcs[0] == pytest.approx(mn, abs=1e-6)
            assert all(rc < 1e-7 for rc in rcs)


@pytest.mark.parametrize("name", _TRAPS)
@pytest.mark.parametrize("ng_size", [None, 3])
def test_limited_memory_pricing_is_consistent_and_a_bound(name, ng_size):
    """Every column the pricer returns has the reduced cost the master
    gives it (same coefficient rule), the most negative one first; and the
    pricer's minimum is no higher than any column's reduced cost — checked
    against a large pool of columns priced without cuts."""
    inst = TRAP_INSTANCES[name]()
    rng = random.Random(zlib.crc32(repr((name, ng_size)).encode()))
    opts = bpc.PricingOptions(ng_size=ng_size, cols_per_type=100_000)
    for _ in range(6):
        prep = bpc._prepare(inst)
        d0, scale = _random_duals(prep, rng, 1.0, 0.0)
        cuts = _random_cuts(inst, rng, 6, full=False)
        d = replace(d0, cuts=tuple(cuts), beta=tuple(-rng.uniform(0, scale) for _ in cuts))
        with_cuts = bpc._NodePricing(prep, opts, frozenset(), {}, frozenset(), frozenset())
        with_cuts.set_cuts(cuts)
        plain = bpc._NodePricing(prep, opts, frozenset(), {}, frozenset(), frozenset())
        for mid in prep.path_types:
            mn, exact, _n, cols = with_cuts.price(mid, d, 1.0, 0.0, 1e-9)
            assert exact
            rcs = [bpc._column_rc(c, d, 1.0, 0.0) for c in cols]
            assert all(rc < 1e-9 for rc in rcs)
            if cols:
                assert rcs[0] == pytest.approx(mn, abs=1e-6)
            pool = plain.price(mid, d0, 1.0, 0.0, 1e-9)[3]
            brute = min([bpc._column_rc(c, d, 1.0, 0.0) for c in pool]
                        + [-d.rho[mid, r] for r in prep.releases[mid]])
            assert mn <= brute + 1e-6


class _Var:
    def __init__(self, x):
        self.X = x


class _State:
    def __init__(self, columns, xs):
        self.columns, self.vars = columns, [_Var(x) for x in xs]


def test_separation_finds_the_cycle_and_its_memory():
    col = lambda *js: Column(type_id=0, jobs=tuple((j, 10 * i) for i, j in enumerate(js)),  # noqa: E731
                             cost1=0.0, cost2=0.0)
    state = _State([col(0, 7, 1), col(0, 2), col(1, 2)], [0.6, 0.6, 0.6])
    cuts = bpc._separate_lm_src(state, 5, 0.05, set())
    assert cuts and cuts[0].S == frozenset({0, 1, 2})
    assert cuts[0].M == frozenset({0, 1, 2, 7})       # job 7 sits between 0 and 1
    lhs = sum(x * _lm_coef(cuts[0], c.jobs) for c, x in zip(state.columns, [0.6] * 3))
    assert lhs == pytest.approx(1.8)
    again = bpc._separate_lm_src(state, 5, 0.05, {frozenset({0, 1, 2})})
    assert frozenset({0, 1, 2}) not in {c.S for c in again}
    assert frozenset({0, 2, 7}) in {c.S for c in again}       # 0.6 + 0.6 > 1 too
    assert not bpc._separate_lm_src(_State(state.columns, [0.3] * 3), 5, 0.05, set())


@pytest.mark.parametrize("name", _TRAPS)
@pytest.mark.parametrize("w", [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
@pytest.mark.parametrize("sr", [dict(sr_cuts=8, sr_per_round=4),
                                dict(sr_cuts=16, sr_per_round=8, sr_depth=3, sr_min_viol=0.01)])
def test_cuts_keep_the_optimum(name, w, sr):
    inst = TRAP_INSTANCES[name]()
    ref, _ = bpc._branch_and_price(inst, *w)
    st: dict = {}
    obj, a = bpc._branch_and_price(inst, *w, pricing=bpc.PricingOptions(**sr), stats=st)
    assert obj == pytest.approx(ref, abs=1e-6)
    inst.check_schedule(a)


def test_cuts_keep_the_optimum_under_bound_rows(trap_medium):
    inst = trap_medium
    f2_min = bpc.solve_f2_bpc(inst).f2
    opts = bpc.PricingOptions(sr_cuts=16, sr_per_round=8, sr_depth=2, sr_min_viol=0.01)
    for b2 in (f2_min, f2_min * 1.5, f2_min * 3):
        b2 = bpc.grid_floor(inst, "f2", b2)
        ref, _ = bpc._branch_and_price(inst, 1.0, 0.0, bound2_value=b2)
        obj, a = bpc._branch_and_price(inst, 1.0, 0.0, bound2_value=b2, pricing=opts)
        assert obj == pytest.approx(ref, abs=1e-6)
        assert inst.f2_of(a) <= b2 + 1e-6
