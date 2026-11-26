"""Branch-and-price tree (Phase 3): reduced-cost fixing, type/time-window
branching, best-first node selection and the open bound on timeout."""
from __future__ import annotations

import math

import pytest

from conftest import TRAP_INSTANCES
from fedhpc import bpc
from fedhpc.formulations import configure_env

configure_env()

pytestmark = pytest.mark.skipif(bpc._bpcprice is None, reason="_bpcprice extension not built")

_TRAPS = sorted(TRAP_INSTANCES)
_W = [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]


def _root_cg(inst, w, cutoff):
    prep = bpc._prepare(inst)
    big_m = bpc._big_m_for(inst)
    seed = {bpc._column_key(c): c for c in prep.static_columns}
    for c in bpc._heuristic_seed_columns(inst, prep):
        seed.setdefault(bpc._column_key(c), c)
    mdl, state = bpc._build_rmp(inst, *w, None, None, big_m, extra_columns=list(seed.values()),
                                prep=prep, activation=True)
    return prep, bpc._run_column_generation(mdl, state, inst, *w, prep, 10_000, 1e-7,
                                            cutoff=cutoff, big_m=big_m)


@pytest.mark.parametrize("name", _TRAPS)
@pytest.mark.parametrize("w", _W)
def test_fixing_keeps_every_optimal_placement(name, w):
    """With a cutoff just above the optimum, the optimal schedule is
    "improving", so none of its placements may be fixed — while a cutoff at
    the root bound fixes something (the test is not vacuous)."""
    inst = TRAP_INSTANCES[name]()
    opt, a = bpc._branch_and_price(inst, *w)
    prep, cg = _root_cg(inst, w, opt + 1e-3)
    assert cg.status in ("converged", "bound")
    placements = {(jid, mid, t) for jid, (mid, t) in a.items()}
    assert not placements & cg.fixed
    _prep, tight = _root_cg(inst, w, cg.objective + 1e-6)
    assert tight.fixed, "a cutoff at the LP bound should fix some placement"


@pytest.mark.parametrize("name", _TRAPS)
def test_time_branch_partitions_the_placements(name):
    inst = TRAP_INSTANCES[name]()
    prep = bpc._prepare(inst)
    jid = inst.jobs[0].id
    mids = [m for m in inst.F[jid] if (jid, m) in prep.starts]
    # split across two types
    m1, m2 = mids[:2]
    t1, t2 = min(prep.starts[jid, m1]), min(prep.starts[jid, m2])
    first, second = bpc._time_branch(inst, prep, {(jid, m1, t1): 0.6, (jid, m2, t2): 0.4}, {jid})
    assert not first & second
    assert (jid, m1, t1) in second and (jid, m2, t2) in first    # m1 holds more mass: kept first
    # split in time on one type
    ts = sorted(prep.starts[jid, m1])
    if len(ts) >= 3:
        agg = {(jid, m1, ts[0]): 0.3, (jid, m1, ts[1]): 0.3, (jid, m1, ts[2]): 0.4}
        first, second = bpc._time_branch(inst, prep, agg, {jid})
        assert not first & second
        assert first | second == {(jid, m1, t) for t in ts}


@pytest.mark.parametrize("name", _TRAPS)
@pytest.mark.parametrize("w", _W)
@pytest.mark.parametrize("tree", [
    dict(node_selection="best", branching="time"),
    dict(node_selection="dfs", branching="time"),
    dict(node_selection="best", branching="rf"),
    dict(node_selection="dfs", branching="rf", pricing=bpc.PricingOptions(fixing=False)),
    dict(node_selection="best", branching="time", heuristic_time_limit=0.0, seed_schedules=[]),
])
def test_tree_options_agree(name, w, tree):
    inst = TRAP_INSTANCES[name]()
    ref, _ = bpc._branch_and_price(inst, *w, node_selection="dfs", branching="rf",
                                   pricing=bpc.PricingOptions(engine="python"))
    obj, a = bpc._branch_and_price(inst, *w, **tree)
    assert obj == pytest.approx(ref, abs=1e-6)
    inst.check_schedule(a)


def test_box_solves_agree_under_bound_rows(trap_medium):
    """Bound rows (Pareto boxes) with fixing and the new tree: same optimum
    as the old tree without fixing."""
    inst = trap_medium
    f2_min = bpc.solve_f2_bpc(inst).f2
    for b2 in (f2_min, f2_min * 1.5, f2_min * 3):
        b2 = bpc.grid_floor(inst, "f2", b2)
        old, _ = bpc._branch_and_price(inst, 1.0, 0.0, bound2_value=b2, node_selection="dfs",
                                       branching="rf", pricing=bpc.PricingOptions(fixing=False))
        new, a = bpc._branch_and_price(inst, 1.0, 0.0, bound2_value=b2)
        assert new == pytest.approx(old, abs=1e-6)
        assert inst.f2_of(a) <= b2 + 1e-6


def test_timeout_reports_a_valid_open_bound(trap_medium):
    inst = trap_medium
    f2_min = bpc.solve_f2_bpc(inst).f2
    opt, _ = bpc._branch_and_price(inst, 1.0, 0.0, bound2_value=f2_min * 1.5)
    stats: dict = {}
    try:
        bpc._branch_and_price(inst, 1.0, 0.0, bound2_value=f2_min * 1.5, time_limit=0.05,
                              stats=stats, heuristic_time_limit=0.0)
    except bpc.BPCTimeLimit as e:
        assert e.bound <= opt + 1e-6
        assert e.objective >= opt - 1e-6
    else:
        assert stats["open_bound"] == pytest.approx(opt)
    assert not math.isnan(stats.get("open_bound", 0.0))


@pytest.mark.parametrize("name", _TRAPS)
@pytest.mark.parametrize("w", _W)
def test_threaded_pricing_matches_sequential(name, w):
    """Pricing the path types concurrently changes neither the columns nor
    the bound: root column generation follows the same path."""
    inst = TRAP_INSTANCES[name]()
    out = []
    for threads in (1, 4):
        stats: dict = {}
        obj, _a = bpc._branch_and_price(inst, *w, stats=stats,
                                        pricing=bpc.PricingOptions(threads=threads))
        out.append((obj, stats["cg_iters"], stats["nodes"], stats["root_bound"]))
    assert out[0] == out[1]


@pytest.mark.parametrize("name", _TRAPS)
@pytest.mark.parametrize("w", _W)
def test_root_heuristic_and_grid_stop_keep_the_optimum(name, w):
    """The RMP heuristic during root column generation and the grid stop
    only change when nodes end, never the optimum."""
    inst = TRAP_INSTANCES[name]()
    base, _a = bpc._branch_and_price(inst, *w, heuristic_time_limit=0.0)
    stats: dict = {}
    obj, a = bpc._branch_and_price(inst, *w, stats=stats)
    assert obj == pytest.approx(base, abs=1e-6)
    assert bpc._schedule_objective(inst, a, *w, None, None) == pytest.approx(obj, abs=1e-6)
