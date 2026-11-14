"""Phase-1 structure of the BPC engine (``bpc._Prep``): exact start-set
reductions, release groups for running jobs, static single-job columns on
unbounded types, schedule decomposition, and reuse of columns, schedules and
preprocessing across solves.
"""
from __future__ import annotations

import pytest

from conftest import TRAP_INSTANCES
from fedhpc import bpc
from fedhpc.formulations import configure_env, unbounded_types
from fedhpc.model import solve_f1, solve_f2

configure_env()

_TRAPS = sorted(TRAP_INSTANCES)


@pytest.mark.parametrize("name", _TRAPS)
def test_prep_structure(name):
    inst = TRAP_INSTANCES[name]()
    prep = bpc._prepare(inst)
    assert prep.unbounded == frozenset(unbounded_types(inst))
    assert set(prep.path_types).isdisjoint(prep.unbounded)
    for mid in prep.path_types:
        itype = next(m for m in inst.instance_types if m.id == mid)
        # every instance is in exactly one release group
        n_late = sum(n for (m, r), n in inst.rj_arrivals.items() if m == mid and r >= inst.horizon)
        assert sum(prep.releases[mid].values()) == itype.capacity - n_late
        assert prep.releases[mid].get(0, 0) == itype.capacity - inst.n_running.get(mid, 0)
    for col in prep.static_columns:
        assert col.type_id in prep.unbounded and len(col.jobs) == 1
    # R1: one start per (job, unbounded type), at the job's release
    for (jid, mid), ts in prep.starts.items():
        if mid in prep.unbounded:
            assert list(ts) == [inst.T[jid, mid].start]
        assert set(ts) <= set(inst.T[jid, mid])


@pytest.mark.parametrize("name", _TRAPS)
def test_schedule_columns_round_trip(name):
    inst = TRAP_INSTANCES[name]()
    prep = bpc._prepare(inst)
    for a in bpc.heuristic_schedules(inst):
        cols = bpc._schedule_columns(prep, a)
        rebuilt = {jid: (c.type_id, t) for c in cols for jid, t in c.jobs}
        assert rebuilt == a
        used: dict[tuple[int, int], int] = {}
        for c in cols:
            if c.type_id in prep.unbounded:
                continue
            used[c.type_id, c.release] = used.get((c.type_id, c.release), 0) + 1
            ends = [c.release] + [t + inst.p_occ[j, c.type_id] for j, t in c.jobs]
            starts = [t for _j, t in c.jobs]
            assert all(e <= s for e, s in zip(ends, starts))   # a chain never overlaps
        for (mid, r), n in used.items():
            assert n <= prep.releases[mid][r]


@pytest.mark.parametrize("primary", ["f1", "f2"])
@pytest.mark.parametrize("name", _TRAPS)
def test_reductions_are_exact(name, primary):
    """Reduced and unreduced start sets give the same lex-min point, and both
    match the compact model."""
    inst = TRAP_INSTANCES[name]()
    solve = bpc.solve_f1_bpc if primary == "f1" else bpc.solve_f2_bpc
    ref = (solve_f1 if primary == "f1" else solve_f2)(inst)
    for reduce in (True, False):
        sol = solve(inst, reduce=reduce)
        assert sol.status == "optimal"
        assert (sol.f1, sol.f2) == pytest.approx((ref.f1, ref.f2), abs=1e-6), reduce


@pytest.mark.parametrize("name", _TRAPS)
def test_reduced_root_bound_not_weaker(name):
    """Restricting starts can only tighten the root LP."""
    inst = TRAP_INSTANCES[name]()
    for w in ((1.0, 0.0), (0.0, 1.0), (0.5, 0.5)):
        full = bpc.solve_root_relaxation(inst, *w, reduce=False).objective
        red = bpc.solve_root_relaxation(inst, *w).objective
        assert red >= full - 1e-6


def test_no_seeds_and_pool_reuse_give_same_optimum(trap_medium):
    inst = trap_medium
    ref, a_ref = bpc._branch_and_price(inst, 1.0, 0.0)
    obj, a = bpc._branch_and_price(inst, 1.0, 0.0, seed_schedules=[])
    assert obj == pytest.approx(ref)
    pool: dict = {}
    bpc._branch_and_price(inst, 0.0, 1.0, pool_out=pool)
    assert pool and all(c.jobs for c in pool.values())
    obj2, _ = bpc._branch_and_price(inst, 1.0, 0.0, column_pool=list(pool.values()),
                                    seed_schedules=[])
    assert obj2 == pytest.approx(ref)


def test_seed_incumbent_respects_bound_rows(trap_small):
    """A seed schedule outside the bound rows must not become the incumbent."""
    inst = trap_small
    f2_min = bpc.solve_f2_bpc(inst).f2
    obj, a = bpc._branch_and_price(inst, 1.0, 0.0, bound2_value=f2_min)
    assert a is not None and inst.f2_of(a) <= f2_min + 1e-6
    expected = solve_f2(inst).f1
    assert obj == pytest.approx(expected)
