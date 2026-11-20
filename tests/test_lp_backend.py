"""The HiGHS backend of :mod:`fedhpc.lpfirst` (``fedhpc.lp_backend``)
against the Gurobi one: same LP values and dual-sign conventions (the
Lagrangian bound at HiGHS's duals equals the LP), and the same proven
optima through every proof step."""
from __future__ import annotations

import pytest

from conftest import TRAP_INSTANCES
from fedhpc import bpc, lp_backend, lpfirst
from fedhpc.formulations import configure_env

pytestmark = pytest.mark.skipif(not lp_backend.available("highs"), reason="highspy not installed")
configure_env()


def _cases():
    from test_lpfirst import _gap_instance
    from test_lpfirst_primal import _dup_instance
    return {**{k: f() for k, f in TRAP_INSTANCES.items()},
            **{f"gap{s}": _gap_instance(s) for s in (0, 5, 24)},
            **{f"dup{s}": _dup_instance(s) for s in (0, 1)}}


@pytest.mark.parametrize("name", ["trap_small", "trap_medium", "gap0", "gap5", "dup0"])
@pytest.mark.parametrize("w", [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
def test_full_master_lp_and_duals_agree(name, w):
    inst = _cases()[name]
    ti = lpfirst._TI(inst, bpc._prepare(inst))
    out = {}
    for be in ("gurobi", "highs"):
        m = lpfirst._Master(ti, *w, None, None, bpc._big_m_for(inst), backend=be)
        m.add(range(ti.n_arcs))
        assert m.solve_lp() == "optimal"
        d = m.duals()
        L, _ = lpfirst._lagrangian(ti, d, lpfirst._price(ti, d, *w), None, None)
        out[be] = (m.objective(), L)
        # strong duality at the full master's optimal duals
        assert L == pytest.approx(m.objective(), abs=1e-6)
    assert out["gurobi"][0] == pytest.approx(out["highs"][0], abs=1e-6)


@pytest.mark.parametrize("name", ["trap_small", "trap_medium", "gap0", "gap5", "gap24", "dup0",
                                  "dup1"])
@pytest.mark.parametrize("w", [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.3e-5, 2.1e-5)])
def test_highs_optimum_matches_gurobi(name, w):
    inst = _cases()[name]
    ref = lpfirst.solve_lpfirst(inst, *w)
    res = lpfirst.solve_lpfirst(inst, *w, options=lpfirst.LPFirstOptions(backend="highs"))
    assert res.status == ref.status == "optimal"
    assert res.objective == pytest.approx(ref.objective, rel=1e-9, abs=1e-9)
    inst.check_schedule(res.assignment)


@pytest.mark.parametrize("opts", [
    dict(heuristics=False, rmp_mip_time=0.0, tree=False),            # eligible MIP alone
    dict(heuristics=False, rmp_mip_time=0.0, max_eligible=0, node_eligible=0),   # tree alone
    dict(heuristics=False, rmp_mip_time=0.0, max_eligible=0, node_eligible=8),   # + node MIPs
], ids=["eligible", "tree", "tree_mip"])
@pytest.mark.parametrize("name", ["gap0", "gap5", "gap24", "dup0", "trap_medium"])
def test_highs_proof_steps(name, opts):
    inst = _cases()[name]
    w = (1.0, 1.0)
    ref, _ = bpc._branch_and_price(inst, *w)
    seeds = bpc.heuristic_schedules(inst)
    worst = max(seeds, key=lambda a: lpfirst._objective_of(inst, a, *w))
    res = lpfirst.solve_lpfirst(inst, *w, seed_schedules=[worst],
                                options=lpfirst.LPFirstOptions(backend="highs", **opts))
    assert res.status == "optimal"
    assert res.objective == pytest.approx(ref, abs=1e-6)


def test_highs_infeasible_box(trap_small):
    f1_min = bpc.solve_f1_bpc(trap_small).f1
    res = lpfirst.solve_lpfirst(trap_small, 0.0, 1.0, f1_min - 1, None,
                                options=lpfirst.LPFirstOptions(backend="highs"))
    assert res.status == "infeasible" and res.assignment is None
