"""Branch and price with fedlp restricted masters (``PricingOptions(master=
"fedlp")``): the same optima as with Gurobi masters, under weights, bound
rows, lex-min phases, fixed costs and a Pareto search."""
from __future__ import annotations

import pytest

from conftest import TRAP_INSTANCES
from fedhpc import bpc, pareto
from fedhpc.formulations import configure_env

configure_env()

pytestmark = pytest.mark.skipif(bpc._bpcprice is None or bpc._fedlp is None,
                                reason="_bpcprice / _fedlp extensions not built")

FEDLP = bpc.PricingOptions(master="fedlp")
GUROBI = bpc.PricingOptions(master="gurobi")
_TRAPS = sorted(TRAP_INSTANCES)


def _rm_m2n10():
    from fedhpc import rm_benchmark as rm
    from test_rm_benchmark import M2N10_1
    data = rm.parse(M2N10_1)
    return rm.to_instance(data, horizon=sum(max(r) for r in data[0]))


@pytest.mark.parametrize("name", _TRAPS)
@pytest.mark.parametrize("w", [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.3, 0.7)])
def test_weighted_optimum_matches_gurobi_master(name, w):
    inst = TRAP_INSTANCES[name]()
    ref, _a = bpc._branch_and_price(inst, *w, pricing=GUROBI)
    stats: dict = {}
    obj, a = bpc._branch_and_price(inst, *w, pricing=FEDLP, stats=stats)
    assert obj == pytest.approx(ref, abs=1e-6)
    assert bpc._schedule_objective(inst, a, *w, None, None) == pytest.approx(obj, abs=1e-6)


@pytest.mark.parametrize("name", _TRAPS)
def test_bound_rows_match_gurobi_master(name):
    inst = TRAP_INSTANCES[name]()
    _o, a1 = bpc._branch_and_price(inst, 1.0, 0.0, pricing=GUROBI)
    f2_cap = inst.f2_of(a1) * 1.5
    ref, _ = bpc._branch_and_price(inst, 1.0, 0.0, bound2_value=f2_cap, pricing=GUROBI)
    obj, _ = bpc._branch_and_price(inst, 1.0, 0.0, bound2_value=f2_cap, pricing=FEDLP)
    assert obj == pytest.approx(ref, abs=1e-6)


@pytest.mark.parametrize("primary", ["f1", "f2"])
def test_lexmin_matches_gurobi_master(primary):
    inst = TRAP_INSTANCES["trap_medium"]()
    solve = bpc.solve_f1_bpc if primary == "f1" else bpc.solve_f2_bpc
    ref = solve(inst, pricing=GUROBI)
    got = solve(inst, pricing=FEDLP)
    assert (got.f1, got.f2) == pytest.approx((ref.f1, ref.f2), abs=1e-6)


def test_fixed_cost_rm_instance_matches_published_optimum():
    inst = _rm_m2n10()
    obj, _ = bpc._branch_and_price(inst, 1.0, 1.0, pricing=FEDLP)
    assert obj == 145220


def test_pareto_front_matches_gurobi_master(monkeypatch):
    inst = TRAP_INSTANCES["trap_small"]()
    monkeypatch.setattr(bpc, "DEFAULT_PRICING", GUROBI)
    ref = sorted((round(s.f1, 6), round(s.f2, 6)) for s in pareto.true_pareto_frontier(inst, engine="bpc"))
    monkeypatch.setattr(bpc, "DEFAULT_PRICING", FEDLP)
    got = sorted((round(s.f1, 6), round(s.f2, 6)) for s in pareto.true_pareto_frontier(inst, engine="bpc"))
    assert got == ref


def test_auto_master_picks_by_size(monkeypatch):
    """"auto" gives small masters to fedlp and large ones to Gurobi."""
    inst = TRAP_INSTANCES["trap_small"]()
    kinds = []
    orig = bpc._build_rmp

    def spy(*a, **k):
        out = orig(*a, **k)
        kinds.append(out[1].master.kind)
        return out
    monkeypatch.setattr(bpc, "_build_rmp", spy)
    bpc._branch_and_price(inst, 1.0, 0.0, pricing=bpc.PricingOptions(master="auto"))
    assert kinds and set(kinds) == {"fedlp"}
    kinds.clear()
    monkeypatch.setattr(bpc, "_FEDLP_MAX_ROWS", 0)
    bpc._branch_and_price(inst, 1.0, 0.0, pricing=bpc.PricingOptions(master="auto"))
    assert set(kinds) == {"gurobi"}
