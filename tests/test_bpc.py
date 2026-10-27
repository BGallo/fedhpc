"""Phase 1 correctness tests for the BPC engine (src/fedhpc/bpc.py).

The load-bearing check: the root LP relaxation of the column-generation
master must equal the LP relaxation of the compact SpaceTimeFormulation MIP
exactly (Geoffrion's integrality property — each type's pricing subproblem
polytope is a pure DAG flow, hence integral). A mismatch means the pricing
arc-cost bookkeeping is wrong, not a branching issue (there is no branching
yet). See src/fedhpc/bpc.py's module docstring for the design.

The trap instances have running jobs, which BPC absorbs as pinned
pseudo-jobs; the compact model handles them through its flow boundary and
ghost completions, so equal bounds also check that rewrite.
"""
from __future__ import annotations

import pytest
from gurobipy import GRB

from conftest import TRAP_INSTANCES
from fedhpc.bpc import solve_root_relaxation
from fedhpc.data import Instance
from fedhpc.formulations import SpaceTimeFormulation, configure_env

configure_env()

_WEIGHTS = [
    (1.0, 0.0),   # lex-min f1 direction
    (0.0, 1.0),   # lex-min f2 direction
    (0.5, 0.5),
    (0.3, 0.7),
]

def _compact_lp_bound(inst: Instance, weight1: float, weight2: float) -> float:
    fmt = SpaceTimeFormulation()
    mdl, vars_ = fmt.build(inst)
    for v in mdl.getVars():
        v.vtype = GRB.CONTINUOUS
    x = vars_["x"]
    f1 = fmt.f1_expr(inst, x)
    f2 = fmt.f2_expr(inst, x)
    mdl.setObjective(weight1 * f1 + weight2 * f2, GRB.MINIMIZE)
    mdl.optimize()
    assert mdl.Status == GRB.OPTIMAL, f"compact LP relaxation not optimal: {mdl.Status}"
    return mdl.ObjVal


@pytest.mark.parametrize("weight1,weight2", _WEIGHTS)
@pytest.mark.parametrize("name", sorted(TRAP_INSTANCES))
def test_root_lp_bound_matches_compact_on_trap_instances(name, weight1, weight2):
    inst = TRAP_INSTANCES[name]()

    compact_bound = _compact_lp_bound(inst, weight1, weight2)
    result = solve_root_relaxation(inst, weight1, weight2)

    assert result.artificial_total == pytest.approx(0.0, abs=1e-6)
    assert result.objective == pytest.approx(compact_bound, rel=1e-6, abs=1e-4)


@pytest.mark.parametrize("weight1,weight2", _WEIGHTS)
def test_root_lp_bound_matches_compact_on_known_pareto_inst(known_pareto_inst, weight1, weight2):
    inst = known_pareto_inst
    compact_bound = _compact_lp_bound(inst, weight1, weight2)
    result = solve_root_relaxation(inst, weight1, weight2)

    assert result.artificial_total == pytest.approx(0.0, abs=1e-6)
    assert result.objective == pytest.approx(compact_bound, rel=1e-6, abs=1e-4)


def test_root_relaxation_with_bound_row(known_pareto_inst):
    """Sanity check the optional bound row (mirrors _make_lexmin_bounded)."""
    inst = known_pareto_inst

    # min f1 s.t. f2 <= 50 should match the compact model bounded the same way.
    fmt = SpaceTimeFormulation()
    mdl, vars_ = fmt.build(inst)
    for v in mdl.getVars():
        v.vtype = GRB.CONTINUOUS
    x = vars_["x"]
    f1, f2 = fmt.f1_expr(inst, x), fmt.f2_expr(inst, x)
    mdl.addConstr(f2 <= 50.0)
    mdl.setObjective(f1, GRB.MINIMIZE)
    mdl.optimize()
    assert mdl.Status == GRB.OPTIMAL
    compact_bound = mdl.ObjVal

    result = solve_root_relaxation(inst, 1.0, 0.0, bound2_value=50.0)
    assert result.artificial_total == pytest.approx(0.0, abs=1e-6)
    assert result.objective == pytest.approx(compact_bound, rel=1e-6, abs=1e-4)


@pytest.mark.parametrize("weight1,weight2", _WEIGHTS)
def test_root_lp_bound_matches_compact_with_job_weights_and_fixed_cost(
    weighted_fixed_cost_inst, weight1, weight2
):
    """job.weight != 1 and a type's fixed_cost > 0 (z_m + per-job linking
    rows) must not break the root LP bound's equality with the compact
    formulation's LP relaxation — see bpc.py's module docstring.
    """
    inst = weighted_fixed_cost_inst
    compact_bound = _compact_lp_bound(inst, weight1, weight2)
    result = solve_root_relaxation(inst, weight1, weight2)

    assert result.artificial_total == pytest.approx(0.0, abs=1e-6)
    assert result.objective == pytest.approx(compact_bound, rel=1e-6, abs=1e-4)
