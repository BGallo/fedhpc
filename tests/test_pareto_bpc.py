"""Phase 4 correctness tests: BPC wired into pareto.py's exact box search.

Validates ``true_pareto_frontier``/``map_pareto_frontier``'s new
``engine="bpc"`` parameter against the existing ``engine="gurobi"`` (compact
MIP) results — same algorithm, same ``_make_engine_lexmin_bounded`` dispatch
point, only the per-box solve differs (see ``pareto.py``'s and
``bpc.make_bpc_lexmin_bounded``'s docstrings). Kept to tiny instances:
``true_pareto_frontier`` issues ~2 box-solves per frontier point, and each
BPC box-solve is a full branch-and-price call, so this suite's runtime scales
with both the front's size and each instance's own solve cost (already
documented as slow on tiny_large/tiny_xlarge for some objective directions
— see bpc.py's module docstring) — larger instances were checked manually,
not included here, to keep this suite's runtime reasonable.
"""
from __future__ import annotations

import pytest

from fedhpc.data import Instance
from fedhpc.formulations import configure_env
from fedhpc.pareto import map_pareto_frontier, true_pareto_frontier

configure_env()


def _points(solutions) -> set[tuple[float, float]]:
    return {(round(s.f1, 3), round(s.f2, 4)) for s in solutions}


@pytest.mark.parametrize("engine_fn", [true_pareto_frontier, map_pareto_frontier])
def test_bpc_engine_matches_gurobi_on_tiny_smallest(engine_fn):
    inst = Instance.from_file("data/tiny_smallest.json")
    gurobi = engine_fn(inst, engine="gurobi")
    bpc = engine_fn(inst, engine="bpc")
    assert _points(bpc) == _points(gurobi)


@pytest.mark.parametrize("engine_fn", [true_pareto_frontier, map_pareto_frontier])
def test_bpc_engine_matches_gurobi_on_known_pareto_inst(engine_fn, known_pareto_inst):
    gurobi = engine_fn(known_pareto_inst, engine="gurobi")
    bpc = engine_fn(known_pareto_inst, engine="bpc")
    assert _points(bpc) == _points(gurobi)


def test_bpc_engine_matches_gurobi_with_job_weights_and_fixed_cost(weighted_fixed_cost_inst):
    """Exercises the box search's engine dispatch together with the two
    formulation extras (Job.weight, InstanceType.fixed_cost) BPC's pricing
    has its own dedicated handling for — see bpc.py's module docstring.
    """
    inst = weighted_fixed_cost_inst
    gurobi = true_pareto_frontier(inst, engine="gurobi")
    bpc = true_pareto_frontier(inst, engine="bpc")
    assert _points(bpc) == _points(gurobi)


def test_unknown_engine_rejected(known_pareto_inst):
    with pytest.raises(ValueError):
        true_pareto_frontier(known_pareto_inst, engine="not-a-real-engine")
