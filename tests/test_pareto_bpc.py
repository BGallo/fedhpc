"""Phase 4 correctness tests: BPC wired into pareto.py's exact box search.

Validates ``true_pareto_frontier``/``map_pareto_frontier``'s new
``engine="bpc"`` parameter against the existing ``engine="gurobi"`` (compact
MIP) results — same algorithm, same ``_make_engine_lexmin_bounded`` dispatch
point, only the per-box solve differs (see ``pareto.py``'s and
``bpc.make_bpc_lexmin_bounded``'s docstrings). Kept to trap_small (10-point
front, running jobs): ``true_pareto_frontier`` issues ~2 box-solves per
frontier point, and each BPC box-solve is a full branch-and-price call, so
this suite's runtime scales with both the front's size and each instance's
own solve cost.
"""
from __future__ import annotations

import pytest

from fedhpc.formulations import configure_env
from fedhpc.pareto import map_pareto_frontier, true_pareto_frontier

configure_env()


def _points(solutions) -> set[tuple[float, float]]:
    return {(round(s.f1, 3), round(s.f2, 4)) for s in solutions}


@pytest.mark.parametrize("engine_fn", [true_pareto_frontier, map_pareto_frontier])
def test_bpc_engine_matches_gurobi_on_trap_small(engine_fn, trap_small):
    inst = trap_small
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


# ---------------------------------------------------------------------------
# Time limits
# ---------------------------------------------------------------------------

def test_branch_and_price_time_limit_raises(trap_small):
    from fedhpc import bpc
    with pytest.raises(bpc.BPCTimeLimit) as exc:
        bpc._branch_and_price(trap_small, 1.0, 0.0, time_limit=0.0)
    assert exc.value.assignment is None


def test_per_solve_time_limit_reaches_the_bpc_engine(trap_small, capsys):
    """A 0 s cap leaves the first anchor inconclusive: no points, a warning."""
    assert true_pareto_frontier(trap_small, engine="bpc", per_solve_time_limit=0.0) == []
    assert "INCONCLUSIVE" in capsys.readouterr().err


def test_timed_out_phases_are_feasible_not_optimal(trap_small, monkeypatch):
    """A phase-1 timeout returns its incumbent and a phase-2 timeout the
    phase-1 schedule, both as "feasible"."""
    from fedhpc import bpc
    real = bpc._branch_and_price
    exact = real(trap_small, 1.0, 0.0)

    def phase1_times_out(*a, **k):
        raise bpc.BPCTimeLimit(*exact)
    monkeypatch.setattr(bpc, "_branch_and_price", phase1_times_out)
    lexmin, _ = bpc.make_bpc_lexmin_bounded(trap_small, time_limit=1.0)
    sol = lexmin("f1", "f2")
    assert sol.status == "feasible" and sol.assignment == exact[1]

    calls = []

    def phase2_times_out(*a, **k):
        calls.append(k.get("time_limit"))
        if len(calls) == 2:
            raise bpc.BPCTimeLimit()
        return real(*a, **k)
    monkeypatch.setattr(bpc, "_branch_and_price", phase2_times_out)
    lexmin, _ = bpc.make_bpc_lexmin_bounded(trap_small, time_limit=60.0)
    sol = lexmin("f1", "f2")
    assert sol.status == "feasible"
    assert sol.f1 == pytest.approx(exact[0])
    assert calls == [60.0, 60.0]
