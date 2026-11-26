"""The VRPSolver engine (fedhpc.vrpsolver).

The exporter is checked on its own: an exported model, solved as the
equivalent compact MIP with HiGHS, must reach the same optimum as BPC. The
Julia model build (``--dry-run``) runs when Julia and the ``vrpsolver/``
project are set up (``FEDHPC_JULIA``), and the full solve only when VRPSolver
itself is available (``BAPCOD_RCSP_LIB``, a BaPCod build with CLP or CPLEX).
"""
import os
import shutil

import pytest

from fedhpc import bpc, vrpsolver

pytest.importorskip("highspy")

WEIGHTS = [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 1e-3)]


def _check_against_bpc(inst, w1, w2, b1=None, b2=None):
    model = vrpsolver.export_model(inst, w1, w2, b1, b2)
    res = vrpsolver.solve_exported_mip(model)
    try:
        ref = bpc.solve_weighted_sum_bpc(inst, w1, w2, bound1_value=b1, bound2_value=b2)
    except RuntimeError:
        ref = None
    if ref is None or ref.objective is None:
        assert res["objective"] is None
        return model, res
    assert res["objective"] == pytest.approx(ref.objective, rel=1e-9, abs=1e-6)
    # the schedule is feasible and its value is the reported one
    inst.check_schedule(res["assignment"])
    comp = {j: t + inst.p_occ[j, m] for j, (m, t) in res["assignment"].items()}
    value = w1 * inst.f1_of(comp) + w2 * inst.f2_of(res["assignment"])
    assert value == pytest.approx(res["objective"], rel=1e-9, abs=1e-6)
    return model, res


@pytest.mark.parametrize("w", WEIGHTS)
def test_export_matches_bpc_trap(trap_inst, w):
    _check_against_bpc(trap_inst, *w)


@pytest.mark.parametrize("w", WEIGHTS)
def test_export_matches_bpc_fixed_cost(weighted_fixed_cost_inst, w):
    model, _ = _check_against_bpc(weighted_fixed_cost_inst, *w)
    assert model["fixed"], "the fixture has a fixed-cost type"


def test_export_with_caps(trap_small):
    free = vrpsolver.solve_exported_mip(vrpsolver.export_model(trap_small, 1.0, 0.0))
    comp = {j: t + trap_small.p_occ[j, m] for j, (m, t) in free["assignment"].items()}
    f2 = trap_small.f2_of(free["assignment"])
    # lex-min (f1, f2): cap f1 at its optimum, minimise f2
    _check_against_bpc(trap_small, 0.0, 1.0, b1=trap_small.f1_of(comp))
    if f2 > 0:   # a binding cost cap moves the f1 optimum
        _check_against_bpc(trap_small, 1.0, 0.0, b2=f2 / 2)


def test_export_structure(trap_medium):
    m = vrpsolver.export_model(trap_medium, 1.0, 0.0)
    assert m["format"] == vrpsolver.FORMAT
    assert m["objective"]["integral"], "f1 with integer weights lies on a grid"
    n_ps = len(m["packing_jobs"])
    for pt in m["path_types"]:
        assert len(pt["dist"]) == n_ps and all(len(r) == n_ps for r in pt["dist"])
        assert all(n > 0 for _r, n in pt["releases"])
        for j, t, p, o, *_ in pt["arcs"]:
            assert 0 <= t and t + p <= m["horizon"]
            assert o == int(o)
    assert m["cutoff"] is not None, "the heuristics give a cutoff"


def test_export_rejects_multinode(trap_small):
    trap_small.jobs[0].nodes = 2
    with pytest.raises(Exception):
        vrpsolver.export_model(trap_small, 1.0, 0.0)


def _julia_ready() -> bool:
    cmd = os.environ.get("FEDHPC_JULIA", "julia").split()[0]
    return shutil.which(cmd) is not None and os.environ.get("FEDHPC_VRPSOLVER_DRYRUN") == "1"


@pytest.mark.skipif(not _julia_ready(),
                    reason="set FEDHPC_VRPSOLVER_DRYRUN=1 with Julia and the vrpsolver/ project")
def test_julia_dry_run(trap_medium):
    model = vrpsolver.export_model(trap_medium, 1.0, 1.0)
    res = vrpsolver.run_julia(model, dry_run=True)
    assert res["status"] == "dry_run"
    assert res["graphs"] == sum(len(pt["releases"]) for pt in model["path_types"])
    assert res["packing_sets"] == len(model["packing_jobs"])


@pytest.mark.skipif(not vrpsolver.available()[0], reason=vrpsolver.available()[1] or "n/a")
@pytest.mark.parametrize("w", [(1.0, 0.0), (1.0, 1.0)])
def test_vrpsolver_matches_bpc(trap_inst, w):
    ref = bpc.solve_weighted_sum_bpc(trap_inst, *w)
    sol = vrpsolver.solve_vrpsolver(trap_inst, *w, cutoff=None)
    assert sol.status == "optimal"
    assert sol.objective == pytest.approx(ref.objective, rel=1e-9, abs=1e-6)
