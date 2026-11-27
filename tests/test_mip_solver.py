"""fedhpc.mip end to end against Gurobi: random MIPs (pure and mixed, binary
and general integer, infeasible ones included), each component switched
off in turn (a stage must not be hiding another's bug), and a small Kramer
arc-flow model; plus the MPS reader against HiGHS's on MIPLIB files."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

gp = pytest.importorskip("gurobipy")

from fedhpc.formulations import configure_env  # noqa: E402

try:
    from fedhpc import _fedlp
except ImportError:  # extension not built
    _fedlp = None

pytestmark = pytest.mark.skipif(_fedlp is None, reason="_fedlp extension not built")
configure_env()
ROOT = Path(__file__).resolve().parents[1]


def gurobi_opt(P):
    from fedhpc import formulations
    g = gp.Model(env=formulations._shared_env)
    g.Params.OutputFlag = 0
    x = g.addMVar(P.n, lb=P.lb, ub=P.ub, vtype=np.where(P.integer, "I", "C"), obj=P.c)
    A = P.A.toarray()
    for i in range(P.m):
        if np.isfinite(P.row_lo[i]):
            g.addConstr(A[i] @ x >= P.row_lo[i])
        if np.isfinite(P.row_hi[i]):
            g.addConstr(A[i] @ x <= P.row_hi[i])
    g.optimize()
    return (g.ObjVal + P.offset) if g.Status == 2 else None


VARIANTS = {
    "default": {},
    "no_presolve": {"presolve": False},
    "no_cuts": {"cut_rounds": 0},
    "no_heuristics": {"heuristics": False},
}


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("seed", range(10))
def test_random_mips_match_gurobi(seed, variant):
    import sys
    sys.path.insert(0, str(ROOT / "tests"))
    from test_mip_components import random_ip

    from fedhpc.mip import solve
    rng = random.Random(3000 + seed)
    P, _ = random_ip(rng, rng.randint(3, 12), rng.randint(4, 22), box=rng.choice([1, 3, 6]),
                     mixed=rng.random() < 0.4)
    want = gurobi_opt(P)
    r = solve(P, time_limit=30, verbose=False, **VARIANTS[variant])
    if want is None:
        assert r.status == "infeasible"
        return
    assert r.status == "optimal"
    assert r.objective == pytest.approx(want, abs=1e-6 * max(1, abs(want)))
    assert P.is_feasible(r.x)
    assert r.bound <= want + 1e-6 * max(1, abs(want))


def test_kramer_arc_flow_n20():
    """A small P|r_j|sum w_j C_j arc-flow model built as bench_prjwc builds
    it, through MPS, against Gurobi's optimum."""
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import bench_prjwc as bp

    from fedhpc import formulations
    from fedhpc.mip import read_mps, solve
    rng = random.Random(7)
    n, m = 10, 2
    p = [rng.randint(1, 10) for _ in range(n)]
    w = [rng.randint(1, 10) for _ in range(n)]
    r = [rng.randint(0, 15) for _ in range(n)]
    T = bp.horizon(p, r, m)
    mdl = bp.build_af(p, w, r, m, T, formulations._shared_env)
    mdl.Params.OutputFlag = 0
    path = ROOT / "results" / "_test_af_n10.mps"
    path.parent.mkdir(exist_ok=True)
    mdl.write(str(path))
    try:
        mdl.optimize()
        want = mdl.ObjVal
        res = solve(read_mps(path), time_limit=60, verbose=False, gap_rel=1e-9)
    finally:
        path.unlink(missing_ok=True)
    assert res.status == "optimal"
    assert res.objective == pytest.approx(want)


@pytest.mark.parametrize("name", ["bppc4-08", "csched008", "fhnw-binpack4-4"])
def test_mps_reader_matches_highs(name):
    highspy = pytest.importorskip("highspy")
    import scipy.sparse as sp

    from fedhpc.mip import read_mps
    path = ROOT / "benchmarks" / "mip" / "miplib2017" / "benchmark-v2" / f"{name}.mps.gz"
    if not path.exists():
        pytest.skip("MIPLIB files not present")
    P = read_mps(path)
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.readModel(str(path))
    lp = h.getLp()
    inf = highspy.kHighsInf

    def fix(v):
        v = np.asarray(v, float)
        return np.where(v >= inf, np.inf, np.where(v <= -inf, -np.inf, v))
    A = sp.csc_matrix((lp.a_matrix_.value_, lp.a_matrix_.index_, lp.a_matrix_.start_),
                      shape=(lp.num_row_, lp.num_col_))
    assert (P.m, P.n) == (lp.num_row_, lp.num_col_)
    assert abs(P.A - A).max() == 0
    assert np.array_equal(P.lb, fix(lp.col_lower_)) and np.array_equal(P.ub, fix(lp.col_upper_))
    assert np.array_equal(P.row_lo, fix(lp.row_lower_)) and np.array_equal(P.row_hi, fix(lp.row_upper_))
    assert np.array_equal(P.integer, np.array([int(t) == 1 for t in lp.integrality_]))
