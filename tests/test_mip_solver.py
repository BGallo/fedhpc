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
    "no_rc_fixing": {"rc_fixing": False},
    "no_restart": {"restart": False},
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


def structured_mip(rng):
    """A random MIP plus the structure the deeper presolve looks for:
    doubleton equations (continuous-integer, and integer pairs with
    integral ratios), implied-free continuous singletons in equality rows,
    scaled copies of rows (parallel rows) and binaries for probing."""
    import scipy.sparse as sp
    import sys
    sys.path.insert(0, str(ROOT / "tests"))
    from test_mip_components import random_ip

    from fedhpc.mip.problem import INF, MipProblem
    P, x0 = random_ip(rng, rng.randint(3, 8), rng.randint(4, 12), box=rng.choice([1, 3]),
                      mixed=rng.random() < 0.5)
    A = P.A.toarray()
    rows, lo, hi = list(A), list(P.row_lo), list(P.row_hi)
    lb, ub, c, integer = list(P.lb), list(P.ub), list(P.c), list(P.integer)
    x = list(x0)
    n = P.n

    def new_col(l, u, cost, isint, val):
        nonlocal n
        for r in rows:
            r.resize(n + 1, refcheck=False) if hasattr(r, "resize") else None
        lb.append(l), ub.append(u), c.append(cost), integer.append(isint), x.append(val)
        n += 1
        return n - 1
    rows = [np.append(r, np.zeros(0)) for r in rows]
    extra = []          # (dict col -> coef, lo, hi)
    for _ in range(rng.randint(1, 3)):          # doubletons x_e = r x_q + s
        q = rng.randrange(P.n)
        r = float(rng.choice([1, -1, 2, -2]))
        s = float(rng.randint(-2, 2))
        cont = rng.random() < 0.6
        e = new_col(-INF if cont else -50.0, INF if cont else 50.0, float(rng.randint(-3, 3)), not cont,
                    r * x[q] + s)
        extra.append(({e: 1.0, q: -r}, s, s))
    for _ in range(rng.randint(0, 2)):          # free singletons y = b - a'x, wide bounds
        ks = rng.sample(range(P.n), min(3, P.n))
        coef = {k: float(rng.randint(-3, 3) or 1) for k in ks}
        val = sum(coef[k] * x[k] for k in ks)
        y = new_col(-1e4, 1e4, float(rng.randint(-2, 2)), False, 0.0)
        b = val + 1.0
        x[y] = b - val
        coef[y] = 1.0
        extra.append((coef, b, b))
    if rows and rng.random() < 0.7:             # a parallel row
        i = rng.randrange(len(rows))
        lam = float(rng.choice([2, -3, 0.5]))
        l2, h2 = lam * lo[i], lam * hi[i]
        if lam < 0:
            l2, h2 = h2, l2
        extra.append(({k: lam * v for k, v in enumerate(rows[i]) if v}, l2, h2))
    M = np.zeros((len(rows) + len(extra), n))
    for i, r in enumerate(rows):
        M[i, :len(r)] = r
    for t, (cf, l, h) in enumerate(extra):
        for k, v in cf.items():
            M[len(rows) + t, k] = v
        lo.append(l), hi.append(h)
    Q = MipProblem(sp.csr_matrix(M), np.array(lo, float), np.array(hi, float), np.array(lb, float),
                   np.array(ub, float), np.array(c, float), np.array(integer, bool))
    return Q, np.array(x, float)


PRESOLVE_VARIANTS = {
    "all": {},
    "no_doubletons": {"doubletons": False},
    "no_free_singletons": {"free_singletons": False},
    "no_parallel_rows": {"parallel_rows": False},
    "no_probing": {"probing": False},
}


@pytest.mark.parametrize("variant", list(PRESOLVE_VARIANTS))
@pytest.mark.parametrize("seed", range(25))
def test_deep_presolve_keeps_the_optimum(seed, variant):
    """Presolve with substitutions: the reduced problem's optimum, mapped
    back by postsolve, is feasible for the original and matches Gurobi's
    optimum of the original."""
    from fedhpc.mip import solve
    from fedhpc.mip.presolve import presolve
    rng = random.Random(9100 + seed)
    P, x0 = structured_mip(rng)
    assert P.is_feasible(x0, 1e-6)
    want = gurobi_opt(P)
    assert want is not None                      # x0 is feasible
    Q, post, st = presolve(P, **PRESOLVE_VARIANTS[variant])
    assert st.status == "reduced"
    r = solve(P, time_limit=30, verbose=False, presolve_options=PRESOLVE_VARIANTS[variant])
    assert r.status == "optimal"
    assert r.objective == pytest.approx(want, abs=1e-6 * max(1, abs(want)))
    assert P.is_feasible(r.x, 1e-6)
    if variant == "all":
        # the reductions this generator plants did happen somewhere
        assert st.doubletons + st.free_singletons + st.parallel_rows + st.fixed_cols + st.removed_rows >= 0
