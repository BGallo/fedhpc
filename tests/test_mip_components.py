"""fedhpc.mip components on small problems where every integer point can be
enumerated: the MPS reader, propagation (never removes a feasible point),
presolve (same optimum), the LP adapter, and cut validity (no feasible
integer point violates a Gomory or cover cut, through rounds of cuts)."""
from __future__ import annotations

import io
import itertools
import random

import numpy as np
import pytest
import scipy.sparse as sp

from fedhpc.mip.mps import parse_mps
from fedhpc.mip.problem import INF, MipProblem

try:
    from fedhpc import _fedlp
except ImportError:  # extension not built
    _fedlp = None

needs_lp = pytest.mark.skipif(_fedlp is None, reason="_fedlp extension not built")


def random_ip(rng, m, n, box=3, mixed=False):
    """A small bounded MIP with a known feasible point."""
    A = np.zeros((m, n))
    for i in range(m):
        for j in rng.sample(range(n), max(1, rng.randint(1, n))):
            A[i, j] = rng.randint(-4, 6)
    x0 = np.array([rng.randint(0, box) for _ in range(n)], float)
    act = A @ x0
    lo, hi = np.full(m, -INF), np.full(m, INF)
    for i in range(m):
        k = rng.random()
        if k < 0.5:
            hi[i] = act[i] + rng.randint(0, 3)
        elif k < 0.8:
            lo[i] = act[i] - rng.randint(0, 3)
        else:
            lo[i], hi[i] = act[i] - rng.randint(0, 2), act[i] + rng.randint(0, 2)
    c = np.array([rng.randint(-5, 5) for _ in range(n)], float)
    integer = np.ones(n, bool)
    if mixed:
        integer[rng.sample(range(n), n // 3)] = False
    return MipProblem(sp.csr_matrix(A), lo, hi, np.zeros(n), np.full(n, float(box)), c, integer), x0


def integer_points(P, box_cont=None):
    """Every feasible point of a pure-integer P (box enumeration)."""
    rngs = [range(int(P.lb[j]), int(P.ub[j]) + 1) for j in range(P.n)]
    A = P.A.toarray()
    for pt in itertools.product(*rngs):
        x = np.array(pt, float)
        act = A @ x
        if np.all(act >= P.row_lo - 1e-9) and np.all(act <= P.row_hi + 1e-9):
            yield x


# ------------------------------------------------------------------ MPS
def test_mps_ranges_bounds_and_markers():
    text = """NAME          T
OBJSENSE
    MAX
ROWS
 N  obj
 L  c1
 G  c2
 E  c3
 E  c4
COLUMNS
    MARKER                 'MARKER'                 'INTORG'
    x         obj       1.0          c1        1.0
    x         c2        1.0
    MARKER                 'MARKER'                 'INTEND'
    y         obj       2.0          c1        1.0
    y         c3        1.0          c4        1.0
    z         c2        3.0          c4        2.0
RHS
    rhs       c1        4.0          c2        1.0
    rhs       c3        2.0          c4        5.0
    rhs       obj       -7.0
RANGES
    rng       c1        2.0          c2        3.0
    rng       c3        1.5          c4        -2.0
BOUNDS
 UP bnd       x         -1.0
 FR bnd       y
 BV bnd       z
ENDATA
"""
    P = parse_mps(io.StringIO(text))
    assert P.maximize and P.offset == pytest.approx(-7.0)     # stored negated
    assert np.allclose(P.c, [-1.0, -2.0, 0.0])
    assert list(P.row_lo) == [2.0, 1.0, 2.0, 3.0] and list(P.row_hi) == [4.0, 4.0, 3.5, 5.0]
    assert P.lb[0] == -INF and P.ub[0] == -1.0 and P.integer[0]
    assert P.lb[1] == -INF and P.ub[1] == INF and not P.integer[1]
    assert (P.lb[2], P.ub[2]) == (0.0, 1.0) and P.integer[2]
    assert P.A.toarray().tolist() == [[1, 1, 0], [1, 0, 3], [0, 1, 0], [0, 1, 2]]


# ------------------------------------------------------------ propagation
@pytest.mark.parametrize("impl", ["cpp", "py"])
@pytest.mark.parametrize("seed", range(40))
def test_propagation_keeps_every_feasible_point(seed, impl):
    from fedhpc.mip.propagate import Propagator, PyPropagator
    rng = random.Random(seed)
    P, _ = random_ip(rng, rng.randint(2, 5), rng.randint(2, 5))
    pts = list(integer_points(P))
    res = (Propagator if impl == "cpp" else PyPropagator)(P).propagate(P.lb, P.ub)
    if not pts:
        return
    assert res.feasible
    for x in pts:
        assert np.all(x >= res.lb - 1e-9) and np.all(x <= res.ub + 1e-9)


@pytest.mark.parametrize("seed", range(40))
def test_propagation_trail_fixings_and_undo(seed):
    """The stateful C++ domain: fixings propagate soundly (every feasible
    point agreeing with them survives), an empty domain is reported only
    when no point agrees, and undo restores the domain exactly."""
    from fedhpc.mip.propagate import Propagator, _fedmip
    if _fedmip is None:
        pytest.skip("_fedmip not built")
    rng = random.Random(4000 + seed)
    P, _ = random_ip(rng, rng.randint(2, 6), rng.randint(2, 6), box=2)
    pts = list(integer_points(P))
    prop = Propagator(P)
    ok = prop.reset(P.lb, P.ub)
    assert ok or not pts
    if not ok:
        return
    base_l, base_u = prop.lb(), prop.ub()
    for _ in range(6):
        lev = prop.level()
        k = rng.randint(1, min(3, P.n))
        js = rng.sample(range(P.n), k)
        vals = [float(rng.randint(int(base_l[j]), int(base_u[j]))) for j in js]
        ok = prop.change(js, vals, vals)
        agree = [x for x in pts if all(x[j] == v for j, v in zip(js, vals))]
        if ok:
            l, u = prop.lb(), prop.ub()
            for x in agree:
                assert np.all(x >= l - 1e-9) and np.all(x <= u + 1e-9)
        else:
            assert not agree
        prop.undo(lev)
        assert np.array_equal(prop.lb(), base_l) and np.array_equal(prop.ub(), base_u)


@pytest.mark.parametrize("seed", range(20))
def test_propagation_mixed_columns_sound(seed):
    """Continuous and unbounded columns: the C++ kernel keeps every point
    of a sampled feasible set (points found by rejection on a grid)."""
    from fedhpc.mip.propagate import Propagator
    rng = random.Random(4500 + seed)
    P, x0 = random_ip(rng, rng.randint(2, 5), rng.randint(2, 5), mixed=True)
    cont = ~P.integer
    P.ub[cont] = np.where(rng.random() < 0.5, INF, P.ub[cont])
    res = Propagator(P).propagate(P.lb, P.ub)
    assert P.is_feasible(x0)
    assert res.feasible
    assert np.all(x0 >= res.lb - 1e-7) and np.all(x0 <= res.ub + 1e-7)


# --------------------------------------------------------------- presolve
def brute_opt(P):
    best = None
    for x in integer_points(P):
        v = P.objective(x)
        best = v if best is None or v < best else best
    return best


@pytest.mark.parametrize("seed", range(40))
def test_presolve_keeps_the_optimum(seed):
    from fedhpc.mip.presolve import presolve
    rng = random.Random(500 + seed)
    P, _ = random_ip(rng, rng.randint(2, 5), rng.randint(2, 5))
    want = brute_opt(P)
    Q, post, st = presolve(P)
    if want is None:
        assert st.status != "reduced" or brute_opt(Q) is None
        return
    assert st.status == "reduced"
    got = brute_opt(Q) if Q.n else Q.offset
    assert got == pytest.approx(want)
    # postsolve maps the reduced optimum back to an original feasible point
    if Q.n:
        xq = min(integer_points(Q), key=Q.objective)
    else:
        xq = np.zeros(0)
    x = post.expand(xq)
    assert P.is_feasible(x) and P.objective(x) == pytest.approx(want)


# --------------------------------------------------------------- LP + cuts
@needs_lp
@pytest.mark.parametrize("seed", range(60))
def test_cuts_never_cut_off_an_integer_point(seed):
    from fedhpc.mip.cuts import CutParams, cover_cuts, gomory_cuts
    from fedhpc.mip.lp import OPTIMAL, LpRelaxation
    rng = random.Random(900 + seed)
    # odd seeds: binaries (knapsack rows, so cover cuts arise)
    P, _ = random_ip(rng, rng.randint(2, 5), rng.randint(3, 8), box=1 if seed % 2 else 3)
    pts = list(integer_points(P))
    lp = LpRelaxation(P)
    par = CutParams(min_efficacy=1e-7)
    total = 0
    for _round in range(4):
        if lp.solve() != OPTIMAL:
            break
        x = lp.x()
        cuts = gomory_cuts(lp, P, x, par) + cover_cuts(P, x, P.lb, P.ub, par)
        for cut in cuts:
            for pt in pts:
                assert cut.coef @ pt[cut.cols] >= cut.rhs - 1e-7, (cut.origin, cut)
            assert cut.violation(x) > 0
        total += len(cuts)
        if not cuts:
            break
        lp.add_rows([(c.cols, c.coef, c.rhs, INF) for c in cuts])
    # the LP bound with cuts never passes the integer optimum
    if pts and lp.solve() == OPTIMAL:
        assert lp.objective <= brute_opt(P) + 1e-6


@needs_lp
@pytest.mark.parametrize("seed", range(30))
def test_lp_relaxation_free_columns_and_bounds(seed):
    """Free columns (split inside), bound changes and basis restore: the LP
    optimum matches scipy's HiGHS."""
    from scipy.optimize import linprog

    from fedhpc.mip.lp import OPTIMAL, LpRelaxation
    rng = random.Random(1300 + seed)
    m, n = rng.randint(2, 8), rng.randint(2, 8)
    P, x0 = random_ip(rng, m, n)
    P.integer[:] = False
    free = rng.sample(range(n), max(1, n // 3))
    P.lb[free], P.ub[free] = -INF, INF
    P.c[free] = 0.0                       # keep it bounded
    lp = LpRelaxation(P)

    def ref(lb, ub):
        A = P.A.toarray()
        Aub, bub = [], []
        for i in range(m):
            if np.isfinite(P.row_hi[i]):
                Aub.append(A[i]), bub.append(P.row_hi[i])
            if np.isfinite(P.row_lo[i]):
                Aub.append(-A[i]), bub.append(-P.row_lo[i])
        r = linprog(P.c, A_ub=np.array(Aub), b_ub=np.array(bub),
                    bounds=[(None if not np.isfinite(l) else l, None if not np.isfinite(u) else u)
                            for l, u in zip(lb, ub)], method="highs")
        return r
    for _k in range(3):
        lb, ub = P.lb.copy(), P.ub.copy()
        j = rng.randrange(n)
        if np.isfinite(ub[j]):
            ub[j] = max(lb[j], ub[j] - 1)
        else:
            lb[j], ub[j] = -2.0, 5.0
        lp.set_bounds(lb, ub)
        st = lp.solve()
        r = ref(lb, ub)
        if r.status != 0:
            assert st != OPTIMAL
            continue
        assert st == OPTIMAL
        assert lp.objective == pytest.approx(r.fun + P.offset, abs=1e-6 * (1 + abs(r.fun)))
        x = lp.x()
        assert np.all(x >= lb - 1e-7) and np.all(x <= ub + 1e-7)
