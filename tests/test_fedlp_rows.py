"""fedlp's general-model and MIP-facing interface: ranged rows, rows added
and removed after construction (cuts), basis save/restore and tableau rows,
each against Gurobi or a dense numpy computation."""
from __future__ import annotations

import random

import numpy as np
import pytest

gp = pytest.importorskip("gurobipy")
from gurobipy import GRB  # noqa: E402

from fedhpc.formulations import configure_env  # noqa: E402

try:
    from fedhpc import _fedlp
except ImportError:  # extension not built
    _fedlp = None

pytestmark = pytest.mark.skipif(_fedlp is None, reason="_fedlp extension not built")
configure_env()
INF = float("inf")
OPT = int(_fedlp.Status.OPTIMAL) if _fedlp else 0


def random_lp(rng, m, n, ranged=0.3):
    """A feasible bounded LP: rows lo <= A x <= hi around a known point."""
    A = np.zeros((m, n))
    for i in range(m):
        for j in rng.sample(range(n), max(1, n // 3)):
            A[i, j] = rng.uniform(-2, 3)
    x0 = np.array([rng.uniform(0, 2) for _ in range(n)])
    act = A @ x0
    lo, hi = np.empty(m), np.empty(m)
    for i in range(m):
        k = rng.random()
        if k < ranged:
            lo[i], hi[i] = act[i] - rng.uniform(0, 1), act[i] + rng.uniform(0, 1)
        elif k < 0.6:
            lo[i], hi[i] = -INF, act[i] + rng.uniform(0, 1)
        elif k < 0.85:
            lo[i], hi[i] = act[i] - rng.uniform(0, 1), INF
        else:
            lo[i] = hi[i] = act[i]
    c = np.array([rng.uniform(-1, 2) for _ in range(n)])
    ub = np.array([rng.choice([INF, x0[j] + rng.uniform(0.5, 2)]) for j in range(n)])
    return A, lo, hi, c, ub


def fedlp_of(A, lo, hi, c, ub):
    lp = _fedlp.Solver.ranged(lo.tolist(), hi.tolist())
    for j in range(A.shape[1]):
        nz = np.nonzero(A[:, j])[0]
        lp.add_column(float(c[j]), 0.0, float(ub[j]), nz.tolist(), A[nz, j].tolist())
    return lp


def gurobi_of(A, lo, hi, c, ub):
    from fedhpc import formulations
    g = gp.Model(env=formulations._shared_env)
    g.Params.OutputFlag = 0
    x = g.addMVar(A.shape[1], lb=0.0, ub=ub, obj=c)
    for i in range(A.shape[0]):
        if lo[i] == hi[i]:
            g.addConstr(A[i] @ x == lo[i])
        else:
            if np.isfinite(lo[i]):
                g.addConstr(A[i] @ x >= lo[i])
            if np.isfinite(hi[i]):
                g.addConstr(A[i] @ x <= hi[i])
    return g, x


def check_feasible(lp, A, lo, hi, ub, tol=1e-6):
    x = np.asarray(lp.x())
    act = A @ x
    assert np.all(x >= -tol) and np.all(x <= ub + tol)
    assert np.all(act >= lo - tol * (1 + np.abs(lo))) and np.all(act <= hi + tol * (1 + np.abs(hi)))


@pytest.mark.parametrize("seed", range(25))
def test_ranged_rows_match_gurobi(seed):
    rng = random.Random(seed)
    A, lo, hi, c, ub = random_lp(rng, rng.randint(3, 20), rng.randint(3, 30))
    lp = fedlp_of(A, lo, hi, c, ub)
    g, _ = gurobi_of(A, lo, hi, c, ub)
    g.optimize()
    st = lp.solve()
    if g.Status != GRB.OPTIMAL:
        assert st != OPT
        return
    assert st == OPT
    assert lp.objective == pytest.approx(g.ObjVal, abs=1e-6 * (1 + abs(g.ObjVal)))
    check_feasible(lp, A, lo, hi, ub)


@pytest.mark.parametrize("seed", range(20))
def test_cut_rounds_add_and_remove_rows(seed):
    """Rows appended after a solve (as cuts are) and removed again: every
    re-solve matches Gurobi on the same rows, keeps the basis (few pivots),
    and column ids stay put."""
    rng = random.Random(100 + seed)
    m, n = rng.randint(4, 15), rng.randint(5, 25)
    A, lo, hi, c, ub = random_lp(rng, m, n)
    ub = np.minimum(ub, 5.0)
    lp = fedlp_of(A, lo, hi, c, ub)
    assert lp.solve() == OPT
    rows_A, rows_lo, rows_hi = [A], [lo], [hi]
    for _round in range(4):
        # a few rows that cut off the current optimum (or nothing at all)
        x = np.asarray(lp.x())
        k = rng.randint(1, 3)
        C = np.zeros((k, n))
        for t in range(k):
            for j in rng.sample(range(n), max(1, n // 4)):
                C[t, j] = rng.uniform(-1, 2)
        a = C @ x
        clo = np.full(k, -INF)
        chi = a - np.array([rng.uniform(-0.2, 0.3) for _ in range(k)])
        start = [0]
        idx, val = [], []
        for t in range(k):
            nz = np.nonzero(C[t])[0]
            idx += nz.tolist()
            val += C[t, nz].tolist()
            start.append(len(idx))
        first = lp.add_rows(clo, chi, np.array(start), np.array(idx, dtype=np.int32), np.array(val))
        assert first == sum(len(r) for r in rows_lo)
        rows_A.append(C), rows_lo.append(clo), rows_hi.append(chi)
        AA, LL, HH = np.vstack(rows_A), np.concatenate(rows_lo), np.concatenate(rows_hi)
        assert lp.rows == AA.shape[0] and lp.cols == n
        g, _ = gurobi_of(AA, LL, HH, c, ub)
        g.optimize()
        lp.reset_stats()
        st = lp.solve()
        if g.Status != GRB.OPTIMAL:
            assert st != OPT
            return
        assert st == OPT, lp.stats()
        assert lp.objective == pytest.approx(g.ObjVal, abs=1e-6 * (1 + abs(g.ObjVal)))
        check_feasible(lp, AA, LL, HH, ub)
    # drop the cuts again: back to the original LP's optimum
    lp.remove_rows(list(range(m, lp.rows)))
    assert lp.rows == m
    g, _ = gurobi_of(A, lo, hi, c, ub)
    g.optimize()
    assert lp.solve() == OPT
    assert lp.objective == pytest.approx(g.ObjVal, abs=1e-6 * (1 + abs(g.ObjVal)))


@pytest.mark.parametrize("seed", range(10))
def test_basis_roundtrip_and_restore(seed):
    """get_basis / set_basis: restoring an optimal basis re-solves in zero
    pivots, after other solves moved the LP elsewhere."""
    rng = random.Random(200 + seed)
    A, lo, hi, c, ub = random_lp(rng, rng.randint(4, 15), rng.randint(5, 25))
    ub = np.minimum(ub, 5.0)   # bounded: an optimum exists
    lp = fedlp_of(A, lo, hi, c, ub)
    assert lp.solve() == OPT
    obj, (rs, cs) = lp.objective, lp.get_basis()
    for j in range(A.shape[1]):   # move it: tighten everything, re-solve
        lp.set_bounds(j, 0.0, float(min(ub[j], 0.5)))
    lp.solve()
    for j in range(A.shape[1]):
        lp.set_bounds(j, 0.0, float(ub[j]))
    assert lp.set_basis(rs, cs)
    lp.reset_stats()
    assert lp.solve() == OPT
    assert lp.stats()["iterations"] == 0
    assert lp.objective == pytest.approx(obj, abs=1e-9 * (1 + abs(obj)))


@pytest.mark.parametrize("seed", range(10))
def test_tableau_rows_equal_dense_inverse(seed):
    """tableau_row(i) is row i of B^-1 [A sigma I] over the nonbasic
    columns, B the basic columns of [A sigma I] (sigma_r = -1 for >= rows)."""
    rng = random.Random(300 + seed)
    m, n = rng.randint(3, 12), rng.randint(4, 20)
    A, lo, hi, c, ub = random_lp(rng, m, n)
    ub = np.minimum(ub, 5.0)
    lp = fedlp_of(A, lo, hi, c, ub)
    assert lp.solve() == OPT
    sigma = np.array([-1.0 if (np.isfinite(lo[r]) and not np.isfinite(hi[r])) else 1.0 for r in range(m)])
    full = np.hstack([np.diag(sigma), A])           # logicals first, as fedlp

    def col(code):
        return code + m if code >= 0 else -1 - code

    heads = lp.basic_heads()
    B = full[:, [col(h) for h in heads]]
    Binv = np.linalg.inv(B)
    rs, cs = lp.get_basis()
    status = np.concatenate([rs, cs])
    for i in range(m):
        ids, vals = lp.tableau_row(i)
        got = {col(k): v for k, v in zip(ids, vals)}
        want = Binv[i] @ full
        for j in range(m + n):
            if status[j] == 0:
                continue
            assert got.get(j, 0.0) == pytest.approx(want[j], abs=1e-8 * (1 + abs(want[j])))
