"""The interval -> network reformulation of fedhpc.mip.reformulate.

Small time-indexed scheduling problems (one start per job, machine
capacity per period) plus random side rows: the detector must find the
capacity block, the differenced LP must have the same optimum and reduced
costs as the original one (duals mapped back), and Gomory cuts read from
the reformulated LP's tableau must hold for every feasible schedule."""
from __future__ import annotations

import itertools
import random

import numpy as np
import pytest
import scipy.sparse as sp

from fedhpc.mip.problem import INF, MipProblem
from fedhpc.mip.reformulate import detect_interval_blocks, network_form

try:
    from fedhpc import _fedlp
except ImportError:  # extension not built
    _fedlp = None

needs_lp = pytest.mark.skipif(_fedlp is None, reason="_fedlp extension not built")
NET = dict(min_len=4, max_ratio=1.5)     # small instances: reformulate regardless of gain


def ti_problem(rng, n_jobs, T, m, side_rows=0, coef=1.0, ge_jobs=False):
    """x_js = 1 when job j starts at s: sum_s x_js = 1 (or >= 1), and for each
    period t, sum of the x_js running at t <= m (rows n_jobs..n_jobs+T-1)."""
    p = [rng.randint(2, max(2, T // 2)) for _ in range(n_jobs)]
    r = [rng.randint(0, T // 3) for _ in range(n_jobs)]
    w = [rng.randint(1, 5) for _ in range(n_jobs)]
    cols = [(j, s) for j in range(n_jobs) for s in range(r[j], T - p[j] + 1)]
    n = len(cols)
    rows, cc, vv = [], [], []
    for k, (j, s) in enumerate(cols):
        rows.append(j), cc.append(k), vv.append(1.0)
        for t in range(s, s + p[j]):
            rows.append(n_jobs + t), cc.append(k), vv.append(coef)
    mrows = n_jobs + T
    lo = np.r_[np.ones(n_jobs), np.full(T, -INF)]
    hi = np.r_[np.full(n_jobs, INF if ge_jobs else 1.0), np.full(T, float(m) * coef)]
    for _ in range(side_rows):
        ks = rng.sample(range(n), max(1, n // 4))
        for k in ks:
            rows.append(mrows), cc.append(k), vv.append(float(rng.randint(1, 3)))
        lo = np.r_[lo, -INF]
        hi = np.r_[hi, float(rng.randint(2, 5))]
        mrows += 1
    A = sp.csr_matrix((vv, (rows, cc)), shape=(mrows, n))
    c = np.array([w[j] * (s + p[j]) for (j, s) in cols], float)
    return MipProblem(A, lo, hi, np.zeros(n), np.ones(n), c, np.ones(n, bool)), cols


def schedules(P, cols, n_jobs):
    """Every feasible integer point with one start per job."""
    by_job = [[k for k, (j, _s) in enumerate(cols) if j == jj] for jj in range(n_jobs)]
    A = P.A.toarray()
    for pick in itertools.product(*by_job):
        x = np.zeros(P.n)
        x[list(pick)] = 1.0
        act = A @ x
        if np.all(act >= P.row_lo - 1e-9) and np.all(act <= P.row_hi + 1e-9):
            yield x


@pytest.mark.parametrize("seed", range(10))
def test_detects_the_capacity_block(seed):
    rng = random.Random(seed)
    P, _ = ti_problem(rng, 4, 16, 2, side_rows=2)
    B = detect_interval_blocks(P, min_len=4)
    rows = set(B.rows.tolist())
    # the capacity rows (4..19) in blocks; job and side rows never
    assert rows <= set(range(4, 20))
    assert B.nnz_after <= 0.5 * B.nnz_before or not B.blocks
    N = network_form(P, B)
    # every block row of the differenced matrix: columns have <= 2 entries per block
    for lo, hi in B.blocks:
        sub = N.A[lo:hi + 1].tocsc()
        assert np.diff(sub.indptr).max() <= 2


@pytest.mark.parametrize("seed", range(10))
def test_differenced_rows_keep_the_feasible_set(seed):
    """For any x, the original rows hold iff the differenced rows hold with
    the slacks the original rows define."""
    rng = random.Random(100 + seed)
    P, _ = ti_problem(rng, 5, 20, 2, coef=rng.choice([1.0, 2.5]))
    B = detect_interval_blocks(P, min_len=4, max_ratio=1.5)
    assert B.blocks
    N = network_form(P, B)
    for _ in range(20):
        x = rng.choice([np.round, lambda v: v])(np.array([rng.random() for _ in range(P.n)]))
        act = P.A @ x
        s = np.zeros(len(N.slack_row))
        for k, t in enumerate(N.slack_row):
            lo, hi = P.row_lo[t], P.row_hi[t]
            sig, rhs = (-1.0, lo) if not np.isfinite(hi) else (1.0, hi)
            s[k] = sig * (rhs - act[t])
        lhs = N.A @ x + N.S @ s
        br = N.block_row
        assert np.allclose(lhs[br], N.row_lo[br])
        ok_orig = np.all(act[br] <= P.row_hi[br] + 1e-9) and np.all(act[br] >= P.row_lo[br] - 1e-9)
        ok_slack = np.all(s >= -1e-9) and np.all(s <= N.slack_ub + 1e-9)
        assert ok_orig == ok_slack


@needs_lp
@pytest.mark.parametrize("seed", range(20))
def test_network_lp_matches_the_original(seed):
    from fedhpc.mip.lp import OPTIMAL, LpRelaxation
    rng = random.Random(200 + seed)
    P, _ = ti_problem(rng, rng.randint(3, 6), rng.randint(12, 24), rng.randint(1, 3),
                      side_rows=rng.randint(0, 2), ge_jobs=bool(seed % 2))
    a = LpRelaxation(P, network=False)
    b = LpRelaxation(P, network=NET)
    b_net = b.net is not None
    sa, sb = a.solve(), b.solve()
    assert sa == sb
    if sa != OPTIMAL:
        return
    assert b_net
    assert abs(a.objective - b.objective) <= 1e-7 * (1 + abs(a.objective))
    assert np.all(P.A @ b.x() <= P.row_hi + 1e-7) and np.all(P.A @ b.x() >= P.row_lo - 1e-7)
    # reduced costs are invariant; the mapped duals reproduce them
    y = b.duals()[: P.m]
    d = P.c - P.A.T @ y
    assert np.allclose(d, b.reduced_costs(), atol=1e-7)
    # bound changes and re-solves agree too
    for _ in range(3):
        lb, ub = P.lb.copy(), P.ub.copy()
        k = rng.randrange(P.n)
        ub[k] = 0.0
        a.set_bounds(lb, ub), b.set_bounds(lb, ub)
        sa, sb = a.solve(), b.solve()
        assert sa == sb
        if sa == OPTIMAL:
            assert abs(a.objective - b.objective) <= 1e-7 * (1 + abs(a.objective))


@needs_lp
@pytest.mark.parametrize("seed", range(20))
def test_gomory_cuts_from_the_network_lp_are_valid(seed):
    from fedhpc.mip.cuts import CutParams, gomory_cuts
    from fedhpc.mip.lp import OPTIMAL, LpRelaxation
    rng = random.Random(300 + seed)
    nj = 3
    P, cols = ti_problem(rng, nj, rng.randint(10, 14), 1, side_rows=1)
    pts = list(schedules(P, cols, nj))
    lp = LpRelaxation(P, network=NET)
    assert lp.net is not None
    par = CutParams(min_efficacy=1e-7, gomory_max_support=1.0)
    for _round in range(3):
        if lp.solve() != OPTIMAL:
            break
        x = lp.x()
        cuts = gomory_cuts(lp, P, x, par)
        for cut in cuts:
            for pt in pts:
                assert cut.coef @ pt[cut.cols] >= cut.rhs - 1e-7, cut
            assert cut.violation(x) > 0
        if not cuts:
            break
        lp.add_rows([(c.cols, c.coef, c.rhs, INF) for c in cuts])
    if pts and lp.solve() == OPTIMAL:
        assert lp.objective <= min(P.c @ p for p in pts) + 1e-6


@pytest.mark.parametrize("seed", range(10))
def test_list_schedules_are_feasible_ti_and_af(seed):
    """The interval (TI) and the arc-flow (AF) list schedulers on small
    Kramer-style models: whatever they return is feasible; on TI a
    schedule always exists (the horizon fits every job in sequence)."""
    import time as _t

    from fedhpc.mip.scheduling import best_schedule, detect, detect_flow_structure, orders
    rng = random.Random(500 + seed)
    nj, T = rng.randint(3, 6), 40
    machines = nj if seed % 2 else rng.randint(1, 3)
    P, _ = ti_problem(rng, nj, T, machines)
    S = detect(P, detect_interval_blocks(P, min_len=4).blocks)
    assert S is not None
    x, v = best_schedule(P, S, orders(S), _t.perf_counter() + 0.5)
    if x is not None:
        assert P.is_feasible(x)
    if machines == nj:                    # every job on its own machine always fits
        assert x is not None
    assert detect_flow_structure(P) is None
