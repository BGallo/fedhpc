"""fedlp (``fedhpc._fedlp``), the simplex for column generation masters:
optimal objective against Gurobi, and the optimality certificate itself
(primal feasibility, dual sign conditions, reduced costs) on random LPs,
degenerate set-partitioning masters re-solved as columns arrive, column
removal and bound changes."""
from __future__ import annotations

import math
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


@pytest.fixture(autouse=True, params=[1, 2, 3], ids=["dense", "pfi", "lu"])
def factor(request, monkeypatch):
    """Every test runs with the dense explicit inverse and with the sparse
    product-form factor."""
    monkeypatch.setattr(Mirror, "FACTOR", request.param)
    return request.param


class Mirror:
    """The same LP in fedlp and in Gurobi."""

    FACTOR = 0

    def __init__(self, sense, rhs):
        from fedhpc import formulations
        self.sense, self.rhs = list(sense), list(rhs)
        self.lp = _fedlp.Solver(self.sense, self.rhs)
        self.lp.options.factor = self.FACTOR
        self.g = gp.Model(env=formulations._shared_env)
        self.g.Params.OutputFlag = 0
        gs = {"<": GRB.LESS_EQUAL, ">": GRB.GREATER_EQUAL, "=": GRB.EQUAL}
        self.rows = [self.g.addLConstr(gp.LinExpr(), gs[s], b) for s, b in zip(sense, rhs)]
        self.vars, self.cols = [], []

    def add(self, c, lb, ub, idx, val):
        k = self.lp.add_column(c, lb, ub, idx, val)
        self.vars.append(self.g.addVar(lb=lb, ub=ub, obj=c,
                                       column=gp.Column(val, [self.rows[i] for i in idx])))
        self.cols.append((c, lb, ub, list(idx), list(val)))
        return k

    def remove(self, ids):
        self.lp.remove_columns(ids)
        for k in ids:
            self.g.remove(self.vars[k])
            self.vars[k] = None

    def bounds(self, k, lb, ub):
        self.lp.set_bounds(k, lb, ub)
        self.vars[k].LB, self.vars[k].UB = lb, ub
        c, _, _, idx, val = self.cols[k]
        self.cols[k] = (c, lb, ub, idx, val)

    def check(self, tol=1e-6):
        st = self.lp.solve()
        self.g.optimize()
        if self.g.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD, GRB.UNBOUNDED):
            self.g.Params.DualReductions = 0
            self.g.optimize()
            want = _fedlp.Status.INFEASIBLE if self.g.Status == GRB.INFEASIBLE else _fedlp.Status.UNBOUNDED
            assert st == int(want)
            return None
        assert self.g.Status == GRB.OPTIMAL
        assert st == int(_fedlp.Status.OPTIMAL), self.lp.stats()
        scale = 1.0 + abs(self.g.ObjVal)
        assert self.lp.objective == pytest.approx(self.g.ObjVal, abs=tol * scale)
        self.certificate(tol)
        return self.lp.objective

    def certificate(self, tol):
        """Primal feasibility, dual signs, reduced-cost signs at bounds."""
        x, y, d, bs = self.lp.x(), self.lp.duals(), self.lp.reduced_costs(), self.lp.basis_status()
        act = np.zeros(len(self.rhs))
        for k, (c, lb, ub, idx, val) in enumerate(self.cols):
            if self.vars[k] is None:
                continue
            assert lb - tol <= x[k] <= ub + tol
            for i, v in zip(idx, val):
                act[i] += v * x[k]
            dk = c - sum(y[i] * v for i, v in zip(idx, val))
            assert dk == pytest.approx(d[k], abs=1e-6 * (1 + abs(c)))
            if lb == ub:
                continue        # a fixed column may have any reduced cost
            if bs[k] == 1:      # at lower bound: d >= 0
                assert dk >= -tol * (1 + abs(c))
            elif bs[k] == 2:    # at upper bound: d <= 0
                assert dk <= tol * (1 + abs(c))
            elif bs[k] == 0:
                assert abs(dk) <= tol * (1 + abs(c))
        for s, b, a, yi in zip(self.sense, self.rhs, act, y):
            sc = tol * (1 + abs(b))
            if s == "<":
                assert a <= b + sc and yi <= tol
            elif s == ">":
                assert a >= b - sc and yi >= -tol
            else:
                assert a == pytest.approx(b, abs=sc)


def _random_lp(rng, m, n, density=0.3):
    sense = [rng.choice("<<=>") for _ in range(m)]
    x0 = [rng.uniform(0, 3) for _ in range(n)]          # a feasible point
    cols = []
    for _j in range(n):
        idx = sorted(rng.sample(range(m), max(1, int(density * m))))
        cols.append((idx, [rng.uniform(-2, 3) for _ in idx]))
    act = [0.0] * m
    for j, (idx, val) in enumerate(cols):
        for i, v in zip(idx, val):
            act[i] += v * x0[j]
    rhs = [a + (rng.uniform(0, 2) if s == "<" else -rng.uniform(0, 2) if s == ">" else 0.0)
           for a, s in zip(act, sense)]
    M = Mirror(sense, rhs)
    for j, (idx, val) in enumerate(cols):
        ub = rng.choice([INF, INF, x0[j] + rng.uniform(0, 2)])
        c = rng.uniform(-1, 2) if ub < INF else rng.uniform(0.1, 2)
        M.add(c, 0.0, ub, idx, val)
    return M


@pytest.mark.parametrize("seed", range(40))
def test_random_lps_match_gurobi(seed):
    rng = random.Random(seed)
    M = _random_lp(rng, rng.randint(3, 25), rng.randint(3, 60))
    M.check()


def _sp_master(rng, n_jobs, n_fleet):
    """A set-partitioning master like bpc's: coverage rows (= 1) with big-M
    artificials, fleet rows (= N) with null columns, and a cost cap."""
    sense = ["="] * n_jobs + ["="] * n_fleet + ["<"]
    rhs = [1.0] * n_jobs + [float(rng.randint(1, 3)) for _ in range(n_fleet)] + [1e9]
    M = Mirror(sense, rhs)
    for j in range(n_jobs):
        M.add(1e6, 0.0, INF, [j], [1.0])
    for f in range(n_fleet):
        M.add(0.0, 0.0, INF, [n_jobs + f], [1.0])
    return M


def _random_route(rng, n_jobs, n_fleet):
    jobs = sorted(rng.sample(range(n_jobs), rng.randint(1, min(6, n_jobs))))
    f = rng.randrange(n_fleet)
    cost = sum(rng.randint(5, 40) for _ in jobs) + rng.randint(0, 10)
    idx = jobs + [n_jobs + f, n_jobs + n_fleet]
    return cost, idx, [1.0] * len(jobs) + [1.0, float(cost)]


@pytest.mark.parametrize("seed", range(12))
def test_degenerate_master_resolves_match_gurobi(seed):
    """Column generation's pattern: append columns, re-solve warm."""
    rng = random.Random(100 + seed)
    nj, nf = rng.randint(8, 40), rng.randint(1, 4)
    M = _sp_master(rng, nj, nf)
    for _round in range(25):
        for _ in range(rng.randint(5, 40)):
            M.add(*_cost_idx_val(rng, nj, nf))
        M.check()


def _cost_idx_val(rng, nj, nf):
    cost, idx, val = _random_route(rng, nj, nf)
    return float(cost), 0.0, INF, idx, val


@pytest.mark.parametrize("seed", range(10))
def test_removals_and_bound_changes_match_gurobi(seed):
    rng = random.Random(300 + seed)
    nj, nf = rng.randint(8, 25), rng.randint(1, 3)
    M = _sp_master(rng, nj, nf)
    for _ in range(80):
        M.add(*_cost_idx_val(rng, nj, nf))
    M.check()
    for _step in range(12):
        live = [k for k in range(nj + nf, len(M.vars)) if M.vars[k] is not None]
        x = M.lp.x()
        used = [k for k in live if x[k] > 1e-9]
        # remove a used (often basic) column and an unused one
        M.remove(sorted({rng.choice(used or live), rng.choice(live)}))
        # fix a column to 0 or raise its lower bound (branching-like)
        live = [k for k in live if M.vars[k] is not None]
        k = rng.choice(live)
        if rng.random() < 0.5:
            M.bounds(k, 0.0, 0.0)
        else:
            M.bounds(k, 0.0, 1.0)
        for _ in range(rng.randint(0, 10)):
            M.add(*_cost_idx_val(rng, nj, nf))
        M.check()


def test_infeasible_lp_is_reported():
    M = Mirror(["<", ">"], [1.0, 2.0])
    M.add(1.0, 0.0, INF, [0, 1], [1.0, 1.0])
    M.check()


def test_batch_add_matches_single_adds():
    rng = random.Random(7)
    a = _fedlp.Solver(["=", "=", "<"], [1.0, 1.0, 50.0])
    b = _fedlp.Solver(["=", "=", "<"], [1.0, 1.0, 50.0])
    cols = [(1e6, [0], [1.0]), (1e6, [1], [1.0])]
    cols += [(float(rng.randint(1, 9)), [0, 1, 2], [1.0, 1.0, float(rng.randint(1, 9))]) for _ in range(5)]
    for c, idx, val in cols:
        a.add_column(c, 0.0, INF, idx, val)
    start = np.cumsum([0] + [len(i) for _c, i, _v in cols])
    b.add_columns(np.array([c for c, _i, _v in cols]), np.zeros(len(cols)), np.full(len(cols), INF),
                  start, np.concatenate([i for _c, i, _v in cols]), np.concatenate([v for _c, _i, v in cols]))
    assert a.solve() == b.solve() == int(_fedlp.Status.OPTIMAL)
    assert a.objective == pytest.approx(b.objective)
    assert math.isfinite(a.objective)


@pytest.mark.parametrize("seed", range(15))
def test_dual_simplex_after_branching_bounds(seed):
    """Closing used columns (ub = 0) and raising one lower bound, with no
    column appended, leaves the basis dual feasible: the dual simplex must
    finish it, with Gurobi's objective (or its infeasibility verdict)."""
    rng = random.Random(500 + seed)
    nj, nf = rng.randint(8, 30), rng.randint(1, 3)
    M = _sp_master(rng, nj, nf)
    for _ in range(120):
        M.add(*_cost_idx_val(rng, nj, nf))
    M.check()
    used_dual = False
    for _step in range(6):
        x = M.lp.x()
        live = [k for k in range(nj + nf, len(M.vars)) if M.vars[k] is not None and M.cols[k][2] > 0]
        used = [k for k in live if x[k] > 1e-9]
        for k in rng.sample(used, min(len(used), rng.randint(1, 3))):
            M.bounds(k, 0.0, 0.0)
        if rng.random() < 0.3 and live:
            k = rng.choice(live)
            if M.cols[k][2] > 0:
                M.bounds(k, min(1.0, M.cols[k][2]), M.cols[k][2])
        M.lp.reset_stats()
        M.check()
        used_dual |= M.lp.stats()["dual_iterations"] > 0
    assert used_dual
