"""The LP relaxation, on fedlp.

``LpRelaxation`` owns one fedlp solver for a ``MipProblem`` and speaks in
the problem's column ids:

- Free columns (both bounds infinite, which fedlp does not take) are split
  as x = x+ - x-; columns with one infinite bound are fedlp's own.
- Bounds change by vector (only the changed columns reach fedlp), so a
  node's domain is installed in one call.
- Cut rows are appended and removed (fedlp rebuilds its model and keeps the
  basis, the new rows' logicals basic: the dual simplex re-solves).
- Bases are saved and restored for node warm starts.
- Tableau rows come back over the problem's columns plus the rows'
  logicals, with each nonbasic variable's bound status, for Gomory cuts.

Every solve's iterations and seconds are added to ``stats``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .problem import INF, MipProblem

try:
    from .. import _fedlp
except ImportError:  # pragma: no cover - the extension is part of the build
    _fedlp = None

OPTIMAL, INFEASIBLE, UNBOUNDED, ITERATION_LIMIT, NUMERICAL = 0, 1, 2, 3, 5
STATUS_NAME = {0: "optimal", 1: "infeasible", 2: "unbounded", 3: "iteration_limit", 5: "numerical"}


@dataclass
class LpStats:
    solves: int = 0
    iterations: int = 0
    seconds: float = 0.0
    rebuilds: int = 0


@dataclass
class TableauRow:
    """x_basic + sum_k coef_k v_k = value, over nonbasic variables v_k that
    are problem columns (kind 0, index j) or row logicals (kind 1, row r)."""
    basic_col: int            # the problem column basic in this row (-1: a logical)
    value: float
    kind: np.ndarray          # 0 column, 1 logical
    index: np.ndarray
    coef: np.ndarray
    at_upper: np.ndarray      # nonbasic at its upper bound


class LpRelaxation:
    def __init__(self, P: MipProblem, options: dict | None = None):
        if _fedlp is None:
            raise RuntimeError("fedlp is not built")
        self.P = P
        n = P.n
        self.stats = LpStats()
        free = ~np.isfinite(P.lb) & ~np.isfinite(P.ub)
        self.free = free
        # fedlp column of x_j (and of its negative part, for free columns)
        self.pos_id = np.arange(n)
        self.neg_id = np.full(n, -1)
        k = n
        for j in np.nonzero(free)[0]:
            self.neg_id[j] = k
            k += 1
        self.n_lp = k
        self.options = dict(options or {})
        self.deadline = float("inf")        # perf_counter time no solve may pass
        self._build(P.row_lo, P.row_hi)
        self.lb = P.lb.copy()
        self.ub = P.ub.copy()
        self.cut_rows: list[int] = []       # fedlp rows that are cuts, in order
        self.cut_coefs: dict[int, tuple[np.ndarray, np.ndarray]] = {}   # row -> (cols, coef)
        self.n_base_rows = P.m
        # the rows' bounds and sigma (fedlp: a'x + sigma s = rhs)
        self._row_lo = list(P.row_lo)
        self._row_hi = list(P.row_hi)

    # ------------------------------------------------------------ building
    def _build(self, row_lo, row_hi):
        P = self.P
        lp = _fedlp.Solver.ranged(list(row_lo), list(row_hi))
        for kk, v in self.options.items():
            setattr(lp.options, kk, v)
        A = P.csc
        lb = np.where(self.free, 0.0, P.lb)
        ub = np.where(self.free, INF, P.ub)
        lp.add_columns(P.c.astype(float), lb, ub, A.indptr.astype(np.int32),
                       A.indices.astype(np.int32), A.data.astype(float))
        fj = np.nonzero(self.free)[0]
        if len(fj):
            Af = A[:, fj]
            lp.add_columns(-P.c[fj], np.zeros(len(fj)), np.full(len(fj), INF),
                           Af.indptr.astype(np.int32), Af.indices.astype(np.int32), -Af.data)
        self.lp = lp

    # -------------------------------------------------------------- bounds
    def set_bounds(self, lb: np.ndarray, ub: np.ndarray) -> None:
        """Install the domain lb <= x <= ub (only changed columns move)."""
        ch = np.nonzero((lb != self.lb) | (ub != self.ub))[0]
        lp = self.lp
        for j in ch:
            l, u = float(lb[j]), float(ub[j])
            if self.free[j]:
                # l <= x+ - x- <= u: x+ in [max(l,0), max(u,0)], x- in
                # [max(-u,0), max(-l,0)] (every sign case; inf stays inf)
                lp.set_bounds(int(self.pos_id[j]), max(l, 0.0), max(u, 0.0))
                lp.set_bounds(int(self.neg_id[j]), max(-u, 0.0), max(-l, 0.0))
            else:
                lp.set_bounds(int(j), l, u)
        self.lb = lb.copy()
        self.ub = ub.copy()

    # --------------------------------------------------------------- solve
    def solve(self, iteration_limit: int | None = None) -> int:
        """ITERATION_LIMIT also when the deadline passes mid-solve."""
        import time
        lp = self.lp
        lp.options.max_iter = int(iteration_limit) if iteration_limit else 1 << 62
        left = self.deadline - time.perf_counter()
        lp.options.time_limit = max(0.01, left) if np.isfinite(left) else 1e100
        lp.reset_stats()
        st = lp.solve()
        s = lp.stats()
        self.stats.solves += 1
        self.stats.iterations += s["iterations"]
        self.stats.seconds += s["seconds"]
        self.status = st
        return st

    def past_deadline(self) -> bool:
        import time
        return time.perf_counter() >= self.deadline

    @property
    def objective(self) -> float:
        return float(self.lp.objective) + self.P.offset

    def x(self) -> np.ndarray:
        v = np.asarray(self.lp.x())
        x = v[: self.P.n].copy()
        if self.n_lp > self.P.n:
            fj = np.nonzero(self.free)[0]
            x[fj] -= v[self.neg_id[fj]]
        return x

    def duals(self) -> np.ndarray:
        return np.asarray(self.lp.duals())

    def reduced_costs(self) -> np.ndarray:
        return np.asarray(self.lp.reduced_costs())[: self.P.n]

    # --------------------------------------------------------------- basis
    def get_basis(self):
        rs, cs = self.lp.get_basis()
        return np.array(rs), np.array(cs)

    def set_basis(self, basis) -> bool:
        rs, cs = basis
        if len(rs) != self.lp.rows:   # rows changed since (cuts): keep the current basis
            return False
        return bool(self.lp.set_basis(rs, cs))

    # ---------------------------------------------------------------- rows
    @property
    def rows(self) -> int:
        return self.lp.rows

    def add_rows(self, rows: list[tuple[np.ndarray, np.ndarray, float, float]]) -> list[int]:
        """Append rows lo <= a'x <= hi (a over problem columns); returns
        their fedlp row indices."""
        if not rows:
            return []
        lo, hi, start, idx, val = [], [], [0], [], []
        for cols, coef, l, u in rows:
            for j, a in zip(cols, coef):
                idx.append(int(self.pos_id[j]))
                val.append(float(a))
                if self.neg_id[j] >= 0:
                    idx.append(int(self.neg_id[j]))
                    val.append(-float(a))
            start.append(len(idx))
            lo.append(l)
            hi.append(u)
        first = self.lp.add_rows(np.array(lo, float), np.array(hi, float), np.array(start, np.int32),
                                 np.array(idx, np.int32), np.array(val, float))
        self.stats.rebuilds += 1
        new = list(range(first, first + len(rows)))
        self.cut_rows += new
        for r, (cols, coef, _l, _u) in zip(new, rows):
            self.cut_coefs[r] = (np.asarray(cols), np.asarray(coef, float))
        self._row_lo += lo
        self._row_hi += hi
        return new

    def remove_rows(self, rows: list[int]) -> None:
        if not rows:
            return
        drop = set(rows)
        self.lp.remove_rows(sorted(drop))
        self.stats.rebuilds += 1
        keep = [r for r in range(len(self._row_lo)) if r not in drop]
        newidx = {r: k for k, r in enumerate(keep)}
        self.cut_rows = [newidx[r] for r in self.cut_rows if r in newidx]
        self.cut_coefs = {newidx[r]: v for r, v in self.cut_coefs.items() if r in newidx}
        self._row_lo = [self._row_lo[r] for r in keep]
        self._row_hi = [self._row_hi[r] for r in keep]

    def row_coefs(self, r: int) -> tuple[np.ndarray, np.ndarray]:
        """Row r's coefficients over problem columns (original or cut row)."""
        if r < self.n_base_rows:
            A = self.P.A
            a, b = A.indptr[r], A.indptr[r + 1]
            return A.indices[a:b], A.data[a:b]
        return self.cut_coefs[r]

    def row_bounds(self, r: int) -> tuple[float, float]:
        return self._row_lo[r], self._row_hi[r]

    def row_activity_slack(self) -> np.ndarray:
        """fedlp's logical values s (a'x + sigma s = rhs) per row."""
        return np.asarray(self.lp.logicals())

    # ------------------------------------------------------------- tableau
    def logical_info(self, r: int):
        """(sigma, rhs, s_lb, s_ub) of row r's logical in fedlp's form."""
        lo, hi = self._row_lo[r], self._row_hi[r]
        if lo == hi:
            return 1.0, hi, 0.0, 0.0
        if not np.isfinite(lo):
            return 1.0, hi, 0.0, INF
        if not np.isfinite(hi):
            return -1.0, lo, 0.0, INF
        return 1.0, hi, 0.0, hi - lo

    def basic_heads(self) -> np.ndarray:
        return np.asarray(self.lp.basic_heads())

    def tableau_row(self, position: int, heads: np.ndarray | None = None) -> TableauRow | None:
        """Row `position` of B^-1 A over problem columns and logicals, or
        None when it involves a split free column (no single bound status)."""
        heads = self.basic_heads() if heads is None else heads
        h = int(heads[position])
        if h >= self.P.n or (h >= 0 and self.free[h]):
            return None
        ids, vals = self.lp.tableau_row(position)
        ids, vals = np.asarray(ids), np.asarray(vals)
        rs, cs = self.lp.get_basis()
        cs, rs = np.asarray(cs), np.asarray(rs)
        kind = np.where(ids >= 0, 0, 1)
        index = np.where(ids >= 0, ids, -1 - ids)
        if np.any((kind == 0) & (index >= self.P.n)) or np.any((kind == 0) & self.free[np.minimum(index, self.P.n - 1)]):
            return None
        at_up = np.where(kind == 0, cs[np.where(kind == 0, index, 0)] == 2, rs[np.where(kind == 1, index, 0)] == 2)
        x = np.asarray(self.lp.x())
        if h >= 0:
            value = float(x[h])
        else:
            value = float(np.asarray(self.lp.logicals())[-1 - h])
        return TableauRow(h, value, kind, index, vals, at_up)
