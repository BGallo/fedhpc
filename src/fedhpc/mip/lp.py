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
- Interval row blocks (time-indexed capacity rows) reach fedlp differenced
  into network rows with explicit slacks (``reformulate``): same LP, same
  tableaux, a fraction of the nonzeros. A slack is reported as its
  original row's logical, so callers see the problem's rows only.

Every solve's iterations and seconds are added to ``stats``.
"""
from __future__ import annotations

import os
import sys
import time as _time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from .problem import INF, MipProblem
from .reformulate import detect_interval_blocks, network_form

try:
    if os.environ.get("FEDLP_DIR"):         # a standalone build (profiling, A/B variants)
        sys.path.insert(0, os.environ["FEDLP_DIR"])
        import _fedlp
    else:
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
    # fedlp's own counters and phase timers (t_*: a -DFEDLP_PROFILE build),
    # summed over solves
    fedlp: dict = field(default_factory=dict)


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
    def __init__(self, P: MipProblem, options: dict | None = None, network: bool | dict = True):
        """``network``: reformulate interval row blocks (a dict passes
        detection parameters to ``detect_interval_blocks``)."""
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
        self.net = None
        if network:
            blocks = detect_interval_blocks(P, **(network if isinstance(network, dict) else {}))
            if blocks.blocks:
                self.net = network_form(P, blocks)
        self.slack0 = k                     # fedlp column of the first slack
        if self.net is not None:
            k += len(self.net.slack_row)
        self.n_lp = k
        self.options = dict(options or {})
        # sum fedlp's full stats over solves (scripts/profile.py turns it on)
        self.collect_stats = bool(os.environ.get("FEDHPC_LP_STATS"))
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
        P, net = self.P, self.net
        if net is not None:
            row_lo, row_hi = net.row_lo, net.row_hi
        lp = _fedlp.Solver.ranged(list(row_lo), list(row_hi))
        for kk, v in self.options.items():
            setattr(lp.options, kk, v)
        A = P.csc if net is None else net.A.tocsc()
        lb = np.where(self.free, 0.0, P.lb)
        ub = np.where(self.free, INF, P.ub)
        lp.add_columns(P.c.astype(float), lb, ub, A.indptr.astype(np.int32),
                       A.indices.astype(np.int32), A.data.astype(float))
        fj = np.nonzero(self.free)[0]
        if len(fj):
            Af = A[:, fj]
            lp.add_columns(-P.c[fj], np.zeros(len(fj)), np.full(len(fj), INF),
                           Af.indptr.astype(np.int32), Af.indices.astype(np.int32), -Af.data)
        if net is not None:
            S = net.S
            k = S.shape[1]
            lp.add_columns(np.zeros(k), np.zeros(k), net.slack_ub, S.indptr.astype(np.int32),
                           S.indices.astype(np.int32), S.data.astype(float))
        self.lp = lp

    # -------------------------------------------------------------- bounds
    def set_bounds(self, lb: np.ndarray, ub: np.ndarray) -> None:
        """Install the domain lb <= x <= ub (only changed columns move)."""
        ch = np.nonzero((lb != self.lb) | (ub != self.ub))[0]
        lp = self.lp
        fr = self.free[ch]
        plain = ch[~fr]
        if len(plain):          # one call for the lot
            lp.set_bounds_many(plain.astype(np.int32), lb[plain].astype(float), ub[plain].astype(float))
        for j in ch[fr]:
            l, u = float(lb[j]), float(ub[j])
            # l <= x+ - x- <= u: x+ in [max(l,0), max(u,0)], x- in
            # [max(-u,0), max(-l,0)] (every sign case; inf stays inf)
            lp.set_bounds(int(self.pos_id[j]), max(l, 0.0), max(u, 0.0))
            lp.set_bounds(int(self.neg_id[j]), max(-u, 0.0), max(-l, 0.0))
        self.lb = lb.copy()
        self.ub = ub.copy()

    # --------------------------------------------------------------- solve
    def solve(self, iteration_limit: int | None = None) -> int:
        """ITERATION_LIMIT also when the deadline passes mid-solve."""
        left = self.deadline - _time.perf_counter()
        st, its, secs = self.lp.solve_counted(int(iteration_limit) if iteration_limit else 1 << 62,
                                              max(0.01, left) if left < 1e100 else 1e100)
        self.stats.solves += 1
        self.stats.iterations += its
        self.stats.seconds += secs
        if self.collect_stats:            # fedlp's full counters and timers (profiling)
            acc = self.stats.fedlp
            for k, v in self.lp.stats().items():
                if k not in ("dual_exit", "dual_status", "post_dual_infeas") and v:
                    acc[k] = acc.get(k, 0) + v
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
        """Row duals in the problem's rows (a differenced block's mapped
        back: y_t = y'_t - y'_{t+1}, the transpose of the differencing)."""
        y = np.asarray(self.lp.duals()).copy()
        if self.net is not None:
            yp = y.copy()
            for lo, hi in self.net.blocks:
                y[lo:hi] = yp[lo:hi] - yp[lo + 1:hi + 1]
        return y

    def reduced_costs(self) -> np.ndarray:
        return np.asarray(self.lp.reduced_costs())[: self.P.n]

    # --------------------------------------------------------------- basis
    def get_basis(self):
        rs, cs = self.lp.get_basis()
        return np.array(rs), np.array(cs)

    def set_basis(self, basis) -> bool:
        """Install (rs, cs) or a state (rs, cs, dse_weights) from get_state."""
        rs, cs = basis[0], basis[1]
        if len(rs) != self.lp.rows:   # rows changed since (cuts): keep the current basis
            return False
        ok = bool(self.lp.set_basis(rs, cs))
        if ok and len(basis) > 2 and basis[2] is not None and len(basis[2]):
            self.lp.set_dual_weights(basis[2])
        return ok

    def get_state(self):
        """The basis with the dual simplex's DSE weights (float32), so a
        restore skips their O(m^2) re-initialisation."""
        rs, cs = self.get_basis()
        w = np.asarray(self.lp.get_dual_weights())
        return rs, cs, (np.minimum(w, 3e38).astype(np.float32) if len(w) else None)

    def slack_basis(self):
        """The basis of the rows' logicals (a reformulated block's slacks in
        place of its differenced rows' logicals: lower bidiagonal, so
        nonsingular, and the original problem's slack basis)."""
        rs = np.zeros(self.lp.rows, np.int8)
        cs = np.ones(self.lp.cols, np.int8)
        if self.net is not None:
            rs[: self.P.m][self.net.block_row] = 1
            cs[self.slack0: self.slack0 + len(self.net.slack_row)] = 0
        return rs, cs

    def basis_from_point(self, x: np.ndarray, tol: float = 1e-9) -> bool:
        """Install a basis whose basic solution is the feasible point x
        (when x is a vertex), so the primal simplex starts in phase 2.

        Columns strictly inside their bounds are basic, the others sit at
        the bound x has them on. Each basic column is matched to a row of
        its own support (fewest free rows first, equality rows preferred:
        their logicals are fixed), and every unmatched row keeps its logical
        basic, at its value under x (degenerate at a tight row). A matching
        makes the basis nonsingular whenever it is triangular; fedlp repairs
        the rest with logicals. Returns False when x has more columns
        inside their bounds than there are rows (not a vertex)."""
        P = self.P
        n, m = P.n, self.lp.rows
        xs = np.asarray(x, float)
        ncol = self.lp.cols
        cs = np.ones(ncol, np.int8)
        rs = np.zeros(m, np.int8)
        val = np.zeros(ncol)                      # fedlp column values under x
        val[:n] = np.where(self.free, np.maximum(xs, 0.0), xs)
        lbv = np.r_[np.where(self.free, 0.0, P.lb), np.zeros(ncol - n)]
        ubv = np.r_[np.where(self.free, INF, P.ub), np.full(ncol - n, INF)]
        if self.n_lp > n:
            fj = np.nonzero(self.free)[0]
            val[self.neg_id[fj]] = np.maximum(-xs[fj], 0.0)
        act = P.A @ xs
        if self.net is not None:
            sl = np.arange(self.slack0, self.slack0 + len(self.net.slack_row))
            t = self.net.slack_row
            lo, hi = P.row_lo[t], P.row_hi[t]
            sig = np.where(~np.isfinite(hi), -1.0, 1.0)
            rhs = np.where(~np.isfinite(hi), lo, hi)
            val[sl] = sig * (rhs - act[t])
            ubv[sl] = self.net.slack_ub
        inside = (val > lbv + tol * (1 + np.abs(lbv))) & (val < ubv - tol * (1 + np.abs(ubv)))
        at_up = ~inside & np.isfinite(ubv) & (np.abs(val - ubv) <= tol * (1 + np.abs(ubv)))
        cs[inside] = 0
        cs[at_up & ~inside] = 2
        basic_cols = np.nonzero(inside)[0]
        if len(basic_cols) > m:
            return False
        # rows: the LP's (block rows differenced; their logicals fixed at 0)
        lo_r = np.asarray(self.lp_row_lo())
        hi_r = np.asarray(self.lp_row_hi())
        fixed_row = lo_r == hi_r
        Acsc = self._lp_csc()
        taken = np.zeros(m, bool)
        sup = [Acsc.indices[Acsc.indptr[j]:Acsc.indptr[j + 1]] for j in basic_cols]
        order = np.argsort([len(r) for r in sup], kind="stable")
        for k in order:
            rows = sup[k][~taken[sup[k]]]
            if len(rows) == 0:
                continue                         # dependent: fedlp's repair decides
            eq = rows[fixed_row[rows]]
            r = int(eq[0] if len(eq) else rows[0])
            taken[r] = True
        rs[taken] = 1
        # basic count must be m: unmatched basic columns displace logicals
        extra = len(basic_cols) - int(taken.sum())
        if extra > 0:
            free_rows = np.nonzero(~taken & ~fixed_row)[0]
            free_rows = np.r_[free_rows, np.nonzero(~taken & fixed_row)[0]]
            rs[free_rows[:extra]] = 1
        # nonbasic logicals at the bound their row sits on
        lact = np.r_[act, [float(self.cut_coefs[r][1] @ xs[self.cut_coefs[r][0]]) for r in range(P.m, m)]] \
            if m > P.m else act
        # a nonbasic logical of a ranged row tight at its lower side sits at
        # its upper bound (a'x + s = hi, s in [0, hi - lo])
        ranged = np.isfinite(lo_r) & np.isfinite(hi_r) & (lo_r < hi_r)
        at_lo_row = np.isfinite(lo_r) & (np.abs(lact - lo_r) <= tol * (1 + np.abs(lo_r)))
        rs[(rs != 0) & ranged & at_lo_row] = 2
        return bool(self.lp.set_basis(rs, cs))

    def lp_row_lo(self):
        return list(self.net.row_lo) + self._row_lo[self.P.m:] if self.net is not None else self._row_lo

    def lp_row_hi(self):
        return list(self.net.row_hi) + self._row_hi[self.P.m:] if self.net is not None else self._row_hi

    def _lp_csc(self) -> sp.csc_matrix:
        """fedlp's matrix over its columns (base rows, then cut rows)."""
        P = self.P
        A = P.A if self.net is None else self.net.A
        blocks = [A.tocsc()]
        if self.n_lp > P.n:
            fj = np.nonzero(self.free)[0]
            blocks.append(-blocks[0][:, fj])
        if self.net is not None:
            blocks.append(self.net.S)
        M = sp.hstack(blocks, format="csr")
        if self.rows > P.m:
            cut = [sp.csr_matrix((cf, (np.zeros(len(cc), int), cc)), shape=(1, P.n)) for cc, cf in
                   (self.cut_coefs[r] for r in range(P.m, self.rows))]
            C = sp.vstack(cut, format="csr")
            pad = sp.csr_matrix((C.shape[0], M.shape[1] - P.n))
            M = sp.vstack([M, sp.hstack([C, pad], format="csr")], format="csr")
        return M.tocsc()

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

    def logical_arrays(self):
        """logical_info for every row at once: (sigma, rhs, s_lb, s_ub)."""
        lo, hi = np.asarray(self._row_lo, float), np.asarray(self._row_hi, float)
        eq = lo == hi
        no_lo, no_hi = ~np.isfinite(lo), ~np.isfinite(hi)
        sigma = np.where(~eq & ~no_lo & no_hi, -1.0, 1.0)
        rhs = np.where(sigma < 0, lo, hi)
        s_ub = np.where(eq, 0.0, np.where(no_lo | no_hi, INF, hi - lo))
        return sigma, rhs, np.zeros(len(lo)), s_ub

    def basic_heads(self) -> np.ndarray:
        return np.asarray(self.lp.basic_heads())

    def tableau_row(self, position: int, heads: np.ndarray | None = None, basis=None,
                    xlp: np.ndarray | None = None) -> TableauRow | None:
        """Row `position` of B^-1 A over problem columns and logicals, or
        None when it involves a split free column (no single bound status).
        A reformulated block's slack is its original row's logical; the
        differenced rows' own logicals are fixed at 0 and left out (their
        term vanishes for every feasible point)."""
        heads = self.basic_heads() if heads is None else heads
        h = int(heads[position])
        n = self.P.n
        if h >= n or (h >= 0 and self.free[h]):
            return None
        ids, vals = self.lp.tableau_row(position)
        ids, vals = np.asarray(ids), np.asarray(vals)
        rs, cs = self.lp.get_basis() if basis is None else basis
        cs, rs = np.asarray(cs), np.asarray(rs)
        col = ids >= 0
        if np.any(col & (ids >= n) & (ids < self.slack0)):
            return None                                     # a free column's negative part
        if np.any(self.free[ids[col & (ids < n)]]):
            return None
        kind = np.where(col, 0, 1)
        index = np.where(col, ids, -1 - ids)
        at_up = np.zeros(len(ids), bool)
        at_up[col] = cs[ids[col]] == 2
        at_up[~col] = rs[index[~col]] == 2
        keep = np.ones(len(ids), bool)
        if self.net is not None:
            sl = col & (ids >= self.slack0)
            kind[sl] = 1
            index[sl] = self.net.slack_row[ids[sl] - self.slack0]
            lg = ~col
            r = index[lg]
            keep[np.nonzero(lg)[0][(r < self.n_base_rows) & self.net.block_row[np.minimum(r, self.P.m - 1)]]] = False
        x = np.asarray(self.lp.x()) if xlp is None else xlp
        if h >= 0:
            value = float(x[h])
        else:
            value = float(np.asarray(self.lp.logicals())[-1 - h])
        return TableauRow(h, value, kind[keep], index[keep], vals[keep], at_up[keep])
