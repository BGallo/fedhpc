"""Activity-based domain propagation (feasibility-based bound tightening).

For row i, lo_i <= sum_j a_ij x_j <= hi_i, the activity bounds are

    minact_i = sum_{a>0} a lb_j + sum_{a<0} a ub_j,
    maxact_i = sum_{a>0} a ub_j + sum_{a<0} a lb_j.

Removing entry j's own contribution gives residual activities, and with
them (Brearley, Mitra & Williams 1975; Achterberg 2007, ch. 7):

    a > 0:  x_j <= (hi_i - minact_i^{-j}) / a,   x_j >= (lo_i - maxact_i^{-j}) / a
    a < 0:  x_j >= (hi_i - minact_i^{-j}) / a,   x_j <= (lo_i - maxact_i^{-j}) / a

Infinite contributions are counted, not summed (an entry's residual is
finite when the only infinite term was its own), so the computation stays
exact with unbounded columns. Integer bounds round inwards with a tolerance;
a continuous bound only moves by more than ``min_rel_change`` of its domain,
which keeps the rounds from creeping. Everything is vectorised over the CSR
entries; per-column reductions go through a precomputed CSR -> CSC
permutation.

The propagator is sound (never removes a feasible point beyond the
tolerances) and incomplete (a fixed point of these rules, not the convex
hull). ``propagate`` returns whether the domain is still nonempty.

``Propagator`` runs on the C++ kernel of ``fedhpc._fedmip`` (event driven:
only rows whose columns changed are re-examined; a bound trail with
``level``/``change``/``undo`` serves dives and backtracking without copying
the domain). ``PyPropagator`` is the vectorised NumPy original, kept as the
reference the tests compare against; it is used when the extension is not
built.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .problem import INF, MipProblem

try:
    from .. import _fedmip
except ImportError:  # pragma: no cover - the extension is part of the build
    _fedmip = None


@dataclass
class PropagationResult:
    feasible: bool
    lb: np.ndarray
    ub: np.ndarray
    tightened: int      # bound changes made
    rounds: int
    infeasible_row: int = -1


class PyPropagator:
    def __init__(self, P: MipProblem, int_tol: float = 1e-6, feas_tol: float = 1e-6,
                 min_rel_change: float = 1e-3, max_abs_bound: float = 1e9):
        A = P.A
        self.P = P
        self.m, self.n = A.shape
        self.indptr = A.indptr
        self.col = A.indices
        self.val = A.data
        self.row = np.repeat(np.arange(self.m), np.diff(A.indptr))
        self.pos = self.val > 0
        self.row_lo, self.row_hi = P.row_lo, P.row_hi
        self.integer = P.integer
        self.int_tol, self.feas_tol = int_tol, feas_tol
        self.min_rel_change = min_rel_change
        self.max_abs_bound = max_abs_bound
        # CSR entry order -> CSC order, and column segment starts
        self.perm = np.argsort(self.col, kind="stable")
        counts = np.bincount(self.col, minlength=self.n)
        self.cstart = np.concatenate(([0], np.cumsum(counts)))
        self.nonempty_col = counts > 0
        self.nonempty_row = np.diff(self.indptr) > 0

    # ----------------------------------------------------------- activities
    def _row_sum(self, v: np.ndarray) -> np.ndarray:
        out = np.zeros(self.m)
        if v.size:
            s = np.add.reduceat(v, self.indptr[:-1][self.nonempty_row])
            out[self.nonempty_row] = s
        return out

    def activities(self, lb: np.ndarray, ub: np.ndarray):
        """Per row: finite parts and infinite counts of minact and maxact,
        and the per-entry contributions they came from."""
        lo_b, hi_b = lb[self.col], ub[self.col]
        cmin = np.where(self.pos, self.val * lo_b, self.val * hi_b)   # entry's share of minact
        cmax = np.where(self.pos, self.val * hi_b, self.val * lo_b)
        imin, imax = ~np.isfinite(cmin), ~np.isfinite(cmax)
        fmin = self._row_sum(np.where(imin, 0.0, cmin))
        fmax = self._row_sum(np.where(imax, 0.0, cmax))
        nmin = self._row_sum(imin.astype(float))
        nmax = self._row_sum(imax.astype(float))
        return fmin, nmin, fmax, nmax, cmin, cmax, imin, imax

    def row_activity_bounds(self, lb, ub):
        fmin, nmin, fmax, nmax, *_ = self.activities(lb, ub)
        return np.where(nmin > 0, -INF, fmin), np.where(nmax > 0, INF, fmax)

    # ------------------------------------------------------------- rounds
    def propagate(self, lb: np.ndarray, ub: np.ndarray, max_rounds: int = 20,
                  rows: np.ndarray | None = None) -> PropagationResult:
        """Tighten (copies of) lb, ub to a fixed point of the activity rules,
        at most max_rounds full passes. rows restricts the first pass."""
        lb = lb.astype(float, copy=True)
        ub = ub.astype(float, copy=True)
        total = 0
        r_active = None if rows is None else np.zeros(self.m, bool)
        if r_active is not None:
            r_active[rows] = True
        for rnd in range(1, max_rounds + 1):
            fmin, nmin, fmax, nmax, cmin, cmax, imin, imax = self.activities(lb, ub)
            # infeasible rows
            minact = np.where(nmin > 0, -INF, fmin)
            maxact = np.where(nmax > 0, INF, fmax)
            tol_hi = self.feas_tol * (1 + np.abs(self.row_hi))
            tol_lo = self.feas_tol * (1 + np.abs(self.row_lo))
            bad = (minact > self.row_hi + tol_hi) | (maxact < self.row_lo - tol_lo)
            if np.any(bad):
                return PropagationResult(False, lb, ub, total, rnd, int(np.argmax(bad)))
            r = self.row
            # residuals without the entry's own contribution
            res_min_n = nmin[r] - imin
            res_min = np.where(imin, fmin[r], fmin[r] - cmin)
            res_max_n = nmax[r] - imax
            res_max = np.where(imax, fmax[r], fmax[r] - cmax)
            hi_r, lo_r = self.row_hi[r], self.row_lo[r]
            a = self.val
            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                # from the row's upper side: a x_j <= hi - res_min
                b1 = np.where((res_min_n == 0) & np.isfinite(hi_r), (hi_r - res_min) / a, np.nan)
                # from the lower side: a x_j >= lo - res_max
                b2 = np.where((res_max_n == 0) & np.isfinite(lo_r), (lo_r - res_max) / a, np.nan)
            if r_active is not None:
                keep = r_active[r]
                b1 = np.where(keep, b1, np.nan)
                b2 = np.where(keep, b2, np.nan)
                r_active = None
            new_ub_e = np.where(self.pos, b1, b2)   # a > 0: from hi; a < 0: from lo
            new_lb_e = np.where(self.pos, b2, b1)
            new_ub = self._col_reduce(new_ub_e, np.fmin, INF)
            new_lb = self._col_reduce(new_lb_e, np.fmax, -INF)
            nub, nlb = self._clean(new_lb, new_ub, lb, ub)
            ch_u = nub < ub
            ch_l = nlb > lb
            nch = int(ch_u.sum() + ch_l.sum())
            if nch == 0:
                return PropagationResult(True, lb, ub, total, rnd)
            ub = np.where(ch_u, nub, ub)
            lb = np.where(ch_l, nlb, lb)
            total += nch
            if np.any(lb > ub + self.feas_tol * (1 + np.abs(ub))):
                return PropagationResult(False, lb, ub, total, rnd)
            # snap crossings within tolerance
            cross = lb > ub
            if np.any(cross):
                mid = np.where(self.integer, np.round(lb), 0.5 * (lb + ub))
                lb = np.where(cross, mid, lb)
                ub = np.where(cross, mid, ub)
        return PropagationResult(True, lb, ub, total, max_rounds)

    # ------------------------------------------ stateful domain (snapshots)
    # The interface of the C++ Propagator's trail, by copying the domain.
    def reset(self, lb, ub, max_rounds: int = 20, propagate: bool = True) -> bool:
        self._stack, self._bad = [], -1
        self._l, self._u = np.asarray(lb, float).copy(), np.asarray(ub, float).copy()
        if not propagate:
            return True
        res = self.propagate(self._l, self._u, max_rounds)
        self._l, self._u, self._bad = res.lb, res.ub, res.infeasible_row
        return res.feasible

    def level(self) -> int:
        return len(self._stack)

    def change(self, cols, lb, ub, max_rounds: int = 20) -> bool:
        self._stack.append((self._l, self._u))
        l, u = self._l.copy(), self._u.copy()
        cols = np.asarray(cols, dtype=np.int64)
        l[cols] = np.maximum(l[cols], lb)
        u[cols] = np.minimum(u[cols], ub)
        if np.any(l[cols] > u[cols] + self.feas_tol):
            self._l, self._u, self._bad = l, u, -1
            return False
        rows = np.unique(self.P.csc[:, cols].indices)
        res = self.propagate(l, u, max_rounds, rows=rows)
        self._l, self._u, self._bad = res.lb, res.ub, res.infeasible_row
        return res.feasible

    def undo(self, level: int) -> None:
        while len(self._stack) > level:
            self._l, self._u = self._stack.pop()

    def lb(self) -> np.ndarray:
        return self._l.copy()

    def ub(self) -> np.ndarray:
        return self._u.copy()

    @property
    def infeasible_row(self) -> int:
        return self._bad

    def _col_reduce(self, v_csr: np.ndarray, op, empty: float) -> np.ndarray:
        v = v_csr[self.perm]
        v = np.where(np.isnan(v), empty, v)
        out = np.full(self.n, empty)
        if v.size:
            red = op.reduceat(v, self.cstart[:-1][self.nonempty_col])
            out[self.nonempty_col] = red
        return out

    def _clean(self, new_lb, new_ub, lb, ub):
        """Round integer bounds inwards, drop huge values and continuous
        moves too small to matter."""
        it = self.int_tol
        nub = np.where(self.integer & np.isfinite(new_ub), np.floor(new_ub + it), new_ub)
        nlb = np.where(self.integer & np.isfinite(new_lb), np.ceil(new_lb - it), new_lb)
        big = self.max_abs_bound
        nub = np.where(np.abs(nub) > big, INF, nub)
        nlb = np.where(np.abs(nlb) > big, -INF, nlb)
        width = np.where(np.isfinite(ub - lb), ub - lb, np.maximum(1.0, np.abs(np.where(np.isfinite(ub), ub, lb))))
        width = np.nan_to_num(width, nan=1.0, posinf=1.0)
        thr = np.where(self.integer, 0.5, self.min_rel_change * np.maximum(width, 1e-6))
        with np.errstate(invalid="ignore"):
            nub = np.where(nub < ub - thr, nub, ub)
            nlb = np.where(nlb > lb + thr, nlb, lb)
        return nub, nlb


class Propagator:
    """The C++ propagator over P's rows, with PyPropagator's interface plus
    a stateful domain (``reset``, ``level``, ``change``, ``undo``, ``lb``,
    ``ub``)."""

    def __new__(cls, P: MipProblem, *args, **kwargs):
        if _fedmip is None:
            return PyPropagator(P, *args, **kwargs)
        return super().__new__(cls)

    def __init__(self, P: MipProblem, int_tol: float = 1e-6, feas_tol: float = 1e-6,
                 min_rel_change: float = 1e-3, max_abs_bound: float = 1e9):
        A = P.A.tocsr()
        self.P = P
        self.m, self.n = A.shape
        self.integer = P.integer
        k = _fedmip.Propagator(self.m, self.n, A.indptr.astype(np.int32), A.indices.astype(np.int32),
                               A.data.astype(float), P.row_lo.astype(float), P.row_hi.astype(float),
                               P.integer.astype(np.int8))
        k.int_tol, k.feas_tol = int_tol, feas_tol
        k.min_rel_change, k.max_abs_bound = min_rel_change, max_abs_bound
        self.k = k
        self.calls = 0

    def propagate(self, lb: np.ndarray, ub: np.ndarray, max_rounds: int = 20,
                  rows: np.ndarray | None = None) -> PropagationResult:
        self.calls += 1
        r = None if rows is None else np.asarray(rows, dtype=np.int32)
        ok, nl, nu, tight, rounds, bad = self.k.propagate(lb, ub, max_rounds, r)
        return PropagationResult(bool(ok), nl, nu, int(tight), int(rounds), int(bad))

    def row_activity_bounds(self, lb, ub):
        self.k.reset(lb, ub)
        return self.k.activity_bounds()

    # ------------------------------------------------- stateful domain
    def reset(self, lb: np.ndarray, ub: np.ndarray, max_rounds: int = 20, propagate: bool = True) -> bool:
        """Install a domain (clearing the trail) and propagate it fully, or
        not, for a domain known to be a fixed point already."""
        self.k.reset(np.asarray(lb, float), np.asarray(ub, float))
        return bool(self.k.propagate_domain(max_rounds)) if propagate else True

    def level(self) -> int:
        return self.k.level()

    def change(self, cols, lb, ub, max_rounds: int = 20) -> bool:
        """Tighten cols to [lb, ub] (never loosens) and propagate."""
        self.calls += 1
        return bool(self.k.change(np.asarray(cols, np.int32), np.asarray(lb, float),
                                  np.asarray(ub, float), max_rounds))

    def undo(self, level: int) -> None:
        self.k.undo(level)

    def lb(self) -> np.ndarray:
        return self.k.lb()

    def ub(self) -> np.ndarray:
        return self.k.ub()

    @property
    def infeasible_row(self) -> int:
        return self.k.infeasible_row
