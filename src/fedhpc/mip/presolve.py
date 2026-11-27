"""MIP presolve: reductions that keep every optimal solution's value.

Each round applies, until nothing changes (or ``max_rounds``):

1. Fixed columns (lb == ub) leave the problem: their activity moves into
   the row bounds, their cost into the offset.
2. Empty columns are fixed at their cost-best bound (an infinite one with a
   cost pointing at it means the problem is unbounded or infeasible).
3. Singleton rows a x_j in [lo, hi] become bounds on x_j.
4. Domain propagation (``Propagator``) tightens bounds; rows whose activity
   range lies inside their bounds are redundant and leave.
5. Dual fixing: a column whose cost and row coefficients all prefer it at a
   bound (no row "locks" it there) is fixed at that bound (Achterberg 2007,
   sec. 10.8).
6. Coefficient tightening on one-sided rows over binaries (Savelsbergh
   1994): for a'x <= b with a_j > 0 and maxact - a_j < b, setting
   a_j' = a_j - d, b' = b - d with d = b - (maxact - a_j) gives the same
   integer points and a tighter relaxation.

Every reduction fixes a column, tightens a bound or a coefficient, or
deletes a row, so postsolve only scatters the kept and fixed values back.
No aggregation or substitution yet (those would need a postsolve stack).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from .problem import INF, MipProblem
from .propagate import Propagator


@dataclass
class Postsolve:
    n_original: int
    kept_cols: np.ndarray            # reduced column k is original kept_cols[k]
    fixed_value: np.ndarray          # value of every original column not kept
    kept_rows: np.ndarray

    def expand(self, x_reduced: np.ndarray) -> np.ndarray:
        x = self.fixed_value.copy()
        x[self.kept_cols] = x_reduced
        return x

    def restrict(self, x_original: np.ndarray) -> np.ndarray:
        return x_original[self.kept_cols]


@dataclass
class PresolveStats:
    rounds: int = 0
    fixed_cols: int = 0
    removed_rows: int = 0
    singleton_rows: int = 0
    dual_fixed: int = 0
    coef_tightened: int = 0
    bounds_tightened: int = 0
    seconds: float = 0.0
    status: str = "reduced"          # or "infeasible", "unbounded"
    notes: list = field(default_factory=list)


def presolve(P: MipProblem, max_rounds: int = 20, feas_tol: float = 1e-9,
             tighten_coefficients: bool = True) -> tuple[MipProblem | None, Postsolve, PresolveStats]:
    import time
    t0 = time.perf_counter()
    st = PresolveStats()
    A = P.A.tocsr().astype(float)
    row_lo, row_hi = P.row_lo.copy(), P.row_hi.copy()
    lb, ub, c = P.lb.copy(), P.ub.copy(), P.c.copy()
    integer = P.integer.copy()
    offset = P.offset
    m, n = A.shape
    col_alive = np.ones(n, bool)
    row_alive = np.ones(m, bool)
    fixed_value = np.zeros(n)
    # integer bounds integral
    lb = np.where(integer & np.isfinite(lb), np.ceil(lb - 1e-9), lb)
    ub = np.where(integer & np.isfinite(ub), np.floor(ub + 1e-9), ub)

    def finish(status: str):
        st.status = status
        st.seconds = time.perf_counter() - t0
        post = Postsolve(n, np.nonzero(col_alive)[0], fixed_value, np.nonzero(row_alive)[0])
        if status != "reduced":
            return None, post, st
        kc, kr = post.kept_cols, post.kept_rows
        Ared = A[kr][:, kc].tocsr()
        Q = MipProblem(Ared, row_lo[kr], row_hi[kr], lb[kc], ub[kc], c[kc], integer[kc],
                       offset=offset, name=P.name, maximize=P.maximize,
                       col_names=[P.col_names[j] for j in kc] if P.col_names else None,
                       row_names=[P.row_names[i] for i in kr] if P.row_names else None)
        return Q, post, st

    for rnd in range(max_rounds):
        st.rounds = rnd + 1
        changed = False
        # restrict to live rows/columns
        Acur = A.multiply(row_alive[:, None]).multiply(col_alive[None, :]).tocsr()
        Acur.eliminate_zeros()
        # --- 1. fixed columns
        fx = col_alive & (lb == ub)
        if np.any(fx):
            idx = np.nonzero(fx)[0]
            v = lb[idx]
            act = Acur[:, idx] @ v
            row_lo -= act
            row_hi -= act
            offset += float(c[idx] @ v)
            fixed_value[idx] = v
            col_alive[idx] = False
            st.fixed_cols += len(idx)
            changed = True
            Acur = Acur.multiply(col_alive[None, :]).tocsr()
            Acur.eliminate_zeros()
        rnnz = np.diff(Acur.indptr)
        # --- empty rows
        er = row_alive & (rnnz == 0)
        if np.any(er):
            if np.any((row_lo[er] > feas_tol * (1 + np.abs(row_lo[er]))) |
                      (row_hi[er] < -feas_tol * (1 + np.abs(row_hi[er])))):
                return finish("infeasible")
            row_alive[er] = False
            st.removed_rows += int(er.sum())
            changed = True
        # --- 2. empty columns
        cnnz = np.bincount(Acur.indices, minlength=n)
        ec = col_alive & (cnnz == 0)
        if np.any(ec):
            for j in np.nonzero(ec)[0]:
                if c[j] > 0:
                    v = lb[j]
                elif c[j] < 0:
                    v = ub[j]
                else:
                    v = lb[j] if np.isfinite(lb[j]) else (ub[j] if np.isfinite(ub[j]) else 0.0)
                if not np.isfinite(v):
                    return finish("unbounded")
                fixed_value[j] = v
                offset += c[j] * v
                col_alive[j] = False
                st.fixed_cols += 1
            changed = True
        # --- 3. singleton rows -> bounds
        sr = row_alive & (rnnz == 1)
        if np.any(sr):
            for i in np.nonzero(sr)[0]:
                a0, a1 = Acur.indptr[i], Acur.indptr[i + 1]
                j, a = Acur.indices[a0], Acur.data[a0]
                lo_i, hi_i = row_lo[i] / a, row_hi[i] / a
                if a < 0:
                    lo_i, hi_i = hi_i, lo_i
                if integer[j]:
                    lo_i = np.ceil(lo_i - 1e-9) if np.isfinite(lo_i) else lo_i
                    hi_i = np.floor(hi_i + 1e-9) if np.isfinite(hi_i) else hi_i
                if lo_i > lb[j]:
                    lb[j] = lo_i
                if hi_i < ub[j]:
                    ub[j] = hi_i
                if lb[j] > ub[j] + feas_tol * (1 + abs(ub[j])):
                    return finish("infeasible")
                ub[j] = max(ub[j], lb[j])
                row_alive[i] = False
                st.singleton_rows += 1
            changed = True
        if changed:
            continue   # restart the round on the smaller problem
        # --- 4. propagation and redundant rows
        live_r, live_c = np.nonzero(row_alive)[0], np.nonzero(col_alive)[0]
        sub = MipProblem(Acur[live_r][:, live_c], row_lo[live_r], row_hi[live_r],
                         lb[live_c], ub[live_c], c[live_c], integer[live_c])
        prop = Propagator(sub, feas_tol=1e-7)
        res = prop.propagate(sub.lb, sub.ub, max_rounds=10)
        if not res.feasible:
            return finish("infeasible")
        nt = int(np.sum(res.lb > sub.lb) + np.sum(res.ub < sub.ub))
        if nt:
            st.bounds_tightened += nt
            lb[live_c], ub[live_c] = res.lb, res.ub
            changed = True
        mina, maxa = prop.row_activity_bounds(res.lb, res.ub)
        tol = 1e-9
        red = (mina >= sub.row_lo - tol * (1 + np.abs(sub.row_lo))) & \
              (maxa <= sub.row_hi + tol * (1 + np.abs(sub.row_hi)))
        if np.any(red):
            row_alive[live_r[red]] = False
            st.removed_rows += int(red.sum())
            changed = True
        # --- 5. dual fixing
        As = sub.A
        down_locks = np.zeros(sub.n)   # rows that forbid decreasing x_j
        up_locks = np.zeros(sub.n)
        rr = np.repeat(np.arange(sub.m), np.diff(As.indptr))
        a = As.data
        fin_lo = np.isfinite(sub.row_lo)[rr]
        fin_hi = np.isfinite(sub.row_hi)[rr]
        np.add.at(down_locks, As.indices, ((a > 0) & fin_lo) | ((a < 0) & fin_hi))
        np.add.at(up_locks, As.indices, ((a > 0) & fin_hi) | ((a < 0) & fin_lo))
        cl, lbl, ubl = sub.c, lb[live_c], ub[live_c]
        to_lb = (down_locks == 0) & (cl >= 0) & np.isfinite(lbl) & (lbl < ubl)
        to_ub = (up_locks == 0) & (cl <= 0) & np.isfinite(ubl) & (lbl < ubl) & ~to_lb
        bad = ((down_locks == 0) & (cl > 0) & ~np.isfinite(lbl)) | ((up_locks == 0) & (cl < 0) & ~np.isfinite(ubl))
        if np.any(bad):
            st.notes.append("a column's cost is unbounded along a lock-free direction")
            return finish("unbounded")
        if np.any(to_lb | to_ub):
            ub[live_c[to_lb]] = lb[live_c[to_lb]]
            lb[live_c[to_ub]] = ub[live_c[to_ub]]
            st.dual_fixed += int(to_lb.sum() + to_ub.sum())
            changed = True
        # --- 6. coefficient tightening (one-sided rows, binaries)
        if tighten_coefficients and not changed:
            nct = _tighten_coefficients(A, row_alive, col_alive, row_lo, row_hi, lb, ub, integer)
            if nct:
                st.coef_tightened += nct
                changed = True
        if not changed:
            break
    return finish("reduced")


def _tighten_coefficients(A, row_alive, col_alive, row_lo, row_hi, lb, ub, integer) -> int:
    """Savelsbergh's coefficient tightening, in place on A and the row
    bounds, for one-sided rows; returns the number of coefficients changed."""
    binary = integer & (lb == 0) & (ub == 1)
    count = 0
    indptr, idx, data = A.indptr, A.indices, A.data
    for i in np.nonzero(row_alive)[0]:
        lo, hi = row_lo[i], row_hi[i]
        if np.isfinite(lo) == np.isfinite(hi):
            continue
        sgn = 1.0 if np.isfinite(hi) else -1.0     # work on sgn*a x <= sgn*bound
        b = hi if sgn > 0 else -lo
        s, e = indptr[i], indptr[i + 1]
        cols = idx[s:e]
        alive = col_alive[cols]
        if not np.any(alive):
            continue
        a = sgn * data[s:e]
        lbc, ubc = lb[cols], ub[cols]
        cont = np.where(a > 0, a * ubc, a * lbc)
        cont = np.where(alive, cont, 0.0)
        if not np.all(np.isfinite(cont)):
            continue
        maxact = float(cont.sum())
        changed_row = False
        for k in range(e - s):
            j = cols[k]
            if not (alive[k] and binary[j] and a[k] > 0):
                continue
            rest = maxact - a[k]
            if rest < b - 1e-9:
                d = b - rest
                if d < a[k] - 1e-12 and d > 1e-9:
                    a[k] -= d
                    b -= d
                    maxact -= d
                    changed_row = True
                    count += 1
        if changed_row:
            data[s:e] = sgn * a
            if sgn > 0:
                row_hi[i] = b
            else:
                row_lo[i] = -b
    return count
