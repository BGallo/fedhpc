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

7. Parallel rows (a_r = lambda a_s): one row keeps the intersection of
   their bounds (rows hashed after normalisation, then compared exactly).
8. Doubleton equations a_i x_i + a_j x_j = b: x_j = (b - a_i x_i) / a_j is
   substituted everywhere and the row leaves; x_j's bounds and cost move to
   x_i. The eliminated column is continuous, or integer with a_i / a_j and
   b / a_j integral and x_i integer (integrality is then implied). A batch
   of disjoint pairs is applied as one product A T (Achterberg, Bixby, Gu,
   Rothberg & Weninger 2020, "Presolve reductions in MIP").
9. Implied-free continuous column singletons in equality rows: x_j's
   bounds follow from the row and the others' bounds, so x_j and its row
   leave; its cost moves onto the row's other columns.
10. Probing on binaries (Savelsbergh 1994), on the C++ propagator's trail
    within a work budget: a value that empties the domain fixes the other;
    bounds both values imply hold globally.

Reductions 1-7 and 10 fix, tighten or delete, and postsolve scatters the
kept and fixed values back; 8 and 9 push affine records
x_j = const + sum coef_k x_k onto the postsolve stack, replayed in reverse.
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
    # substitutions in the order applied: x_j = const + coef' x[cols]
    stack: list = field(default_factory=list)

    def expand(self, x_reduced: np.ndarray) -> np.ndarray:
        x = self.fixed_value.copy()
        x[self.kept_cols] = x_reduced
        for j, cols, coef, const in reversed(self.stack):
            x[j] = const + float(np.dot(coef, x[cols]))
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
    parallel_rows: int = 0
    doubletons: int = 0
    free_singletons: int = 0
    probed: int = 0
    probe_fixed: int = 0
    probe_tightened: int = 0
    seconds: float = 0.0
    status: str = "reduced"          # or "infeasible", "unbounded"
    notes: list = field(default_factory=list)


def presolve(P: MipProblem, max_rounds: int = 20, feas_tol: float = 1e-9,
             tighten_coefficients: bool = True, parallel_rows: bool = True, doubletons: bool = True,
             free_singletons: bool = True, probing: bool = True, probe_budget: int = 2_000_000,
             max_ratio: float = 1e3) -> tuple[MipProblem | None, Postsolve, PresolveStats]:
    """The reductions of the module notes; each one can be switched off.
    ``probe_budget``: propagator entry visits for probing; ``max_ratio``:
    largest |a_i / a_j| a doubleton substitution accepts."""
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

    stack: list = []
    probed_once = False

    def finish(status: str):
        st.status = status
        st.seconds = time.perf_counter() - t0
        post = Postsolve(n, np.nonzero(col_alive)[0], fixed_value, np.nonzero(row_alive)[0], stack)
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
        # --- 7. parallel rows
        if parallel_rows and not changed:
            k = _parallel_rows(Acur, row_alive, row_lo, row_hi)
            if k < 0:
                return finish("infeasible")
            if k:
                st.parallel_rows += k
                changed = True
        # --- 8. doubleton equations
        if doubletons and not changed:
            A, offset, k = _doubletons(A, Acur, row_alive, col_alive, row_lo, row_hi, lb, ub, c, integer,
                                       offset, stack, max_ratio)
            if k < 0:
                return finish("infeasible")
            if k:
                st.doubletons += k
                changed = True
        # --- 9. implied-free continuous column singletons in equalities
        if free_singletons and not changed:
            A, offset, k = _free_singletons(A, Acur, row_alive, col_alive, row_lo, row_hi, lb, ub, c, integer,
                                            offset, stack)
            if k:
                st.free_singletons += k
                changed = True
        # --- 10. probing on binaries, once the rest is done
        if probing and not changed and not probed_once:
            probed_once = True
            r = _probe(Acur, row_alive, col_alive, row_lo, row_hi, lb, ub, integer, probe_budget, st)
            if r < 0:
                return finish("infeasible")
            if r:
                changed = True
        if not changed:
            break
    return finish("reduced")


# ----------------------------------------------------------------- rows
def _parallel_rows(Acur, row_alive, row_lo, row_hi) -> int:
    """Rows equal up to a nonzero factor keep one copy with the intersected
    bounds (in that copy's scale); -1 when the intersection is empty."""
    rows = np.nonzero(row_alive & (np.diff(Acur.indptr) > 0))[0]
    if len(rows) < 2:
        return 0
    A = Acur
    rn = np.diff(A.indptr)
    rr = np.repeat(np.arange(A.shape[0]), rn)
    first = A.data[A.indptr[:-1].clip(max=max(len(A.data) - 1, 0))] if len(A.data) else np.zeros(A.shape[0])
    scale = np.where(rn > 0, first, 1.0)
    v = A.data / scale[rr]
    # a hash of (columns, normalised coefficients) per row, int64 arithmetic
    q = np.round(v * 1e6).astype(np.int64)
    h_e = (A.indices.astype(np.int64) * 1000003 + q * 7919) & ((1 << 40) - 1)
    h = np.zeros(A.shape[0], np.int64)
    np.add.at(h, rr, h_e)
    key = h[rows] * 131 + rn[rows]
    order = np.argsort(key, kind="stable")
    ks = key[order]
    same = np.nonzero(ks[1:] == ks[:-1])[0]
    if len(same) == 0:
        return 0
    removed = 0
    keep_of: dict[int, int] = {}
    for t in same.tolist():
        r1, r2 = int(rows[order[t]]), int(rows[order[t + 1]])
        r1 = keep_of.get(r1, r1)
        if not (row_alive[r1] and row_alive[r2]) or r1 == r2:
            continue
        a1, b1 = A.indptr[r1], A.indptr[r1 + 1]
        a2, b2 = A.indptr[r2], A.indptr[r2 + 1]
        if not np.array_equal(A.indices[a1:b1], A.indices[a2:b2]):
            continue
        lam = A.data[a2] / A.data[a1]                 # row2 = lam row1
        if not np.allclose(A.data[a2:b2], lam * A.data[a1:b1], rtol=1e-12, atol=1e-14):
            continue
        lo2, hi2 = row_lo[r2] / lam, row_hi[r2] / lam
        if lam < 0:
            lo2, hi2 = hi2, lo2
        lo, hi = max(row_lo[r1], lo2), min(row_hi[r1], hi2)
        if lo > hi + 1e-9 * (1 + abs(hi)):
            return -1
        row_lo[r1], row_hi[r1] = lo, max(hi, lo)
        row_alive[r2] = False
        keep_of[r2] = r1
        removed += 1
    return removed


# ----------------------------------------------------------- substitution
def _substitute(A, sub_cols, keep_cols, coef, const, row_lo, row_hi, c, offset):
    """x_e = const_e + sum coef x_keep for every e in sub_cols (batched as
    x = T x' + t): returns A T and the new offset; row bounds and costs move
    in place. ``keep_cols``/``coef``: per substitution, arrays."""
    n = A.shape[1]
    ri, ci, vi = [], [], []
    e_set = np.zeros(n, bool)
    e_set[sub_cols] = True
    keep = np.nonzero(~e_set)[0]
    ri.append(keep), ci.append(keep), vi.append(np.ones(len(keep)))
    t = np.zeros(n)
    for e, ks, cf, k0 in zip(sub_cols, keep_cols, coef, const):
        ri.append(np.full(len(ks), e)), ci.append(np.asarray(ks)), vi.append(np.asarray(cf, float))
        t[e] = k0
    T = sp.csr_matrix((np.concatenate(vi), (np.concatenate(ri), np.concatenate(ci))), shape=(n, n))
    shift = A @ t
    row_lo -= shift
    row_hi -= shift
    offset += float(c @ t)
    c_new = T.T @ c
    c[:] = c_new
    A2 = (A @ T).tocsr()
    small = np.abs(A2.data) < 1e-12
    if np.any(small):
        A2.data[small] = 0.0
        A2.eliminate_zeros()
    return A2, offset


def _doubletons(A, Acur, row_alive, col_alive, row_lo, row_hi, lb, ub, c, integer, offset, stack, max_ratio):
    rn = np.diff(Acur.indptr)
    eq = row_alive & (rn == 2) & (row_lo == row_hi) & np.isfinite(row_hi)
    rows = np.nonzero(eq)[0]
    if len(rows) == 0:
        return A, offset, 0
    cnnz = np.bincount(Acur.indices, minlength=A.shape[1])
    used = np.zeros(A.shape[1], bool)
    subs, keeps, coefs, consts, drop = [], [], [], [], []
    for i in rows.tolist():
        s0 = Acur.indptr[i]
        (j, k), (aj, ak) = Acur.indices[s0:s0 + 2], Acur.data[s0:s0 + 2]
        b = row_hi[i]
        if used[j] or used[k]:
            continue
        pick = None
        for e, q, ae, aq in ((j, k, aj, ak), (k, j, ak, aj)):
            r, sft = -aq / ae, b / ae
            if not (1.0 / max_ratio <= abs(r) <= max_ratio) or cnnz[e] > 200:
                continue
            if integer[e]:
                if not (integer[q] and abs(r - round(r)) < 1e-9 and abs(sft - round(sft)) < 1e-9):
                    continue
            if pick is None or (integer[pick[0]] and not integer[e]):
                pick = (e, q, r, sft)
        if pick is None:
            continue
        e, q, r, sft = pick
        # x_e = r x_q + sft in [lb_e, ub_e]  ->  bounds on x_q
        lo_q, hi_q = (lb[e] - sft) / r, (ub[e] - sft) / r
        if r < 0:
            lo_q, hi_q = hi_q, lo_q
        if integer[q]:
            lo_q = np.ceil(lo_q - 1e-9) if np.isfinite(lo_q) else lo_q
            hi_q = np.floor(hi_q + 1e-9) if np.isfinite(hi_q) else hi_q
        nl, nu = max(lb[q], lo_q), min(ub[q], hi_q)
        if nl > nu + 1e-9 * (1 + abs(nu)):
            return A, offset, -1
        lb[q], ub[q] = nl, max(nu, nl)
        used[e] = used[q] = True
        subs.append(e), keeps.append([q]), coefs.append([r]), consts.append(sft), drop.append(i)
    if not subs:
        return A, offset, 0
    A, offset = _substitute(A, subs, keeps, coefs, consts, row_lo, row_hi, c, offset)
    for e, ks, cf, k0 in zip(subs, keeps, coefs, consts):
        stack.append((e, np.asarray(ks), np.asarray(cf, float), float(k0)))
        col_alive[e] = False
    row_alive[drop] = False
    return A, offset, len(subs)


def _free_singletons(A, Acur, row_alive, col_alive, row_lo, row_hi, lb, ub, c, integer, offset, stack):
    C = Acur.tocsc()
    cn = np.diff(C.indptr)
    cand = np.nonzero(col_alive & ~integer & (cn == 1))[0]
    if len(cand) == 0:
        return A, offset, 0
    used_row = np.zeros(A.shape[0], bool)
    subs, keeps, coefs, consts, drop = [], [], [], [], []
    for j in cand.tolist():
        i = int(C.indices[C.indptr[j]])
        aj = C.data[C.indptr[j]]
        if used_row[i] or not (row_alive[i] and row_lo[i] == row_hi[i] and np.isfinite(row_hi[i])):
            continue
        s0, s1 = Acur.indptr[i], Acur.indptr[i + 1]
        cols, a = Acur.indices[s0:s1], Acur.data[s0:s1]
        other = cols != j
        oc, oa = cols[other], a[other]
        if len(oc) == 0 or np.any(np.isin(oc, subs)):
            continue
        # the rest's activity range, then x_j's implied range
        mn = np.where(oa > 0, oa * lb[oc], oa * ub[oc]).sum()
        mx = np.where(oa > 0, oa * ub[oc], oa * lb[oc]).sum()
        if not (np.isfinite(mn) and np.isfinite(mx)):
            continue
        b = row_hi[i]
        lo_j, hi_j = (b - mx) / aj, (b - mn) / aj
        if aj < 0:
            lo_j, hi_j = hi_j, lo_j
        tol = 1e-9
        if lo_j < lb[j] - tol * (1 + abs(lb[j])) or hi_j > ub[j] + tol * (1 + abs(ub[j])):
            continue                                    # not implied free
        used_row[i] = True
        subs.append(j), keeps.append(oc), coefs.append(-oa / aj), consts.append(b / aj), drop.append(i)
    if not subs:
        return A, offset, 0
    A, offset = _substitute(A, subs, keeps, coefs, consts, row_lo, row_hi, c, offset)
    for e, ks, cf, k0 in zip(subs, keeps, coefs, consts):
        stack.append((e, np.asarray(ks), np.asarray(cf, float), float(k0)))
        col_alive[e] = False
    row_alive[drop] = False
    return A, offset, len(subs)


# ---------------------------------------------------------------- probing
def _probe(Acur, row_alive, col_alive, row_lo, row_hi, lb, ub, integer, budget, st) -> int:
    """Probe the open binaries (most rows first) until ``budget`` entry
    visits; returns the number of bound changes, -1 when infeasible."""
    live_r, live_c = np.nonzero(row_alive)[0], np.nonzero(col_alive)[0]
    if len(live_c) == 0:
        return 0
    sub = MipProblem(Acur[live_r][:, live_c].tocsr(), row_lo[live_r], row_hi[live_r],
                     lb[live_c], ub[live_c], np.zeros(len(live_c)), integer[live_c])
    prop = Propagator(sub, feas_tol=1e-7)
    if not hasattr(prop, "changed_since"):
        return 0
    if not prop.reset(sub.lb, sub.ub, propagate=True):
        return -1
    l, u = prop.lb(), prop.ub()
    binary = sub.integer & (l == 0) & (u == 1)
    cn = np.diff(sub.csc.indptr)
    order = np.nonzero(binary)[0]
    order = order[np.argsort(-cn[order], kind="stable")]
    changes, work = 0, 0
    for j in order.tolist():
        if work > budget:
            break
        lj, uj = prop.k.bound(j)
        if lj == uj:
            continue
        st.probed += 1
        lev = prop.level()
        ok0 = prop.change([j], [0.0], [0.0], max_rounds=20)
        work += prop.k.work
        c0 = prop.changed_since(lev) if ok0 else None
        prop.undo(lev)
        ok1 = prop.change([j], [1.0], [1.0], max_rounds=20)
        work += prop.k.work
        c1 = prop.changed_since(lev) if ok1 else None
        prop.undo(lev)
        if not ok0 and not ok1:
            return -1
        if not ok0 or not ok1:
            v = 1.0 if not ok0 else 0.0
            if not prop.change([j], [v], [v], max_rounds=20):
                return -1
            st.probe_fixed += 1
            changes += 1
            continue
        # bounds implied by both values
        j0, l0, u0 = c0
        j1, l1, u1 = c1
        common, i0, i1 = np.intersect1d(j0, j1, assume_unique=True, return_indices=True)
        if len(common):
            nl = np.minimum(l0[i0], l1[i1])
            nu = np.maximum(u0[i0], u1[i1])
            cur = np.array([prop.k.bound(int(k)) for k in common])
            tight = (nl > cur[:, 0] + 1e-9) | (nu < cur[:, 1] - 1e-9)
            if np.any(tight):
                if not prop.change(common[tight], nl[tight], nu[tight], max_rounds=20):
                    return -1
                st.probe_tightened += int(tight.sum())
                changes += int(tight.sum())
    if changes:
        nl, nu = prop.lb(), prop.ub()
        lb[live_c] = np.maximum(lb[live_c], nl)
        ub[live_c] = np.minimum(ub[live_c], nu)
    return changes


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
            if not (alive[k] and binary[j]):
                continue
            if a[k] > 0:
                rest = maxact - a[k]
                if rest < b - 1e-9:
                    d = b - rest
                    if d < a[k] - 1e-12 and d > 1e-9:
                        a[k] -= d
                        b -= d
                        maxact -= d
                        changed_row = True
                        count += 1
            elif a[k] < 0:
                # complemented: y' = 1 - y has coefficient -a_k > 0 and the
                # row reads ... + (-a_k) y' <= b - a_k; the same reduction
                # d = (b - a_k) - (maxact' - (-a_k)) with maxact' = maxact - a_k
                # (y at 0 gives the largest activity), so a big-M link
                # x - M y <= 0 with x <= U becomes x - U y <= 0
                ak = -a[k]
                bp = b + ak
                rest = maxact           # maxact' - ak, as y' = 1 (y = 0) contributes ak
                if rest < bp - 1e-9:
                    d = bp - rest
                    if d < ak - 1e-12 and d > 1e-9:
                        a[k] += d       # -a_k shrinks by d
                        changed_row = True
                        count += 1
        if changed_row:
            data[s:e] = sgn * a
            if sgn > 0:
                row_hi[i] = b
            else:
                row_lo[i] = -b
    return count
