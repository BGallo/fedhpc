"""Cutting planes: separators and the pool that selects among them.

A cut is ``Cut(cols, coef, rhs)`` meaning  sum_k coef_k x_{cols_k} >= rhs,
over the problem's columns. Every separator returns valid inequalities for
the problem's integer hull (not only for the current node).

Separators
----------
Gomory mixed-integer (Gomory 1960; Balas, Ceria, Cornuejols & Natraj 1996;
the formula as in Cornuejols 2008). From a tableau row
x_B + sum_k a_k v_k = beta, with v_k >= 0 the nonbasic variables shifted to
their current bound, x_B integer and f0 = frac(beta):

    sum_{int, f_k <= f0} f_k/f0 v_k + sum_{int, f_k > f0} (1-f_k)/(1-f0) v_k
      + sum_{cont, a_k > 0} a_k/f0 v_k + sum_{cont, a_k < 0} -a_k/(1-f0) v_k >= 1

f_k = frac(a_k). Row logicals are treated as continuous (valid, weaker
than using integral slacks). v_k is substituted back: for columns through
their bound, for logicals through s_r = sigma_r (rhs_r - a_r'x).

Knapsack covers (Balas 1975; Crowder, Johnson & Padberg 1983). For a row
sum a_j x_j <= b over binaries (negative coefficients complemented), a
cover C (sum_C a_j > b) gives sum_C x_j <= |C| - 1, extended by the items
at least as heavy as C's heaviest. C is chosen greedily by (1 - x*_j)/a_j.

Numerical safety
----------------
Cuts with dynamism max|coef| / min|coef| above ``max_dynamism`` are
dropped; coefficients below ``tiny`` relative to the largest are removed by
relaxing the right-hand side through the variable's bounds; the right-hand
side is relaxed by ``rhs_relax`` (relative). Efficacy is the Euclidean
distance from the point to the cut's hyperplane.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .lp import LpRelaxation
from .problem import INF, MipProblem


@dataclass
class Cut:
    cols: np.ndarray
    coef: np.ndarray
    rhs: float
    origin: str = ""

    def violation(self, x: np.ndarray) -> float:
        return float(self.rhs - self.coef @ x[self.cols])

    def efficacy(self, x: np.ndarray) -> float:
        nrm = float(np.linalg.norm(self.coef))
        return self.violation(x) / nrm if nrm > 0 else 0.0


@dataclass
class CutParams:
    max_dynamism: float = 1e6
    tiny: float = 1e-9
    rhs_relax: float = 1e-9
    min_efficacy: float = 1e-4
    max_parallelism: float = 0.99
    gomory_rows: int = 100
    gomory_min_frac: float = 0.01
    gomory_max_support: float = 0.5       # fraction of columns


def _finish(lb: np.ndarray, ub: np.ndarray, cols: np.ndarray, coef: np.ndarray, rhs: float,
            par: CutParams, origin: str) -> Cut | None:
    """Clean a raw cut coef'x >= rhs: tiny coefficients out (rhs relaxed
    through bounds), dynamism check, rhs relaxation."""
    if len(cols) == 0:
        return None
    amax = float(np.abs(coef).max())
    if amax == 0 or not np.isfinite(amax) or not np.isfinite(rhs):
        return None
    small = np.abs(coef) < par.tiny * amax
    if np.any(small):
        # coef_j x_j >= -max over the domain of (-coef_j x_j)... : drop
        # term t = coef_j x_j, then rhs -= max(t) over [lb, ub]
        cj, aj = cols[small], coef[small]
        tmax = np.maximum(aj * lb[cj], aj * ub[cj])
        if not np.all(np.isfinite(tmax)):
            return None
        rhs -= float(tmax.sum())
        cols, coef = cols[~small], coef[~small]
        if len(cols) == 0:
            return None
    amin = float(np.abs(coef).min())
    if amax / amin > par.max_dynamism:
        return None
    rhs -= par.rhs_relax * max(1.0, abs(rhs))
    scale = 1.0 / amax
    return Cut(cols, coef * scale, rhs * scale, origin)


# ---------------------------------------------------------------- Gomory
def gomory_cuts(lp: LpRelaxation, P: MipProblem, x: np.ndarray, par: CutParams,
                impint: np.ndarray | None = None) -> list[Cut]:
    """GMI cuts from the tableau rows of fractional integer basics;
    ``impint`` (implied integers) strengthens the nonbasic coefficients."""
    heads = lp.basic_heads()
    cand = []
    for pos, h in enumerate(heads):
        if h < 0 or h >= P.n or not P.integer[h]:
            continue
        f = x[h] - np.floor(x[h])
        if par.gomory_min_frac <= f <= 1 - par.gomory_min_frac:
            cand.append((abs(f - 0.5), pos))
    cand.sort()
    out = []
    if not cand:
        return out
    dense = np.zeros(P.n)
    ctx = _GmiContext(lp, P)
    ctx.integer = P.integer if impint is None else impint
    for _score, pos in cand[: par.gomory_rows]:
        row = lp.tableau_row(pos, heads, ctx.basis, ctx.xlp)
        if row is None:
            continue
        cut = _gmi_from_row(lp, P, row, dense, par, ctx)
        if cut is not None and cut.efficacy(x) >= par.min_efficacy:
            out.append(cut)
    return out


class _GmiContext:
    """What every tableau row of one separation round shares: the basis,
    the LP point, the rows' logical data and their matrix."""

    def __init__(self, lp: LpRelaxation, P: MipProblem):
        self.basis = lp.get_basis()
        self.xlp = np.asarray(lp.lp.x())
        self.sigma, self.rhs, self.s_lo, self.s_hi = lp.logical_arrays()
        self.m0 = lp.n_base_rows
        self.A = P.A
        self.integer = P.integer


def _gmi_from_row(lp: LpRelaxation, P: MipProblem, row, dense: np.ndarray, par: CutParams,
                  ctx: _GmiContext | None = None) -> Cut | None:
    """The GMI cut of one tableau row, vectorised. Nonbasic columns sit at
    the LP's installed bounds (lp.lb, lp.ub), which the cut is valid for:
    global at the root. Logicals are substituted through
    s_r = sigma_r (rhs_r - a_r'x), a sparse product with their rows."""
    ctx = ctx or _GmiContext(lp, P)
    lb, ub = lp.lb, lp.ub
    beta = row.value
    f0 = beta - np.floor(beta)
    if f0 < par.gomory_min_frac or f0 > 1 - par.gomory_min_frac:
        return None
    # complemented coefficients: v = bound + v' (lower) or bound - v' (upper)
    a = np.where(row.at_upper, -row.coef, row.coef)
    if np.any(np.abs(a) > 1e8):
        return None
    is_col = row.kind == 0
    is_int = np.zeros(len(a), bool)
    is_int[is_col] = ctx.integer[row.index[is_col]]
    fk = a - np.floor(a)
    g = np.where(is_int,
                 np.where(fk <= f0, fk / f0, (1 - fk) / (1 - f0)),
                 np.where(a >= 0, a / f0, -a / (1 - f0)))
    nz = g > 1e-12
    cmask, lmask = nz & is_col, nz & ~is_col
    jc, gc, uc = row.index[cmask], g[cmask], row.at_upper[cmask]
    if len(jc) > par.gomory_max_support * P.n + 10:
        return None                           # too dense before the logicals add more
    # sum g_k v'_k >= 1  ->  pi'x >= 1 - const
    if np.any(uc & ~np.isfinite(ub[jc])) or np.any(~uc & ~np.isfinite(lb[jc])):
        return None
    const = float(np.sum(np.where(uc, gc * np.where(uc, ub[jc], 0.0), -gc * np.where(uc, 0.0, lb[jc]))))
    dense[jc] += np.where(uc, -gc, gc)          # a tableau row names each column once
    rr, gl, ul = row.index[lmask], g[lmask], row.at_upper[lmask]
    if len(rr):
        sig, rhs, slo, shi = ctx.sigma[rr], ctx.rhs[rr], ctx.s_lo[rr], ctx.s_hi[rr]
        if np.any(ul & ~np.isfinite(shi)):
            dense[jc] = 0.0
            return None
        # upper: v' = s_hi - s = s_hi - sigma rhs + sigma a'x; lower: v' = sigma rhs - sigma a'x - s_lo
        w = np.where(ul, gl * sig, -gl * sig)
        const += float(np.sum(np.where(ul, gl * (np.where(ul, shi, 0.0) - sig * rhs), gl * (sig * rhs - slo))))
        base = rr < ctx.m0
        if np.any(base):
            sub = ctx.A[rr[base]]
            dense += sub.T @ w[base]
        for r, wr in zip(rr[~base].tolist(), w[~base].tolist()):
            cc, cf = lp.cut_coefs[r]
            np.add.at(dense, cc, wr * cf)
    cols = np.flatnonzero(dense)
    coef = dense[cols].copy()
    dense[cols] = 0.0
    if len(cols) == 0 or len(cols) > par.gomory_max_support * P.n + 10:
        return None
    return _finish(lb, ub, cols, coef, 1.0 - const, par, "gomory")


# ---------------------------------------------------------- knapsack covers
def cover_cuts(P: MipProblem, x: np.ndarray, lb: np.ndarray, ub: np.ndarray, par: CutParams,
               max_rows: int = 2000) -> list[Cut]:
    """Extended cover cuts from one-sided rows over binaries. Columns fixed
    by lb == ub move to the right-hand side, so the cuts are valid for the
    domain [lb, ub]: pass root bounds for global cuts."""
    A = P.A
    binary = P.integer & (lb >= 0) & (ub <= 1)
    out = []
    frac_cols = np.abs(x - np.round(x)) > 1e-6
    if not np.any(frac_cols & binary):
        return out
    rows = np.unique(P.csc[:, np.nonzero(frac_cols & binary)[0]].indices)
    for i in rows[:max_rows]:
        lo, hi = P.row_lo[i], P.row_hi[i]
        for sgn, b in ((1.0, hi), (-1.0, -lo)):
            if not np.isfinite(b):
                continue
            s, e = A.indptr[i], A.indptr[i + 1]
            cols, a = A.indices[s:e], sgn * A.data[s:e]
            fixed = lb[cols] == ub[cols]
            if not np.all(binary[cols] | fixed):
                continue
            bb = b - float(a[fixed] @ lb[cols[fixed]])
            cols, a = cols[~fixed], a[~fixed]
            if len(cols) < 2:
                continue
            neg = a < 0
            xs = np.where(neg, 1 - x[cols], x[cols])     # complemented values
            w = np.abs(a)
            bb += float(-a[neg].sum())                   # b' = b - sum_{a<0} a
            if bb < 0 or w.sum() <= bb + 1e-9:
                continue
            order = np.argsort((1 - xs) / w, kind="stable")
            tot, C = 0.0, []
            for k in order:
                C.append(k)
                tot += w[k]
                if tot > bb + 1e-9:
                    break
            if tot <= bb + 1e-9:
                continue
            C = np.array(C)
            if float((1 - xs[C]).sum()) >= 1 - 1e-6:
                continue                                  # not violated
            amax = w[C].max()
            ext = np.setdiff1d(np.nonzero(w >= amax)[0], C)
            S = np.concatenate([C, ext])
            # sum_S xs_k <= |C| - 1, xs = x or 1 - x: as  -sum ... >= 1 - |C|
            coef = np.where(neg[S], 1.0, -1.0)
            rhs = 1.0 - len(C) + float(neg[S].sum())
            cut = _finish(lb, ub, cols[S], coef, rhs, par, "cover")
            if cut is not None and cut.efficacy(x) >= par.min_efficacy:
                out.append(cut)
    return out


# ---------------------------------------------------------------- c-MIR
def _mir(a: np.ndarray, b: float, isint: np.ndarray, xs: np.ndarray, delta: float, min_frac: float):
    """MIR of  sum a_k v_k <= b  (v >= 0; integers where isint, continuous
    otherwise) divided by delta: coefficients g and right-hand side beta of
    sum g_k v_k <= beta, or None when frac(b / delta) is too close to 0/1.
    Continuous terms with a > 0 relax away; those with a < 0 make
    s = sum |a| v and enter as -s / (delta (1 - f0))."""
    bd = b / delta
    f0 = bd - np.floor(bd)
    if f0 < min_frac or f0 > 1 - min_frac:
        return None
    ad = a / delta
    fl = np.floor(ad)
    fj = ad - fl
    g = np.where(isint, fl + np.maximum(fj - f0, 0.0) / (1 - f0),
                 np.where(a < 0, a / (delta * (1 - f0)), 0.0))
    return g, float(np.floor(bd))


def cmir_cuts(P: MipProblem, x: np.ndarray, lb: np.ndarray, ub: np.ndarray, par: CutParams,
              max_rows: int = 300, max_aggr: int = 5, deadline: float = np.inf,
              vb: VarBounds | None = None) -> list[Cut]:
    """Complemented mixed-integer rounding cuts (Marchand & Wolsey 2001),
    the separator SCIP's and Gurobi's flow-cover and MIR cuts share.

    From each tight row (side) holding fractional integers, a path of
    aggregations: while the aggregated row has a continuous column strictly
    inside its bounds, the tight row through the farthest-inside one with
    the fewest entries is added with the multiplier that cancels it (any
    sign for an equality, the sign that keeps a <= row valid otherwise),
    up to ``max_aggr`` rows. After each step: bound substitution (every
    column to its nearest finite bound, integers complemented when nearer
    the upper one), then MIR with the divisors delta taken from the
    integers strictly inside their bounds (their |a|, up to 8, and 1), the
    best one also halved up to three times; the most efficacious cut of the
    path is kept. Bounds are the given (root) ones: the cuts are global."""
    A, C = P.A, P.csc
    m, n = P.m, P.n
    act = A @ x
    tol_r = 1e-6 * (1 + np.abs(np.where(np.isfinite(P.row_hi), P.row_hi, 0.0)))
    tight_hi = np.isfinite(P.row_hi) & (P.row_hi - act <= 1e-6 * (1 + np.abs(np.nan_to_num(P.row_hi, posinf=0))))
    tight_lo = np.isfinite(P.row_lo) & (act - P.row_lo <= 1e-6 * (1 + np.abs(np.nan_to_num(P.row_lo, neginf=0))))
    del tol_r
    fracint = P.integer & (np.abs(x - np.round(x)) > 1e-6)
    if not np.any(fracint):
        return []
    # base rows: tight rows holding a fractional integer, or a continuous
    # column whose variable bound is on one (fixed-charge flow rows hold
    # only flows; their binaries enter through the substitution)
    seed = fracint.copy()
    if vb is not None:
        for kk in (vb.vlb_k, vb.vub_k):
            has = kk >= 0
            seed |= has & fracint[np.maximum(kk, 0)]
    rows = np.unique(C[:, np.nonzero(seed)[0]].indices)
    rows = rows[tight_hi[rows] | tight_lo[rows]]
    if len(rows) == 0:
        return []
    # variable-bound rows (two entries, a continuous and an integer) are
    # substitutions, not bases; then fewest entries first: short rows give
    # sparse cuts
    if vb is not None:
        rn2 = np.diff(A.indptr)[rows] == 2
        if np.any(rn2):
            s0 = A.indptr[rows[rn2]]
            c0, c1 = A.indices[s0], A.indices[s0 + 1]
            vbrow = P.integer[c0] != P.integer[c1]
            drop = np.zeros(len(rows), bool)
            drop[np.nonzero(rn2)[0][vbrow]] = True
            rows = rows[~drop]
    rows = rows[np.argsort(np.diff(A.indptr)[rows], kind="stable")][:max_rows]
    rn = np.diff(A.indptr)
    # a continuous column is aggregated away when it sits strictly inside
    # every bound it could be substituted by: simple and variable ones
    with np.errstate(invalid="ignore"):
        dist = np.minimum(np.where(np.isfinite(lb), x - lb, np.inf), np.where(np.isfinite(ub), ub - x, np.inf))
        if vb is not None:
            kl, ku = vb.vlb_k, vb.vub_k
            dl = np.where(kl >= 0, x - (vb.vlb_d * x[np.maximum(kl, 0)] + vb.vlb_e), np.inf)
            du = np.where(ku >= 0, vb.vub_d * x[np.maximum(ku, 0)] + vb.vub_e - x, np.inf)
            dist = np.minimum(dist, np.minimum(dl, du))
    inside = ~P.integer & (dist > 1e-6) & np.isfinite(dist)
    dist = np.where(inside, dist, 0.0)
    dense = np.zeros(n)
    out = []

    def side_rows(i):
        a, b = A.indices[A.indptr[i]:A.indptr[i + 1]], A.data[A.indptr[i]:A.indptr[i + 1]]
        res = []
        if tight_hi[i]:
            res.append((a, b, float(P.row_hi[i]), P.row_lo[i] == P.row_hi[i]))
        if tight_lo[i] and P.row_lo[i] != P.row_hi[i]:
            res.append((a, -b, -float(P.row_lo[i]), False))
        return res

    for i in rows:
        if time.perf_counter() > deadline:
            break
        for cols0, a0, b0, _eq in side_rows(i):
            used = {int(i)}
            dense[cols0] += a0
            touched = set(cols0.tolist())
            rhs = b0
            best = None
            for _step in range(max_aggr + 1):
                cols = np.fromiter(touched, dtype=np.int64)
                coef = dense[cols]
                nzm = np.abs(coef) > 1e-12
                cols, coef = cols[nzm], coef[nzm]
                cut = _cmir_from_row(cols, coef, rhs, P, x, lb, ub, par, vb)
                if cut is not None and (best is None or cut[0] > best[0]):
                    best = cut
                # aggregate along the continuous column farthest inside
                cand = cols[inside[cols]]
                if len(cand) == 0 or _step == max_aggr:
                    break
                j = int(cand[np.argmax(dist[cand])])
                aj = dense[j]
                rr = C.indices[C.indptr[j]:C.indptr[j + 1]]
                vv = C.data[C.indptr[j]:C.indptr[j + 1]]
                pick = None
                for r, v in sorted(zip(rr.tolist(), vv.tolist()), key=lambda t: rn[t[0]]):
                    if r in used:
                        continue
                    if P.row_lo[r] == P.row_hi[r] and tight_hi[r]:
                        pick = (r, -aj / v, float(P.row_hi[r]))
                        break
                    lam = -aj / v                     # on the <= form a_r x <= hi
                    if lam > 0 and tight_hi[r]:
                        pick = (r, lam, float(P.row_hi[r]))
                        break
                    if lam < 0 and tight_lo[r]:     # -a_r x <= -lo with multiplier -lam
                        pick = (r, lam, float(P.row_lo[r]))
                        break
                if pick is None:
                    break
                r, lam, bound = pick
                ar = A.indices[A.indptr[r]:A.indptr[r + 1]]
                dense[ar] += lam * A.data[A.indptr[r]:A.indptr[r + 1]]
                dense[j] = 0.0                        # exact cancellation
                rhs += lam * bound
                touched.update(ar.tolist())
                used.add(r)
            dense[np.fromiter(touched, dtype=np.int64)] = 0.0
            if best is not None and best[0] >= par.min_efficacy:
                out.append(best[1])
    return out


@dataclass
class VarBounds:
    """Variable bounds x_j >= d y_k + e (vlb) and x_j <= d y_k + e (vub) of
    continuous columns, read off two-entry rows over a continuous and an
    integer column (fixed-charge links x <= u y). k = -1: none."""
    vlb_k: np.ndarray
    vlb_d: np.ndarray
    vlb_e: np.ndarray
    vub_k: np.ndarray
    vub_d: np.ndarray
    vub_e: np.ndarray


def variable_bounds(P: MipProblem) -> VarBounds:
    n = P.n
    vb = VarBounds(np.full(n, -1), np.zeros(n), np.zeros(n), np.full(n, -1), np.zeros(n), np.zeros(n))
    A = P.A
    rn = np.diff(A.indptr)
    for i in np.nonzero(rn == 2)[0].tolist():
        s0 = A.indptr[i]
        (c0, c1), (a0, a1) = A.indices[s0:s0 + 2], A.data[s0:s0 + 2]
        i0, i1 = P.integer[c0], P.integer[c1]
        if i0 == i1:
            continue
        (j, aj, k, ak) = (c1, a1, c0, a0) if i0 else (c0, a0, c1, a1)
        for side, bound in ((1, P.row_hi[i]), (-1, P.row_lo[i])):
            if not np.isfinite(bound):
                continue
            d, e = -ak / aj, bound / aj
            upper = (aj > 0) == (side > 0)            # aj x + ak y <= hi with aj > 0: x <= ...
            if upper and vb.vub_k[j] < 0:
                vb.vub_k[j], vb.vub_d[j], vb.vub_e[j] = k, d, e
            elif not upper and vb.vlb_k[j] < 0:
                vb.vlb_k[j], vb.vlb_d[j], vb.vlb_e[j] = k, d, e
    return vb


def _cmir_from_row(cols, coef, rhs, P, x, lb, ub, par, vb: VarBounds | None = None):
    """(efficacy, Cut) of the best c-MIR of  coef'x <= rhs, or None.

    Bound substitution: each continuous column goes to the closest of its
    bounds - simple, or variable (x_j = d y_k + e -+ w with y_k integer,
    whose coefficient then joins the integer part: the flow-cover
    mechanism of fixed-charge models) - leaving a slack w >= 0; integers
    are complemented to their nearer bound (z >= 0). MIR on
    sum a_z z + sum a_w w <= b: the integer coefficients round with the
    MIR function, only slacks with a_w < 0 stay (Marchand & Wolsey 2001)."""
    if len(cols) == 0:
        return None
    intc: dict[int, float] = {}
    wterms = []                         # (a_w, const, [(col, coef)]): w = const + sum coef x_col
    b = float(rhs)
    for j, a in zip(cols.tolist(), coef.tolist()):
        if P.integer[j]:
            intc[j] = intc.get(j, 0.0) + a
            continue
        xj, l, u = x[j], lb[j], ub[j]
        opts = []
        if np.isfinite(l):
            opts.append((xj - l, 1, "lb"))
        if np.isfinite(u):
            opts.append((u - xj, 1, "ub"))
        if vb is not None and vb.vlb_k[j] >= 0:
            k, d, e = int(vb.vlb_k[j]), vb.vlb_d[j], vb.vlb_e[j]
            opts.append((xj - (d * x[k] + e), 0, "vlb"))
        if vb is not None and vb.vub_k[j] >= 0:
            k, d, e = int(vb.vub_k[j]), vb.vub_d[j], vb.vub_e[j]
            opts.append((d * x[k] + e - xj, 0, "vub"))
        if not opts:
            return None
        _dist, _p, kind = min(opts, key=lambda t: (t[0] - 1e-9 * (t[1] == 0), t[1]))
        if kind == "lb":
            b -= a * l
            wterms.append((a, -l, [(j, 1.0)]))
        elif kind == "ub":
            b -= a * u
            wterms.append((-a, u, [(j, -1.0)]))
        elif kind == "vlb":
            k, d, e = int(vb.vlb_k[j]), vb.vlb_d[j], vb.vlb_e[j]
            intc[k] = intc.get(k, 0.0) + a * d
            b -= a * e
            wterms.append((a, -e, [(j, 1.0), (k, -d)]))
        else:
            k, d, e = int(vb.vub_k[j]), vb.vub_d[j], vb.vub_e[j]
            intc[k] = intc.get(k, 0.0) + a * d
            b -= a * e
            wterms.append((-a, e, [(j, -1.0), (k, d)]))
    if not intc:
        return None
    ic = np.fromiter(intc.keys(), dtype=np.int64)
    ia = np.fromiter(intc.values(), dtype=float)
    keep_i = np.abs(ia) > 1e-12
    ic, ia = ic[keep_i], ia[keep_i]
    if len(ic) == 0:
        return None
    l, u, xv = lb[ic], ub[ic], x[ic]
    use_up = np.isfinite(u) & (~np.isfinite(l) | (u - xv < xv - l))
    if np.any(~use_up & ~np.isfinite(l)):
        return None
    bnd = np.where(use_up, u, l)
    az = np.where(use_up, -ia, ia)
    b -= float(ia @ bnd)
    zs = np.where(use_up, u - xv, xv - l)
    if not np.isfinite(b):
        return None
    inner = (zs > 1e-6) & ((u - l) - zs > 1e-6)
    if not np.any(inner):
        return None
    aw = np.array([t[0] for t in wterms], float)
    a_all = np.concatenate([az, aw])
    isint_all = np.concatenate([np.ones(len(az), bool), np.zeros(len(aw), bool)])
    cand = np.unique(np.abs(az[inner]))
    cand = cand[cand > 1e-6][-8:]
    deltas = list(cand) + [1.0]
    n = P.n

    def evaluate(delta):
        r = _mir(a_all, b, isint_all, None, delta, par.gomory_min_frac)
        if r is None:
            return None
        g, beta = r
        gz, gw = g[:len(az)], g[len(az):]
        dense: dict[int, float] = {}
        bt = beta
        # z = x - l -> g x (rhs + g l); z = u - x -> -g x (rhs - g u)
        for k, gk, up, lk, uk in zip(ic.tolist(), gz.tolist(), use_up.tolist(), l.tolist(), u.tolist()):
            if gk == 0.0:
                continue
            if up:
                dense[k] = dense.get(k, 0.0) - gk
                bt -= gk * uk
            else:
                dense[k] = dense.get(k, 0.0) + gk
                bt += gk * lk
        # w = const + sum coef x
        for (aw_t, const, terms), gt in zip(wterms, gw.tolist()):
            if gt == 0.0:
                continue
            bt -= gt * const
            for cc, cf in terms:
                dense[cc] = dense.get(cc, 0.0) + gt * cf
        cc = np.fromiter(dense.keys(), dtype=np.int64)
        pi = np.fromiter(dense.values(), dtype=float)
        nz = np.abs(pi) > 1e-12
        cc, pi = cc[nz], pi[nz]
        if len(cc) == 0:
            return None
        viol = float(pi @ x[cc]) - bt
        nrm = float(np.linalg.norm(pi))
        if nrm <= 0 or viol <= 0:
            return None
        return viol / nrm, cc, pi, bt

    best = None
    for d in deltas:
        e = evaluate(d)
        if e is not None and (best is None or e[0] > best[0]):
            best = (e[0], d) + e[1:]
    if best is None:
        return None
    d0 = best[1]
    for k in (2.0, 4.0, 8.0):
        e = evaluate(d0 / k)
        if e is not None and e[0] > best[0]:
            best = (e[0], d0 / k) + e[1:]
    _eff, _d, cc, pi, bt = best
    del n
    # pi'x <= bt  as  -pi'x >= -bt
    cut = _finish(lb, ub, cc, -pi, -bt, par, "cmir")
    if cut is None:
        return None
    return cut.efficacy(x), cut


# ------------------------------------------------------- implied integers
def implied_integers(P: MipProblem, rounds: int = 5) -> np.ndarray:
    """Columns integral in every feasible point: the integers, and a
    continuous column with integral bounds that sits in an equality row
    with an integral right-hand side whose other entries are integral
    coefficients on such columns, its own coefficient +-1 (an arc-flow
    model's loss arcs: flow conservation over integral arcs). Repeated, as
    one detection can enable another. For cut separation only: branching
    never needs them."""
    A = P.A
    imp = P.integer.copy()
    with np.errstate(invalid="ignore"):
        eq = (P.row_lo == P.row_hi) & np.isfinite(P.row_hi) & (np.abs(P.row_hi - np.round(P.row_hi)) < 1e-9)
    intcoef_e = np.abs(A.data - np.round(A.data)) < 1e-9
    rr = np.repeat(np.arange(P.m), np.diff(A.indptr))
    unit = np.abs(np.abs(A.data) - 1.0) < 1e-12
    with np.errstate(invalid="ignore"):
        int_bounds = ((~np.isfinite(P.lb)) | (np.abs(P.lb - np.round(P.lb)) < 1e-9)) & \
            ((~np.isfinite(P.ub)) | (np.abs(P.ub - np.round(P.ub)) < 1e-9))
    for _ in range(rounds):
        # per row: number of entries not (integral coefficient on an implied-integer column)
        bad_e = ~(intcoef_e & imp[A.indices])
        nbad = np.bincount(rr, weights=bad_e.astype(float), minlength=P.m)
        # an equality row with exactly one bad entry, that entry unit on a
        # continuous column with integral bounds: the column is implied
        cand = eq[rr] & (nbad[rr] == 1) & bad_e & unit & ~imp[A.indices] & int_bounds[A.indices]
        new = np.unique(A.indices[cand])
        if len(new) == 0:
            break
        imp[new] = True
    return imp


# ------------------------------------------------------------- zero-half
def zerohalf_cuts(P: MipProblem, x: np.ndarray, lb: np.ndarray, ub: np.ndarray, par: CutParams,
                  impint: np.ndarray | None = None, max_rows: int = 4000, max_cols: int = 4000,
                  deadline: float = np.inf, max_work: int = 2_000_000) -> list[Cut]:
    """{0, 1/2}-Chvatal-Gomory cuts (Caprara & Fischetti 1996) by Gaussian
    elimination over GF(2) (Koster, Zymolka & Kutschka 2009).

    Rows with integral coefficients over (implied) integer columns and an
    integral right-hand side, in <= form, columns complemented to their
    nearest bound (v >= 0, value y*). Half the sum of a row set S, rounded
    down, is valid; with u = sum_S a_i (v-space) and beta = sum_S b_i odd,

        sum_j floor(u_j / 2) v_j <= (beta - 1) / 2,

    violated by (1 - sum_S slack_i - sum_{u_j odd} y*_j) / 2. So S is
    sought with odd beta and a small weight sum_S slack + sum_odd y*: rows
    of slack >= 1 never help; columns at their bound (y* = 0) are free. The
    columns with y* > 0 are eliminated in order of decreasing y* (each by
    XOR of the pivot row - least slack - into the others holding it); after
    every elimination, rows of odd parity and weight < 1 give cuts.
    ``max_work`` (row visits) bounds the elimination deterministically; the
    deadline is only a safety net."""
    impint = P.integer if impint is None else impint
    A = P.A
    act = A @ x
    use_up = np.isfinite(ub) & (~np.isfinite(lb) | (ub - x < x - lb))
    y = np.where(use_up, ub - x, x - lb)
    y = np.where(np.abs(y) < 1e-9, 0.0, y)
    rr = np.repeat(np.arange(P.m), np.diff(A.indptr))
    ok_e = (np.abs(A.data - np.round(A.data)) < 1e-9) & impint[A.indices] & \
        np.isfinite(np.where(use_up, ub, lb)[A.indices])
    ok_row = np.bincount(rr, weights=(~ok_e).astype(float), minlength=P.m) == 0
    ok_row &= np.diff(A.indptr) > 0
    # sides: (row, sign, rhs, slack)
    sides = []
    for sgn, bnd in ((1.0, P.row_hi), (-1.0, P.row_lo)):
        with np.errstate(invalid="ignore"):
            fin = np.isfinite(bnd) & ok_row & (np.abs(bnd - np.round(bnd)) < 1e-9)
        if sgn < 0:
            fin &= P.row_lo != P.row_hi                     # an equality once
        sl = sgn * (bnd - act)
        good = fin & (sl < 1 - 1e-6)
        for i in np.nonzero(good)[0]:
            sides.append((int(i), sgn, float(sgn * bnd[i]), max(float(sl[i]), 0.0)))
    if not sides:
        return []
    sides = sorted(sides, key=lambda t: t[3])[:max_rows]
    # v-space row: a_v = a (lower) or -a (upper); b_v = b - a'bound
    bnd_v = np.where(use_up, ub, lb)
    fr = np.nonzero(y > 1e-9)[0]
    if len(fr) > max_cols:
        fr = fr[np.argsort(-y[fr])[:max_cols]]
    pos = np.full(P.n, -1, np.int64)
    pos[fr] = np.arange(len(fr))
    rows = []      # [colbits, parity, slack, rowbits]
    for k, (i, sgn, b, sl) in enumerate(sides):
        cols = A.indices[A.indptr[i]:A.indptr[i + 1]]
        a = sgn * A.data[A.indptr[i]:A.indptr[i + 1]]
        av = np.where(use_up[cols], -a, a)
        bv = b - float(a @ bnd_v[cols])
        odd = (np.round(av).astype(np.int64) % 2) != 0
        bits = 0
        for c in pos[cols[odd]].tolist():
            if c >= 0:
                bits |= 1 << c
        # odd columns at their bound (y* = 0) cost nothing: not tracked
        par_ = int(round(bv)) % 2
        rows.append([bits, par_, sl, 1 << k])
    out, seen = [], set()
    yw = y[fr]

    def weight(r):
        w, bits = r[2], r[0]
        while bits and w < 1:
            low = bits & -bits
            w += yw[low.bit_length() - 1]
            bits ^= low
        return w

    def emit(r):
        if r[3] in seen:
            return
        seen.add(r[3])
        cut = _zerohalf_from_set(r[3], sides, A, P, x, lb, ub, use_up, par)
        if cut is not None and cut.efficacy(x) >= par.min_efficacy:
            out.append(cut)

    active = [r for r in rows if r[2] < 1 - 1e-6]
    for r in active:
        if r[1] == 1 and r[0] == 0:
            emit(r)
    order = np.argsort(-yw)
    work = 0
    for c in order.tolist():
        work += len(active)
        if work > max_work or time.perf_counter() > deadline or len(out) >= 200:
            break
        bit = 1 << c
        if not active:
            break
        holders = [r for r in active if r[0] & bit]
        if not holders:
            continue
        piv = min(holders, key=lambda r: r[2])
        nxt = []
        for r in active:
            if r is piv:
                continue
            if r[0] & bit:
                r = [r[0] ^ piv[0], r[1] ^ piv[1], r[2] + piv[2], r[3] ^ piv[3]]
                if r[2] >= 1 - 1e-6:
                    continue
            nxt.append(r)
        active = nxt
        for r in active:
            if r[1] == 1 and r[2] < 1 - 1e-6 and (r[0] == 0 or weight(r) < 1 - 1e-6):
                emit(r)
    return out


def _zerohalf_from_set(rowbits: int, sides, A, P, x, lb, ub, use_up, par) -> Cut | None:
    """The {0,1/2}-cut of the sides whose bits are set, in x-space."""
    ks = []
    k = 0
    while rowbits:
        if rowbits & 1:
            ks.append(k)
        rowbits >>= 1
        k += 1
    dense: dict[int, float] = {}
    beta = 0.0
    for k in ks:
        i, sgn, b, _sl = sides[k]
        cols = A.indices[A.indptr[i]:A.indptr[i + 1]]
        a = sgn * A.data[A.indptr[i]:A.indptr[i + 1]]
        for j, v in zip(cols.tolist(), a.tolist()):
            dense[j] = dense.get(j, 0.0) + v
        beta += b
    cols = np.fromiter(dense.keys(), dtype=np.int64)
    u = np.fromiter(dense.values(), dtype=float)
    up = use_up[cols]
    bnd = np.where(up, ub[cols], lb[cols])
    uv = np.where(up, -u, u)                          # v-space coefficients
    bv = beta - float(u @ bnd)
    bvi = int(round(bv))
    if bvi % 2 == 0:
        return None
    g = np.floor(np.round(uv) / 2.0)                  # sum g v <= (bv - 1) / 2
    rhs_v = (bvi - 1) / 2
    # back to x: v = x - l -> g x, rhs + g l;  v = u - x -> -g x, rhs - g u
    pi = np.where(up, -g, g)
    bt = rhs_v + float(np.sum(np.where(up, -g * ub[cols], g * lb[cols])))
    keep = pi != 0
    if not np.any(keep):
        return None
    return _finish(lb, ub, cols[keep], -pi[keep], -bt, par, "zerohalf")


# ------------------------------------------------------------------- pool
@dataclass
class CutPool:
    params: CutParams = field(default_factory=CutParams)
    seen: set = field(default_factory=set)

    def _key(self, cut: Cut):
        o = np.argsort(cut.cols)
        return (tuple(cut.cols[o].tolist()), tuple(np.round(cut.coef[o], 9).tolist()), round(cut.rhs, 9))

    def select(self, cuts: list[Cut], x: np.ndarray, max_cuts: int) -> list[Cut]:
        """The most efficacious cuts, skipping duplicates and near-parallel
        ones (|cos| > max_parallelism). A candidate's dots with the chosen
        cuts read its scattered dense copy: one gather per chosen cut."""
        scored = sorted(((c.efficacy(x), k) for k, c in enumerate(cuts)), reverse=True)
        chosen: list[Cut] = []
        norms: list[float] = []
        dense = np.zeros(len(x))
        for eff, k in scored:
            if eff < self.params.min_efficacy or len(chosen) >= max_cuts:
                break
            c = cuts[k]
            key = self._key(c)
            if key in self.seen:
                continue
            nc = float(np.linalg.norm(c.coef))
            dense[c.cols] = c.coef
            lim = self.params.max_parallelism * nc
            parallel = any(abs(float(d.coef @ dense[d.cols])) > lim * nd for d, nd in zip(chosen, norms))
            dense[c.cols] = 0.0
            if parallel:
                continue
            self.seen.add(key)
            chosen.append(c)
            norms.append(nc)
        return chosen
