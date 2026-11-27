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
def gomory_cuts(lp: LpRelaxation, P: MipProblem, x: np.ndarray, par: CutParams) -> list[Cut]:
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
    dense = np.zeros(P.n)
    for _score, pos in cand[: par.gomory_rows]:
        row = lp.tableau_row(pos, heads)
        if row is None:
            continue
        cut = _gmi_from_row(lp, P, row, dense, par)
        if cut is not None and cut.efficacy(x) >= par.min_efficacy:
            out.append(cut)
    return out


def _gmi_from_row(lp: LpRelaxation, P: MipProblem, row, dense: np.ndarray, par: CutParams) -> Cut | None:
    """The GMI cut of one tableau row. Nonbasic columns sit at the LP's
    installed bounds (lp.lb, lp.ub), which the cut is valid for: global at
    the root."""
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
    is_int[is_col] = P.integer[row.index[is_col]]
    fk = a - np.floor(a)
    g = np.where(is_int,
                 np.where(fk <= f0, fk / f0, (1 - fk) / (1 - f0)),
                 np.where(a >= 0, a / f0, -a / (1 - f0)))
    # sum g_k v'_k >= 1  ->  pi'x >= 1 - const
    const = 0.0
    touched = []
    for k in np.nonzero(g > 1e-12)[0]:
        gk = float(g[k])
        if is_col[k]:
            j = int(row.index[k])
            if row.at_upper[k]:
                if not np.isfinite(ub[j]):
                    return None
                dense[j] -= gk
                const += gk * ub[j]
            else:
                if not np.isfinite(lb[j]):
                    return None
                dense[j] += gk
                const -= gk * lb[j]
            touched.append(j)
        else:
            r = int(row.index[k])
            sigma, rhs, s_lo, s_hi = lp.logical_info(r)
            cols, coefs = lp.row_coefs(r)
            # s = sigma (rhs - a_r x)
            if row.at_upper[k]:     # v' = s_hi - s = s_hi - sigma rhs + sigma a x
                if not np.isfinite(s_hi):
                    return None
                dense[cols] += gk * sigma * coefs
                const += gk * (s_hi - sigma * rhs)
            else:                   # v' = s - s_lo = sigma rhs - sigma a x - s_lo
                dense[cols] -= gk * sigma * coefs
                const += gk * (sigma * rhs - s_lo)
            touched.extend(cols.tolist())
    if not touched:
        return None
    cols = np.unique(np.asarray(touched, dtype=int))
    coef = dense[cols].copy()
    dense[cols] = 0.0
    nz = coef != 0
    cols, coef = cols[nz], coef[nz]
    if len(cols) > par.gomory_max_support * P.n + 10:
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
        ones (|cos| > max_parallelism)."""
        scored = sorted(((c.efficacy(x), k) for k, c in enumerate(cuts)), reverse=True)
        chosen: list[Cut] = []
        dense_chosen: list[dict] = []
        for eff, k in scored:
            if eff < self.params.min_efficacy or len(chosen) >= max_cuts:
                break
            c = cuts[k]
            key = self._key(c)
            if key in self.seen:
                continue
            dc = dict(zip(c.cols.tolist(), c.coef.tolist()))
            nc = float(np.linalg.norm(c.coef))
            parallel = False
            for d, nd in dense_chosen:
                dot = sum(v * d.get(j, 0.0) for j, v in dc.items())
                if abs(dot) > self.params.max_parallelism * nc * nd:
                    parallel = True
                    break
            if parallel:
                continue
            self.seen.add(key)
            chosen.append(c)
            dense_chosen.append((dc, nc))
        return chosen
