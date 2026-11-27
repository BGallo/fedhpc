"""Clique table: conflicts between binary literals, and what they give.

A literal is a binary column or its complement: lit = 2 j (x_j) or 2 j + 1
(1 - x_j). A clique is a set of literals at most one of which can be 1 in a
feasible point. Sources (Atamturk, Nemhauser & Savelsbergh 2000; Achterberg
2007, sec. 3.3.4; HiGHS's HighsCliqueTable):

- rows: a <= row over binaries (other columns at their least activity),
  negative coefficients complemented so every literal coefficient c_l > 0
  and the row reads sum c_l l <= b'. Two literals conflict iff c_i + c_j >
  b'. Sorted by c descending, the longest prefix whose last two conflict is
  a clique (every pair in it conflicts); a set-packing row gives itself.
- probing: x_j = v forcing x_k = w makes {x_j = v, x_k = 1 - w} a 2-clique
  (literals l(j, v) and l(k, 1 - w) cannot both hold).

Uses:
- clique cuts: sum_{l in K} l <= 1 for a clique K of the conflict graph
  heavier than 1 at the LP point, found greedily (literals by LP weight,
  each joining when it conflicts with every member) and extended to a
  maximal clique by zero-weight literals;
- propagation: cliques that are not rows already are handed to the
  propagator as extra rows (sum l <= 1).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from .cuts import Cut, CutParams, _finish
from .problem import INF, MipProblem


@dataclass
class CliqueTable:
    n: int
    cliques: list = field(default_factory=list)        # arrays of literals
    from_row: list = field(default_factory=list)       # whether the clique is a whole row already
    _lit_cliques: dict | None = None

    def add(self, lits, is_row: bool = False) -> None:
        lits = np.unique(np.asarray(lits, dtype=np.int64))
        if len(lits) >= 2 and len(np.unique(lits // 2)) == len(lits):
            self.cliques.append(lits)
            self.from_row.append(is_row)
            self._lit_cliques = None

    def lit_cliques(self) -> dict:
        if self._lit_cliques is None:
            d: dict[int, list[int]] = {}
            for k, c in enumerate(self.cliques):
                for lit in c.tolist():
                    d.setdefault(lit, []).append(k)
            self._lit_cliques = {k: set(v) for k, v in d.items()}
        return self._lit_cliques

    @property
    def size(self) -> int:
        return len(self.cliques)

    def neighbours(self, lit: int) -> set:
        lc = self.lit_cliques()
        out = set()
        for k in lc.get(lit, ()):
            out.update(self.cliques[k].tolist())
        out.discard(lit)
        return out


def _literal_value(x: np.ndarray, lits: np.ndarray) -> np.ndarray:
    j, neg = lits // 2, lits % 2
    return np.where(neg == 1, 1.0 - x[j], x[j])


def row_cliques(P: MipProblem, lb: np.ndarray, ub: np.ndarray, table: CliqueTable,
                max_row_len: int = 5000) -> int:
    """Cliques from every row side (see the module notes); returns how many."""
    A = P.A
    binary = P.integer & (lb == 0) & (ub == 1)
    rn = np.diff(A.indptr)
    found = 0
    for i in np.nonzero((rn >= 2) & (rn <= max_row_len))[0].tolist():
        s0, s1 = A.indptr[i], A.indptr[i + 1]
        cols, a = A.indices[s0:s1], A.data[s0:s1]
        for sgn, bound in ((1.0, P.row_hi[i]), (-1.0, P.row_lo[i])):
            if not np.isfinite(bound):
                continue
            c = sgn * a
            b = sgn * bound
            isb = binary[cols]
            # other columns at their least contribution
            oc, oa = cols[~isb], c[~isb]
            if len(oc):
                mn = np.where(oa > 0, oa * lb[oc], oa * ub[oc])
                if not np.all(np.isfinite(mn)):
                    continue
                b -= float(mn.sum())
            bc, ba = cols[isb], c[isb]
            if len(bc) < 2:
                continue
            neg = ba < 0
            b -= float(ba[neg].sum())            # complemented: a x = a - a (1 - x)
            w = np.abs(ba)
            lits = 2 * bc + neg
            order = np.argsort(-w, kind="stable")
            ws = w[order]
            if ws[0] + ws[1] <= b + 1e-9:
                continue                          # no pair conflicts
            k = 1
            while k + 1 < len(ws) and ws[k] + ws[k + 1] > b + 1e-9:
                k += 1
            K = lits[order[:k + 1]]
            # the clique is the row itself when it is a set-packing side
            # (sum l <= b'/w with 1 < b'/w < 2 is weaker in the LP than the
            # clique inequality, so only b' = w is the clique row itself)
            is_row = (k + 1 == len(bc)) and len(oc) == 0 and np.allclose(w, w[0]) and abs(b - w[0]) <= 1e-9 * (1 + w[0])
            table.add(K, is_row)
            found += 1
    return found


def probe_cliques(P: MipProblem, prop, lb: np.ndarray, ub: np.ndarray, table: CliqueTable,
                  budget: int = 1_000_000, max_probes: int = 2000, deadline: float = np.inf) -> int:
    """2-cliques from probing implications on binaries (most rows first)."""
    if not hasattr(prop, "changed_since"):
        return 0
    if not prop.reset(lb, ub, propagate=False):
        return 0
    binary = P.integer & (lb == 0) & (ub == 1)
    cn = np.diff(P.csc.indptr)
    order = np.nonzero(binary)[0]
    order = order[np.argsort(-cn[order], kind="stable")][:max_probes]
    found, work = 0, 0
    for j in order.tolist():
        if work > budget or time.perf_counter() > deadline:
            break
        for v in (0.0, 1.0):
            lev = prop.level()
            ok = prop.change([j], [v], [v], max_rounds=10)
            work += prop.k.work
            if ok:
                ks, kl, ku = prop.changed_since(lev)
                fixed = (kl == ku) & binary[ks] & (ks != j)
                lit_j = 2 * j + (1 if v == 0.0 else 0)          # the literal "x_j = v"
                for k, val in zip(ks[fixed].tolist(), kl[fixed].tolist()):
                    # x_j = v forces x_k = val: "x_j = v" and "x_k = 1 - val" conflict
                    lit_k = 2 * k + (1 if (1 - val) == 0 else 0)
                    table.add([lit_j, lit_k])
                    found += 1
            prop.undo(lev)
    return found


def clique_cuts(table: CliqueTable, x: np.ndarray, lb: np.ndarray, ub: np.ndarray, par: CutParams,
                max_cuts: int = 200, tol: float = 1e-6) -> list[Cut]:
    """Greedy heavy cliques of the conflict graph at x (see the module notes)."""
    if table.size == 0:
        return []
    lc = table.lit_cliques()
    lits = np.fromiter(lc.keys(), dtype=np.int64)
    if len(lits) == 0:
        return []
    w = _literal_value(x, lits)
    frac = lits[(w > tol)]
    if len(frac) < 2:
        return []
    wf = _literal_value(x, frac)
    order = frac[np.argsort(-wf, kind="stable")]
    weight = dict(zip(frac.tolist(), wf.tolist()))
    out, seen = [], set()
    for s in order.tolist()[:max_cuts * 4]:
        if len(out) >= max_cuts:
            break
        cand = [u for u in table.neighbours(s) if weight.get(u, 0.0) > tol]
        cand.sort(key=lambda u: -weight[u])
        K = [s]
        common = table.neighbours(s)
        for u in cand:
            if u in common:
                K.append(u)
                common &= table.neighbours(u)
        total = sum(weight.get(u, 0.0) for u in K)
        if total <= 1 + 1e-6 or len(K) < 2:
            continue
        # maximal: add zero-weight literals adjacent to all
        for u in sorted(common)[:50]:
            if u in common and u not in K:          # common: adjacent to every member
                K.append(u)
                common &= table.neighbours(u)
        key = tuple(sorted(K))
        if key in seen:
            continue
        seen.add(key)
        Ka = np.asarray(K, dtype=np.int64)
        j, neg = Ka // 2, Ka % 2
        if len(np.unique(j)) != len(j):
            continue
        # sum_{pos} x_j + sum_{neg} (1 - x_j) <= 1  ->  -sum_pos x + sum_neg x >= #neg - 1
        coef = np.where(neg == 1, 1.0, -1.0)
        cut = _finish(lb, ub, j, coef, float(neg.sum()) - 1.0, par, "clique")
        if cut is not None and cut.efficacy(x) >= par.min_efficacy:
            out.append(cut)
    return out


def clique_rows(table: CliqueTable, n: int):
    """The cliques that are not rows already, as rows sum l <= 1 over x:
    (A, row_lo, row_hi) to append to the propagator's problem."""
    ri, ci, vi, hi = [], [], [], []
    r = 0
    for c, is_row in zip(table.cliques, table.from_row):
        if is_row:
            continue
        j, neg = c // 2, c % 2
        ri.extend([r] * len(c)), ci.extend(j.tolist()), vi.extend(np.where(neg == 1, -1.0, 1.0).tolist())
        hi.append(1.0 - float(neg.sum()))
        r += 1
    if r == 0:
        return None
    A = sp.csr_matrix((vi, (ri, ci)), shape=(r, n))
    return A, np.full(r, -INF), np.asarray(hi)
