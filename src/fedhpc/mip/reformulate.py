"""LP-side reformulation of interval (consecutive-ones) rows as network rows.

A block of consecutive rows t0..t1 in which every column's entries form a
single run of consecutive rows with a constant coefficient (an interval
matrix: time-indexed scheduling's capacity rows, where x_js covers the
periods s..s+p_j-1) is equivalent, after explicit slacks and differencing
adjacent rows, to a network-like block (Veinott & Wagner 1962; Nemhauser &
Wolsey, III.1.3). With the rows written as  a_t'x + sigma_t s_t = rhs_t,

    row t0:         a_t0'x + sigma_t0 s_t0                       = rhs_t0
    row t > t0:  (a_t - a_{t-1})'x + sigma_t s_t - sigma_{t-1} s_{t-1}
                                                             = rhs_t - rhs_{t-1}

which is the block multiplied by a nonsingular lower bidiagonal matrix. A
column's run [a, b] with coefficient v leaves +v at row a and -v at row
b + 1 (when inside the block): two entries instead of b - a + 1. On
Kramer's TI models (p_j ~ 45) the matrix shrinks from 1.7 M to ~0.1 M
nonzeros and the LP is the time-expanded network flow of the arc-flow
model.

The feasible set in x is unchanged, and so is every simplex tableau (row
operations leave B^-1 A alone): slack s_t is the original row t's logical,
in the same sign convention as fedlp's, and the differenced rows' own
logicals are fixed at 0. Only the LP sees the reformulation; propagation,
heuristics and cut separation keep the original rows.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .problem import MipProblem


@dataclass
class IntervalBlocks:
    blocks: list[tuple[int, int]]       # inclusive row ranges [t0, t1]
    nnz_before: int                     # the blocks' nonzeros in A
    nnz_after: int                      # in the differenced rows (slacks included)

    @property
    def rows(self) -> np.ndarray:
        if not self.blocks:
            return np.zeros(0, dtype=np.int64)
        return np.concatenate([np.arange(a, b + 1) for a, b in self.blocks])


def detect_interval_blocks(P: MipProblem, min_len: int = 8, max_ratio: float = 0.5,
                           link_share: float = 0.5) -> IntervalBlocks:
    """Blocks of consecutive rows with the interval property whose
    differencing at least halves (``max_ratio``) their nonzeros.

    Adjacent rows t-1, t are chained when at least ``link_share`` of the
    smaller row's entries continue a column's run (same column, coefficient
    equal, rows adjacent). A chain is split where a column re-enters it
    after a gap or changes coefficient, so that every column has at most
    one run in a block."""
    m = P.m
    if m < min_len:
        return IntervalBlocks([], 0, 0)
    C = P.csc.copy()
    C.sort_indices()
    rows, vals = C.indices.astype(np.int64), C.data
    rn = np.diff(P.A.indptr)
    col = np.repeat(np.arange(P.n), np.diff(C.indptr))
    same_col = col[1:] == col[:-1]                              # entries k, k+1 in one column
    nxt_row = rows[1:]
    cont = same_col & (nxt_row == rows[:-1] + 1) & (vals[1:] == vals[:-1])
    link = np.bincount(nxt_row[cont], minlength=m)            # link[t]: runs t-1 -> t
    ok_row = np.isfinite(P.row_lo) | np.isfinite(P.row_hi)
    chained = np.zeros(m, bool)
    chained[1:] = (link[1:] >= link_share * np.minimum(rn[:-1], rn[1:])) & (link[1:] > 0) \
        & ok_row[1:] & ok_row[:-1]
    bad_next = same_col & ~cont                                 # pair (k, k+1) not a continuation
    # a bad pair (row_k, row_k+1) inside a chain needs a boundary in
    # (row_k, row_k+1]: cut before row_k+1. Sorted by that row.
    bad_lo, bad_hi = rows[:-1][bad_next], nxt_row[bad_next]
    o = np.argsort(bad_hi, kind="stable")
    bad_lo, bad_hi = bad_lo[o], bad_hi[o]
    A = P.A.tocsr()
    # chains: maximal runs of chained[i] = True; chain = [first - 1, last]
    edge = np.diff(np.concatenate([[0], chained.astype(np.int8), [0]]))
    first, last = np.nonzero(edge == 1)[0], np.nonzero(edge == -1)[0] - 1
    blocks, before, after = [], 0, 0
    for a, b in zip(first - 1, last):
        if b - a + 1 < min_len:
            continue
        p0, p1 = np.searchsorted(bad_hi, a + 1), np.searchsorted(bad_hi, b, side="right")
        sel = bad_lo[p0:p1] >= a
        if np.count_nonzero(sel) > 200000:
            continue
        cuts = _stab(bad_lo[p0:p1][sel], bad_hi[p0:p1][sel])
        lo = int(a)
        for hi_excl in [*cuts, int(b) + 1]:
            hi = hi_excl - 1
            if hi - lo + 1 >= min_len:
                nb = int(rn[lo:hi + 1].sum())
                # differenced nonzeros (2 per run, 1 when it ends at the
                # block's last row) plus the slacks' 2 per row
                na = int(_difference(A[lo:hi + 1]).nnz) + 2 * (hi - lo + 1)
                if na <= max_ratio * nb:
                    blocks.append((lo, hi))
                    before += nb
                    after += na
            lo = max(lo, hi_excl)
    return IntervalBlocks(blocks, before, after)


def _stab(lo: np.ndarray, hi: np.ndarray) -> list[int]:
    """The fewest rows c with every interval (lo_k, hi_k] holding one
    (intervals sorted by hi): the greedy by right end, which is optimal for
    interval stabbing. A cut at c starts a new block at row c."""
    cuts, last = [], -1
    for l, h in zip(lo.tolist(), hi.tolist()):
        if last > l:                    # last <= h by the order
            continue
        cuts.append(h)
        last = h
    return cuts


def _difference(B: sp.csr_matrix) -> sp.csr_matrix:
    """Rows of B differenced: row 0 kept, row t -> B_t - B_{t-1}."""
    k = B.shape[0]
    L = sp.eye(k, format="csr") - sp.eye(k, k=-1, format="csr")
    D = (L @ B).tocsr()
    D.eliminate_zeros()
    return D


@dataclass
class NetworkForm:
    """The LP's rows after differencing the blocks: A_lp x + S s = rhs-part.

    ``A`` (m x n) replaces P.A; the block rows become equalities with
    ``row_lo == row_hi``; slack column k belongs to original row
    ``slack_row[k]`` with bounds [0, slack_ub[k]] and coefficients
    (``S`` m x k_slacks)."""
    A: sp.csr_matrix
    row_lo: np.ndarray
    row_hi: np.ndarray
    S: sp.csc_matrix
    slack_row: np.ndarray
    slack_ub: np.ndarray
    block_row: np.ndarray          # bool per row: in a differenced block
    blocks: list[tuple[int, int]]


def network_form(P: MipProblem, blocks: IntervalBlocks) -> NetworkForm:
    m = P.m
    A = P.A.tocsr().copy()
    row_lo, row_hi = P.row_lo.astype(float).copy(), P.row_hi.astype(float).copy()
    block_row = np.zeros(m, bool)
    s_row, s_ub, s_i, s_j, s_v = [], [], [], [], []
    pieces = []
    last = 0
    for (lo, hi) in blocks.blocks:
        pieces.append(A[last:lo])
        pieces.append(_difference(A[lo:hi + 1]))
        last = hi + 1
        block_row[lo:hi + 1] = True
        rhs, sig = np.empty(hi - lo + 1), np.empty(hi - lo + 1)
        for k, t in enumerate(range(lo, hi + 1)):
            l, u = P.row_lo[t], P.row_hi[t]
            if l == u:
                sig[k], rhs[k], ub = 1.0, u, 0.0
            elif not np.isfinite(l):
                sig[k], rhs[k], ub = 1.0, u, np.inf
            elif not np.isfinite(u):
                sig[k], rhs[k], ub = -1.0, l, np.inf
            else:
                sig[k], rhs[k], ub = 1.0, u, u - l
            col = len(s_row)
            s_row.append(t)
            s_ub.append(ub)
            s_i.append(t), s_j.append(col), s_v.append(sig[k])
            if t < hi:
                s_i.append(t + 1), s_j.append(col), s_v.append(-sig[k])
        d = rhs.copy()
        d[1:] -= rhs[:-1]
        row_lo[lo:hi + 1] = d
        row_hi[lo:hi + 1] = d
    pieces.append(A[last:])
    A2 = sp.vstack(pieces, format="csr")
    S = sp.csc_matrix((s_v, (s_i, s_j)), shape=(m, len(s_row)))
    return NetworkForm(A2, row_lo, row_hi, S, np.asarray(s_row, dtype=np.int64),
                       np.asarray(s_ub, float), block_row, list(blocks.blocks))
