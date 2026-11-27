"""The mixed-integer linear program every stage of fedhpc.mip works on.

    min  c'x + offset
    s.t. row_lo <= A x <= row_hi
         lb <= x <= ub,   x_j integer for integer[j]

Always a minimisation: a maximisation is read with its objective negated and
``maximize`` set, so reported objectives can be given back in the model's
sense. Infinite bounds are +-inf. A is kept by rows (CSR); the CSC copy is
built on demand and cached, as stages need one or the other.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

INF = float("inf")


@dataclass
class MipProblem:
    A: sp.csr_matrix
    row_lo: np.ndarray
    row_hi: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    c: np.ndarray
    integer: np.ndarray
    offset: float = 0.0
    name: str = ""
    maximize: bool = False
    col_names: list | None = None
    row_names: list | None = None
    _csc: sp.csc_matrix | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.A = sp.csr_matrix(self.A, dtype=float)
        self.A.sum_duplicates()
        self.A.eliminate_zeros()
        m, n = self.A.shape
        self.row_lo = np.asarray(self.row_lo, dtype=float).reshape(m)
        self.row_hi = np.asarray(self.row_hi, dtype=float).reshape(m)
        self.lb = np.asarray(self.lb, dtype=float).reshape(n)
        self.ub = np.asarray(self.ub, dtype=float).reshape(n)
        self.c = np.asarray(self.c, dtype=float).reshape(n)
        self.integer = np.asarray(self.integer, dtype=bool).reshape(n)
        if np.any(self.row_lo > self.row_hi):
            raise ValueError("a row's lower bound exceeds its upper bound")

    # ------------------------------------------------------------- shape
    @property
    def m(self) -> int:
        return self.A.shape[0]

    @property
    def n(self) -> int:
        return self.A.shape[1]

    @property
    def nnz(self) -> int:
        return self.A.nnz

    @property
    def csc(self) -> sp.csc_matrix:
        if self._csc is None:
            self._csc = self.A.tocsc()
        return self._csc

    def copy(self) -> "MipProblem":
        return MipProblem(self.A.copy(), self.row_lo.copy(), self.row_hi.copy(), self.lb.copy(),
                          self.ub.copy(), self.c.copy(), self.integer.copy(), self.offset,
                          self.name, self.maximize, self.col_names, self.row_names)

    # ------------------------------------------------------------ queries
    def objective(self, x: np.ndarray) -> float:
        """c'x + offset, in this (minimisation) form."""
        return float(self.c @ x + self.offset)

    def reported(self, value: float) -> float:
        """An objective value in the model's own sense."""
        return -value if self.maximize else value

    def violations(self, x: np.ndarray) -> tuple[float, float, float]:
        """Largest (row, bound, integrality) violation of x, each relative to
        1 + |the bound|."""
        act = self.A @ x
        rv = np.maximum(_side(self.row_lo, act), _side(act, self.row_hi, upper=True))
        bv = np.maximum(_side(self.lb, x), _side(x, self.ub, upper=True))
        xi = x[self.integer]
        iv = np.abs(xi - np.round(xi))
        return (float(rv.max(initial=0.0)), float(bv.max(initial=0.0)), float(iv.max(initial=0.0)))

    def is_feasible(self, x: np.ndarray, tol: float = 1e-6, int_tol: float = 1e-6) -> bool:
        r, b, i = self.violations(x)
        return r <= tol and b <= tol and i <= int_tol

    def objective_is_integral(self) -> bool:
        """Every feasible point has an integer objective (up to the offset):
        integer costs on integer columns, none on continuous ones. A node's
        bound may then be rounded up."""
        cost = self.c != 0
        if np.any(cost & ~self.integer):
            return False
        ci = self.c[cost]
        return bool(np.all(np.abs(ci - np.round(ci)) < 1e-9))

    def summary(self) -> str:
        ni = int(self.integer.sum())
        nb = int((self.integer & (self.lb == 0) & (self.ub == 1)).sum())
        return (f"{self.name or 'problem'}: {self.m} rows, {self.n} columns ({nb} binary, "
                f"{ni - nb} general integer, {self.n - ni} continuous), {self.nnz} nonzeros")


def _side(a: np.ndarray, b: np.ndarray, upper: bool = False) -> np.ndarray:
    """(a - b) / (1 + |bound|) where the bound (b for an upper side, a for a
    lower one) is finite, -inf where it is infinite. An infinite bound must
    not turn into NaN: np.maximum propagates NaN, and nan_to_num then hid
    the other side's violation (a violated <= row read as satisfied)."""
    bound = b if upper else a
    fin = np.isfinite(bound)
    out = np.full(np.shape(bound), -np.inf)
    out[fin] = (np.asarray(a)[fin] - np.asarray(b)[fin]) / (1 + np.abs(bound[fin]))
    return out
