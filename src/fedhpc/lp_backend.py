"""LP/MIP backends for :mod:`fedhpc.lpfirst`'s restricted masters and
placement MIPs: Gurobi (default) and HiGHS (``highspy``, licence-free).

The interface is the little column generation needs: rows with bounds,
columns added in batches as CSC slices, an LP solve with row duals, and a
MIP solve with a time limit, an objective cutoff, an absolute gap and a
(partial) start. Statuses are normalised to ``"optimal"``, ``"infeasible"``
(also: nothing below the cutoff), ``"limit"`` and ``"other"``.

Sign convention: row duals are those of ``min c x``: a binding ``<=`` row has
a dual ``<= 0``, as Gurobi's ``Pi`` (HiGHS's ``row_dual`` agrees; checked in
``tests/test_lp_backend.py``).
"""
from __future__ import annotations

import math

import numpy as np

INF = math.inf


def make(name: str, threads: int = 0):
    if name == "gurobi":
        return GurobiBackend(threads)
    if name == "highs":
        return HighsBackend(threads)
    if name == "fedlp":
        return FedlpBackend(threads)
    raise ValueError(f"unknown LP backend {name!r} (gurobi, highs, fedlp)")


def available(name: str) -> bool:
    try:
        if name == "gurobi":
            import gurobipy  # noqa: F401
        elif name == "highs":
            import highspy  # noqa: F401
        elif name == "fedlp":
            from . import _fedlp  # noqa: F401
        else:
            return False
    except ImportError:
        return False
    return True


class GurobiBackend:
    """Gurobi, built lazily: rows and columns added before the first solve
    are buffered, then built as one ``addMVar`` and one ``addLConstr`` per
    row (CSC transposed to CSR in NumPy). One ``addVar`` per placement cost
    more than the MIPs themselves on pos_congestion's LNS (28 s of 132 s);
    rows are a few thousand where placements are 10^5. Later additions,
    the column generation's, go in column by column."""

    name = "gurobi"

    def __init__(self, threads: int = 0):
        import gurobipy as gp
        from . import formulations
        self.gp = gp
        from gurobipy import GRB
        self.GRB = GRB
        formulations.configure_env()
        self.m = gp.Model("lpfirst", env=formulations._shared_env)
        self.m.setParam("OutputFlag", 0)
        if threads:
            self.m.setParam("Threads", threads)
        self.rows: list = []
        self.cols: list = []
        self.n_rows = self.n_cols = 0
        self.built = False
        self._rlo: list = []
        self._rhi: list = []
        self._pend: list = []   # buffered (cost, ub, integer, starts, index, value)

    def add_rows(self, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        start = self.n_rows
        self.n_rows += len(lo)
        if not self.built:
            self._rlo.append(lo), self._rhi.append(hi)
            return np.arange(start, self.n_rows)
        gp, m = self.gp, self.m
        for a, b in zip(lo.tolist(), hi.tolist()):
            self.rows.append(m.addConstr(self._expr_bound(gp.LinExpr(), a, b)))
        m.update()
        return np.arange(start, self.n_rows)

    @staticmethod
    def _expr_bound(e, a: float, b: float):
        if a == b:
            return e == a
        if a == -INF:
            return e <= b
        return e >= a

    def add_cols(self, cost, ub, integer, starts, index, value) -> None:
        n = len(cost)
        if not n:
            return
        self.n_cols += n
        if not self.built:
            self._pend.append((np.asarray(cost, float), np.asarray(ub, float),
                               np.asarray(integer, dtype=bool), np.asarray(starts, dtype=np.int64),
                               np.asarray(index, dtype=np.int64), np.asarray(value, float)))
            return
        gp, GRB, m, rows = self.gp, self.GRB, self.m, self.rows
        cost, ub, integer = cost.tolist(), ub.tolist(), np.asarray(integer).tolist()
        starts, index, value = starts.tolist(), index.tolist(), value.tolist()
        for c in range(len(cost)):
            lo, hi = starts[c], starts[c + 1]
            col = gp.Column(value[lo:hi], [rows[r] for r in index[lo:hi]])
            self.cols.append(m.addVar(obj=cost[c], ub=ub[c],
                                      vtype=GRB.INTEGER if integer[c] else GRB.CONTINUOUS,
                                      column=col))

    def _build(self) -> None:
        if self.built:
            return
        gp, m, GRB = self.gp, self.m, self.GRB
        lo = np.concatenate(self._rlo) if self._rlo else np.zeros(0)
        hi = np.concatenate(self._rhi) if self._rhi else np.zeros(0)
        nr = len(lo)
        if self._pend:
            cost = np.concatenate([c[0] for c in self._pend])
            ub = np.concatenate([c[1] for c in self._pend])
            integer = np.concatenate([c[2] for c in self._pend])
            index = np.concatenate([c[4] for c in self._pend])
            value = np.concatenate([c[5] for c in self._pend])
            nnz = np.concatenate([np.diff(c[3]) for c in self._pend])
        else:
            cost = ub = value = np.zeros(0)
            integer = np.zeros(0, dtype=bool)
            index = nnz = np.zeros(0, dtype=np.int64)
        nc = len(cost)
        if nc:
            x = m.addMVar(nc, lb=0.0, ub=ub, obj=cost,
                          vtype=np.where(integer, GRB.INTEGER, GRB.CONTINUOUS))
            m.update()
            self.cols = x.tolist()
        try:
            import scipy.sparse as sps
        except ImportError:
            sps = None
        if sps is not None and nc and nr:
            # one sparse matrix build, all in C
            eq = lo == hi
            le = ~eq & (lo == -INF)
            sense = np.where(eq, GRB.EQUAL, np.where(le, GRB.LESS_EQUAL, GRB.GREATER_EQUAL))
            rhs = np.where(le, hi, lo)
            A = sps.csc_matrix((value, index, np.r_[0, np.cumsum(nnz)]), shape=(nr, nc)).tocsr()
            self.rows = m.addMConstr(A, x, sense, rhs).tolist()
        else:
            # CSC -> CSR, one LinExpr per row
            col_of = np.repeat(np.arange(nc), nnz)
            order = np.argsort(index, kind="stable")
            rcols, rvals = col_of[order].tolist(), value[order].tolist()
            rptr = np.r_[0, np.cumsum(np.bincount(index, minlength=nr))].tolist()
            cols = self.cols
            for r, (a, b) in enumerate(zip(lo.tolist(), hi.tolist())):
                p, q = rptr[r], rptr[r + 1]
                e = gp.LinExpr(rvals[p:q], [cols[j] for j in rcols[p:q]])
                self.rows.append(m.addLConstr(self._expr_bound(e, a, b)))
        m.update()
        self.built = True
        self._rlo = self._rhi = self._pend = None

    def change_costs(self, idx: np.ndarray, cost: np.ndarray) -> None:
        if not self.built:
            # the buffered columns, in order
            flat = np.concatenate([c[0] for c in self._pend])
            flat[np.asarray(idx)] = cost
            k = 0
            for c in self._pend:
                n = len(c[0])
                c[0][:] = flat[k:k + n]
                k += n
            return
        self.m.setAttr("Obj", [self.cols[i] for i in np.asarray(idx).tolist()],
                       np.asarray(cost, float).tolist())

    def solve_lp(self) -> str:
        self._build()
        self.m.optimize()
        return "optimal" if self.m.Status == self.GRB.OPTIMAL else "other"

    def solve_mip(self, *, time_limit, cutoff, gap_abs, int_tol=1e-9, focus=0,
                  start: dict[int, float] | None = None) -> str:
        self._build()
        m, GRB = self.m, self.GRB
        m.setParam("TimeLimit", max(0.0, time_limit))
        if math.isfinite(cutoff):
            m.setParam("Cutoff", cutoff)
        m.setParam("MIPGap", 0.0)
        m.setParam("MIPGapAbs", gap_abs)
        m.setParam("IntFeasTol", int_tol)
        if focus:
            m.setParam("MIPFocus", focus)
        if start:
            m.setAttr("Start", [self.cols[c] for c in start], list(start.values()))
        m.optimize()
        if m.Status in (GRB.CUTOFF, GRB.INFEASIBLE) or (m.Status == GRB.OPTIMAL and m.SolCount == 0):
            return "infeasible"
        if m.Status == GRB.OPTIMAL:
            return "optimal"
        return "limit"

    def has_solution(self) -> bool:
        return self.built and self.m.SolCount > 0

    def objective(self) -> float:
        return self.m.ObjVal

    def col_values(self) -> np.ndarray:
        return np.array(self.m.getAttr("X", self.cols)) if self.cols else np.zeros(0)

    def row_duals(self) -> np.ndarray:
        return np.array(self.m.getAttr("Pi", self.rows)) if self.rows else np.zeros(0)

    def write(self, path: str) -> None:
        self._build()
        self.m.write(path)


class HighsBackend:
    name = "highs"

    def __init__(self, threads: int = 0):
        import highspy
        self.hs = highspy
        h = self.h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        if threads:
            h.setOptionValue("threads", int(threads))
        self.n_rows = 0
        self.n_cols = 0
        self.integer = False

    def add_rows(self, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        n = len(lo)
        lo = np.where(np.isfinite(lo), lo, -self.hs.kHighsInf).astype(float)
        hi = np.where(np.isfinite(hi), hi, self.hs.kHighsInf).astype(float)
        self.h.addRows(n, lo, hi, 0, np.zeros(n, dtype=np.int32), np.zeros(0, dtype=np.int32),
                       np.zeros(0))
        start = self.n_rows
        self.n_rows += n
        return np.arange(start, self.n_rows)

    def add_cols(self, cost, ub, integer, starts, index, value) -> None:
        n = len(cost)
        if not n:
            return
        self.h.addCols(n, np.asarray(cost, float), np.zeros(n), np.asarray(ub, float),
                       int(starts[-1]), np.asarray(starts[:-1], dtype=np.int32),
                       np.asarray(index, dtype=np.int32), np.asarray(value, float))
        integer = np.asarray(integer, dtype=bool)
        if integer.any():
            idx = (self.n_cols + np.nonzero(integer)[0]).astype(np.int32)
            kinds = np.array([self.hs.HighsVarType.kInteger] * len(idx))
            self.h.changeColsIntegrality(len(idx), idx, kinds)
            self.integer = True
        self.n_cols += n

    def change_costs(self, idx: np.ndarray, cost: np.ndarray) -> None:
        self.h.changeColsCost(len(idx), np.asarray(idx, dtype=np.int32), np.asarray(cost, float))

    def _status(self) -> str:
        S = self.hs.HighsModelStatus
        st = self.h.getModelStatus()
        if st == S.kOptimal:
            return "optimal"
        if st in (S.kInfeasible, S.kObjectiveBound, S.kObjectiveTarget):
            return "infeasible"
        if st in (S.kTimeLimit, S.kIterationLimit, S.kSolutionLimit, S.kInterrupt):
            return "limit"
        return "other"

    def solve_lp(self) -> str:
        self.h.setOptionValue("solver", "simplex")
        self.h.run()
        st = self._status()
        if st != "optimal" and self.h.getModelStatus() in (
                self.hs.HighsModelStatus.kNotset, self.hs.HighsModelStatus.kSolveError,
                self.hs.HighsModelStatus.kUnknown):
            # A warm-started run on a badly scaled master (big-M artificials
            # next to costs scaled to 1e4 per unit) sometimes ends in kError;
            # from scratch it solves (B_fedhpc_n100_lE_iH, f2 at f1_T).
            self.h.clearSolver()
            self.h.run()
            st = self._status()
        self.last_status = self.h.modelStatusToString(self.h.getModelStatus())
        return "optimal" if st == "optimal" else "other"

    def solve_mip(self, *, time_limit, cutoff, gap_abs, int_tol=1e-9, focus=0,
                  start: dict[int, float] | None = None) -> str:
        h = self.h
        h.setOptionValue("time_limit", float(max(0.0, time_limit)))
        h.setOptionValue("mip_rel_gap", 0.0)
        h.setOptionValue("mip_abs_gap", float(gap_abs))
        # HiGHS's integrality tolerance floor is looser than Gurobi's; the
        # caller re-checks every schedule it extracts.
        h.setOptionValue("mip_feasibility_tolerance", float(max(int_tol, 1e-7)))
        if math.isfinite(cutoff):
            h.setOptionValue("objective_bound", float(cutoff))
        if start:
            idx = np.array(sorted(start), dtype=np.int32)
            val = np.array([start[i] for i in idx.tolist()], dtype=float)
            try:
                h.setSolution(len(idx), idx, val)      # sparse start (HiGHS >= 1.8)
            except TypeError:
                pass
        h.run()
        st = self._status()
        if st == "optimal" and not self.has_solution():
            return "infeasible"
        return st

    def has_solution(self) -> bool:
        info = self.h.getInfo()
        return bool(info.primal_solution_status == 2) or (
            self.integer and info.mip_node_count >= 0 and info.primal_solution_status != 0)

    def objective(self) -> float:
        return self.h.getInfo().objective_function_value

    def col_values(self) -> np.ndarray:
        return np.asarray(self.h.getSolution().col_value, dtype=float)

    def row_duals(self) -> np.ndarray:
        return np.asarray(self.h.getSolution().row_dual, dtype=float)

    def write(self, path: str) -> None:
        self.h.writeModel(path)


# Options applied to every fedlp solver (e.g. {"factor": 3}); see _fedlp.Options.
FEDLP_OPTIONS: dict = {}


class FedlpBackend:
    """fedlp (:mod:`fedhpc._fedlp`), LPs only: the column generation master.
    Rows are buffered until the first column batch builds the solver (fedlp
    fixes its rows at construction); MIPs stay on another backend."""

    name = "fedlp"

    def __init__(self, threads: int = 0):
        from . import _fedlp
        self._fedlp = _fedlp
        self.lp = None
        self._sense: list[str] = []
        self._rhs: list[float] = []
        self.n_cols = 0
        self._status = None

    def add_rows(self, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        if self.lp is not None:
            raise NotImplementedError("fedlp takes no rows after its first columns")
        start = len(self._rhs)
        for a, b in zip(np.asarray(lo, float).tolist(), np.asarray(hi, float).tolist()):
            if a == b:
                self._sense.append("="), self._rhs.append(a)
            elif a == -INF:
                self._sense.append("<"), self._rhs.append(b)
            elif b == INF:
                self._sense.append(">"), self._rhs.append(a)
            else:
                raise NotImplementedError("fedlp has no ranged rows")
        return np.arange(start, len(self._rhs))

    def _ensure(self) -> None:
        if self.lp is None:
            self.lp = self._fedlp.Solver(self._sense, self._rhs)
            for k, v in FEDLP_OPTIONS.items():
                setattr(self.lp.options, k, v)

    def add_cols(self, cost, ub, integer, starts, index, value) -> None:
        n = len(cost)
        if not n:
            return
        if np.any(np.asarray(integer, dtype=bool)):
            raise NotImplementedError("fedlp solves LPs only")
        self._ensure()
        self.lp.add_columns(np.asarray(cost, float), np.zeros(n), np.asarray(ub, float),
                            np.asarray(starts, dtype=np.int32), np.asarray(index, dtype=np.int32),
                            np.asarray(value, float))
        self.n_cols += n

    def change_costs(self, idx: np.ndarray, cost: np.ndarray) -> None:
        self._ensure()
        self.lp.set_costs(np.asarray(idx).tolist(), np.asarray(cost, float).tolist())

    def crash_hint(self, cols) -> None:
        """Columns to start the basis from (a feasible solution's)."""
        self._ensure()
        self.lp.set_crash_hint([int(c) for c in cols])

    def solve_lp(self) -> str:
        self._ensure()
        st = self.lp.solve()
        self._status = st
        return "optimal" if st == int(self._fedlp.Status.OPTIMAL) else "other"

    def solve_mip(self, **kw) -> str:
        raise NotImplementedError("fedlp solves LPs only")

    def has_solution(self) -> bool:
        return self._status == int(self._fedlp.Status.OPTIMAL)

    def objective(self) -> float:
        return self.lp.objective

    def col_values(self) -> np.ndarray:
        return self.lp.x() if self.lp is not None else np.zeros(0)

    def row_duals(self) -> np.ndarray:
        return self.lp.duals() if self.lp is not None else np.zeros(0)

    def write(self, path: str) -> None:
        raise NotImplementedError
