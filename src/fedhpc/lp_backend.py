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
    raise ValueError(f"unknown LP backend {name!r} (gurobi, highs)")


def available(name: str) -> bool:
    try:
        if name == "gurobi":
            import gurobipy  # noqa: F401
        elif name == "highs":
            import highspy  # noqa: F401
        else:
            return False
    except ImportError:
        return False
    return True


class GurobiBackend:
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

    def add_rows(self, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        gp, m = self.gp, self.m
        start = len(self.rows)
        for a, b in zip(lo.tolist(), hi.tolist()):
            if a == b:
                self.rows.append(m.addConstr(gp.LinExpr() == a))
            elif a == -INF:
                self.rows.append(m.addConstr(gp.LinExpr() <= b))
            else:
                self.rows.append(m.addConstr(gp.LinExpr() >= a))
        m.update()
        return np.arange(start, len(self.rows))

    def add_cols(self, cost, ub, integer, starts, index, value) -> None:
        gp, GRB, m, rows = self.gp, self.GRB, self.m, self.rows
        cost, ub, integer = cost.tolist(), ub.tolist(), np.asarray(integer).tolist()
        starts, index, value = starts.tolist(), index.tolist(), value.tolist()
        for c in range(len(cost)):
            lo, hi = starts[c], starts[c + 1]
            col = gp.Column(value[lo:hi], [rows[r] for r in index[lo:hi]])
            self.cols.append(m.addVar(obj=cost[c], ub=ub[c],
                                      vtype=GRB.INTEGER if integer[c] else GRB.CONTINUOUS,
                                      column=col))

    def change_costs(self, idx: np.ndarray, cost: np.ndarray) -> None:
        for i, c in zip(idx.tolist(), cost.tolist()):
            self.cols[i].Obj = c

    def solve_lp(self) -> str:
        self.m.optimize()
        return "optimal" if self.m.Status == self.GRB.OPTIMAL else "other"

    def solve_mip(self, *, time_limit, cutoff, gap_abs, int_tol=1e-9, focus=0,
                  start: dict[int, float] | None = None) -> str:
        m, GRB = self.m, self.GRB
        m.setParam("TimeLimit", max(0.0, time_limit))
        if math.isfinite(cutoff):
            m.setParam("Cutoff", cutoff)
        m.setParam("MIPGap", 0.0)
        m.setParam("MIPGapAbs", gap_abs)
        m.setParam("IntFeasTol", int_tol)
        if focus:
            m.setParam("MIPFocus", focus)
        for c, v in (start or {}).items():
            self.cols[c].Start = v
        m.optimize()
        if m.Status in (GRB.CUTOFF, GRB.INFEASIBLE) or (m.Status == GRB.OPTIMAL and m.SolCount == 0):
            return "infeasible"
        if m.Status == GRB.OPTIMAL:
            return "optimal"
        return "limit"

    def has_solution(self) -> bool:
        return self.m.SolCount > 0

    def objective(self) -> float:
        return self.m.ObjVal

    def col_values(self) -> np.ndarray:
        return np.array(self.m.getAttr("X", self.cols)) if self.cols else np.zeros(0)

    def row_duals(self) -> np.ndarray:
        return np.array(self.m.getAttr("Pi", self.rows)) if self.rows else np.zeros(0)

    def write(self, path: str) -> None:
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
