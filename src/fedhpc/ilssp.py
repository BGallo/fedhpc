"""ILS-SP matheuristic for Rm||TWCT+TMUC (Chen, Chu, Sahli & Li, EJOR 2024).

The C++ engine (``_ilssp.IlsSp``) runs an Iterated Local Search over job→machine
assignments (each machine in WSPT order) and harvests the per-machine job sets
of near-best local optima. Those are columns of the paper's set-partitioning
model (Problem P):

    min Σ_s c_s x_s   s.t.  Σ_{s∋j} x_s = 1 ∀j,   Σ_{s on k} x_s ≤ 1 ∀k,   x binary

which this module solves periodically with Gurobi over the growing pool; a
better recombination is handed back to the engine as its new incumbent.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import gurobipy as gp
from gurobipy import GRB

from . import _ilssp  # type: ignore[attr-defined]
from .rm_benchmark import RmData

__all__ = ["IlsSpResult", "SetPartitioningPool", "solve"]


@dataclass
class IlsSpResult:
    cost: float
    assignment: list[int]
    time_s: float
    time_to_best_s: float
    iterations: int
    pool_size: int
    sp_calls: int
    sp_improvements: int
    history: list[tuple[float, float]] = field(default_factory=list)  # (t, best)
    # every harvested column (machine, jobs, cost) — seeds the LP bound
    columns: list[tuple[int, list[int], int]] = field(default_factory=list)


class SetPartitioningPool:
    """Restricted set-partitioning MIP grown column by column."""

    def __init__(self, n: int, m: int, threads: int = 1):
        self.n, self.m = n, m
        self.env = gp.Env(empty=True)
        self.env.setParam("OutputFlag", 0)
        self.env.start()
        self.mdl = gp.Model("ilssp_sp", env=self.env)
        self.mdl.Params.Threads = threads
        self.job_rows = [self.mdl.addConstr(gp.LinExpr() == 1, name=f"j{j}") for j in range(n)]
        self.mach_rows = [self.mdl.addConstr(gp.LinExpr() <= 1, name=f"k{k}") for k in range(m)]
        self.cols: list[tuple[int, tuple[int, ...]]] = []
        self.vars: list[gp.Var] = []
        self.index: dict[tuple[int, tuple[int, ...]], int] = {}

    def add(self, columns) -> int:
        added = 0
        for k, jobs, cost in columns:
            key = (k, tuple(jobs))
            if key in self.index:
                continue
            col = gp.Column([1.0] * (len(jobs) + 1),
                            [self.job_rows[j] for j in jobs] + [self.mach_rows[k]])
            self.index[key] = len(self.cols)
            self.cols.append(key)
            self.vars.append(self.mdl.addVar(obj=float(cost), vtype=GRB.BINARY, column=col))
            added += 1
        return added

    def solve(self, incumbent: list[int], time_limit: float) -> tuple[float, list[int]] | None:
        """Best combination found within ``time_limit`` (warm-started from
        ``incumbent``), or ``None`` if Gurobi found no feasible cover."""
        per: list[list[int]] = [[] for _ in range(self.m)]
        for j, k in enumerate(incumbent):
            per[k].append(j)
        for v in self.vars:
            v.Start = 0.0
        for k, js in enumerate(per):
            idx = self.index.get((k, tuple(js)))
            if js and idx is not None:
                self.vars[idx].Start = 1.0
        self.mdl.Params.TimeLimit = max(time_limit, 0.01)
        self.mdl.optimize()
        if self.mdl.SolCount == 0:
            return None
        assign = [-1] * self.n
        for v, (k, jobs) in zip(self.vars, self.cols):
            if v.X > 0.5:
                for j in jobs:
                    assign[j] = k
        return self.mdl.ObjVal, assign

    def close(self) -> None:
        self.mdl.dispose()
        self.env.dispose()


def solve(
    data: RmData,
    time_limit: float = 60.0,
    *,
    seed: int = 0,
    use_sp: bool = True,
    sp_interval: float | None = None,
    sp_time: float | None = None,
    target: float | None = None,
    initial: list[int] | None = None,
    verbose: bool = False,
    **engine_params,
) -> IlsSpResult:
    """Minimise Σ w_j C_j + Σ_{k used}(U_k + L_k load_k) with ILS-SP.

    ``sp_interval``: ILS seconds between set-partitioning solves (default
    ``time_limit / 10``); ``sp_time``: Gurobi time per solve (default
    ``sp_interval / 2``). Stops early when ``target`` is reached.
    ``engine_params`` go to ``_ilssp.IlsSp`` (destroy_min, destroy_max,
    accept_tol, pool_tol, restart_after, p_machine_kick, swap_k,
    deep_every).
    """
    p, w, U, L = data
    n, m = len(w), len(U)
    t0 = time.perf_counter()
    eng = _ilssp.IlsSp(p, w, U, L, seed=seed, **engine_params)
    if initial is not None:
        eng.set_solution(initial)
    else:
        eng.construct()

    sp_interval = sp_interval if sp_interval is not None else max(time_limit / 10, 0.05)
    sp_time = sp_time if sp_time is not None else sp_interval / 2
    pool = SetPartitioningPool(n, m) if use_sp else None

    best = eng.best_cost
    t_best = time.perf_counter() - t0
    history = [(t_best, float(best))]
    sp_calls = sp_improvements = 0
    columns: list = []

    def elapsed() -> float:
        return time.perf_counter() - t0

    def note(tag: str) -> None:
        nonlocal best, t_best
        if eng.best_cost < best:
            best, t_best = eng.best_cost, elapsed()
            history.append((t_best, float(best)))
            if verbose:
                print(f"  {t_best:8.2f}s  {best}  ({tag})")

    try:
        while elapsed() < time_limit and (target is None or best > target):
            left = time_limit - elapsed()
            eng.run(time_limit=min(sp_interval, left))
            note("ils")
            new = eng.take_new_columns()
            columns.extend(new)
            if pool is None:
                continue
            pool.add(new)
            left = time_limit - elapsed()
            if left <= 0 or (target is not None and best <= target):
                break
            sp_calls += 1
            res = pool.solve(eng.best_assignment(), min(sp_time, left))
            if res is not None and res[0] < eng.best_cost - 1e-6:
                sp_improvements += 1
                eng.set_solution(res[1])  # VND polish of the recombination
                new = eng.take_new_columns()
                columns.extend(new)
                pool.add(new)
                note("sp")
    finally:
        columns.extend(eng.take_new_columns())
        size = len(columns)
        if pool is not None:
            pool.close()

    return IlsSpResult(
        cost=float(eng.best_cost), assignment=list(eng.best_assignment()),
        time_s=elapsed(), time_to_best_s=t_best, iterations=eng.iterations,
        pool_size=size, sp_calls=sp_calls, sp_improvements=sp_improvements,
        history=history, columns=columns,
    )
