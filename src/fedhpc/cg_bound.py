"""Column-generation lower bound for Rm||TWCT+TMUC (root LP of Chen et al.'s BaP).

Restricted master (LP relaxation of the paper's Problem P):

    min Σ_s c_s x_s   s.t.  Σ_{s∋j} x_s = 1  (β_j),   Σ_{s on k} x_s ≤ 1  (α_k ≤ 0),   x ≥ 0

Pricing is the paper's WSPT DP (C++ ``_rmprice``) with each machine's load
capped at ``rm.machine_horizons(data, ub)``, which every solution of cost ≤ UB
respects. The reported bound is the Lagrangian bound

    L(β) = Σ_j β_j + Σ_k min(0, min_S [c(k,S) − β(S)])

which is valid for *any* β (each machine picks its best subset or none), so it
certifies a bound at every iteration and allows Wentges dual smoothing. On the
restricted column set it lower-bounds every solution of cost ≤ UB, hence
``min(UB, ⌈L⌉)`` is a valid lower bound on the optimum (costs are integral).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import gurobipy as gp
from gurobipy import GRB

from . import _rmprice  # type: ignore[attr-defined]
from . import rm_benchmark as rm
from .rm_benchmark import RmData

__all__ = ["BoundResult", "lp_bound"]


@dataclass
class BoundResult:
    lb: int                 # valid integer lower bound on the optimum
    lagrangian: float       # best Lagrangian bound (before rounding / UB cap)
    rmp: float              # last restricted-master LP value (≥ root LP)
    lp_optimal: bool        # column generation converged (rmp == root LP)
    proven_optimal: bool    # lb == ub
    iterations: int
    columns: int
    time_s: float
    pricing_s: float
    lp_s: float
    subgradient_lb: float = -math.inf   # bound reached by the subgradient phase


def lp_bound(
    data: RmData,
    ub: float,
    columns=(),
    *,
    time_limit: float = 300.0,
    smoothing: float = 0.8,
    per_machine: int = 5,
    horizons: list[int] | None = None,
    incumbent: list[int] | None = None,
    method: int = 0,
    subgradient_iters: int = 200000,
    threads: int = 1,
    verbose: bool = False,
) -> BoundResult:
    """Root column-generation bound, stopping at convergence, at ``lb == ub``
    or after ``time_limit`` seconds (the bound is valid at any stop).

    ``columns``: initial (machine, jobs, cost) patterns, e.g. the ILS-SP pool.
    ``incumbent``: job→machine assignment of the UB solution; its cost
    shares β_j = w_j C_j + L_k p_jk + U_k/|S_k| seed the smoothing center,
    which avoids the long dual-convergence phase of a degenerate master.
    ``method``: Gurobi LP method for the master (0 primal, 1 dual simplex,
    2 barrier without crossover — interior duals, best on degenerate masters).
    ``subgradient_iters``: Polyak subgradient steps on L(β) before column
    generation (0 disables). Each step is one pricing call — orders of
    magnitude cheaper than a master LP. It alone proves most instances;
    subproblem solutions within the remaining gap seed the master.
    """
    t0 = time.perf_counter()
    p, w, U, L = data
    n, m = len(w), len(U)
    ub_int = int(round(ub))
    H = horizons if horizons is not None else rm.machine_horizons(data, ub)
    pricer = _rmprice.Pricer(p, w, U, L)

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    mdl = gp.Model("rm_cg", env=env)
    mdl.Params.Threads = threads
    mdl.Params.Method = method
    mdl.Params.Presolve = 0        # keep the simplex warm start between solves
    if method == 2:
        mdl.Params.Crossover = 0   # interior duals: well-centred, less degenerate
    job_rows = [mdl.addConstr(gp.LinExpr() == 1) for _ in range(n)]
    mach_rows = [mdl.addConstr(gp.LinExpr() <= 1) for _ in range(m)]
    seen: set[tuple[int, tuple[int, ...]]] = set()

    def add(cols) -> int:
        added = 0
        for k, jobs, cost, *_ in cols:
            key = (k, tuple(jobs))
            if key in seen:
                continue
            seen.add(key)
            col = gp.Column([1.0] * (len(jobs) + 1),
                            [job_rows[j] for j in jobs] + [mach_rows[k]])
            mdl.addVar(obj=float(cost), lb=0.0, column=col)
            added += 1
        return added

    add(columns)
    # Expensive per-job slacks keep the master feasible whatever the seed
    # pool; they drop out of the basis once real columns cover every job.
    big = float(sum(w) * sum(max(r) for r in p) + sum(U) + sum(L) * sum(max(r) for r in p))
    for j in range(n):
        mdl.addVar(obj=big, lb=0.0, column=gp.Column([1.0], [job_rows[j]]))

    best_lb = -math.inf
    center: list[float] | None = None
    if incumbent is not None:
        center = _cost_shares(data, incumbent)
    sg_lb = -math.inf
    if subgradient_iters > 0:
        t1 = time.perf_counter()
        beta0 = center if center is not None else [0.0] * n
        sg_beta, sg_lb, sg_cols = _subgradient(
            pricer, beta0, H, ub_int, subgradient_iters,
            deadline=t0 + time_limit, verbose=verbose)
        t_sg = time.perf_counter() - t1
        # Keep only columns within the duality gap of the best duals: any
        # column of a solution cheaper than UB has c − β(S) ≤ UB − L(β) + v_k
        # (v_k = its machine's Lagrangian subproblem value, ≤ 0).
        mins, _ = pricer.price(sg_beta, [0.0] * m, H, 0)
        slack = ub_int - sg_lb
        n_sg = len(sg_cols)
        sg_cols = [c for c in sg_cols
                   if c[2] - sum(sg_beta[j] for j in c[1]) <= min(0.0, mins[c[0]]) + slack + 1e-6]
        add(sg_cols)
        best_lb, center = sg_lb, sg_beta
        if verbose:
            print(f"  subgradient: lb {sg_lb:.2f}  {t_sg:.1f}s  kept {len(sg_cols)}/{n_sg} cols")
    rmp = math.inf
    it = 0
    lp_optimal = False
    t_price = t_lp = 0.0

    def lagrangian(beta, mins) -> float:
        return sum(beta) + sum(min(0.0, v) for v in mins)

    def done() -> bool:
        return best_lb > -math.inf and math.ceil(best_lb - 1e-6) >= ub_int

    try:
        while time.perf_counter() - t0 < time_limit and not done():
            it += 1
            mdl.Params.TimeLimit = max(time_limit - (time.perf_counter() - t0), 0.01)
            t1 = time.perf_counter()
            mdl.optimize()
            t_lp += time.perf_counter() - t1
            if mdl.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL) or mdl.SolCount == 0:
                break
            rmp = mdl.ObjVal
            beta = [c.Pi for c in job_rows]
            alpha = [c.Pi for c in mach_rows]

            added = 0
            if center is not None and smoothing > 0:
                sb = [smoothing * c + (1 - smoothing) * b for c, b in zip(center, beta)]
                t1 = time.perf_counter()
                mins, cols = pricer.price(sb, alpha, H, per_machine)
                t_price += time.perf_counter() - t1
                lag = lagrangian(sb, mins)
                if lag > best_lb:
                    best_lb, center = lag, sb
                # keep only columns that price out at the true duals
                added = add(c for c in cols
                            if c[2] - alpha[c[0]] - sum(beta[j] for j in c[1]) < -1e-6)
            if added == 0:  # no smoothing, or a mispricing: price at the RMP duals
                t1 = time.perf_counter()
                mins, cols = pricer.price(beta, alpha, H, per_machine)
                t_price += time.perf_counter() - t1
                lag = lagrangian(beta, mins)
                if lag > best_lb:
                    best_lb, center = lag, beta
                added = add(cols)
                if added == 0:
                    # no column prices out: L(β) == master value at exact
                    # optimal duals; with barrier duals only L(β) is certified
                    lp_optimal = True
                    break
            if verbose:
                print(f"  it {it:4d}  rmp {rmp:14.2f}  lb {best_lb:14.2f}  "
                      f"cols {len(seen):6d}  +{added}  simplex {int(mdl.IterCount):6d} "
                      f"in {mdl.Runtime:.3f}s  {time.perf_counter() - t0:7.1f}s")
    finally:
        mdl.dispose()
        env.dispose()

    lb = min(ub_int, math.ceil(best_lb - 1e-6)) if best_lb > -math.inf else 0
    return BoundResult(
        lb=lb, lagrangian=best_lb, rmp=rmp, lp_optimal=lp_optimal,
        proven_optimal=lb >= ub_int, iterations=it, columns=len(seen),
        time_s=time.perf_counter() - t0, pricing_s=t_price, lp_s=t_lp,
        subgradient_lb=sg_lb,
    )


def _cost_shares(data: RmData, machine_of: list[int]) -> list[float]:
    """β_j = w_j C_j + L_k p_jk + U_k/|S_k| in the given WSPT schedule: the
    shares sum to each machine's column cost."""
    p, w, U, L = data
    beta = [0.0] * len(w)
    for k, js in enumerate(rm._wspt_sequences(data, machine_of)):
        t = 0
        for j in js:
            t += p[j][k]
            beta[j] = w[j] * t + L[k] * p[j][k] + U[k] / len(js)
    return beta


def _subgradient(pricer, beta0, H, ub, iters, *, deadline, verbose=False,
                 theta=0.1, patience=100, theta_min=1e-8):
    """Polyak subgradient ascent on L(β) = Σβ + Σ_k min(0, min_S c − β(S)).

    Returns (best β, best L, every distinct subproblem column seen). The
    subgradient is g_j = 1 − #{chosen machine subsets containing j}; θ halves
    after ``patience`` steps without improving the best bound. The Polyak
    target is UB + 1: aiming exactly at UB lets the step vanish just short of
    the ⌈L⌉ ≥ UB proof when UB is optimal and the LP gap is zero.
    """
    n, m = len(beta0), len(H)
    zero = [0.0] * m
    beta = list(beta0)
    best_beta, best = list(beta), -math.inf
    cols: dict[tuple[int, tuple[int, ...]], tuple] = {}
    stall = 0
    for it in range(iters):
        if time.perf_counter() >= deadline:
            break
        mins, chosen = pricer.price(beta, zero, H, 1)
        lag = sum(beta) + sum(min(0.0, v) for v in mins)
        for c in chosen:
            cols.setdefault((c[0], tuple(c[1])), c)
        if lag > best + 1e-9:
            best, best_beta, stall = lag, list(beta), 0
        else:
            stall += 1
            if stall >= patience:
                theta, stall = theta / 2, 0
                beta = list(best_beta)       # restart from the best point
                if theta < theta_min:
                    break
                continue
        if math.ceil(best - 1e-6) >= ub:
            break
        g = [1.0] * n
        for c in chosen:
            for j in c[1]:
                g[j] -= 1.0
        norm = sum(x * x for x in g)
        if norm == 0:                     # subproblem solution is a partition
            best = max(best, lag)
            break
        step = theta * (ub + 1 - lag) / norm
        beta = [b + step * x for b, x in zip(beta, g)]
        if verbose and it % 200 == 0:
            print(f"    sg {it:5d}  L {lag:14.2f}  best {best:14.2f}  θ {theta:.2e}")
    return best_beta, best, list(cols.values())
