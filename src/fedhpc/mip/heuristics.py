"""Primal heuristics: each returns a feasible point of the problem or None.

- ``simple_rounding``: round each fractional integer in a direction no row
  locks (Achterberg 2007, sec. 9.1). Trivially cheap; finds a point only
  when the locks allow it.
- ``fix_and_propagate``: fix integers one decision at a time with
  propagation (binaries by LP value, at 1 first), a depth-first search with
  conflict-directed backjumping on dead ends; the continuous part is an
  LP. Close to SCIP's and Gurobi's fix-and-propagate family.
- ``guided_dive``: repeatedly fix the integer closest to integrality (in
  the incumbent's direction when there is one), propagate, re-solve the LP
  warm with an iteration limit; one flip on infeasibility (Achterberg 2007,
  sec. 9.2; Danna, Rothberg & Le Pape 2005 for the guidance).
- ``rens``: a sub-MIP with the LP-integral integers fixed and the others
  bounded to their LP value's floor and ceiling (Berthold 2014).
- ``rins``: a sub-MIP with the integers fixed where the LP point and the
  incumbent agree (Danna, Rothberg & Le Pape 2005).

All work on the problem the solver holds (presolved), never on its LP's
state permanently: the dive restores bounds and basis.
"""
from __future__ import annotations

import time

import numpy as np

from .lp import OPTIMAL, LpRelaxation
from .problem import INF, MipProblem
from .propagate import Propagator


def locks(P: MipProblem) -> tuple[np.ndarray, np.ndarray]:
    """Per column: rows that forbid decreasing it (down) and increasing it (up)."""
    A = P.A
    rr = np.repeat(np.arange(P.m), np.diff(A.indptr))
    a = A.data
    fin_lo = np.isfinite(P.row_lo)[rr]
    fin_hi = np.isfinite(P.row_hi)[rr]
    down = np.bincount(A.indices, weights=(((a > 0) & fin_lo) | ((a < 0) & fin_hi)).astype(float), minlength=P.n)
    up = np.bincount(A.indices, weights=(((a > 0) & fin_hi) | ((a < 0) & fin_lo)).astype(float), minlength=P.n)
    return down, up


def simple_rounding(P: MipProblem, x: np.ndarray, down_locks, up_locks, tol: float = 1e-6):
    """Round each fractional integer in a direction no row locks; None when
    some fractional column is locked both ways (vectorised: it runs at every
    node)."""
    ii = P.int_idx if hasattr(P, "int_idx") else np.nonzero(P.integer)[0]
    xi = x[ii]
    f = xi - np.floor(xi)
    frac = (f > tol) & (f < 1 - tol)
    if not np.any(frac):
        y = x.copy()
        y[ii] = np.round(xi)
        return y if P.is_feasible(y, tol) else None
    fj = ii[frac]
    dn, upk = down_locks[fj] == 0, up_locks[fj] == 0
    if not np.all(dn | upk):
        return None
    y = x.copy()
    y[ii] = np.round(xi)
    y[fj] = np.where(dn, np.floor(x[fj]), np.ceil(x[fj]))
    return y if P.is_feasible(y, tol) else None


def _continuous_completion(P: MipProblem, lp: LpRelaxation, lb, ub, iteration_limit=None):
    """With every integer fixed in [lb, ub], solve the LP for the rest; the
    LP's bounds and basis come back as they were."""
    if not np.any(~P.integer):
        y = lb.copy()
        return y if P.is_feasible(y) else None
    saved_lb, saved_ub, basis = lp.lb.copy(), lp.ub.copy(), lp.get_state()
    try:
        lp.set_bounds(lb, ub)
        if lp.solve(iteration_limit) != OPTIMAL:
            return None
        y = lp.x()
        y[P.integer] = np.round(y[P.integer])
        return y if P.is_feasible(y) else None
    finally:
        lp.set_bounds(saved_lb, saved_ub)
        lp.set_basis(basis)


def fix_and_propagate(P: MipProblem, prop: Propagator, lp: LpRelaxation, x: np.ndarray,
                      lb: np.ndarray, ub: np.ndarray, deadline: float, tol: float = 1e-6,
                      max_backjumps: int = 200, stats: dict | None = None):
    """Fix the integers one decision at a time with propagation, then
    complete the continuous part by LP; a depth-first search with
    conflict-directed backjumping (Prosser 1993) on dead ends.

    Order and values: the LP-integral binaries at 1 first, in one batch;
    then binaries by LP value descending, each tried at 1 then 0 (in
    assignment and partitioning rows a member at 1 settles the row, where
    rounding every member below 1/2 down empties it); general integers
    nearest-rounded, then the other rounding; the remaining binaries at 0
    in one batch (one by one if that fails). A decision whose both values
    propagate to an empty domain is a dead end: the search jumps back to
    the latest earlier decision on a column of the rows that failed (their
    conflict set; chronologically when none is), flips it, and resumes from
    there. ``max_backjumps`` bounds the search; ``lb, ub`` must be a
    propagation fixed point (the caller's domain)."""
    prop.reset(lb, ub, propagate=False)
    A = P.A
    ii = np.nonzero(P.integer & (lb < ub))[0]
    binary = (lb[ii] == 0) & (ub[ii] == 1)
    xb = x[ii]
    one = ii[binary & (xb >= 1 - tol)]
    up = ii[binary & (xb > tol) & (xb < 1 - tol)]
    up = up[np.argsort(-x[up], kind="stable")]
    gen = ii[~binary]
    gen = gen[np.argsort(np.abs(x[gen] - np.round(x[gen])), kind="stable")]
    rest = ii[binary & (xb <= tol)]
    if len(one) and not prop.change(one, np.ones(len(one)), np.ones(len(one))):
        prop.reset(lb, ub, propagate=False)
        up = np.r_[one, up]                       # not all together: decide them singly
    order = np.r_[up, gen].astype(np.int64)
    first = np.where(np.isin(order, ii[binary]), 1.0, np.clip(np.round(x[order]), lb[order], ub[order]))

    def alt(k, v):
        j = order[k]
        if lb[j] == 0 and ub[j] == 1 and P.integer[j]:
            return 1.0 - v
        xj = x[j]
        w = np.floor(xj) if v > xj else np.ceil(xj)
        if w == v:
            w = v - 1 if v - 1 >= lb[j] else v + 1
        return float(np.clip(w, lb[j], ub[j]))

    stack: list[tuple[int, int, float, bool]] = []   # (trail level, order index, value, flipped)
    k, jumps = 0, 0

    def conflict_cols():
        r = prop.infeasible_row
        if r < 0 or r >= P.m:
            return set()
        return set(A.indices[A.indptr[r]:A.indptr[r + 1]].tolist())

    def decide(kk, v):
        j = int(order[kk])
        return prop.change([j], [v], [v])

    while True:
        while k < len(order):
            if time.perf_counter() > deadline:
                return None
            j = int(order[k])
            l_j, u_j = prop.k.bound(j) if hasattr(prop, "k") else (prop.lb()[j], prop.ub()[j])
            if l_j == u_j:
                k += 1
                continue
            v = float(np.clip(first[k], l_j, u_j))
            lev = prop.level()
            if decide(k, v):
                stack.append((lev, k, v, False))
                k += 1
                continue
            conflict = conflict_cols()
            prop.undo(lev)
            w = alt(k, v)
            if l_j <= w <= u_j and w != v and decide(k, w):
                stack.append((lev, k, w, True))
                k += 1
                continue
            conflict |= conflict_cols()
            prop.undo(lev)
            # dead end at k: jump back to the latest unflipped decision in the conflict
            jumps += 1
            if jumps > max_backjumps:
                return None
            target = None
            for t in range(len(stack) - 1, -1, -1):
                if not stack[t][3] and int(order[stack[t][1]]) in conflict:
                    target = t
                    break
            if target is None:
                for t in range(len(stack) - 1, -1, -1):
                    if not stack[t][3]:
                        target = t
                        break
            if target is None:
                return None                         # every decision flipped: give up
            lev_t, k_t, v_t, _f = stack[target]
            del stack[target:]
            prop.undo(lev_t)
            w = alt(k_t, v_t)
            if not decide(k_t, w):
                # the flip fails at once: revisit k_t trying w first, then
                # v_t again (as flipped), so the next dead end jumps past it
                prop.undo(lev_t)
                first[k_t] = w
                k = k_t
                continue
            stack.append((lev_t, k_t, w, True))
            k = k_t + 1
        # the binaries left open at 0 (one batch; singly when it fails)
        lnow, unow = prop.lb(), prop.ub()
        open_rest = rest[lnow[rest] < unow[rest]]
        if len(open_rest):
            lev = prop.level()
            if not prop.change(open_rest, np.zeros(len(open_rest)), np.zeros(len(open_rest))):
                prop.undo(lev)
                ok = True
                for j in open_rest:
                    if time.perf_counter() > deadline:
                        return None
                    lj, uj = prop.k.bound(j) if hasattr(prop, "k") else (prop.lb()[j], prop.ub()[j])
                    if lj == uj:
                        continue
                    lv = prop.level()
                    if not prop.change([j], [0.0], [0.0]):
                        prop.undo(lv)
                        if not prop.change([j], [1.0], [1.0]):
                            ok = False
                            break
                if not ok:
                    return None
        if stats is not None:
            stats["fp_backjumps"] = stats.get("fp_backjumps", 0) + jumps
        return _continuous_completion(P, lp, prop.lb(), prop.ub())


def guided_dive(P: MipProblem, prop: Propagator, lp: LpRelaxation, lb, ub, deadline: float,
                incumbent: np.ndarray | None = None, max_depth: int = 400,
                iteration_limit: int = 2000, tol: float = 1e-6, rule: str = "up"):
    """rule (without an incumbent): "up" rounds binaries up, the largest
    first (partitioning rows); "nearest" rounds the most nearly integral
    integer to its nearest value (fractional diving)."""
    saved_lb, saved_ub, basis = lp.lb.copy(), lp.ub.copy(), lp.get_state()
    lb, ub = lb.copy(), ub.copy()
    prop.reset(lb, ub, propagate=False)
    try:
        for _depth in range(max_depth):
            if time.perf_counter() > deadline:
                return None
            x = lp.x()
            ii = np.nonzero(P.integer)[0]
            f = x[ii] - np.floor(x[ii])
            frac = (f > tol) & (f < 1 - tol)
            if not np.any(frac):
                y = x.copy()
                y[ii] = np.round(y[ii])
                if P.is_feasible(y):
                    return y
                return _continuous_completion(P, lp, *_fix_integers(P, y, lb, ub))
            cand = ii[frac]
            fc = f[frac]
            if incumbent is not None:
                up = incumbent[cand] > x[cand]
                dist = np.where(up, 1 - fc, fc)
            else:
                # binaries round up, the largest first (a partitioning
                # row's member at 1 settles the row); others to nearest
                isbin = (lb[cand] == 0) & (ub[cand] == 1) & (rule == "up")
                up = np.where(isbin, True, fc >= 0.5)
                dist = np.where(isbin, 1 - fc, np.minimum(fc, 1 - fc))
            k = int(np.argmin(dist))
            j, go_up = int(cand[k]), bool(up[k])
            ok = False
            for side in (go_up, not go_up):
                lev = prop.level()
                v = np.ceil(x[j]) if side else np.floor(x[j])
                if not prop.change([j], [v if side else -INF], [INF if side else v], max_rounds=5):
                    prop.undo(lev)
                    continue
                nl, nu = prop.lb(), prop.ub()
                lp.set_bounds(nl, nu)
                if lp.solve(iteration_limit) == OPTIMAL:
                    lb, ub = nl, nu
                    ok = True
                    break
                prop.undo(lev)
            if not ok:
                return None
        return None
    finally:
        lp.set_bounds(saved_lb, saved_ub)
        lp.set_basis(basis)


def _fix_integers(P, y, lb, ub):
    l, u = lb.copy(), ub.copy()
    l[P.integer] = u[P.integer] = np.clip(np.round(y[P.integer]), lb[P.integer], ub[P.integer])
    return l, u


def rens_problem(P: MipProblem, x: np.ndarray, lb, ub, tol: float = 1e-6, min_fixed: float = 0.5):
    """The RENS sub-MIP's bounds, or None when too few integers are fixed."""
    ii = np.nonzero(P.integer & (lb < ub))[0]
    if len(ii) == 0:
        return None
    fl, ce = np.floor(x[ii] + tol), np.ceil(x[ii] - tol)
    nl, nu = lb.copy(), ub.copy()
    nl[ii] = np.maximum(lb[ii], fl)
    nu[ii] = np.minimum(ub[ii], ce)
    if np.mean(nl[ii] == nu[ii]) < min_fixed:
        return None
    Q = P.copy()
    Q.lb, Q.ub = nl, nu
    Q._csc = P._csc
    return Q


def rins_problem(P: MipProblem, x: np.ndarray, incumbent: np.ndarray, lb, ub, tol: float = 1e-6,
                 min_fixed: float = 0.5):
    """The RINS sub-MIP (Danna, Rothberg & Le Pape 2005): integers on which
    the LP point and the incumbent agree are fixed there; the rest keep the
    domain. None when fewer than ``min_fixed`` of the open integers agree
    (the sub-MIP would be the problem again)."""
    ii = np.nonzero(P.integer & (lb < ub))[0]
    if len(ii) == 0:
        return None
    agree = np.abs(x[ii] - incumbent[ii]) <= tol
    if np.mean(agree) < min_fixed or np.all(agree):
        return None
    Q = P.copy()
    Q.lb, Q.ub = lb.copy(), ub.copy()
    fj = ii[agree]
    Q.lb[fj] = Q.ub[fj] = np.clip(np.round(incumbent[fj]), lb[fj], ub[fj])
    Q._csc = P._csc
    return Q


def feasibility_pump(P: MipProblem, lp: LpRelaxation, x: np.ndarray, lb, ub, deadline: float,
                     max_rounds: int = 200, alpha0: float = 1.0, decay: float = 0.9,
                     seed: int = 0, tol: float = 1e-6, prop: Propagator | None = None):
    """The objective feasibility pump (Fischetti, Glover & Lodi 2005;
    Achterberg & Berthold 2007) over the binaries.

    Each round rounds the LP point to x~ and solves the LP whose objective
    is the L1 distance to x~ on binaries (x_j where x~_j = 0, 1 - x_j where
    x~_j = 1), mixed with the original objective at weight alpha (scaled to
    the distance's norm), alpha decaying by ``decay``. The pump stops when
    the LP point is integral on the binaries. When the rounding repeats, the
    binaries farthest from it are flipped (T in [10, 30]); a cycle gets a
    random flip of a handful. General integers are rounded, never part of
    the distance. With a propagator, a point with few fractional binaries
    (at most 20, or 5 %) is rounded by fix-and-propagate instead (the
    propagation rounding of Fischetti & Salvagnin's FP 2.0). The LP's costs,
    bounds and basis come back as they were."""
    binary = np.nonzero(P.integer & (lb == 0) & (ub == 1))[0]
    if len(binary) == 0:
        return None
    rng = np.random.default_rng(seed)
    n = P.n
    ids = list(range(n))
    c0 = P.c.copy()
    cnorm = float(np.linalg.norm(c0))
    scale = np.sqrt(len(binary)) / cnorm if cnorm > 0 else 0.0
    neg = np.nonzero(lp.neg_id >= 0)[0]
    saved_lb, saved_ub, basis = lp.lb.copy(), lp.ub.copy(), lp.get_state()

    def set_obj(c):
        lp.lp.set_costs(ids, c.tolist())
        if len(neg):
            lp.lp.set_costs(lp.neg_id[neg].tolist(), (-c[neg]).tolist())

    seen = set()
    alpha = alpha0
    xt = np.round(x)
    try:
        lp.set_bounds(lb, ub)
        for _rnd in range(max_rounds):
            if time.perf_counter() > deadline:
                return None
            xb = xt[binary]
            cost = alpha * scale * c0
            cost[binary] += np.where(xb > 0.5, -1.0, 1.0)
            set_obj(cost)
            if lp.solve(5000) != OPTIMAL:
                return None
            xs = lp.x()
            fb = np.abs(xs[binary] - np.round(xs[binary]))
            nfrac = int(np.sum(fb > tol))
            if prop is not None and 0 < nfrac <= max(20, len(binary) // 20):
                set_obj(c0)
                y = fix_and_propagate(P, prop, lp, xs, lb, ub, deadline, tol)
                if y is not None:
                    return y
                set_obj(cost)
            if np.all(fb <= tol):
                y = xs.copy()
                y[P.integer] = np.round(y[P.integer])
                if P.is_feasible(y):
                    return y
                fixed = _continuous_completion_costs(P, lp, y, lb, ub, set_obj, c0)
                if fixed is not None:
                    return fixed
            alpha *= decay
            new = np.round(xs)
            if np.array_equal(new[binary], xt[binary]):
                # stalled: flip the T binaries farthest from the rounding
                d = np.abs(xs[binary] - xt[binary])
                T = int(rng.integers(10, 31))
                flip = binary[np.argsort(-d)[:T]]
                new[flip] = 1 - xt[flip]
            key = hash(new[binary].tobytes())
            if key in seen:
                k = max(1, len(binary) // 20)
                flip = rng.choice(binary, size=min(k, len(binary)), replace=False)
                new[flip] = 1 - new[flip]
            seen.add(hash(new[binary].tobytes()))
            xt = new
        return None
    finally:
        set_obj(c0)
        lp.set_bounds(saved_lb, saved_ub)
        lp.set_basis(basis)


def _continuous_completion_costs(P, lp, y, lb, ub, set_obj, c0):
    """The pump's point is integral on binaries but infeasible for the rest:
    fix the integers and solve the true objective's LP."""
    set_obj(c0)
    l, u = _fix_integers(P, y, lb, ub)
    lp.set_bounds(l, u)
    try:
        if lp.solve(20000) != OPTIMAL:
            return None
        z = lp.x()
        z[P.integer] = np.round(z[P.integer])
        return z if P.is_feasible(z) else None
    finally:
        lp.set_bounds(lb, ub)
