"""LP-first exact engine for large, near-integral FED-HPC instances (BPC plan,
Phase 5).

On the real-trace instances (964 jobs, on-prem types with hundreds of
identical instances, ``H`` of a few hundred slots) the compact model's LP
bound is essentially the integer optimum: a gap of about 4e-4 % for min f1
and 7e-3 % for a Pareto box step (the "Bespoke FED-HPC solver" note). What
makes Gurobi slow there is the size of the compact LP (7.6 M variables;
404 s of barrier for one box step), not branching. This engine solves the
same LP by column generation and then proves integer optimality without a
search tree.

Master
------
The time-indexed form of the compact model. On a capacitated type the
space-time network's flow through slot ``τ`` is ``N_m(τ)``; its wait arcs
carry ``N_m(τ) - busy(τ)``, so a placement vector is a feasible flow exactly
when no slot is over capacity. The LP and integer optima therefore match
``SpaceTimeFormulation``'s on the same start sets. Rows:

- coverage ``sum_{m,t} x_jmt = 1`` (dual ``σ_j``), each with a big-M
  artificial;
- capacity ``sum_{j, t <= τ < t + p_jm} x_jmt <= N_m(τ)`` (dual
  ``π_mτ <= 0``) per path type and slot. ``N_m(τ)`` counts the instances
  released by ``τ``, the idle ones plus those freed by running jobs (the
  release groups of :func:`fedhpc.bpc._prepare`). Unbounded types (cloud)
  have no capacity row;
- the optional bound rows ``f1 <= B1`` and ``f2 <= B2`` (duals ``ν <= 0``).

A column is one placement ``(j, m, t)``. Start sets are the exact
reductions of :class:`fedhpc.bpc._Prep`, which stay valid under both bound
rows. Instances with fixed activation costs are not supported; use
:mod:`fedhpc.bpc` for those.

Pricing and the Lagrangian bound
--------------------------------
A placement's reduced cost is
``(w1 - ν1) c1 + (w2 - ν2) c2 - σ_j - sum_{τ in [t, t+p)} π_mτ``. All of
them (about 1.3 M on the 964-job trace) are priced at once with NumPy
prefix sums. Keeping coverage as "each job picks one placement" gives the
Lagrangian bound

    L(π, ν) = sum_j min_{m,t} [rc_jmt + σ_j] + sum π_mτ N_m(τ) + ν1 B1 + ν2 B2,

valid for any ``π, ν <= 0``. Pricing uses Wentges smoothing around the
best-bound duals, as the BPC engine does, and falls back to the RMP duals
on a mispricing.

Proof of optimality without branching
-------------------------------------
For any feasible schedule ``x`` and the duals of any bound ``L``,
``obj(x) >= L + sum_j (rc_j(x) - min_j)``, where ``min_j`` is job ``j``'s
least reduced cost. Every term in the sum is nonnegative. With an incumbent
of value ``UB`` and the objective on a grid of step ``s``
(:func:`fedhpc.bpc._objective_step`):

1. If ``L > UB - s``, the incumbent is optimal. Column generation stops as
   soon as this holds, usually long before the LP converges.
2. Otherwise, the restricted master's columns are solved as a MIP, with
   cutoff ``UB - s``, to improve the incumbent; then step 1 is retried.
3. Otherwise, every schedule better than ``UB`` uses only placements with
   ``rc - min_j < UB - L`` (*eligible* placements). A MIP over the eligible
   placements, with cutoff ``UB - s``, is therefore exact. Either it finds
   the optimum, or it proves that nothing beats ``UB``.

Step 3 is where the tiny LP gap pays off. The eligible set is a small
fraction of all placements, so this MIP is far smaller than the compact
model.

Results
-------
Measured 2026-11 with ``scripts/lpfirst_bench.py``. Each instance gets
five lex-min solves: both anchors and three f1-boxes. Every solve proves
the same (f1, f2) as the compact Gurobi model.

============================  ==========  =========
Instances                     lpfirst     Gurobi
============================  ==========  =========
100-400-job trace slices (5)  11.5 s      136 s
full 964-job trace            7.0 s       107 s
pos_congestion (3340 jobs)    16.7 s      421 s
============================  ==========  =========

``true_pareto_frontier(engine="lpfirst")`` proves 461 points of the
964-job front in 30 min, about 4 s per point, with no inconclusive box.
That front is dense, about one point per f1 unit.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from . import bpc, formulations
from .data import Instance
from .model import Solution, grid_floor

__all__ = ["LPFirstOptions", "LPFirstResult", "solve_lpfirst", "solve_weighted_sum_lpfirst",
           "solve_f1_lpfirst", "solve_f2_lpfirst", "make_lpfirst_lexmin_bounded"]


@dataclass(frozen=True)
class LPFirstOptions:
    """``smoothing`` — Wentges weight on the best-bound duals (0 disables).
    ``per_job`` — columns added per job per iteration (its most negative
    placements). ``max_cols`` — columns added per iteration in total.
    ``tol`` — reduced-cost tolerance. ``rmp_mip_time`` — seconds for the
    restricted-master MIP (step 2). ``max_eligible`` — the most placements
    the exact MIP (step 3) may have; past it the result is not certified.
    ``mip_threads`` — Gurobi threads for both MIPs (0: Gurobi's default).
    """
    smoothing: float = 0.5
    per_job: int = 3
    max_cols: int = 20_000
    tol: float = 1e-6
    rmp_mip_time: float = 30.0
    max_eligible: int = 400_000
    mip_threads: int = 0


@dataclass
class LPFirstResult:
    """``status``: ``"optimal"`` (proven), ``"feasible"`` (``assignment`` is
    the best schedule found, ``bound`` a valid lower bound) or
    ``"infeasible"``. ``how`` — which step proved it (``"bound"``,
    ``"rmp_mip"``, ``"eligible_mip"``)."""
    status: str
    objective: float
    assignment: dict[int, tuple[int, int]] | None
    bound: float
    how: str = ""
    stats: dict = field(default_factory=dict)


class _TI:
    """Every placement of an instance as flat arrays, sorted by job (then
    type, then start), plus each path type's capacity per slot."""

    def __init__(self, inst: Instance, prep: bpc._Prep):
        if any(m.fixed_cost for m in inst.instance_types):
            raise NotImplementedError("lpfirst does not model fixed activation costs; use fedhpc.bpc")
        self.inst, self.prep = inst, prep
        H = self.H = inst.horizon
        self.jobs = [j.id for j in inst.jobs]
        self.jpos = {jid: i for i, jid in enumerate(self.jobs)}
        self.types = tuple(prep.path_types)
        self.kpos = {m: k for k, m in enumerate(self.types)}
        job_of = {j.id: j for j in inst.jobs}
        static_by_job: dict[int, list] = {}
        for col in prep.static_columns:
            static_by_job.setdefault(col.jobs[0][0], []).append(col)
        arcs_by_job: dict[int, list] = {}
        for m in self.types:
            for arc in prep.arcs_by_type[m]:
                arcs_by_job.setdefault(arc.job_id, []).append((m, arc))
        J, K, T, P, C1, C2, MID = [], [], [], [], [], [], []
        self.pair: dict[tuple[int, int], tuple[int, np.ndarray]] = {}
        n = 0
        for jid in self.jobs:
            j = job_of[jid]
            blocks = []
            for m, arc in sorted(arcs_by_job.get(jid, ()), key=lambda x: x[0]):
                ts = np.array(sorted(t for t in arc.starts if 0 <= t < H and t + arc.p_occ <= H),
                              dtype=np.int64)
                blocks.append((m, self.kpos[m], ts, arc.p_occ, inst.c[jid, m]))
            for m in sorted({c.type_id for c in static_by_job.get(jid, ())}):
                ts = np.array(sorted(c.jobs[0][1] for c in static_by_job[jid] if c.type_id == m),
                              dtype=np.int64)
                blocks.append((m, -1, ts, inst.p_occ[jid, m], inst.c[jid, m]))
            if not blocks:
                raise ValueError(f"job {jid} has no placement")
            for m, k, ts, p, c2 in blocks:
                if not len(ts):
                    continue
                self.pair[jid, m] = (n, ts)
                J.append(np.full(len(ts), self.jpos[jid], dtype=np.int64))
                K.append(np.full(len(ts), k, dtype=np.int64))
                MID.append(np.full(len(ts), m, dtype=np.int64))
                T.append(ts)
                P.append(np.full(len(ts), p, dtype=np.int64))
                C1.append(j.weight * (ts + p - j.arrival))
                C2.append(np.full(len(ts), c2, dtype=float))
                n += len(ts)
        cat = np.concatenate
        self.J, self.K, self.T, self.P, self.MID = cat(J), cat(K), cat(T), cat(P), cat(MID)
        self.C1, self.C2 = cat(C1).astype(float), cat(C2)
        self.n_arcs = n
        self.first = np.searchsorted(self.J, np.arange(len(self.jobs)))   # reduceat offsets
        if np.any(np.diff(np.append(self.first, n)) == 0):
            raise ValueError("some job has no placement")
        self.capped = self.K >= 0
        # capacity N_m(τ): instances released by slot τ
        self.N = np.zeros((len(self.types), H), dtype=float)
        for k, m in enumerate(self.types):
            for r, cnt in prep.releases[m].items():
                self.N[k, min(max(r, 0), H):] += cnt
        # slots some placement can occupy (only those need a row)
        used = np.zeros((len(self.types), H + 1), dtype=np.int64)
        kc, tc, pc = self.K[self.capped], self.T[self.capped], self.P[self.capped]
        np.add.at(used, (kc, tc), 1)
        np.add.at(used, (kc, tc + pc), -1)
        self.row_needed = np.cumsum(used, axis=1)[:, :H] > 0

    def index(self, jid: int, mid: int, t: int) -> int | None:
        hit = self.pair.get((jid, mid))
        if hit is None:
            return None
        lo, ts = hit
        i = int(np.searchsorted(ts, t))
        return lo + i if i < len(ts) and ts[i] == t else None

    def arcs_of(self, assignment: dict[int, tuple[int, int]]) -> list[int]:
        out = []
        for jid, (mid, t) in assignment.items():
            a = self.index(jid, mid, t)
            if a is not None:
                out.append(a)
        return out

    def assignment_of(self, arcs) -> dict[int, tuple[int, int]]:
        return {self.jobs[self.J[a]]: (int(self.MID[a]), int(self.T[a])) for a in arcs}


@dataclass
class _D:
    sigma: np.ndarray
    pi: np.ndarray        # (types, H), <= 0
    nu1: float
    nu2: float

    def mix(self, other: "_D", a: float) -> "_D":
        return _D(a * self.sigma + (1 - a) * other.sigma, a * self.pi + (1 - a) * other.pi,
                  a * self.nu1 + (1 - a) * other.nu1, a * self.nu2 + (1 - a) * other.nu2)


class _Master:
    """The restricted master LP (or, with ``vtype=BINARY`` and no
    artificials, a MIP over a fixed set of placements)."""

    def __init__(self, ti: _TI, w1: float, w2: float, b1: float | None, b2: float | None,
                 big_m: float | None, vtype=GRB.CONTINUOUS):
        formulations.configure_env()
        self.ti, self.w1, self.w2, self.vtype = ti, w1, w2, vtype
        mdl = self.mdl = gp.Model("lpfirst", env=formulations._shared_env)
        mdl.setParam("OutputFlag", 0)
        self.cover = [mdl.addConstr(gp.LinExpr() == 1.0, name=f"cover[{jid}]") for jid in ti.jobs]
        self.cap: dict[tuple[int, int], gp.Constr] = {}
        for k, tau in zip(*np.nonzero(ti.row_needed)):
            self.cap[int(k), int(tau)] = mdl.addConstr(gp.LinExpr() <= float(ti.N[k, tau]),
                                                       name=f"cap[{k},{tau}]")
        self.b1 = mdl.addConstr(gp.LinExpr() <= b1, name="bound1") if b1 is not None else None
        self.b2 = mdl.addConstr(gp.LinExpr() <= b2, name="bound2") if b2 is not None else None
        self.B1, self.B2 = b1, b2
        mdl.update()
        self.art = ([mdl.addVar(obj=big_m, column=gp.Column([1.0], [c])) for c in self.cover]
                    if big_m is not None else [])
        self.arcs: list[int] = []
        self.vars: list[gp.Var] = []
        self.has = np.zeros(ti.n_arcs, dtype=bool)

    def add(self, arcs) -> None:
        ti, mdl = self.ti, self.mdl
        for a in arcs:
            a = int(a)
            if self.has[a]:
                continue
            self.has[a] = True
            coefs, rows = [1.0], [self.cover[ti.J[a]]]
            k = ti.K[a]
            if k >= 0:
                t = ti.T[a]
                for tau in range(t, t + ti.P[a]):
                    coefs.append(1.0)
                    rows.append(self.cap[int(k), int(tau)])
            if self.b1 is not None:
                coefs.append(float(ti.C1[a]))
                rows.append(self.b1)
            if self.b2 is not None:
                coefs.append(float(ti.C2[a]))
                rows.append(self.b2)
            self.arcs.append(a)
            self.vars.append(mdl.addVar(obj=self.w1 * ti.C1[a] + self.w2 * ti.C2[a],
                                        ub=1.0, vtype=self.vtype,
                                        column=gp.Column(coefs, rows)))

    def duals(self) -> _D:
        ti = self.ti
        pi = np.zeros((len(ti.types), ti.H))
        for (k, tau), c in self.cap.items():
            pi[k, tau] = min(0.0, c.Pi)
        return _D(np.array([c.Pi for c in self.cover]), pi,
                  min(0.0, self.b1.Pi) if self.b1 is not None else 0.0,
                  min(0.0, self.b2.Pi) if self.b2 is not None else 0.0)

    def artificial(self) -> float:
        return sum(v.X for v in self.art)

    def assignment(self) -> dict[int, tuple[int, int]] | None:
        chosen = [a for a, v in zip(self.arcs, self.vars) if v.X > 0.5]
        out = self.ti.assignment_of(chosen)
        return out if len(out) == len(self.ti.jobs) == len(chosen) else None


def _price(ti: _TI, d: _D, w1: float, w2: float) -> np.ndarray:
    """Reduced cost of every placement at ``d``."""
    pre = np.zeros((len(ti.types), ti.H + 1))
    np.cumsum(d.pi, axis=1, out=pre[:, 1:])
    rc = (w1 - d.nu1) * ti.C1 + (w2 - d.nu2) * ti.C2 - d.sigma[ti.J]
    c = ti.capped
    k, t, p = ti.K[c], ti.T[c], ti.P[c]
    rc[c] -= pre[k, t + p] - pre[k, t]
    return rc


def _lagrangian(ti: _TI, d: _D, rc: np.ndarray, b1: float | None, b2: float | None
                ) -> tuple[float, np.ndarray]:
    """``(L, min_j)``: the bound at ``d`` and each job's least ``rc + σ_j``."""
    mins = np.minimum.reduceat(rc + d.sigma[ti.J], ti.first)
    L = float(mins.sum() + (d.pi * ti.N).sum())
    if b1 is not None:
        L += d.nu1 * b1
    if b2 is not None:
        L += d.nu2 * b2
    return L, mins


def _select(ti: _TI, rc: np.ndarray, has: np.ndarray, tol: float, per_job: int,
            max_cols: int) -> np.ndarray:
    """Up to ``per_job`` most negative placements per job, ``max_cols`` in all."""
    cand = np.nonzero((rc < -tol) & ~has)[0]
    if not len(cand):
        return cand
    order = np.lexsort((rc[cand], ti.J[cand]))
    cand = cand[order]
    jobs = ti.J[cand]
    start = np.r_[0, np.nonzero(np.diff(jobs))[0] + 1]
    rank = np.arange(len(cand)) - np.repeat(start, np.diff(np.r_[start, len(cand)]))
    cand = cand[rank < per_job]
    if len(cand) > max_cols:
        cand = cand[np.argsort(rc[cand], kind="stable")[:max_cols]]
    return cand


def _objective_of(inst: Instance, a: dict[int, tuple[int, int]], w1: float, w2: float) -> float:
    completion = {jid: t + inst.p_occ[jid, mid] for jid, (mid, t) in a.items()}
    return w1 * inst.f1_of(completion) + w2 * inst.f2_of(a)


def _within(inst: Instance, a: dict[int, tuple[int, int]], b1, b2) -> bool:
    if inst.schedule_violations(a):
        return False
    completion = {jid: t + inst.p_occ[jid, mid] for jid, (mid, t) in a.items()}
    return ((b1 is None or inst.f1_of(completion) <= b1 + 1e-6)
            and (b2 is None or inst.f2_of(a) <= b2 + 1e-6))


def _mip(ti: _TI, arcs, w1, w2, b1, b2, cutoff: float, time_limit: float, step: float,
         threads: int, start: dict | None = None):
    """Solve the placement MIP over ``arcs`` for a schedule below ``cutoff``.
    Returns ``(status, objective, assignment)``: ``"optimal"`` (proven on the
    grid), ``"none"`` (proven: nothing below the cutoff) or ``"limit"``."""
    m = _Master(ti, w1, w2, b1, b2, None, vtype=GRB.BINARY)
    m.add(arcs)
    mdl = m.mdl
    mdl.setParam("TimeLimit", max(0.0, time_limit))
    mdl.setParam("Cutoff", cutoff)
    mdl.setParam("MIPGap", 0.0)
    mdl.setParam("MIPGapAbs", 0.999 * step if step > 0 else 1e-9)
    mdl.setParam("IntFeasTol", 1e-9)
    if threads:
        mdl.setParam("Threads", threads)
    if start:
        idx = {a: v for a, v in zip(m.arcs, m.vars)}
        for a in ti.arcs_of(start):
            if a in idx:
                idx[a].Start = 1.0
    mdl.optimize()
    if mdl.Status in (GRB.CUTOFF, GRB.INFEASIBLE) or (mdl.Status == GRB.OPTIMAL and mdl.SolCount == 0):
        return "none", math.inf, None
    a = m.assignment() if mdl.SolCount else None
    obj = _objective_of(ti.inst, a, w1, w2) if a else math.inf
    if mdl.Status == GRB.OPTIMAL and a is not None:
        return "optimal", obj, a
    return "limit", obj, a


def solve_lpfirst(
    inst: Instance, weight1: float, weight2: float,
    bound1_value: float | None = None, bound2_value: float | None = None,
    *, time_limit: float | None = None, options: LPFirstOptions | None = None,
    ti: _TI | None = None, seed_schedules: list[dict] | None = None,
    column_pool: set[int] | None = None, stats: dict | None = None,
) -> LPFirstResult:
    """Minimise ``weight1·f1 + weight2·f2`` s.t. the optional bound rows,
    proving optimality as the module docstring describes.

    ``ti`` — a shared :class:`_TI` (built here otherwise). ``seed_schedules``
    — complete schedules whose placements seed the master and whose best one
    within the bounds is the first incumbent (default:
    :func:`fedhpc.bpc.heuristic_schedules`). ``column_pool`` — placement
    indices to seed the master with; when given, it receives the support of
    the final LP and of every schedule found, for the next solve.
    """
    opts = options or LPFirstOptions()
    t0 = time.perf_counter()
    deadline = None if time_limit is None else t0 + time_limit
    st = stats if stats is not None else {}
    if ti is None:
        ti = _TI(inst, bpc._prepare(inst))
    w1, w2, b1, b2 = weight1, weight2, bound1_value, bound2_value
    step = bpc._objective_step(inst, w1, w2)
    if seed_schedules is None:
        seed_schedules = bpc.heuristic_schedules(inst)

    ub, best = math.inf, None
    for a in seed_schedules:
        if len(a) == len(ti.jobs) and _within(inst, a, b1, b2):
            obj = _objective_of(inst, a, w1, w2)
            if obj < ub:
                ub, best = obj, dict(a)
    st["seed_obj"] = ub

    def certified(L: float) -> bool:
        if not math.isfinite(ub):
            return False
        margin = 1e-7 * max(1.0, abs(ub))
        return L > ub - step + margin if step > 0 else L >= ub - margin

    def left() -> float:
        return math.inf if deadline is None else deadline - time.perf_counter()

    def result(status, how, L):
        st["time_s"] = time.perf_counter() - t0
        st["bound"] = L
        if column_pool is not None:
            try:
                column_pool.update(int(a) for a, v in zip(master.arcs, master.vars) if v.X > 1e-9)
            except (gp.GurobiError, AttributeError):   # no LP solution yet
                pass
            if best is not None:
                column_pool.update(ti.arcs_of(best))
        return LPFirstResult(status, ub if best is not None else math.inf, best, L, how, st)

    master = _Master(ti, w1, w2, b1, b2, bpc._big_m_for(inst))
    seed = set(column_pool or ())
    for a in seed_schedules:
        seed.update(ti.arcs_of(a))
    seed.update(np.nonzero(~ti.capped)[0].tolist())
    master.add(sorted(seed))

    # --- column generation, stopped by the certificate when possible
    best_L, center, center_rc = -math.inf, None, None
    rmp = math.inf
    it = 0
    while True:
        if left() <= 0:
            return result("feasible", "", best_L)
        it += 1
        tl = time.perf_counter()
        master.mdl.optimize()
        st["lp_s"] = st.get("lp_s", 0.0) + time.perf_counter() - tl
        if master.mdl.Status != GRB.OPTIMAL:
            raise RuntimeError(f"lpfirst master LP status {master.mdl.Status}")
        rmp = master.mdl.ObjVal
        d = master.duals()
        tp = time.perf_counter()
        pick = np.array([], dtype=np.int64)
        if center is not None and opts.smoothing > 0:
            sd = center.mix(d, opts.smoothing)
            rc_s = _price(ti, sd, w1, w2)
            L, _m = _lagrangian(ti, sd, rc_s, b1, b2)
            if L > best_L:
                best_L, center, center_rc = L, sd, rc_s
            pick = _select(ti, rc_s, master.has, opts.tol, opts.per_job, opts.max_cols)
            if len(pick):
                rc_d = _price_subset(ti, d, w1, w2, pick)
                pick = pick[rc_d < -opts.tol]
        if not len(pick):
            rc = _price(ti, d, w1, w2)
            L, _m = _lagrangian(ti, d, rc, b1, b2)
            if L > best_L:
                best_L, center, center_rc = L, d, rc
            pick = _select(ti, rc, master.has, opts.tol, opts.per_job, opts.max_cols)
        st["pricing_s"] = st.get("pricing_s", 0.0) + time.perf_counter() - tp
        st["cg_iters"] = it
        if certified(best_L):
            st["columns"] = len(master.arcs)
            return result("optimal", "bound", best_L)
        if not len(pick) or best_L >= rmp - max(opts.tol, 1e-9 * abs(rmp)):
            break
        master.add(pick)
    st["columns"] = len(master.arcs)
    st["lp"] = rmp
    if master.artificial() > 1e-6:   # the LP optimum needs artificials: no schedule
        best = None
        return result("infeasible", "lp", best_L)

    # --- step 2: the restricted master as a MIP
    if math.isfinite(rmp) and opts.rmp_mip_time > 0:
        cut = ub - step + 1e-7 * max(1.0, abs(ub)) if math.isfinite(ub) else GRB.INFINITY
        tm = time.perf_counter()
        status, obj, a = _mip(ti, master.arcs, w1, w2, b1, b2, cut,
                              min(opts.rmp_mip_time, left()), step, opts.mip_threads, best)
        st["rmp_mip_s"] = time.perf_counter() - tm
        if a is not None and obj < ub and _within(inst, a, b1, b2):
            ub, best = obj, a
        if certified(best_L):
            return result("optimal", "rmp_mip", best_L)
    if not math.isfinite(ub):
        return result("feasible", "", best_L)

    # --- step 3: the exact MIP over the eligible placements
    _L, mins = _lagrangian(ti, center, center_rc, b1, b2)
    gap = ub - best_L
    eligible = np.nonzero(center_rc + center.sigma[ti.J] - mins[ti.J]
                          < gap + 1e-7 * max(1.0, abs(ub)))[0]
    st["eligible"] = int(len(eligible))
    if len(eligible) > opts.max_eligible or left() <= 0:
        return result("feasible", "", best_L)
    cut = ub - step + 1e-7 * max(1.0, abs(ub)) if step > 0 else ub - 1e-7 * max(1.0, abs(ub))
    tm = time.perf_counter()
    status, obj, a = _mip(ti, eligible, w1, w2, b1, b2, cut, left(), step, opts.mip_threads)
    st["eligible_mip_s"] = time.perf_counter() - tm
    if status == "limit":
        if a is not None and obj < ub and _within(inst, a, b1, b2):
            ub, best = obj, a
        return result("feasible", "", best_L)
    if status == "optimal" and obj < ub and _within(inst, a, b1, b2):
        ub, best = obj, a
    return result("optimal", "eligible_mip", best_L)


def _price_subset(ti: _TI, d: _D, w1: float, w2: float, idx: np.ndarray) -> np.ndarray:
    pre = np.zeros((len(ti.types), ti.H + 1))
    np.cumsum(d.pi, axis=1, out=pre[:, 1:])
    rc = (w1 - d.nu1) * ti.C1[idx] + (w2 - d.nu2) * ti.C2[idx] - d.sigma[ti.J[idx]]
    k, t, p = ti.K[idx], ti.T[idx], ti.P[idx]
    c = k >= 0
    rc[c] -= pre[k[c], t[c] + p[c]] - pre[k[c], t[c]]
    return rc


# ---------------------------------------------------------------------------
# Solution wrappers and the Pareto-search engine
# ---------------------------------------------------------------------------

def _solution(inst: Instance, res: LPFirstResult, status: str | None = None) -> Solution:
    if res.assignment is None:
        return Solution(status="infeasible" if res.status == "infeasible" else "feasible",
                        objective=None, f1=None, f2=None, assignment={}, completion={})
    a = res.assignment
    inst.check_schedule(a)
    completion = {jid: t + inst.p_occ[jid, mid] for jid, (mid, t) in a.items()}
    return Solution(status=status or res.status, objective=res.objective,
                    f1=inst.f1_of(completion), f2=inst.f2_of(a), assignment=dict(a),
                    completion=completion)


def solve_weighted_sum_lpfirst(inst: Instance, weight1: float, weight2: float, **kw) -> Solution:
    """Min ``weight1·f1 + weight2·f2`` (see :func:`solve_lpfirst`)."""
    return _solution(inst, solve_lpfirst(inst, weight1, weight2, **kw))


def _lexmin(inst: Instance, primary: str, b1=None, b2=None, **kw) -> Solution:
    w = (1.0, 0.0) if primary == "f1" else (0.0, 1.0)
    if kw.get("ti") is None:
        kw["ti"] = _TI(inst, bpc._prepare(inst))
    first = solve_lpfirst(inst, *w, b1, b2, **kw)
    if first.status != "optimal":
        return _solution(inst, first, "feasible" if first.assignment else None)
    sol1 = _solution(inst, first)
    if primary == "f1":
        b1 = min(b1, sol1.f1) if b1 is not None else sol1.f1
    else:
        b2 = min(b2, sol1.f2) if b2 is not None else sol1.f2
    seeds = [first.assignment, *(kw.pop("seed_schedules", None) or [])]
    second = solve_lpfirst(inst, *w[::-1], b1, b2, seed_schedules=seeds, **kw)
    if second.status != "optimal":
        return replace(sol1, status="feasible")
    return _solution(inst, second)


def solve_f1_lpfirst(inst: Instance, **kw) -> Solution:
    """Lex-min (f1, f2)."""
    return _lexmin(inst, "f1", **kw)


def solve_f2_lpfirst(inst: Instance, **kw) -> Solution:
    """Lex-min (f2, f1)."""
    return _lexmin(inst, "f2", **kw)


def make_lpfirst_lexmin_bounded(inst: Instance, time_limit: float | None = None,
                                options: LPFirstOptions | None = None, ea_seeds: bool = True):
    """A ``lexmin_bounded`` closure for ``pareto.true_pareto_frontier`` /
    ``map_pareto_frontier`` (``engine="lpfirst"``), with the contract of
    :func:`fedhpc.bpc.make_bpc_lexmin_bounded`. All box solves share one
    :class:`_TI`, one column pool (every placement stays valid under any
    bound rows) and the schedules found so far. ``time_limit`` caps each
    phase; an unproven phase gives a ``"feasible"`` result, which the box
    search treats as inconclusive."""
    counters = {"n_solves": 0, "build_time": 0.0}
    t0 = time.perf_counter()
    ti = _TI(inst, bpc._prepare(inst))
    schedules = bpc.heuristic_schedules(inst, ea=ea_seeds)
    counters["build_time"] = time.perf_counter() - t0
    pool: set[int] = set()

    def solve(w1, w2, b1, b2) -> LPFirstResult:
        counters["n_solves"] += 1
        res = solve_lpfirst(inst, w1, w2, b1, b2, time_limit=time_limit, options=options,
                            ti=ti, seed_schedules=schedules, column_pool=pool)
        if res.assignment is not None:
            schedules.insert(0, res.assignment)
        return res

    def lexmin_bounded(primary: str, secondary: str, bound: str | None = None,
                       bound_value: float | None = None, *, start: Solution | None = None
                       ) -> Solution:
        if start is not None and start.assignment and len(start.assignment) == len(inst.jobs):
            schedules.insert(0, dict(start.assignment))
        b1 = grid_floor(inst, "f1", bound_value) if bound == "f1" else None
        b2 = grid_floor(inst, "f2", bound_value) if bound == "f2" else None
        w = (1.0, 0.0) if primary == "f1" else (0.0, 1.0)
        first = solve(*w, b1, b2)
        if first.status == "infeasible":
            return _solution(inst, first)
        if first.status != "optimal":
            return _solution(inst, first, "feasible")
        sol1 = _solution(inst, first)
        if primary == "f1":
            b1 = min(b1, sol1.f1) if b1 is not None else sol1.f1
        else:
            b2 = min(b2, sol1.f2) if b2 is not None else sol1.f2
        ws = (1.0, 0.0) if secondary == "f1" else (0.0, 1.0)
        second = solve(*ws, b1, b2)
        if second.status != "optimal":
            return replace(sol1, status="feasible")
        return _solution(inst, second)

    return lexmin_bounded, counters
