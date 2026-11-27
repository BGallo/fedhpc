"""Branch-and-cut over fedlp.

    presolve -> root propagation -> root LP -> cut rounds -> root heuristics
    -> branch and bound (best bound, with plunging) -> postsolve

Root. The LP is solved; then up to ``cut_rounds`` rounds separate Gomory
mixed-integer and cover cuts, the pool keeps the efficacious and
non-parallel ones, they are appended to the LP (fedlp keeps the basis; the
dual simplex re-solves) and the loop stops when the bound stalls. Cuts whose
logicals end basic (slack) are removed again, keeping node LPs small. All
root cuts are global: they hold for the whole tree.

Tree. A node is its path of branching bounds; activating one rebuilds its
domain from the root bounds, propagates it, installs it in the LP and, when
it is not the child of the node just solved, restores its parent's basis.
Nodes are taken by best bound, except that after branching the solver
plunges into a child while that child's bound stays within ``plunge_gap`` of
the best one (depth-first search keeps the LP warm). An integral objective
(integer costs on integer columns only) lets every bound round up.

Primal side. Simple rounding at every node; fix-and-propagate, a guided
dive and RENS at the root; a guided dive every ``dive_frequency`` nodes. A
candidate is accepted only if it is feasible for the presolved problem.

Statuses: optimal (gap closed), infeasible, unbounded (presolve or root LP),
time_limit, node_limit.
"""
from __future__ import annotations

import heapq
import itertools
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace

import numpy as np
import scipy.sparse as sp

from .branching import Pseudocosts, select_branch
from .cuts import (CutParams, CutPool, cmir_cuts, cover_cuts, gomory_cuts, implied_integers, variable_bounds,
                   zerohalf_cuts)
from .heuristics import (feasibility_jump, feasibility_pump, fix_and_propagate, guided_dive, locks,
                         rens_problem, rins_problem, simple_rounding)
from .lp import INFEASIBLE, ITERATION_LIMIT, OPTIMAL, UNBOUNDED, LpRelaxation
from .presolve import presolve
from .problem import INF, MipProblem
from .propagate import Propagator
from .cliques import CliqueTable, clique_cuts, clique_rows, probe_cliques, row_cliques
from .scheduling import best_schedule, detect as detect_schedule, orders as schedule_orders


@dataclass
class MipOptions:
    time_limit: float = 3600.0
    node_limit: int = 10**9
    gap_rel: float = 1e-4
    gap_abs: float = 1e-6
    int_tol: float = 1e-6
    feas_tol: float = 1e-6
    presolve: bool = True
    presolve_options: dict = field(default_factory=dict)   # presolve()'s switches (doubletons=False, ...)
    cut_rounds: int = 50
    cuts_per_round: int = 200
    gomory: bool = True
    covers: bool = True
    cmir: bool = True
    zerohalf: bool = True
    # clique table (cliques.py): cliques from rows and from probing
    # implications (within clique_probe_budget propagator entry visits),
    # clique cuts in the root loop, and the cliques that are not rows handed
    # to the propagator
    # Feasibility Jump (heuristics.feasibility_jump): before the root LP
    # from the domain's zero point, after it from the LP point; work in
    # entry visits per call (deterministic)
    fj: bool = True
    fj_work: int = 20_000_000
    # conflict analysis on nodes that propagation proves infeasible
    conflicts: bool = True
    conflict_max_depth: int = 40         # binary decisions on the path analysed at most
    conflict_max_size: int = 12          # longer nogoods are dropped
    conflict_batch: int = 25             # nogoods per propagator rebuild
    cliques: bool = True
    clique_probe_budget: int = 2_000_000
    clique_propagation: bool = True
    # rounds without bound progress before the loop stops: on degenerate
    # LPs violated cuts move the vertex for several rounds before the bound
    # moves (neos-911970: 23.3 -> 48.2 with 12, flat with 3), so LPs that
    # solve in under cut_cheap_lp_seconds get cut_stall_rounds_cheap
    cut_stall_rounds: int = 3
    cut_stall_rounds_cheap: int = 12
    cut_cheap_lp_seconds: float = 0.5
    cut_rounds_expensive: int = 15       # the round cap when the root LP is not cheap
    cut_cheap_max_nnz: int = 50000       # "cheap" also needs a small matrix (Kramer's TI: 16 s of rounds for nothing)
    cut_min_improvement: float = 1e-4     # relative bound gain counting as progress
    cut_time_share: float = 0.1           # the root cut loop stops past this share of the limit
    heuristics: bool = True
    rens: bool = True
    rens_node_limit: int = 300
    rens_time_share: float = 0.2
    # RINS at the root once there is an incumbent, then every
    # rins_frequency nodes after the incumbent changed (or 4x that), all
    # sub-MIPs together within rins_total_share of the time limit
    rins: bool = True
    rins_frequency: int = 50
    rins_node_limit: int = 500
    rins_total_share: float = 0.15
    dive_frequency: int = 25
    fp_time_share: float = 0.05           # fix-and-propagate's budget at the root
    # list scheduling on interval structure (scheduling.py): at the root,
    # orders from the LP's start times plus insertion local search within
    # schedule_root_share of the time limit (at most 5 s); then every
    # schedule_frequency nodes from the node LP, with a short search
    schedule_heuristic: bool = True
    schedule_root_share: float = 0.03
    schedule_frequency: int = 10
    # a Smith-rule schedule before the root LP, crashing the primal's basis
    # from it (off: on TI_N50_M2_R0 the crash from a near-optimal schedule
    # took the primal 9.9 s against 8.0 s from the slack basis)
    schedule_crash: bool = False
    # node dives' LP pivots stay within this share of the node LPs' (plus a
    # start-up allowance): unbudgeted, csched008 spent 18.5 M pivots diving
    dive_budget_ratio: float = 0.1
    dive_budget_start: int = 5000
    reliability: int = 4
    strong_candidates: int = 6
    strong_iteration_limit: int = 100
    # strong branching's pivots stay within this share of the node LPs'
    # (plus a start-up allowance), as SCIP bounds them; past it, pseudocosts
    # (the average for uninitialised ones) choose alone
    strong_budget_ratio: float = 0.5
    strong_budget_start: int = 20000
    plunge_gap: float = 0.02              # relative to the incumbent gap
    max_stored_bases: int = 4000
    lp_options: dict = field(default_factory=dict)
    # LP: interval row blocks differenced into network rows (reformulate.py)
    network_lp: bool = True
    # Root LP algorithm: "dual", "primal" (partial pricing; from a heuristic
    # schedule's basis when one is known, else the slack basis), or "auto":
    # the primal on wide LPs (columns >= root_primal_ratio x rows), where
    # Kramer's masters solve 2-5x faster with it, falling back to the dual
    # when its phase 1 passes root_phase1_cap x rows pivots (AF_N50_M2_R4's
    # composite phase 1 wandered for 7.6 m pivots)
    # (on a network-reformulated LP the cap is root_phase1_cap_network: TI's
    # phase 1 repairs only the assignment rows, 3-6 x rows pivots at a cheap
    # pivot each, and the primal still won 2.5-6x on the R0 instances)
    root_lp: str = "auto"
    root_primal_ratio: float = 20.0
    root_phase1_cap: float = 3.0
    root_phase1_cap_network: float = 10.0
    verbose: bool = True
    log_interval: float = 5.0
    sub_mip: bool = False                 # a heuristic's sub-MIP: no recursion, quiet
    # reduced-cost fixing from the root LP at every new incumbent (_rc_fix)
    rc_fixing: bool = True
    # Root restart: when reduced-cost fixing (with the root's propagation)
    # has fixed at least restart_min_fixed of the integer columns, the
    # problem restricted to the root domain is presolved again (the fixed
    # columns leave) and solved with the incumbent's value as cutoff - the
    # MIP over lpfirst's eligible placements. No better solution there proves
    # the incumbent optimal.
    restart: bool = True
    restart_min_fixed: float = 0.3
    # an objective value (this problem's minimisation form) only strictly
    # better solutions count against: the restart's sub-solve gets the
    # parent's incumbent value
    cutoff: float = INF


@dataclass
class MipResult:
    status: str
    objective: float | None           # in the model's sense
    bound: float                      # in the model's sense
    x: np.ndarray | None              # original columns
    gap: float
    nodes: int
    seconds: float
    stats: dict


@dataclass(order=True)
class _Node:
    """Ordered by bound, then deepest first, then age. With flat bounds
    (csched's makespan-like objective) the age tie-break alone made the
    search breadth-first: thousands of open nodes, no leaf reached."""
    bound: float
    neg_depth: int
    seq: int
    depth: int = field(compare=False)
    path: tuple = field(compare=False)          # ((j, is_upper, value), ...)
    basis: tuple | None = field(compare=False, default=None)
    branch: tuple | None = field(compare=False, default=None)   # (j, up, frac) that made it
    parent_obj: float = field(compare=False, default=-INF)


class MipSolver:
    def __init__(self, problem: MipProblem, options: MipOptions | None = None):
        self.original = problem
        self.opt = options or MipOptions()
        self.t0 = time.perf_counter()
        self.stats: dict = {"nodes": 0, "lp_iterations": 0, "cuts_added": 0, "cut_rounds": 0,
                            "root_bound_no_cuts": None, "root_bound": None, "heuristic_solutions": {},
                            "strong_branches": 0, "infeasible_nodes": 0, "pruned_nodes": 0}
        self.incumbent = None
        self.inc_obj = INF
        self._last_log = 0.0

    # --------------------------------------------------------------- utils
    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def deadline(self) -> float:
        return self.t0 + self.opt.time_limit

    def log(self, msg: str, force: bool = False) -> None:
        if not self.opt.verbose or self.opt.sub_mip:
            return
        t = self.elapsed()
        if force or t - self._last_log >= self.opt.log_interval:
            self._last_log = t
            print(f"[{t:7.1f}s] {msg}", flush=True)

    def _round_bound(self, v: float) -> float:
        if self.int_obj and np.isfinite(v):
            off = self.P.offset
            return math.ceil(v - off - 1e-6) + off
        return v

    def _gap_closed(self, bound: float) -> bool:
        if not np.isfinite(self.inc_obj):
            return False
        g = self.inc_obj - bound
        return g <= max(self.opt.gap_abs, self.opt.gap_rel * max(1.0, abs(self.inc_obj)))

    def _cutoff(self) -> float:
        if not np.isfinite(self.inc_obj):
            return INF
        if self.int_obj:
            return self.inc_obj - 1.0 + 1e-6       # only strictly better integers
        return self.inc_obj - max(self.opt.gap_abs, self.opt.gap_rel * max(1.0, abs(self.inc_obj)))

    @contextmanager
    def _timed(self, key: str):
        """Accumulate wall time (and LP pivots) under stats["time"][key]."""
        t0 = time.perf_counter()
        it0 = self.lp.stats.iterations if getattr(self, "lp", None) is not None else 0
        try:
            yield
        finally:
            tt = self.stats.setdefault("time", {})
            tt[key] = tt.get(key, 0.0) + time.perf_counter() - t0
            if getattr(self, "lp", None) is not None:
                ti = self.stats.setdefault("pivots", {})
                ti[key] = ti.get(key, 0) + self.lp.stats.iterations - it0

    def _try(self, x: np.ndarray | None, source: str) -> bool:
        if x is None:
            return False
        P = self.P
        y = x.copy()
        y[P.integer] = np.round(y[P.integer])
        y = np.clip(y, P.lb, P.ub)
        if not P.is_feasible(y, self.opt.feas_tol, self.opt.int_tol):
            return False
        v = P.objective(y)
        if v < self.inc_obj - 1e-9:
            self.inc_obj, self.incumbent = v, y
            hs = self.stats["heuristic_solutions"]
            hs[source] = hs.get(source, 0) + 1
            self.log(f"incumbent {P.reported(v):.6g} from {source}", force=True)
            self._rc_fix()
            return True
        return False

    # ------------------------------------------------- reduced-cost fixing
    def _save_root_duals(self) -> None:
        """The root LP's reduced costs, the bounds they were taken at and
        its value: what reduced-cost fixing reads for the rest of the solve."""
        if not self.opt.rc_fixing:
            return
        lp = self.lp
        _rs, cs = lp.get_basis()
        n = self.P.n
        self._rc = dict(d=lp.reduced_costs().copy(), z=float(lp.objective), lb=lp.lb.copy(), ub=lp.ub.copy(),
                        st=np.asarray(cs[:n]).copy())
        self._fixed_before_rc = int(np.sum(self.P.integer & (self.root_lb == self.root_ub)))
        self._rc_fix()

    def _rc_fix(self) -> None:
        """Reduced-cost fixing on the global domain (the LP-first proof
        carried into the tree). With the root LP's value z, reduced costs d
        and the cutoff U (the incumbent, less one objective step when the
        objective is integral), every solution better than the incumbent
        satisfies, for a column nonbasic at its lower bound l with d_j > 0,

            x_j <= l + (U - z) / d_j

        (and symmetrically at an upper bound): its LP objective moves by at
        least d_j per unit. lpfirst's step 3 is this with the eligible
        placements as the columns left free. The root domain shrinks, is
        propagated, and every node rebuilds its domain from it. Applied once
        the root LP is done and at every new incumbent."""
        rc = getattr(self, "_rc", None)
        if rc is None or not np.isfinite(self.inc_obj):
            return
        P = self.P
        gap = self._cutoff() - rc["z"]
        if gap < 0:
            return                      # the bound is at the incumbent: the tree closes it
        d, st, l0, u0 = rc["d"], rc["st"], rc["lb"], rc["ub"]
        tol = 1e-9
        lb, ub = self.root_lb.copy(), self.root_ub.copy()
        free = self.lp.free
        at_lo = (st == 1) & (d > tol) & ~free & np.isfinite(l0)
        at_up = (st == 2) & (d < -tol) & ~free & np.isfinite(u0)
        with np.errstate(divide="ignore", invalid="ignore"):
            nu = l0 + gap / np.where(at_lo, d, 1.0)
            nl = u0 - gap / np.where(at_up, -d, 1.0)
        nu = np.where(P.integer, np.floor(nu + 1e-9), nu)
        nl = np.where(P.integer, np.ceil(nl - 1e-9), nl)
        tu = at_lo & (nu < ub - 1e-9)
        tl = at_up & (nl > lb + 1e-9)
        if not (np.any(tu) or np.any(tl)):
            return
        ub[tu] = np.maximum(nu[tu], lb[tu])
        lb[tl] = np.minimum(nl[tl], ub[tl])
        res = self.prop.propagate(lb, ub, max_rounds=20)
        self.stats["rc_fixed"] = self.stats.get("rc_fixed", 0) + int(tu.sum() + tl.sum())
        if not res.feasible:
            # nothing beats the incumbent anywhere: the solve is over
            self._rc_closed = True
            return
        self.root_lb, self.root_ub = res.lb, res.ub
        self.stats["rc_fixed_cols"] = int(np.sum(self.root_lb == self.root_ub))

    # --------------------------------------------------------------- solve
    def solve(self) -> MipResult:
        opt = self.opt
        P0 = self.original
        if opt.presolve:
            P, self.post, pst = presolve(P0, **opt.presolve_options)
            self.stats["presolve"] = {k: v for k, v in vars(pst).items()}
            if pst.status != "reduced":
                return self._result(pst.status, None, INF if pst.status == "infeasible" else -INF)
        else:
            from .presolve import Postsolve
            P = P0
            self.post = Postsolve(P0.n, np.arange(P0.n), np.zeros(P0.n), np.arange(P0.m))
        self.P = P
        if np.isfinite(opt.cutoff):
            self.inc_obj = float(opt.cutoff)
        self.log(f"presolved: {P.summary()}", force=True)
        if P.n == 0:
            self.incumbent, self.inc_obj = np.zeros(0), P.offset
            return self._result("optimal", self.incumbent, self.inc_obj)
        self.int_obj = P.objective_is_integral()
        self.prop = Propagator(P, feas_tol=opt.feas_tol)
        res = self.prop.propagate(P.lb, P.ub)
        if not res.feasible:
            return self._result("infeasible", None, INF)
        self.root_lb, self.root_ub = res.lb, res.ub
        self.down_locks, self.up_locks = locks(P)
        self.lp = LpRelaxation(P, opt.lp_options, network=opt.network_lp)
        if self.lp.net is not None:
            self.stats["network_rows"] = int(self.lp.net.block_row.sum())
        self.lp.deadline = self.deadline()
        self.lp.set_bounds(self.root_lb, self.root_ub)
        self.sched = None
        if opt.heuristics and opt.schedule_heuristic:
            self.sched = detect_schedule(P, self.lp.net.blocks if self.lp.net is not None else [])
            if self.sched is not None:
                self.stats["schedule_structure"] = type(self.sched).__name__
            if self.sched is not None and opt.schedule_crash:
                with self._timed("list_schedule"):
                    xs, _v = best_schedule(P, self.sched, schedule_orders(self.sched), 0.0, improve=False)
                self._try(xs, "list_schedule")
        if opt.heuristics and opt.fj and not opt.sub_mip and self.sched is None:
            # (scheduling structure: list scheduling after the LP does better)
            with self._timed("feasibility_jump"):
                self._try(feasibility_jump(P, self.root_lb, self.root_ub, None, self._fj_work()), "feasibility_jump")
        with self._timed("root_lp"):
            st = self._root_lp()
        if opt.cliques:
            with self._timed("cliques"):
                self._build_cliques()
        if st != OPTIMAL and self.lp.past_deadline():
            return self._result("time_limit", None, -INF)
        if st == INFEASIBLE:
            return self._result("infeasible", None, INF)
        if st == UNBOUNDED:
            return self._result("unbounded", None, -INF)
        if st != OPTIMAL:
            return self._result("error", None, -INF)
        root_obj = self._round_bound(self.lp.objective)
        self.stats["root_bound_no_cuts"] = P.reported(root_obj)
        self.log(f"root LP {P.reported(self.lp.objective):.6g} ({self.lp.stats.iterations} its, "
                 f"{self.lp.stats.seconds:.2f}s)", force=True)
        with self._timed("root_cuts"):
            root_obj = self._root_cuts(root_obj)
        self._save_root_duals()
        self.stats["root_bound"] = P.reported(root_obj)
        if opt.heuristics:
            self._root_heuristics()
        if not self._gap_closed(root_obj) and self._restart_due():
            return self._restart()
        return self._branch_and_bound(root_obj)

    def _restart_due(self) -> bool:
        opt, P = self.opt, self.P
        if not (opt.restart and opt.rc_fixing) or not np.isfinite(self.inc_obj):
            return False
        ii = P.integer
        n_int = int(ii.sum())
        if n_int == 0 or self.elapsed() > 0.8 * opt.time_limit:
            return False
        fixed = int(np.sum(ii & (self.root_lb == self.root_ub)))
        return fixed - self._fixed_before_rc >= opt.restart_min_fixed * n_int

    def _restart(self) -> MipResult:
        """Solve the problem restricted to the root domain again, from
        presolve, with the incumbent's value as cutoff."""
        opt, P = self.opt, self.P
        Q = P.copy()
        Q.lb, Q.ub = self.root_lb.copy(), self.root_ub.copy()
        Q._csc = P._csc
        left = self.deadline() - time.perf_counter()
        self.log(f"restart: {int(np.sum(Q.lb == Q.ub))} of {P.n} columns fixed", force=True)
        sub = MipSolver(Q, replace(opt, time_limit=max(1.0, left), restart=False, cutoff=self.inc_obj,
                                   lp_options=dict(opt.lp_options)))
        sub.t0 = time.perf_counter()
        r = sub.solve()
        self.stats["restart"] = {k: r.stats.get(k) for k in ("nodes", "root_bound", "lp_iterations",
                                                             "presolve", "time", "heuristic_solutions")}
        self.stats["restart"]["columns"] = int(sub.P.n) if hasattr(sub, "P") else None
        self.stats["nodes"] = self.stats.get("nodes", 0) + r.nodes
        for key in ("time", "pivots"):          # the sub-solve's phases count as ours
            mine = self.stats.setdefault(key, {})
            for k, v in (r.stats.get(key) or {}).items():
                mine[k] = mine.get(k, 0) + v
        if r.x is not None:
            self._try(r.x, "restart")
        if r.status in ("optimal", "infeasible"):
            # nothing beats the incumbent outside what the sub-solve covered
            return self._result("optimal", self.incumbent, self.inc_obj)
        bound = Q.reported(r.bound) if np.isfinite(r.bound) else -INF
        status = r.status if r.status != "error" else "error"
        return self._result(status, self.incumbent, min(bound, self.inc_obj))

    def _fj_work(self) -> int:
        """Feasibility Jump's budget: fj_work, at most 100 passes over A."""
        return int(min(self.opt.fj_work, 100 * max(1, self.P.A.nnz)))

    def _build_cliques(self) -> None:
        """The clique table from rows and probing; cliques that are not rows
        go to a propagator over P plus those rows."""
        opt, P = self.opt, self.P
        t = CliqueTable(P.n)
        nr = row_cliques(P, self.root_lb, self.root_ub, t)
        npb = probe_cliques(P, self.prop, self.root_lb, self.root_ub, t, budget=opt.clique_probe_budget,
                            deadline=time.perf_counter() + 0.05 * opt.time_limit) if opt.clique_probe_budget else 0
        self.cliques = t
        self.stats["cliques"] = {"rows": nr, "probing": npb, "total": t.size}
        if opt.clique_propagation:
            extra = clique_rows(t, P.n)
            if extra is not None:
                A2, lo2, hi2 = extra
                Q = MipProblem(sp.vstack([P.A, A2], format="csr"), np.r_[P.row_lo, lo2], np.r_[P.row_hi, hi2],
                               P.lb, P.ub, P.c, P.integer)
                self.prop = Propagator(Q, feas_tol=opt.feas_tol)
                self.stats["cliques"]["propagator_rows"] = int(A2.shape[0])
                res = self.prop.propagate(self.root_lb, self.root_ub)
                if res.feasible:
                    self.root_lb, self.root_ub = res.lb, res.ub

    def _root_lp(self) -> int:
        """The cold root LP, by ``root_lp`` (see MipOptions)."""
        opt, P, lp = self.opt, self.P, self.lp
        mode = opt.root_lp
        if mode == "auto":
            mode = "primal" if P.n >= opt.root_primal_ratio * max(1, P.m) else "dual"
        self.stats["root_lp_algorithm"] = mode

        def cold():
            if lp.net is not None:
                lp.set_basis(lp.slack_basis())
            else:   # fedlp's crash (logicals, singleton columns on equality rows)
                lp.set_basis((np.ones(lp.rows, np.int8), np.ones(lp.lp.cols, np.int8)))

        if mode != "primal":
            if lp.net is not None:
                cold()
            return lp.solve()
        # a crash from a point only when a schedule heuristic produced it
        # (schedule_crash): Feasibility Jump's points are merely feasible, and
        # the primal from them took 65 s on TI_N50_M2_R9 against 0.6 s cold
        crash_pt = self.incumbent if (opt.schedule_crash and self.stats["heuristic_solutions"].get("list_schedule")) else None
        if crash_pt is None or not lp.basis_from_point(crash_pt):
            cold()
        o = lp.lp.options
        saved = (o.dual, o.partial_min_cols, o.phase1_limit)
        cap = opt.root_phase1_cap_network if lp.net is not None else opt.root_phase1_cap
        o.dual, o.partial_min_cols, o.phase1_limit = False, 2000, int(cap * lp.rows)
        try:
            st = lp.solve()
        finally:
            o.dual, o.partial_min_cols, o.phase1_limit = saved
        if st == ITERATION_LIMIT and not lp.past_deadline():
            self.stats["root_lp_algorithm"] = "primal->dual"
            cold()
            st = lp.solve()
        return st

    # ------------------------------------------------------------ root cuts
    def _root_cuts(self, root_obj: float) -> float:
        opt, P, lp = self.opt, self.P, self.lp
        par = CutParams()
        pool = CutPool(par)
        stall, best = 0, root_obj
        # (scheduling structures keep the short loop: the list-scheduling
        # path does not need the long one, which cost AF_N20 1.3 s for nothing)
        cheap = (self.stats.get("time", {}).get("root_lp", INF) < opt.cut_cheap_lp_seconds
                 and P.A.nnz < opt.cut_cheap_max_nnz and getattr(self, "sched", None) is None)
        patience = opt.cut_stall_rounds_cheap if cheap else opt.cut_stall_rounds
        max_rounds = opt.cut_rounds if cheap else min(opt.cut_rounds, opt.cut_rounds_expensive)
        t_cuts = time.perf_counter()
        impint = implied_integers(P)
        vb = variable_bounds(P) if opt.cmir else None
        self.stats["implied_integers"] = int(impint.sum() - P.integer.sum())
        for rnd in range(max_rounds):
            if self.elapsed() > 0.5 * opt.time_limit or time.perf_counter() - t_cuts > opt.cut_time_share * opt.time_limit:
                break
            x = lp.x()
            if not self._fractional(x).size:
                self._try(x, "root_lp")
                break
            self._try(simple_rounding(P, x, self.down_locks, self.up_locks), "rounding")
            cuts = []
            if opt.gomory:
                cuts += gomory_cuts(lp, P, x, par, impint)
            if opt.covers:
                cuts += cover_cuts(P, x, self.root_lb, self.root_ub, par)
            if opt.cliques and self.cliques.size:
                with self._timed("clique_cuts"):
                    cuts += clique_cuts(self.cliques, x, self.root_lb, self.root_ub, par)
            if opt.zerohalf:
                with self._timed("zerohalf"):
                    cuts += zerohalf_cuts(P, x, self.root_lb, self.root_ub, par, impint,
                                          deadline=time.perf_counter() + 0.05 * opt.time_limit)
            if opt.cmir:
                with self._timed("cmir"):
                    cuts += cmir_cuts(P, x, self.root_lb, self.root_ub, par,
                                      deadline=time.perf_counter() + 0.02 * opt.time_limit, vb=vb)
            sel = pool.select(cuts, x, opt.cuts_per_round)
            if not sel:
                break
            lp.add_rows([(c.cols, c.coef, c.rhs, INF) for c in sel])
            self.stats["cuts_added"] += len(sel)
            self.stats["cut_rounds"] = rnd + 1
            st = lp.solve()
            if st != OPTIMAL and lp.past_deadline():
                break
            if st != OPTIMAL:
                # numerical trouble with cuts: drop them all, keep the plain LP
                lp.remove_rows(list(lp.cut_rows))
                lp.solve()
                break
            obj = self._round_bound(lp.objective)
            self.log(f"cut round {rnd + 1}: {len(sel)} cuts, bound {P.reported(obj):.6g}", force=True)
            if obj - best <= opt.cut_min_improvement * max(1.0, abs(best)):
                stall += 1
                if stall >= patience:
                    best = max(best, obj)
                    break
            else:
                stall = 0
            best = max(best, obj)
            if self._gap_closed(best):
                break
        # slack cuts out (their logical basic): smaller node LPs
        if lp.cut_rows:
            rs, _cs = lp.get_basis()
            slack = [r for r in lp.cut_rows if rs[r] == 0]
            if slack:
                lp.remove_rows(slack)
                lp.solve()
        return best

    def _fractional(self, x: np.ndarray) -> np.ndarray:
        ii = np.nonzero(self.P.integer)[0]
        f = np.abs(x[ii] - np.round(x[ii]))
        return ii[f > self.opt.int_tol]

    # ------------------------------------------------------- root heuristics
    def _root_heuristics(self) -> None:
        opt, P, lp = self.opt, self.P, self.lp
        x = lp.x()
        end = min(self.deadline(), time.perf_counter() + 0.3 * opt.time_limit)
        self._try(simple_rounding(P, x, self.down_locks, self.up_locks), "rounding")
        if opt.fj and not opt.sub_mip and self.sched is None:
            with self._timed("feasibility_jump"):
                self._try(feasibility_jump(P, self.root_lb, self.root_ub, x, self._fj_work(), seed=1,
                                           obj_weight=1.0 if self.incumbent is not None else 0.0), "feasibility_jump")
        if self.sched is not None:
            budget = min(5.0, opt.schedule_root_share * opt.time_limit)
            with self._timed("list_schedule"):
                xs, _v = best_schedule(P, self.sched, schedule_orders(self.sched, x),
                                       min(end, time.perf_counter() + budget))
            self._try(xs, "list_schedule")
        with self._timed("fix_and_propagate"):
            self._try(fix_and_propagate(P, self.prop, lp, x, self.root_lb, self.root_ub,
                                        min(end, time.perf_counter() + opt.fp_time_share * opt.time_limit),
                                        stats=self.stats), "fix_and_propagate")
        with self._timed("root_dive"):
            self._try(guided_dive(P, self.prop, lp, self.root_lb, self.root_ub, end, self.incumbent), "dive")
        if self.incumbent is None:
            with self._timed("root_dive"):
                self._try(guided_dive(P, self.prop, lp, self.root_lb, self.root_ub,
                                      min(end, time.perf_counter() + 0.05 * opt.time_limit), None,
                                      rule="nearest"), "dive_nearest")
        if self.incumbent is None:
            with self._timed("feasibility_pump"):
                self._try(feasibility_pump(P, lp, x, self.root_lb, self.root_ub, end, prop=self.prop),
                          "feasibility_pump")
        if opt.rens and not opt.sub_mip:
            Q = rens_problem(P, x, self.root_lb, self.root_ub)
            if Q is not None:
                budget = max(1.0, opt.rens_time_share * (self.deadline() - time.perf_counter()))
                sub = MipSolver(Q, MipOptions(time_limit=budget, node_limit=opt.rens_node_limit,
                                              sub_mip=True, verbose=False, rens=False,
                                              cut_rounds=3, lp_options=opt.lp_options))
                with self._timed("rens"):
                    r = sub.solve()
                if r.x is not None:
                    self._try(r.x, "rens")
        self._rins(x, self.root_lb, self.root_ub)

    def _rins(self, x, lb, ub) -> None:
        """One RINS sub-MIP around the incumbent, within the budget."""
        opt, P = self.opt, self.P
        if not opt.rins or opt.sub_mip or self.incumbent is None:
            return
        used = self.stats.get("time", {}).get("rins", 0.0)
        left = opt.rins_total_share * opt.time_limit - used
        if left < 0.5:
            return
        Q = rins_problem(P, x, self.incumbent, lb, ub)
        if Q is None:
            return
        self._rins_inc = self.inc_obj
        budget = min(left, 0.25 * (self.deadline() - time.perf_counter()))
        sub = MipSolver(Q, MipOptions(time_limit=max(0.5, budget), node_limit=opt.rins_node_limit, sub_mip=True,
                                      verbose=False, rens=False, rins=False, cut_rounds=3,
                                      lp_options=opt.lp_options, network_lp=opt.network_lp))
        with self._timed("rins"):
            r = sub.solve()
        self.stats["rins_calls"] = self.stats.get("rins_calls", 0) + 1
        if r.x is not None:
            self._try(r.x, "rins")

    # ------------------------------------------------------- branch and bound
    def _domain(self, node: _Node):
        """The node's domain: the root's (a propagation fixed point) with
        the path's branching bounds, propagated from their rows only."""
        prop = self.prop
        prop.reset(self.root_lb, self.root_ub, propagate=False)
        if not node.path:
            return prop.lb(), prop.ub()
        js = np.fromiter((j for j, _u, _v in node.path), dtype=np.int64, count=len(node.path))
        lo = np.array([v if up else -INF for _j, up, v in node.path])
        hi = np.array([INF if up else v for _j, up, v in node.path])
        if not prop.change(js, lo, hi, max_rounds=10):
            return None
        return prop.lb(), prop.ub()

    # ----------------------------------------------------- conflict analysis
    def _infeasible(self, decisions) -> bool:
        prop = self.prop
        prop.reset(self.root_lb, self.root_ub, propagate=False)
        if not decisions:
            return False
        js = np.fromiter((j for j, _u, _v in decisions), dtype=np.int64, count=len(decisions))
        lo = np.array([v if up else -INF for _j, up, v in decisions])
        hi = np.array([INF if up else v for _j, up, v in decisions])
        return not prop.change(js, lo, hi, max_rounds=10)

    def _learn_conflict(self, node: _Node) -> None:
        """Conflict analysis by propagation (the deletion filter of
        conflict-set minimisation; Achterberg 2007, ch. 11, without the
        implication graph): the node's binary branching decisions are
        reduced, one at a time, to a set D that still propagates to an empty
        domain, giving the nogood

            sum_{x_j = 1 in D} (1 - x_j) + sum_{x_j = 0 in D} x_j >= 1,

        valid for the whole problem. Nogoods are batched into the
        propagator's rows (never the LP's), so later nodes fail earlier."""
        opt, P = self.opt, self.P
        binary = P.integer & (self.root_lb == 0) & (self.root_ub == 1)
        dec = [d for d in node.path if binary[d[0]]]
        if len(dec) != len(node.path) or not dec or len(dec) > opt.conflict_max_depth:
            return                                  # general-integer bounds: no clause
        if not self._infeasible(dec):
            return
        D = list(dec)
        k = 0
        while k < len(D):
            trial = D[:k] + D[k + 1:]
            if trial and self._infeasible(trial):
                D = trial
            else:
                k += 1
        if len(D) > opt.conflict_max_size:
            return
        key = tuple(sorted((j, up) for j, up, _v in D))
        pool = self.__dict__.setdefault("_conflicts", {})
        if key in pool:
            return
        pool[key] = D
        self.stats["conflicts"] = len(pool)
        pending = self.__dict__.setdefault("_conflicts_pending", [])
        pending.append(D)
        if len(pending) >= opt.conflict_batch:
            self._install_conflicts()

    def _install_conflicts(self) -> None:
        pending = self.__dict__.get("_conflicts_pending") or []
        if not pending:
            return
        P = self.P
        ri, ci, vi, lo = [], [], [], []
        for r, D in enumerate(pending):
            # up (x = 1) contributes 1 - x, down (x = 0) contributes x
            n1 = sum(1 for _j, up, _v in D if up)
            for j, up, _v in D:
                ri.append(r), ci.append(j), vi.append(-1.0 if up else 1.0)
            lo.append(1.0 - n1)
        C = sp.csr_matrix((vi, (ri, ci)), shape=(len(pending), P.n))
        base = self.prop.P
        Q = MipProblem(sp.vstack([base.A, C], format="csr"), np.r_[base.row_lo, lo],
                       np.r_[base.row_hi, np.full(len(pending), INF)], P.lb, P.ub, P.c, P.integer)
        self.prop = Propagator(Q, feas_tol=self.opt.feas_tol)
        self._conflicts_pending = []
        self.stats["conflict_rows"] = self.stats.get("conflict_rows", 0) + len(pending)

    def _branch_and_bound(self, root_obj: float) -> MipResult:
        opt, P, lp = self.opt, self.P, self.lp
        pc = Pseudocosts(P.n, opt.reliability)
        seq = itertools.count()
        heap: list[_Node] = []
        root = _Node(root_obj, 0, next(seq), 0, (), None)
        heapq.heappush(heap, root)
        nodes, stored = 0, 0
        nxt: _Node | None = None          # a plunge child, LP state continuing from its parent
        last_solved_seq = None
        status = "optimal"
        while heap or nxt is not None:
            if getattr(self, "_rc_closed", False):
                heap, nxt = [], None
                break
            best_bound = min([n.bound for n in heap] + ([nxt.bound] if nxt else []))
            if self._gap_closed(best_bound):
                break
            if self.elapsed() > opt.time_limit:
                status = "time_limit"
                break
            if nodes >= opt.node_limit:
                status = "node_limit"
                break
            if nxt is not None:
                node, warm = nxt, True
                nxt = None
            else:
                node, warm = heapq.heappop(heap), False
            if node.basis is not None:
                stored -= 1
            if node.bound >= self._cutoff():
                self.stats["pruned_nodes"] += 1
                continue
            nodes += 1
            dom = self._domain(node)
            if dom is None:
                self.stats["infeasible_nodes"] += 1
                if opt.conflicts:
                    self._learn_conflict(node)
                continue
            lb, ub = dom
            lp.set_bounds(lb, ub)
            if not warm and node.basis is not None and node.seq != last_solved_seq:
                lp.set_basis(node.basis)
            with self._timed("node_lp"):
                st = lp.solve()
            if st not in (OPTIMAL, INFEASIBLE) and lp.past_deadline():
                heapq.heappush(heap, node)       # still open: its bound stands
                status = "time_limit"
                break
            if st not in (OPTIMAL, INFEASIBLE):
                lp.set_basis((np.ones(lp.rows, np.int8), np.ones(lp.lp.cols, np.int8)))   # crash
                st = lp.solve()
                if st not in (OPTIMAL, INFEASIBLE) and lp.past_deadline():
                    heapq.heappush(heap, node)
                    status = "time_limit"
                    break
            if st == INFEASIBLE:
                self.stats["infeasible_nodes"] += 1
                continue
            if st != OPTIMAL:
                # the LP failed (numerics): the node keeps its parent's bound,
                # which is valid, and is split on its widest open integer
                # column; with none open its bound is kept for the result
                self.stats["lp_failures"] = self.stats.get("lp_failures", 0) + 1
                open_int = np.nonzero(P.integer & (ub > lb))[0]
                if len(open_int) == 0:
                    self._lost_bound = min(getattr(self, "_lost_bound", INF), node.bound)
                    continue
                j = int(open_int[np.argmax(ub[open_int] - lb[open_int])])
                v = 0.5 * (lb[j] + ub[j]) if np.isfinite(ub[j] - lb[j]) else (lb[j] + 1 if np.isfinite(lb[j]) else ub[j] - 1)
                fv = math.floor(v)
                for side in ((j, False, fv), (j, True, fv + 1)):
                    heapq.heappush(heap, _Node(node.bound, -(node.depth + 1), next(seq), node.depth + 1,
                                               node.path + (side,), None, None, node.parent_obj))
                continue
            obj = self._round_bound(lp.objective)
            if node.branch is not None and np.isfinite(node.parent_obj):
                j, up, chg = node.branch
                pc.update(j, up, lp.objective - node.parent_obj, chg)
            if obj >= self._cutoff():
                self.stats["pruned_nodes"] += 1
                continue
            x = lp.x()
            frac = self._fractional(x)
            if frac.size == 0:
                self._try(x, "node_lp")
                continue
            if opt.heuristics:
                self._try(simple_rounding(P, x, self.down_locks, self.up_locks), "rounding")
                if (opt.rins_frequency and self.incumbent is not None and nodes % opt.rins_frequency == 0
                        and (self.inc_obj < getattr(self, "_rins_inc", INF) or nodes % (4 * opt.rins_frequency) == 0)):
                    self._rins(x, lb, ub)
                    lp.set_bounds(lb, ub)          # the sub-MIP has its own LP: ours is untouched
                if self.sched is not None and opt.schedule_frequency and nodes % opt.schedule_frequency == 0:
                    with self._timed("list_schedule"):
                        xs, _v = best_schedule(P, self.sched, schedule_orders(self.sched, x)[:2],
                                               time.perf_counter() + 0.5, max_evals=100)
                    self._try(xs, "list_schedule")
                dive_used = self.stats.get("dive_iterations", 0)
                node_its = lp.stats.iterations - dive_used - self.stats.get("strong_iterations", 0)
                if (opt.dive_frequency and nodes % opt.dive_frequency == 0
                        and dive_used <= opt.dive_budget_start + opt.dive_budget_ratio * node_its):
                    it0 = lp.stats.iterations
                    with self._timed("node_dive"):
                        self._try(guided_dive(P, self.prop, lp, lb, ub,
                                              min(self.deadline(), time.perf_counter() + 0.05 * opt.time_limit),
                                              self.incumbent, max_depth=50, iteration_limit=300), "dive")
                    self.stats["dive_iterations"] = dive_used + lp.stats.iterations - it0
            it0, t0 = lp.stats.iterations, lp.stats.seconds
            sb_used = self.stats.get("strong_iterations", 0)
            node_its = lp.stats.iterations - sb_used
            sb_ok = sb_used <= opt.strong_budget_start + opt.strong_budget_ratio * node_its
            with self._timed("branching"):
                dec = select_branch(pc, lp, x, frac, lb, ub, lp.objective, self._cutoff(),
                                    opt.strong_candidates if sb_ok else 0, opt.strong_iteration_limit)
            self.stats["strong_iterations"] = self.stats.get("strong_iterations", 0) + lp.stats.iterations - it0
            self.stats["strong_seconds"] = self.stats.get("strong_seconds", 0.0) + lp.stats.seconds - t0
            self.stats["strong_branches"] += 1
            j, v = dec.var, dec.value
            if dec.fix_down and dec.fix_up:
                continue
            if dec.fix_down or dec.fix_up:
                # one side proven empty: the node continues as the other
                side = (j, True, math.ceil(v)) if dec.fix_down else (j, False, math.floor(v))
                child = _Node(obj, -node.depth, next(seq), node.depth, node.path + (side,), None, None, lp.objective)
                nxt = child
                continue
            basis = lp.get_state() if stored < opt.max_stored_bases else None
            fj = v - math.floor(v)
            down = _Node(obj, -(node.depth + 1), next(seq), node.depth + 1, node.path + ((j, False, math.floor(v)),),
                         basis, (j, False, fj), lp.objective)
            upn = _Node(obj, -(node.depth + 1), next(seq), node.depth + 1, node.path + ((j, True, math.ceil(v)),),
                        basis, (j, True, 1 - fj), lp.objective)
            if basis is not None:
                stored += 2
            first, second = (upn, down) if fj >= 0.5 else (down, upn)
            heapq.heappush(heap, second)
            # plunge while the child is close to the best bound
            gap_ref = max(1.0, abs(self.inc_obj) if np.isfinite(self.inc_obj) else abs(obj))
            others = min([n.bound for n in heap], default=INF)
            if obj <= others + opt.plunge_gap * gap_ref:
                nxt = first
            else:
                heapq.heappush(heap, first)
            last_solved_seq = node.seq
            self.stats["nodes"] = nodes
            self.log(f"nodes {nodes} open {len(heap) + (nxt is not None)} bound "
                     f"{P.reported(best_bound):.6g} incumbent "
                     f"{P.reported(self.inc_obj) if np.isfinite(self.inc_obj) else '-'} "
                     f"lp its {lp.stats.iterations}")
        self.stats["nodes"] = nodes
        open_bounds = [n.bound for n in heap] + ([nxt.bound] if nxt else [])
        bound = min(open_bounds) if open_bounds else (self.inc_obj if np.isfinite(self.inc_obj) else INF)
        if status == "optimal" and open_bounds and not self._gap_closed(bound):
            status = "time_limit"
        if status == "optimal" and not np.isfinite(self.inc_obj):
            status = "infeasible"
        lost = getattr(self, "_lost_bound", INF)
        if lost < INF:
            bound = min(bound, lost)
            if status == "optimal" and not self._gap_closed(bound):
                status = "time_limit"        # a node that could not be solved stays open
        if np.isfinite(self.inc_obj):
            bound = min(bound, self.inc_obj)
        return self._result(status, self.incumbent, bound)

    # --------------------------------------------------------------- result
    def _result(self, status: str, x_red, bound_min: float) -> MipResult:
        P0 = self.original
        lp_stats = getattr(self, "lp", None)
        if lp_stats is not None:
            self.stats["lp_iterations"] = lp_stats.stats.iterations
            self.stats["lp_seconds"] = lp_stats.stats.seconds
            self.stats["fedlp"] = dict(lp_stats.stats.fedlp)
        x = obj = None
        if x_red is not None and np.isfinite(self.inc_obj):
            x = self.post.expand(x_red)
            obj = P0.reported(P0.objective(x))
        bound = P0.reported(bound_min) if np.isfinite(bound_min) else (P0.reported(bound_min))
        if obj is not None:
            gap = abs(obj - bound) / max(1e-10, abs(obj)) if np.isfinite(bound) else INF
        else:
            gap = INF
        r = MipResult(status, obj, bound, x, gap, self.stats.get("nodes", 0), self.elapsed(), self.stats)
        if self.opt.verbose and not self.opt.sub_mip:
            print(f"[{r.seconds:7.1f}s] {status}: objective {obj} bound {bound} gap {gap:.2e} "
                  f"nodes {r.nodes}", flush=True)
        return r


def solve(problem: MipProblem, **options) -> MipResult:
    return MipSolver(problem, MipOptions(**options)).solve()
