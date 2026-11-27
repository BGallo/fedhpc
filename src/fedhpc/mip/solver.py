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
from dataclasses import dataclass, field

import numpy as np

from .branching import Pseudocosts, select_branch
from .cuts import CutParams, CutPool, cover_cuts, gomory_cuts
from .heuristics import (feasibility_pump, fix_and_propagate, guided_dive, locks,
                         rens_problem, simple_rounding)
from .lp import INFEASIBLE, OPTIMAL, UNBOUNDED, LpRelaxation
from .presolve import presolve
from .problem import INF, MipProblem
from .propagate import Propagator


@dataclass
class MipOptions:
    time_limit: float = 3600.0
    node_limit: int = 10**9
    gap_rel: float = 1e-4
    gap_abs: float = 1e-6
    int_tol: float = 1e-6
    feas_tol: float = 1e-6
    presolve: bool = True
    cut_rounds: int = 15
    cuts_per_round: int = 200
    gomory: bool = True
    covers: bool = True
    cut_stall_rounds: int = 3
    cut_min_improvement: float = 1e-4     # relative bound gain counting as progress
    heuristics: bool = True
    rens: bool = True
    rens_node_limit: int = 300
    rens_time_share: float = 0.2
    dive_frequency: int = 25
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
    verbose: bool = True
    log_interval: float = 5.0
    sub_mip: bool = False                 # a heuristic's sub-MIP: no recursion, quiet


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
            return True
        return False

    # --------------------------------------------------------------- solve
    def solve(self) -> MipResult:
        opt = self.opt
        P0 = self.original
        if opt.presolve:
            P, self.post, pst = presolve(P0)
            self.stats["presolve"] = {k: v for k, v in vars(pst).items()}
            if pst.status != "reduced":
                return self._result(pst.status, None, INF if pst.status == "infeasible" else -INF)
        else:
            from .presolve import Postsolve
            P = P0
            self.post = Postsolve(P0.n, np.arange(P0.n), np.zeros(P0.n), np.arange(P0.m))
        self.P = P
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
        self.lp = LpRelaxation(P, opt.lp_options)
        self.lp.deadline = self.deadline()
        self.lp.set_bounds(self.root_lb, self.root_ub)
        st = self.lp.solve()
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
        root_obj = self._root_cuts(root_obj)
        self.stats["root_bound"] = P.reported(root_obj)
        if opt.heuristics:
            self._root_heuristics()
        return self._branch_and_bound(root_obj)

    # ------------------------------------------------------------ root cuts
    def _root_cuts(self, root_obj: float) -> float:
        opt, P, lp = self.opt, self.P, self.lp
        par = CutParams()
        pool = CutPool(par)
        stall, best = 0, root_obj
        for rnd in range(opt.cut_rounds):
            if self.elapsed() > 0.5 * opt.time_limit:
                break
            x = lp.x()
            if not self._fractional(x).size:
                self._try(x, "root_lp")
                break
            self._try(simple_rounding(P, x, self.down_locks, self.up_locks), "rounding")
            cuts = []
            if opt.gomory:
                cuts += gomory_cuts(lp, P, x, par)
            if opt.covers:
                cuts += cover_cuts(P, x, self.root_lb, self.root_ub, par)
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
                if stall >= opt.cut_stall_rounds:
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
        self._try(fix_and_propagate(P, self.prop, lp, x, self.root_lb, self.root_ub, end), "fix_and_propagate")
        self._try(guided_dive(P, self.prop, lp, self.root_lb, self.root_ub, end, self.incumbent), "dive")
        if self.incumbent is None:
            self._try(feasibility_pump(P, lp, x, self.root_lb, self.root_ub, end, prop=self.prop),
                      "feasibility_pump")
        if opt.rens and not opt.sub_mip:
            Q = rens_problem(P, x, self.root_lb, self.root_ub)
            if Q is not None:
                budget = max(1.0, opt.rens_time_share * (self.deadline() - time.perf_counter()))
                sub = MipSolver(Q, MipOptions(time_limit=budget, node_limit=opt.rens_node_limit,
                                              sub_mip=True, verbose=False, rens=False,
                                              cut_rounds=3, lp_options=opt.lp_options))
                r = sub.solve()
                if r.x is not None:
                    self._try(r.x, "rens")

    # ------------------------------------------------------- branch and bound
    def _domain(self, node: _Node):
        lb, ub = self.root_lb.copy(), self.root_ub.copy()
        for j, up, v in node.path:
            if up:
                lb[j] = max(lb[j], v)
            else:
                ub[j] = min(ub[j], v)
        if np.any(lb > ub):
            return None
        res = self.prop.propagate(lb, ub, max_rounds=10)
        return (res.lb, res.ub) if res.feasible else None

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
                continue
            lb, ub = dom
            lp.set_bounds(lb, ub)
            if not warm and node.basis is not None and node.seq != last_solved_seq:
                lp.set_basis(node.basis)
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
                # cannot bound this node: keep it unpruned by branching on
                # the most fractional of the last LP point would be unsafe
                # without a bound, so it stays with its parent's bound
                self.stats.setdefault("lp_failures", 0)
                self.stats["lp_failures"] += 1
                status = "error"
                heapq.heappush(heap, node)
                break
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
                dive_used = self.stats.get("dive_iterations", 0)
                node_its = lp.stats.iterations - dive_used - self.stats.get("strong_iterations", 0)
                if (opt.dive_frequency and nodes % opt.dive_frequency == 0
                        and dive_used <= opt.dive_budget_start + opt.dive_budget_ratio * node_its):
                    it0 = lp.stats.iterations
                    self._try(guided_dive(P, self.prop, lp, lb, ub,
                                          min(self.deadline(), time.perf_counter() + 0.05 * opt.time_limit),
                                          self.incumbent, max_depth=50, iteration_limit=300), "dive")
                    self.stats["dive_iterations"] = dive_used + lp.stats.iterations - it0
            it0, t0 = lp.stats.iterations, lp.stats.seconds
            sb_used = self.stats.get("strong_iterations", 0)
            node_its = lp.stats.iterations - sb_used
            sb_ok = sb_used <= opt.strong_budget_start + opt.strong_budget_ratio * node_its
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
            basis = lp.get_basis() if stored < opt.max_stored_bases else None
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
