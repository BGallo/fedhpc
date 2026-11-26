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
- capacity ``sum_{j, t <= τ < t + p_jm} q_j x_jmt <= N_m(τ)`` (dual
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
``(w1 - ν1) c1 + (w2 - ν2) c2 - σ_j - q_j sum_{τ in [t, t+p)} π_mτ``, where
``q_j`` is the job's node count (``Job.nodes``, its capacity units). All of
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

Objective scaling
-----------------
Every solve works in units where one step of the objective's coarser term
is 1 (:func:`fedhpc.model.weighted_sum_scale`). The normalised weighted
sums have weights near 1e-5; unscaled, cost differences sat below Gurobi's
OptimalityTol and the eligible MIP "proved" schedules 1.7e-5 worse than the
optimum (B_pos_n100_lE_* at λ = 0.75). The artificials' big-M starts at
1e3 times the largest objective coefficient and grows 100-fold while
artificials stay in a converged LP (HiGHS fails on a fixed 1e6 factor).

Primal heuristics (Phase 7, ``LPFirstOptions.heuristics``)
-----------------------------------------------------------
Between column generation and step 2, from the converged LP:

- α-point list scheduling (α = 0.5, 0.25, 0.75, 0): jobs in α-point
  order, each at the earliest start with room on its LP type (or a
  no-dearer capacitated type that completes it sooner);
- fix-and-solve (RINS-like): fix every job the LP places integrally (about
  95 % of them on the generated pos instances) and solve the MIP over the
  rest, widening the free set to overlapping fixed jobs if it fails;
- LNS: sub-MIPs over sliding time windows of the incumbent (about
  ``lns_jobs`` jobs each), at most ``lns_sweeps`` sweeps.

The sub-MIPs use only the freed jobs' master columns plus their
``cand_per_job`` placements of least reduced-cost gap. On B_pos_n100_lE_iM
(f1^0, 3340 jobs) they reach 108211.67 in 26 s; the optimum is 108210.67,
the seed was 267085.67.

Job aggregation (Phase 8)
-------------------------
Every MIP (:func:`_mip`) merges identical free jobs (:meth:`_TI.classes`:
same weight, release slot, units and per-type start sets, durations and
costs) into one coverage row ``sum x = n_k`` with integer ``x <= n_k``, and
spreads the solution back over the members. Exact, since such jobs are
interchangeable. With cloud costs every job is distinct (I/O volumes
differ), but once f2 is capped at 0 the cloud placements leave the
eligible set and 3340 jobs collapse into about 730 classes (up to 156
members). On B_pos_n100_lE_iM the eligible MIP fell from > 780 s
(unproven) to 407 s.

Placement tree (Phase 9, ``LPFirstOptions.tree``)
-------------------------------------------------
When more than ``max_eligible`` placements are eligible, or there is no
incumbent, step 3 becomes a best-first branch and price over placements
(:class:`_Tree`): nodes restrict the allowed placements, :func:`_cg`
prices only those, reduced-cost fixing shrinks each node's set, a node
small enough is solved by its MIP, and branching splits the most
fractional job by type or at the half-mass start (Kramer et al. 2020).

It is a fallback, not the default proof. On B_pos_n100_lE_iM, starting
from the full 250 k eligible set, 175 nodes in 800 s left the bound at the
LP value (108206.97), where the aggregated eligible MIP proves 108210.67
in about 400 s. Branching one job of a class lets an identical twin take
over its LP mass, so the bound does not move. A useful tree there would
branch on class counts. Each node also rebuilds its LP from scratch
(614 s of LP time over 214 iterations).

Backends
--------
``LPFirstOptions.backend`` picks Gurobi or HiGHS (:mod:`fedhpc.lp_backend`)
for every LP and MIP. Columns go in as CSC batches. HiGHS's duals give the
same Lagrangian bound (``tests/test_lp_backend.py``). Measured on the
B_pos_n100_lE_iM master (4.5 k rows, 141 k columns, 4 threads): the LP from
scratch takes 15.5 s with HiGHS and 6.5 s with Gurobi. At 180 s the MIP has
the same dual bound (108210.67) but a worse incumbent (108221.67 vs
108213.67).

Proof first, and the model build (2026-11-26)
---------------------------------------------
Profiling showed lpfirst's time was not in its LPs: on pos_congestion's five
bench solves only 6 s of 132 s were LP. The rest was the LNS sub-MIPs and
building their Gurobi models.

- **Proof first** (``LPFirstOptions.quick_proof``). After the α-points, the
  restricted-master MIP runs for at most ``quick_rmp_time`` (5 s). Then the
  eligible MIP, when at most ``quick_eligible`` placements are eligible,
  runs for at most ``quick_proof_time``. Only then fix-and-solve and LNS.
  The restricted-master MIP runs again after the heuristics if they improved
  the incumbent: on the zero-cost phases of B_pos_n100_lE_* that is what
  brings the incumbent to the optimum the eligible MIP then proves. Neither
  the gap nor the eligible-set size predicts whether the early MIP pays.
  pos_congestion's boxes start 86-254 % above the bound with every
  placement eligible, and it finds the near-optimum in 1-2 s. So it always
  runs, with a short cap.
- **Batch model build** (:class:`fedhpc.lp_backend.GurobiBackend`). Rows
  and columns added before the first solve become one ``addMVar`` plus one
  sparse ``addMConstr`` (scipy), instead of one ``addVar`` per placement
  (28 s of the 132).
- **Vectorised placement lookup.** :meth:`_TI.arcs_of` is one
  ``searchsorted`` over sorted (job, type, start) keys, replacing 3.2 M
  Python lookups.

Results, five bench solves each, all the same proven optima:

=====================  =========  =========
Instance               before     after
=====================  =========  =========
fedhpc_known (964)     13.6 s     9.3 s
pos_congestion (3340)  109 s      27 s
=====================  =========  =========

On the 45 B_pos_n100_* generated solves (3 workers × 4 threads, as P78), the
total went from 2844 s to 2414 s, and the unproven solves from 2 to 1.
lex_f1 is 5-15x faster. lex_f2 on the lE instances is slower (137 -> 322 s
on lE_iH), as their eligible MIP is sensitive to the exact incumbent that
LNS and the restricted-master MIP reach.

Reference points (:func:`reference_points_lpfirst`)
---------------------------------------------------
The weighted-sum reference points of ``pareto.weighted_sum_reference``,
licence-free on HiGHS (``engine="lpfirst"`` there). Each phase runs on the
placements its optima can use: the lex-min (f1, f2) phases drop every
placement completing after the job's earliest uncapacitated completion
(:meth:`_TI.lex_f1_mask`, 2-8 % of the placements remain), and the f1_0
phase keeps only free ones (:meth:`_TI.free_mask`). Without the cloud,
identical jobs abound (3340 jobs in 730 classes on the pos instances), so
:data:`REFERENCE_OPTIONS` also runs column generation on the
class-aggregated master (``aggregate_lp``, :meth:`_TI.aggregated`) and skips
LNS, whose sweeps cost more than the proof steps they spare. Measured
2026-11-23 (HiGHS, all four points proven):

==========================  ==========  =================  ===============
Instance                    Gurobi      lpfirst/HiGHS,     this path
                            (compact)   unrestricted
==========================  ==========  =================  ===============
fedhpc_known (964 jobs)     12.6 s      4.2 s              1.1 s
A_fedhpc_n100_lE_iM         8.0 s       3.9 s              0.6 s
B_pos_n100_lH_iM (3340)     > 13 min    f1_0 unproven      45-59 s
                                        after 600 s
==========================  ==========  =================  ===============

On B_pos the f1_0 phase dominates: the aggregated LP takes 5 s (32 s
unaggregated), the rest is HiGHS MIPs. The incumbent entering the eligible
MIP matters most: one f1 unit worse (no restricted-master MIP) and that
MIP takes 229 s instead of 11 s.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace

import gurobipy as gp
import numpy as np
from gurobipy import GRB

INF = math.inf

from . import bpc, formulations, lp_backend
from .data import Instance
from .model import Solution, grid_floor, weighted_sum_scale

__all__ = ["LPFirstOptions", "LPFirstResult", "solve_lpfirst", "solve_weighted_sum_lpfirst",
           "solve_f1_lpfirst", "solve_f2_lpfirst", "make_lpfirst_lexmin_bounded",
           "ReferencePoints", "REFERENCE_OPTIONS", "reference_points_lpfirst"]


@dataclass(frozen=True)
class LPFirstOptions:
    """``smoothing`` — Wentges weight on the best-bound duals (0 disables).
    ``per_job`` — columns added per job per iteration (its most negative
    placements). ``max_cols`` — columns added per iteration in total.
    ``tol`` — reduced-cost tolerance. ``rmp_mip_time`` — seconds for the
    restricted-master MIP (step 2). ``max_eligible`` — the most placements
    the exact MIP (step 3) may have; past it the result is not certified.
    ``mip_threads`` — Gurobi threads for both MIPs (0: Gurobi's default).
    ``lp_threads`` — Gurobi threads for the master LP (0: Gurobi's default).

    Primal heuristics (step 1b, Phase 7): ``heuristics`` turns them on;
    ``heuristic_time`` caps them in all. ``cand_per_job`` — placements of
    least reduced-cost gap a freed job may take in a sub-MIP. ``fix_time`` —
    seconds per fix-and-solve MIP. ``lns_jobs`` — jobs starting in one LNS
    window; ``lns_time_per`` — seconds per LNS sub-MIP; ``lns_sweeps`` —
    full passes over the horizon.

    Search tree (step 3', Phase 9): with ``tree``, an eligible set larger
    than ``max_eligible`` (where step 3 would give up), or a solve without
    an incumbent, is branched on instead; a node whose eligible set has at
    most ``node_eligible`` placements gets its MIP, for ``node_mip_time``
    seconds at most. Below ``max_eligible`` the aggregated eligible MIP is
    faster (see the module docstring).

    ``backend`` — LP/MIP solver for every master and MIP: ``"gurobi"`` or
    ``"highs"`` (:mod:`fedhpc.lp_backend`).

    Proof first (``quick_proof``): after the α-point schedules, the
    restricted-master MIP (for up to ``quick_rmp_time`` seconds) and, when at
    most ``quick_eligible`` placements are then eligible, the exact eligible
    MIP (``quick_proof_time``) run before fix-and-solve and LNS; a proof
    there skips them.
    Otherwise the restricted-master MIP runs again after the heuristics, with
    their incumbent as cutoff — on the zero-cost phases of B_pos_n100_lE_* it
    is what brings the incumbent to the optimum the eligible MIP then proves. The heuristics only pay when the gap is too
    wide for that MIP: on pos_congestion they took the five bench solves
    from 17 s to over 100 s, and on B_pos_n100_lE_iL 30 LNS sub-MIPs (20 s)
    ended above the optimum the 0.06 s eligible MIP then proved.

    ``aggregate_lp`` — run column generation on the class-aggregated master
    (:meth:`_TI.aggregated`) when identical jobs cut the job count by at
    least a fifth, then map its duals and solution back to the jobs. Exact:
    the aggregated LP has the same value and its duals, spread over a class,
    are optimal for the full LP.
    """
    smoothing: float = 0.5
    per_job: int = 3
    max_cols: int = 20_000
    tol: float = 1e-6
    rmp_mip_time: float = 30.0
    max_eligible: int = 400_000
    mip_threads: int = 0
    lp_threads: int = 0
    heuristics: bool = True
    heuristic_time: float = 120.0
    cand_per_job: int = 30
    fix_time: float = 30.0
    lns_jobs: int = 400
    lns_time_per: float = 10.0
    lns_sweeps: int = 2
    tree: bool = True
    backend: str = "gurobi"
    aggregate_lp: bool = False
    node_eligible: int = 60_000
    node_mip_time: float = 60.0
    quick_proof: bool = True
    quick_eligible: int = 60_000
    quick_proof_time: float = 15.0
    quick_rmp_time: float = 5.0


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
        J, K, T, P, Q, C1, C2, MID = [], [], [], [], [], [], [], []
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
                Q.append(np.full(len(ts), j.nodes, dtype=float))
                C1.append(j.weight * (ts + p - j.arrival))
                C2.append(np.full(len(ts), c2, dtype=float))
                n += len(ts)
        cat = np.concatenate
        self.J, self.K, self.T, self.P, self.MID = cat(J), cat(K), cat(T), cat(P), cat(MID)
        self.Q = cat(Q)   # capacity units a placement holds per slot (Job.nodes)
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
        self._rows_needed()

    def _rows_needed(self) -> None:
        """Slots some placement can occupy (only those need a capacity row)."""
        used = np.zeros((len(self.types), self.H + 1), dtype=np.int64)
        kc, tc, pc = self.K[self.capped], self.T[self.capped], self.P[self.capped]
        np.add.at(used, (kc, tc), 1)
        np.add.at(used, (kc, tc + pc), -1)
        self.row_needed = np.cumsum(used, axis=1)[:, :self.H] > 0

    def restrict(self, keep: np.ndarray) -> "_TI":
        """A copy holding only the placements ``keep`` marks (a boolean mask
        over this one's). Each job must keep at least one; capacities and
        capacity rows stay those of the full instance, so the restricted LP is
        the full one with the dropped columns fixed to 0."""
        keep = np.asarray(keep, dtype=bool)
        new = object.__new__(_TI)
        new.__dict__.update({k: v for k, v in self.__dict__.items()
                             if k not in ("_classes", "class_rep", "class_offset")})
        for name in ("J", "K", "T", "P", "MID", "Q", "C1", "C2"):
            setattr(new, name, getattr(self, name)[keep])
        new.n_arcs = int(keep.sum())
        new.first = np.searchsorted(new.J, np.arange(len(new.jobs)))
        if np.any(np.diff(np.append(new.first, new.n_arcs)) == 0):
            raise ValueError("some job has no placement")
        new.capped = new.K >= 0
        new._rows_needed()
        new.pair = {}
        # runs of equal (job, type) are contiguous: the arrays are sorted by job, then type
        if new.n_arcs:
            brk = np.nonzero((np.diff(new.J) != 0) | (np.diff(new.MID) != 0))[0] + 1
            for lo, hi in zip(np.r_[0, brk].tolist(), np.r_[brk, new.n_arcs].tolist()):
                new.pair[new.jobs[new.J[lo]], int(new.MID[lo])] = (lo, new.T[lo:hi])
        return new

    def lex_f1_mask(self) -> np.ndarray:
        """Placements that can occur in a minimum-f1 schedule: for a job of
        weight > 0 with an uncapacitated placement, none completing after its
        earliest uncapacitated completion ``e_j``. Swapping such a placement
        for the one completing at ``e_j`` frees capacity and lowers f1, so
        every f1-optimal schedule, and hence the lex-min (f1, f2) one and any
        schedule with f1 <= f1_T, avoids them (the lex deadlines of
        ``SpaceTimeFormulation``). Requires nonnegative weights."""
        comp = self.T + self.P
        e = np.full(len(self.jobs), np.inf)
        np.minimum.at(e, self.J[~self.capped], comp[~self.capped])
        wt = np.array([j.weight for j in self.inst.jobs], dtype=float)
        pos = np.array([self.jpos[j.id] for j in self.inst.jobs])
        w = np.empty(len(self.jobs))
        w[pos] = wt
        return (comp <= e[self.J]) | (w[self.J] <= 0)

    def free_mask(self) -> np.ndarray:
        """Placements that cost nothing: with nonnegative costs, the only ones a
        schedule with f2 <= 0 can use."""
        return self.C2 <= 0

    def representable(self, assignment: dict[int, tuple[int, int]]) -> bool:
        """Whether every placement of a complete schedule is one of ours."""
        return len(assignment) == len(self.jobs) and len(self.arcs_of(assignment)) == len(assignment)

    def aggregated(self) -> "_TI":
        """The class-aggregated placement set (Phase 8 applied to the master):
        one job per class (its representative, :meth:`classes`), with
        ``mult`` (the class size) as its coverage right-hand side. Extra
        attributes map it back: ``full`` — the full set this came from;
        ``cls`` — each full job position's class; ``agg_of`` — each full
        placement's counterpart here; ``offset`` — each full job's ``C1``
        minus its representative's at the same placement."""
        cls, canon = self.classes()
        reps = self.class_rep
        keep = np.zeros(self.n_arcs, dtype=bool)
        is_rep = np.zeros(len(self.jobs), dtype=bool)
        is_rep[reps] = True
        keep[is_rep[self.J]] = True
        new = object.__new__(_TI)
        new.__dict__.update({k: v for k, v in self.__dict__.items()
                             if k not in ("_classes", "class_rep", "class_offset")})
        new.jobs = [self.jobs[r] for r in reps.tolist()]
        new.jpos = {jid: i for i, jid in enumerate(new.jobs)}
        for name in ("K", "T", "P", "MID", "Q", "C1", "C2"):
            setattr(new, name, getattr(self, name)[keep])
        new.J = cls[self.J[keep]]
        new.n_arcs = int(keep.sum())
        new.first = np.searchsorted(new.J, np.arange(len(new.jobs)))
        new.capped = new.K >= 0
        new._rows_needed()
        new.pair = {}
        brk = np.nonzero((np.diff(new.J) != 0) | (np.diff(new.MID) != 0))[0] + 1
        for lo, hi in zip(np.r_[0, brk].tolist(), np.r_[brk, new.n_arcs].tolist()):
            new.pair[new.jobs[new.J[lo]], int(new.MID[lo])] = (lo, new.T[lo:hi])
        new.mult = np.bincount(cls, minlength=len(reps)).astype(float)
        new.class_rep_full = reps
        new.full, new.cls, new.offset = self, cls, self.class_offset
        new.full_index = np.nonzero(keep)[0]
        new.agg_of = np.searchsorted(new.full_index, canon)
        return new

    def disaggregate(self, x_agg: np.ndarray) -> np.ndarray:
        """An aggregated solution as one over the full placements (of
        ``self.full``) with the same capacity use and objective: each class's
        mass, in start order, fills its members one unit at a time, so a
        member is integral wherever the class's mass allows."""
        full = self.full
        x = np.zeros(full.n_arcs)
        nz = np.nonzero(x_agg > 1e-12)[0]
        if not len(nz):
            return x
        nz = nz[np.lexsort((self.T[nz], self.J[nz]))]
        members = np.argsort(self.cls, kind="stable")
        mstart = np.r_[0, np.cumsum(self.mult.astype(np.int64))]
        brk = np.r_[0, np.nonzero(np.diff(self.J[nz]))[0] + 1, len(nz)]
        for lo, hi in zip(brk[:-1].tolist(), brk[1:].tolist()):
            idx = nz[lo:hi]
            k = int(self.J[idx[0]])
            mem = members[mstart[k]:mstart[k + 1]]
            v = x_agg[idx]
            cum = np.r_[0.0, np.cumsum(v)]
            i = np.arange(len(mem), dtype=float)[:, None]
            share = np.clip(np.minimum(i + 1, cum[None, 1:]) - np.maximum(i, cum[None, :-1]),
                            0.0, None)                         # members x placements
            # member j's copy of the representative's placement
            rep_arc = self.full_index[idx]
            rep_first = full.first[self.class_rep_full[k]]
            arcs = full.first[mem][:, None] + (rep_arc - rep_first)[None, :]
            m = share > 1e-12
            x[arcs[m]] += share[m]
        return x

    def classes(self) -> tuple[np.ndarray, np.ndarray]:
        """``(cls, canon)``: each job position's class and each placement's
        counterpart on its class representative (Phase 8).

        Two jobs are in one class when they have the same weight, release
        slot, capacity units and, type by type, the same start set, duration
        and cost. Such jobs are interchangeable: swapping them changes f1 by
        nothing (their ``C1`` differ by the constant ``w (a_rep - a_j)`` at
        every start) and f2 not at all, and the start-set reductions treat
        them alike because their start sets are compared directly."""
        if getattr(self, "_classes", None) is None:
            job_of = {j.id: j for j in self.inst.jobs}
            ends = np.append(self.first[1:], self.n_arcs)
            key_cls: dict[tuple, int] = {}
            cls = np.empty(len(self.jobs), dtype=np.int64)
            rep: list[int] = []
            for i, jid in enumerate(self.jobs):
                lo, hi = int(self.first[i]), int(ends[i])
                j = job_of[jid]
                key = (j.weight, max(0, math.ceil(j.arrival)), float(self.Q[lo]),
                       self.MID[lo:hi].tobytes(), self.T[lo:hi].tobytes(),
                       self.P[lo:hi].tobytes(), self.C2[lo:hi].tobytes())
                c = key_cls.setdefault(key, len(rep))
                if c == len(rep):
                    rep.append(i)
                cls[i] = c
            rep_arr = np.array(rep, dtype=np.int64)
            canon = np.arange(self.n_arcs) - self.first[self.J] + self.first[rep_arr[cls[self.J]]]
            # C1_j = C1_rep + offset_j at every start: offset_j = w (a_rep - a_j)
            arr = np.array([job_of[jid].arrival for jid in self.jobs], dtype=float)
            wt = np.array([job_of[jid].weight for jid in self.jobs], dtype=float)
            self.class_rep = rep_arr
            self.class_offset = wt * (arr[rep_arr[cls]] - arr)
            self._classes = (cls, canon)
        return self._classes

    def index(self, jid: int, mid: int, t: int) -> int | None:
        hit = self.pair.get((jid, mid))
        if hit is None:
            return None
        lo, ts = hit
        i = int(np.searchsorted(ts, t))
        return lo + i if i < len(ts) and ts[i] == t else None

    def _keys(self, jpos, mid_rank, t) -> np.ndarray:
        return (np.asarray(jpos, dtype=np.int64) * (len(self._mid_rank) + 1)
                + np.asarray(mid_rank, dtype=np.int64)) * (self.H + 1) + np.asarray(t, dtype=np.int64)

    def arcs_of(self, assignment: dict[int, tuple[int, int]]) -> list[int]:
        """The placements of ``assignment`` (those that exist), in one
        vectorised lookup over the sorted (job, type, start) keys."""
        if not assignment:
            return []
        if not hasattr(self, "_sorted_keys"):
            self._mid_rank = {m: r for r, m in enumerate(sorted({int(v) for v in np.unique(self.MID)}))}
            rank = np.array([self._mid_rank[int(v)] for v in np.unique(self.MID)])
            mid_rank = rank[np.searchsorted(np.unique(self.MID), self.MID)]
            keys = self._keys(self.J, mid_rank, self.T)
            self._key_order = np.argsort(keys, kind="stable")
            self._sorted_keys = keys[self._key_order]
        jp, mr, ts = [], [], []
        jpos, mrank = self.jpos, self._mid_rank
        for jid, (mid, t) in assignment.items():
            i, r = jpos.get(jid), mrank.get(mid)
            if i is None or r is None or not 0 <= t <= self.H:
                continue
            jp.append(i), mr.append(r), ts.append(t)
        if not jp:
            return []
        q = self._keys(jp, mr, ts)
        pos = np.searchsorted(self._sorted_keys, q)
        pos = np.minimum(pos, len(self._sorted_keys) - 1)
        hit = self._sorted_keys[pos] == q
        return self._key_order[pos[hit]].tolist()

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
    """The restricted master LP (or, with ``integer=True`` and no
    artificials, a MIP over a fixed set of placements), on an
    :mod:`fedhpc.lp_backend` backend (``"gurobi"`` or ``"highs"``)."""

    def __init__(self, ti: _TI, w1: float, w2: float, b1: float | None, b2: float | None,
                 big_m: float | None, integer: bool = False, *,
                 free: np.ndarray | None = None, occupy: np.ndarray | None = None,
                 cover_rhs: np.ndarray | None = None, backend: str = "gurobi",
                 threads: int = 0):
        """``free`` (bool per job position) — only these jobs get a coverage
        row; the others are fixed elsewhere. ``occupy`` (types × H) —
        capacity those fixed jobs already hold. Callers shift the bound
        rows themselves. ``cover_rhs`` (per job position) — coverage
        right-hand sides, and the placements' upper bounds, for an
        aggregated model (Phase 8); default 1."""
        self.ti, self.w1, self.w2, self.integer = ti, w1, w2, integer
        self.be = lp_backend.make(backend, threads)
        nj = len(ti.jobs)
        self.rhs = np.ones(nj) if cover_rhs is None else np.asarray(cover_rhs, float)
        self.free = np.ones(nj, dtype=bool) if free is None else np.asarray(free, dtype=bool)
        fj = np.nonzero(self.free)[0]
        self.cover_row = np.full(nj, -1, dtype=np.int64)
        self.cover_row[fj] = self.be.add_rows(self.rhs[fj], self.rhs[fj])
        cap_rhs = ti.N if occupy is None else ti.N - occupy
        kk, tt = np.nonzero(ti.row_needed)
        self.cap_row = np.full((len(ti.types), ti.H), -1, dtype=np.int64)
        self.cap_row[kk, tt] = self.be.add_rows(np.full(len(kk), -INF), cap_rhs[kk, tt].astype(float))
        self.b1_row = int(self.be.add_rows(np.array([-INF]), np.array([float(b1)]))[0]) \
            if b1 is not None else -1
        self.b2_row = int(self.be.add_rows(np.array([-INF]), np.array([float(b2)]))[0]) \
            if b2 is not None else -1
        self.B1, self.B2 = b1, b2
        self.n_art = 0
        self.big_m = big_m
        self.big_m_cap = (big_m or 0.0) * 1e6
        if big_m is not None:
            rows = self.cover_row[fj]
            self.be.add_cols(np.full(len(fj), float(big_m)), np.full(len(fj), INF),
                             np.zeros(len(fj), dtype=bool), np.arange(len(fj) + 1),
                             rows, np.ones(len(fj)))
            self.n_art = len(fj)
        self.arcs: list[int] = []
        self.has = np.zeros(ti.n_arcs, dtype=bool)
        self._x = None

    def add(self, arcs) -> None:
        ti = self.ti
        arcs = np.unique(np.asarray(arcs, dtype=np.int64))
        arcs = arcs[~self.has[arcs]]
        if not len(arcs):
            return
        self.has[arcs] = True
        n = len(arcs)
        K, T, P, J = ti.K[arcs], ti.T[arcs], ti.P[arcs], ti.J[arcs]
        p_eff = np.where(K >= 0, P, 0)
        nb = (self.b1_row >= 0) + (self.b2_row >= 0)
        per = 1 + p_eff + nb
        starts = np.r_[0, np.cumsum(per)]
        index = np.empty(int(starts[-1]), dtype=np.int64)
        value = np.empty(int(starts[-1]))
        index[starts[:-1]] = self.cover_row[J]
        value[starts[:-1]] = 1.0
        rep = np.repeat(np.arange(n), p_eff)
        off = np.arange(len(rep)) - np.repeat(np.cumsum(p_eff) - p_eff, p_eff)
        pos = starts[:-1][rep] + 1 + off
        index[pos] = self.cap_row[K[rep], T[rep] + off]
        value[pos] = ti.Q[arcs][rep]
        nxt = starts[:-1] + 1 + p_eff
        if self.b1_row >= 0:
            index[nxt] = self.b1_row
            value[nxt] = ti.C1[arcs]
            nxt = nxt + 1
        if self.b2_row >= 0:
            index[nxt] = self.b2_row
            value[nxt] = ti.C2[arcs]
        if np.any(index < 0):
            raise ValueError("a placement of a fixed job, or outside the capacity rows")
        cost = self.w1 * ti.C1[arcs] + self.w2 * ti.C2[arcs]
        self.be.add_cols(cost, self.rhs[J], np.full(n, self.integer), starts, index, value)
        self.arcs.extend(arcs.tolist())
        self._x = None

    def raise_big_m(self, factor: float = 100.0) -> None:
        """Make the coverage artificials ``factor`` times dearer (adaptive
        big-M: start moderate, which HiGHS needs, and grow only while
        artificials stay in the LP)."""
        self.big_m *= factor
        self.be.change_costs(np.arange(self.n_art), np.full(self.n_art, float(self.big_m)))
        self._x = None

    def col_of(self) -> dict[int, int]:
        return {a: self.n_art + i for i, a in enumerate(self.arcs)}

    def solve_lp(self) -> str:
        self._x = None
        return self.be.solve_lp()

    def solve_mip(self, **kw) -> str:
        self._x = None
        return self.be.solve_mip(**kw)

    def objective(self) -> float:
        return self.be.objective()

    def duals(self) -> _D:
        ti = self.ti
        rd = self.be.row_duals()
        sigma = np.zeros(len(ti.jobs))
        fj = np.nonzero(self.free)[0]
        sigma[fj] = rd[self.cover_row[fj]]
        pi = np.zeros((len(ti.types), ti.H))
        m = self.cap_row >= 0
        pi[m] = np.minimum(0.0, rd[self.cap_row[m]])
        return _D(sigma, pi,
                  min(0.0, float(rd[self.b1_row])) if self.b1_row >= 0 else 0.0,
                  min(0.0, float(rd[self.b2_row])) if self.b2_row >= 0 else 0.0)

    def _vals(self) -> np.ndarray:
        if self._x is None:
            self._x = self.be.col_values()
        return self._x

    def artificial(self) -> float:
        return float(self._vals()[:self.n_art].sum())

    def arc_values(self) -> np.ndarray:
        """Values of the placement columns, in ``self.arcs`` order."""
        return self._vals()[self.n_art:]

    def x(self) -> np.ndarray:
        """The current solution as a dense vector over all placements."""
        out = np.zeros(self.ti.n_arcs)
        if self.arcs:
            out[np.asarray(self.arcs)] = self.arc_values()
        return out

    def chosen(self) -> list[int]:
        return [a for a, v in zip(self.arcs, self.arc_values().tolist()) if v > 0.5]

    def assignment(self) -> dict[int, tuple[int, int]] | None:
        chosen = self.chosen()
        out = self.ti.assignment_of(chosen)
        return out if len(out) == len(self.ti.jobs) == len(chosen) else None


def _price(ti: _TI, d: _D, w1: float, w2: float) -> np.ndarray:
    """Reduced cost of every placement at ``d``."""
    pre = np.zeros((len(ti.types), ti.H + 1))
    np.cumsum(d.pi, axis=1, out=pre[:, 1:])
    rc = (w1 - d.nu1) * ti.C1 + (w2 - d.nu2) * ti.C2 - d.sigma[ti.J]
    c = ti.capped
    k, t, p = ti.K[c], ti.T[c], ti.P[c]
    rc[c] -= ti.Q[c] * (pre[k, t + p] - pre[k, t])
    return rc


def _lagrangian(ti: _TI, d: _D, rc: np.ndarray, b1: float | None, b2: float | None
                ) -> tuple[float, np.ndarray]:
    """``(L, min_j)``: the bound at ``d`` and each job's least ``rc + σ_j``."""
    mins = np.minimum.reduceat(rc + d.sigma[ti.J], ti.first)
    mult = getattr(ti, "mult", None)
    L = float((mins.sum() if mult is None else mins @ mult) + (d.pi * ti.N).sum())
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
         threads: int, start: dict | None = None, *, fixed=None, focus: int = 0,
         aggregate: bool = True, backend: str = "gurobi"):
    """Solve the placement MIP over ``arcs`` for a schedule below ``cutoff``.
    Returns ``(status, objective, assignment)``: ``"optimal"`` (proven on the
    grid), ``"none"`` (proven: nothing below the cutoff) or ``"limit"``.

    ``fixed`` — placements held fixed (one per job, none of them in
    ``arcs``): only the other jobs get coverage rows, the fixed ones' capacity
    and bound-row use is taken off the right-hand sides, and the result is a
    sub-MIP of the full one (a large-neighbourhood step). ``"none"`` then only
    means nothing below the cutoff *with those jobs fixed*.

    ``aggregate`` (Phase 8) — identical free jobs (:meth:`_TI.classes`) share
    one coverage row ``sum x = n_k`` over their representative's placements,
    with integer ``x <= n_k``; the solution is spread back over the members.
    Exact, since members are interchangeable; it removes the symmetry that
    makes these MIPs slow on the zero-cost (all on-prem) phase."""
    fixed = np.asarray(fixed if fixed is not None else [], dtype=np.int64)
    free = occ = None
    const = 0.0
    if len(fixed):
        free = np.ones(len(ti.jobs), dtype=bool)
        free[ti.J[fixed]] = False
        occ = _occupancy(ti, fixed)
        const = float(w1 * ti.C1[fixed].sum() + w2 * ti.C2[fixed].sum())
        if b1 is not None:
            b1 = b1 - float(ti.C1[fixed].sum())
        if b2 is not None:
            b2 = b2 - float(ti.C2[fixed].sum())
        arcs = [a for a in arcs if free[ti.J[a]]]
    arcs = np.asarray(arcs, dtype=np.int64)
    members = None
    rhs = None
    if aggregate and len(arcs):
        cls, canon = ti.classes()
        fmask = np.ones(len(ti.jobs), dtype=bool) if free is None else free
        fj = np.nonzero(fmask)[0]
        counts = np.bincount(cls[fj], minlength=len(ti.class_rep))
        if counts.max() > 1:
            rhs = np.zeros(len(ti.jobs))
            rhs[ti.class_rep] = counts
            free = rhs > 0
            arcs = np.unique(canon[arcs])
            off = float(ti.class_offset[fj].sum())   # Σ C1_j - C1_rep over the free jobs
            const += w1 * off
            if b1 is not None:
                b1 = b1 - off
            members = {}
            for i in fj:
                members.setdefault(int(cls[i]), []).append(int(i))
    m = _Master(ti, w1, w2, b1, b2, None, integer=True, free=free, occupy=occ,
                cover_rhs=rhs, backend=backend, threads=threads)
    m.add(arcs)
    start_vals: dict[int, float] = {}
    if start:
        col = m.col_of()
        canon = ti.classes()[1] if members is not None else None
        for a in ti.arcs_of(start):
            if canon is not None:
                a = int(canon[a])
            if a in col:
                start_vals[col[a]] = start_vals.get(col[a], 0.0) + 1.0
    status = m.solve_mip(time_limit=time_limit,
                         cutoff=cutoff - const if math.isfinite(cutoff) else cutoff,
                         gap_abs=0.999 * step if step > 0 else 1e-9, focus=focus,
                         start=start_vals)
    if status == "infeasible":
        return "none", math.inf, None
    a = None
    if m.be.has_solution():
        if members is None:
            chosen = m.chosen()
        else:
            cls, _canon = ti.classes()
            chosen = []
            left = {c: list(v) for c, v in members.items()}
            for a, v in zip(m.arcs, m.arc_values().tolist()):
                n = int(round(v))
                if n <= 0:
                    continue
                c = int(cls[ti.J[a]])
                shift = a - int(ti.first[ti.J[a]])
                for _ in range(n):
                    if not left[c]:
                        break
                    i = left[c].pop()
                    chosen.append(int(ti.first[i]) + shift)
        chosen = chosen + fixed.tolist()
        a = ti.assignment_of(chosen)
        if not len(a) == len(ti.jobs) == len(chosen) or ti.inst.schedule_violations(a):
            a = None
    obj = _objective_of(ti.inst, a, w1, w2) if a else math.inf
    if status == "optimal" and a is not None:
        return "optimal", obj, a
    return "limit", obj, a


def _occupancy(ti: _TI, arcs) -> np.ndarray:
    """Capacity (types × H) the placements ``arcs`` hold."""
    arcs = np.asarray(arcs, dtype=np.int64)
    arcs = arcs[ti.K[arcs] >= 0]
    d = np.zeros((len(ti.types), ti.H + 1))
    np.add.at(d, (ti.K[arcs], ti.T[arcs]), ti.Q[arcs])
    np.add.at(d, (ti.K[arcs], ti.T[arcs] + ti.P[arcs]), -ti.Q[arcs])
    return np.cumsum(d, axis=1)[:, :ti.H]


# ---------------------------------------------------------------------------
# Primal heuristics (BPC plan, Phase 7)
# ---------------------------------------------------------------------------

def _per_job_argmax(ti: _TI, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(arc, value)``: each job's largest placement in ``x``."""
    mx = np.maximum.reduceat(x, ti.first)
    hit = np.nonzero(x >= mx[ti.J] - 1e-12)[0]
    _, first = np.unique(ti.J[hit], return_index=True)
    return hit[first], mx


def _alpha_schedule(ti: _TI, x: np.ndarray, alpha: float) -> dict[int, tuple[int, int]] | None:
    """α-point list scheduling from a fractional placement vector (Hall,
    Schulz, Shmoys & Wein 1997; Savelsbergh, Uma & Wein 2005).

    A job's α-point is the first start by which an ``alpha`` share of its LP
    mass has begun; jobs are placed in α-point order, each on the type its
    α-point uses, at the earliest start where the type has room. A
    capacitated job may move to another capacitated type that costs no more
    when that completes it sooner, so bound rows on f2 stay safe."""
    sup = np.nonzero(x > 1e-9)[0]
    if not len(sup):
        return None
    order = np.lexsort((ti.T[sup], ti.J[sup]))
    sup = sup[order]
    jobs = ti.J[sup]
    cum = np.cumsum(x[sup])
    seg = np.r_[0, np.nonzero(np.diff(jobs))[0] + 1]
    base = np.repeat(np.r_[0.0, cum[seg[1:] - 1]], np.diff(np.r_[seg, len(sup)]))
    reached = (cum - base) >= alpha - 1e-9
    reached_idx = np.nonzero(reached)[0]
    _, first = np.unique(jobs[reached_idx], return_index=True)
    pick = sup[reached_idx[first]]
    if len(pick) != len(ti.jobs):
        return None
    inst = ti.inst
    used = np.zeros_like(ti.N)
    out: dict[int, tuple[int, int]] = {}
    for a in pick[np.lexsort((ti.P[pick], ti.T[pick]))]:
        i = int(ti.J[a])
        jid = ti.jobs[i]
        k0, c0 = int(ti.K[a]), float(ti.C2[a])
        lo, hi = ti.first[i], (ti.first[i + 1] if i + 1 < len(ti.first) else ti.n_arcs)
        best = None
        for m in {int(v) for v in ti.MID[lo:hi]}:
            n0, ts = ti.pair[jid, m]
            k = int(ti.K[n0])
            if k0 < 0 and m != int(ti.MID[a]):
                continue                     # a cloud job keeps its LP type
            if k0 >= 0 and (k < 0 or ti.C2[n0] > c0 + 1e-9):
                continue                     # only capacitated types, no dearer
            p, q = int(ti.P[n0]), float(ti.Q[n0])
            if k < 0:
                t = int(ts[0])
            else:
                bad = (ti.N[k] - used[k]) < q - 1e-9
                pre = np.r_[0, np.cumsum(bad)]
                ok = np.nonzero(pre[ts + p] - pre[ts] == 0)[0]
                if not len(ok):
                    continue
                t = int(ts[ok[0]])
            if best is None or t + p < best[0]:
                best = (t + p, m, k, t, p, q)
        if best is None:
            return None
        _, m, k, t, p, q = best
        if k >= 0:
            used[k, t:t + p] += q
        out[jid] = (m, t)
    return out


def _fix_and_solve(ti: _TI, x: np.ndarray, gap_rc: np.ndarray, has: np.ndarray, w1, w2, b1, b2,
                   cutoff: float, deadline: float, step: float, opts: "LPFirstOptions",
                   start: dict | None):
    """RINS-style dive: fix every job the LP places integrally, solve the
    MIP over the fractional ones (their master columns plus their
    ``cand_per_job`` placements of least ``gap_rc``), and widen the free set
    to the fixed jobs overlapping them when that has no solution."""
    arg, mx = _per_job_argmax(ti, x)
    free = mx < 1 - 1e-6
    for _round in range(3):
        if not free.any() or time.perf_counter() >= deadline:
            return None
        cand = _candidates(ti, free, gap_rc, has, opts.cand_per_job, start)
        fixed = arg[~free]
        status, obj, a = _mip(ti, cand, w1, w2, b1, b2, cutoff,
                              min(opts.fix_time, deadline - time.perf_counter()), step,
                              opts.mip_threads, start, fixed=fixed, backend=opts.backend)
        if a is not None:
            return a
        # widen: free the fixed jobs whose placement overlaps a free job's window
        span = np.zeros((len(ti.types), ti.H + 1))
        c = cand[ti.K[cand] >= 0]
        np.add.at(span, (ti.K[c], ti.T[c]), 1)
        np.add.at(span, (ti.K[c], ti.T[c] + ti.P[c]), -1)
        busy = np.cumsum(span, axis=1)[:, :ti.H] > 0
        f = fixed[ti.K[fixed] >= 0]
        pre = np.c_[np.zeros(len(ti.types)), np.cumsum(busy, axis=1)]
        hit = pre[ti.K[f], ti.T[f] + ti.P[f]] - pre[ti.K[f], ti.T[f]] > 0
        if not hit.any():
            return None
        free[ti.J[f[hit]]] = True
    return None


def _candidates(ti: _TI, free: np.ndarray, gap_rc: np.ndarray, has: np.ndarray, per_job: int,
                start: dict | None) -> np.ndarray:
    """The free jobs' placements: master columns, the incumbent's and the
    ``per_job`` of least ``gap_rc`` each."""
    idx = np.nonzero(free[ti.J])[0]
    order = np.lexsort((gap_rc[idx], ti.J[idx]))
    idx = idx[order]
    jobs = ti.J[idx]
    seg = np.r_[0, np.nonzero(np.diff(jobs))[0] + 1]
    rank = np.arange(len(idx)) - np.repeat(seg, np.diff(np.r_[seg, len(idx)]))
    keep = (rank < per_job) | has[idx]
    out = idx[keep]
    if start:
        sa = np.asarray(ti.arcs_of(start), dtype=np.int64)
        if len(sa):
            out = np.union1d(out, sa[free[ti.J[sa]]])
            return out
    return np.unique(out)


def _lns(ti: _TI, best: dict, ub: float, gap_rc: np.ndarray, has: np.ndarray, w1, w2, b1, b2,
         step: float, deadline: float, opts: "LPFirstOptions", certified, st: dict):
    """Large-neighbourhood search over time windows: free the jobs that the
    incumbent runs during a window (on any capacitated type, plus the cloud
    jobs released in it), re-solve that sub-MIP below the incumbent, slide on.
    Stops after a sweep without improvement, at the deadline, or once the
    incumbent is certified. Returns ``(ub, best)``."""
    arcs = np.array(ti.arcs_of(best), dtype=np.int64)
    if len(arcs) != len(ti.jobs):
        return ub, best
    rel = np.array([max(0, math.ceil(j.arrival)) for j in ti.inst.jobs])
    pos = np.array([ti.jpos[j.id] for j in ti.inst.jobs])
    release = np.empty(len(ti.jobs), dtype=np.int64)
    release[pos] = rel
    n_target = opts.lns_jobs
    improved_any = True
    sweeps = 0
    while improved_any and time.perf_counter() < deadline and sweeps < opts.lns_sweeps:
        improved_any = False
        sweeps += 1
        # windows holding about n_target job starts each
        starts = np.sort(ti.T[arcs])
        edges = np.unique(starts[::max(1, n_target // 2)])
        for lo_i in range(len(edges)):
            if time.perf_counter() >= deadline or certified(ub):
                return ub, best
            t_lo = int(edges[lo_i])
            t_hi = int(edges[min(lo_i + 2, len(edges) - 1)]) if lo_i + 2 < len(edges) else ti.H
            a_t, a_e = ti.T[arcs], ti.T[arcs] + ti.P[arcs]
            in_win = (a_t < t_hi) & (a_e > t_lo)
            in_win |= (ti.K[arcs] < 0) & (release[ti.J[arcs]] < t_hi) & (release[ti.J[arcs]] >= t_lo)
            free = np.zeros(len(ti.jobs), dtype=bool)
            free[ti.J[arcs[in_win]]] = True
            if free.sum() < 2:
                continue
            cand = _candidates(ti, free, gap_rc, has, opts.cand_per_job, best)
            fixed = arcs[~free[ti.J[arcs]]]
            cut = ub - step + 1e-7 * max(1.0, abs(ub)) if step > 0 else ub - 1e-7 * max(1.0, abs(ub))
            status, obj, a = _mip(ti, cand, w1, w2, b1, b2, cut,
                                  min(opts.lns_time_per, deadline - time.perf_counter()), step,
                                  opts.mip_threads, best, fixed=fixed, backend=opts.backend)
            st["lns_mips"] = st.get("lns_mips", 0) + 1
            if a is not None and obj < ub - 1e-9 and _within(ti.inst, a, b1, b2):
                ub, best = obj, a
                arcs = np.array(ti.arcs_of(best), dtype=np.int64)
                improved_any = True
                st["lns_improvements"] = st.get("lns_improvements", 0) + 1
    return ub, best


@dataclass
class _CGOut:
    status: str               # "converged", "certified" (stop(L) held) or "time"
    L: float                  # best Lagrangian bound
    center: "_D | None"       # its duals
    center_rc: np.ndarray | None
    rmp: float                # last restricted-master LP value


def _cg(ti: _TI, master: "_Master", w1, w2, b1, b2, opts: LPFirstOptions,
        deadline: float | None, allowed: np.ndarray | None, stop, st: dict) -> _CGOut:
    """Column generation on ``master`` until it converges, ``stop(L)`` holds
    or the deadline passes. ``allowed`` (bool per placement) restricts
    pricing, for a node of the search tree: a forbidden placement gets an
    infinite reduced cost, so it is never added and the Lagrangian bound is
    the bound of the restricted problem (infinite when some job has no
    allowed placement)."""
    best_L, center, center_rc = -math.inf, None, None
    rmp = math.inf
    it = 0

    def price(d):
        rc = _price(ti, d, w1, w2)
        if allowed is not None:
            rc[~allowed] = np.inf
        return rc

    while True:
        if deadline is not None and time.perf_counter() >= deadline:
            return _CGOut("time", best_L, center, center_rc, rmp)
        it += 1
        tl = time.perf_counter()
        status = master.solve_lp()
        st["lp_s"] = st.get("lp_s", 0.0) + time.perf_counter() - tl
        if status != "optimal":
            raise RuntimeError(f"lpfirst master LP status {status} ({master.be.name}: "
                               f"{getattr(master.be, 'last_status', '')})")
        rmp = master.objective()
        d = master.duals()
        tp = time.perf_counter()
        pick = np.array([], dtype=np.int64)
        if center is not None and opts.smoothing > 0:
            sd = center.mix(d, opts.smoothing)
            rc_s = price(sd)
            L, _m = _lagrangian(ti, sd, rc_s, b1, b2)
            if L > best_L:
                best_L, center, center_rc = L, sd, rc_s
            pick = _select(ti, rc_s, master.has, opts.tol, opts.per_job, opts.max_cols)
            if len(pick):
                rc_d = _price_subset(ti, d, w1, w2, pick)
                pick = pick[rc_d < -opts.tol]
        if not len(pick):
            rc = price(d)
            L, _m = _lagrangian(ti, d, rc, b1, b2)
            if L > best_L:
                best_L, center, center_rc = L, d, rc
            pick = _select(ti, rc, master.has, opts.tol, opts.per_job, opts.max_cols)
        st["pricing_s"] = st.get("pricing_s", 0.0) + time.perf_counter() - tp
        st["cg_iters"] = st.get("cg_iters", 0) + 1
        if stop(best_L):
            return _CGOut("certified", best_L, center, center_rc, rmp)
        if not len(pick) or best_L >= rmp - max(opts.tol, 1e-9 * abs(rmp)):
            if master.n_art and master.artificial() > 1e-6 and master.big_m < master.big_m_cap:
                master.raise_big_m()          # artificials left: maybe M is too small
                best_L, center, center_rc = -math.inf, None, None
                continue
            return _CGOut("converged", best_L, center, center_rc, rmp)
        master.add(pick)


class _AggMaster:
    """The full-placement view of an aggregated master's solution, with the
    attributes :func:`solve_lpfirst` reads after column generation."""

    def __init__(self, ti: _TI, agg: _TI, inner: "_Master"):
        self.inner = inner
        self._xfull = agg.disaggregate(inner.x())
        self.has = inner.has[agg.agg_of]
        self.arcs = np.nonzero(self.has)[0].tolist()
        self.n_art = inner.n_art
        self.be = inner.be
        self._art = inner.artificial()

    def x(self) -> np.ndarray:
        return self._xfull

    def arc_values(self) -> np.ndarray:
        return self._xfull[np.asarray(self.arcs, dtype=np.int64)]

    def artificial(self) -> float:
        return self._art


def _cg_aggregated(ti: _TI, agg: _TI, full_master: "_Master", w1, w2, b1, b2, big_m,
                   opts: LPFirstOptions, deadline, certified, st) -> tuple[_CGOut, _AggMaster]:
    """Column generation on the aggregated master, mapped back to ``ti``.

    A member's placement costs ``w1·offset_j`` more than its representative's
    (``C1`` differs by a constant, ``C2`` not at all), so with the class dual
    ``σ_k`` the member dual ``σ_k + (w1 - ν1)·offset_j`` gives every member
    placement its representative's reduced cost. Those duals are feasible
    and optimal for the full LP, and the Lagrangian bound is recomputed from
    them on the full placement set. The f1 bound row moves by the summed
    offsets."""
    off = agg.offset
    off_sum = float(off.sum())
    b1a = None if b1 is None else b1 - off_sum
    m = _Master(agg, w1, w2, b1a, b2, big_m, cover_rhs=agg.mult, backend=opts.backend,
                threads=opts.lp_threads)
    seed = set(agg.agg_of[np.asarray(full_master.arcs, dtype=np.int64)].tolist())
    m.add(sorted(seed))
    full_master.be = None   # never solved: drop the full model

    def to_full(d: _D) -> _D:
        return _D(d.sigma[agg.cls] + (w1 - d.nu1) * off, d.pi, d.nu1, d.nu2)

    def stop(L_agg: float) -> bool:
        # the bound of the aggregated problem, in full units
        # L_full = L_agg + (w1 - ν1)·Σoff + ν1·Σoff (the shifted f1 row)
        return certified(L_agg + w1 * off_sum)

    cg = _cg(agg, m, w1, w2, b1a, b2, opts, deadline, None, stop, st)
    if cg.center is None:
        return _CGOut(cg.status, -math.inf, None, None, math.inf), _AggMaster(ti, agg, m)
    center = to_full(cg.center)
    center_rc = _price(ti, center, w1, w2)
    L, _ = _lagrangian(ti, center, center_rc, b1, b2)
    status = cg.status
    if status == "certified" and not certified(L):
        status = "converged"
    rmp = cg.rmp + w1 * off_sum if math.isfinite(cg.rmp) else cg.rmp
    return _CGOut(status, L, center, center_rc, rmp), _AggMaster(ti, agg, m)


class _Tree:
    """Branch and price over placements (BPC plan, Phase 9).

    A node is a set of allowed placements. Its relaxation is the
    time-indexed master restricted to them, solved by :func:`_cg`; its
    Lagrangian bound prunes it against the incumbent. After the bound:

    - **reduced-cost fixing**: every placement whose gap exceeds
      ``UB - L`` leaves the node's allowed set (the eligibility argument of
      step 3, applied per node), so children inherit only eligible ones;
    - **node MIP**: once at most ``node_eligible`` remain, the aggregated
      MIP over them (cutoff ``UB - step``) solves the node outright; if it
      hits ``node_mip_time`` the node branches instead;
    - **branching** (Kramer et al. 2020): the most fractional job is split
      by type when its LP mass spans types (its heaviest type against the
      rest), else by start time at the point where its mass first reaches
      one half, so both children cut off the node's LP solution;
    - an α-point rounding of each node's LP tries to improve the incumbent.

    Best-first; on a timeout the smallest open bound is a valid bound.
    """

    def __init__(self, ti, inst, w1, w2, b1, b2, big_m, opts, step, proves, st):
        self.ti, self.inst, self.w = ti, inst, (w1, w2)
        self.b1, self.b2, self.big_m, self.opts = b1, b2, big_m, opts
        self.step, self.proves, self.st = step, proves, st

    def _cut(self, ub):
        m = 1e-7 * max(1.0, abs(ub))
        return ub - self.step + m if self.step > 0 else ub - m

    def _branch(self, x: np.ndarray, allowed: np.ndarray):
        ti = self.ti
        arg, mx = _per_job_argmax(ti, x)
        frac = np.nonzero(mx < 1 - 1e-6)[0]
        if not len(frac):
            return None
        i = int(frac[np.argmin(mx[frac])])
        lo = int(ti.first[i])
        hi = int(ti.first[i + 1]) if i + 1 < len(ti.first) else ti.n_arcs
        seg = np.arange(lo, hi)
        sup = seg[x[seg] > 1e-9]
        types = ti.MID[sup]
        side = np.zeros(ti.n_arcs, dtype=bool)
        if len(set(types.tolist())) > 1:
            mass = {m: float(x[sup[types == m]].sum()) for m in set(types.tolist())}
            m_star = max(mass, key=mass.get)
            side[seg[ti.MID[seg] == m_star]] = True
            self.st["tree_type_branches"] = self.st.get("tree_type_branches", 0) + 1
        else:
            order = sup[np.argsort(ti.T[sup], kind="stable")]
            cum = np.cumsum(x[order])
            k = int(np.searchsorted(cum, 0.5 * cum[-1] - 1e-12))
            k = min(k, len(order) - 2)
            t_cut = int(ti.T[order[k]])
            side[seg[ti.T[seg] <= t_cut]] = True
        left, right = allowed.copy(), allowed.copy()
        left[seg[~side[seg]]] = False
        right[seg[side[seg]]] = False
        return left, right

    def run(self, ub, best, allowed, root_master, root_cg, deadline):
        """Returns ``(ub, best, open_bound, proven)``."""
        import heapq
        ti, opts, st = self.ti, self.opts, self.st
        w1, w2 = self.w
        heap = [(root_cg.L, 0, allowed, np.asarray(root_master.arcs, dtype=np.int64), root_master,
                 root_cg)]
        counter = 1
        st["tree_nodes"] = 0
        while heap:
            if deadline is not None and time.perf_counter() >= deadline:
                return ub, best, heap[0][0], False
            L0, _, allow, cols, master, cg = heapq.heappop(heap)
            if self.proves(L0, ub):
                continue
            st["tree_nodes"] += 1
            if master is None:
                master = _Master(ti, w1, w2, self.b1, self.b2, self.big_m,
                                 backend=opts.backend, threads=opts.lp_threads)
                seed = set(cols[allow[cols]].tolist())
                if best is not None:
                    seed.update(a for a in ti.arcs_of(best) if allow[a])
                seed.update(np.nonzero(~ti.capped & allow)[0].tolist())
                master.add(sorted(seed))
                cg = _cg(ti, master, w1, w2, self.b1, self.b2, opts, deadline, allow,
                         lambda L: self.proves(L, ub), st)
                if cg.status == "time":
                    return ub, best, min(L0, heap[0][0]) if heap else L0, False
                if cg.status == "certified" or not math.isfinite(cg.L):
                    continue
                if master.artificial() > 1e-6:
                    continue          # the node's LP is infeasible
            L = max(L0, cg.L)
            if self.proves(L, ub):
                continue
            x = master.x()
            # incumbent from the node: integral LP, or its α-point rounding
            for cand in ([master.assignment()] if master.assignment() else []) + \
                    [_alpha_schedule(ti, x, 0.5)]:
                if cand is not None and _within(self.inst, cand, self.b1, self.b2):
                    obj = _objective_of(self.inst, cand, w1, w2)
                    if obj < ub - 1e-9:
                        ub, best = obj, cand
                        st["tree_improvements"] = st.get("tree_improvements", 0) + 1
            if self.proves(L, ub):
                continue
            # reduced-cost fixing at the node
            _L, mins = _lagrangian(ti, cg.center, cg.center_rc, self.b1, self.b2)
            gap_rc = cg.center_rc + cg.center.sigma[ti.J] - mins[ti.J]
            keep = allow & (gap_rc < (ub - cg.L) + 1e-7 * max(1.0, abs(ub)))
            n_keep = int(keep.sum())
            if n_keep <= opts.node_eligible:
                status, obj, a = _mip(ti, np.nonzero(keep)[0], w1, w2, self.b1, self.b2,
                                      self._cut(ub), min(opts.node_mip_time,
                                                         (deadline - time.perf_counter())
                                                         if deadline else opts.node_mip_time),
                                      self.step, opts.mip_threads, best, backend=opts.backend)
                st["tree_node_mips"] = st.get("tree_node_mips", 0) + 1
                if a is not None and obj < ub - 1e-9 and _within(self.inst, a, self.b1, self.b2):
                    ub, best = obj, a
                if status in ("optimal", "none"):
                    continue
            kids = self._branch(x, keep)
            if kids is None:      # integral LP but not better than ub: done
                continue
            cols = np.asarray(master.arcs, dtype=np.int64)
            for kid in kids:
                heapq.heappush(heap, (L, counter, kid, cols, None, None))
                counter += 1
        return ub, best, math.inf, True


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
    # Scale the objective so one step of its coarser term is 1 (as
    # model.weighted_sum_scale does for the compact model): normalised
    # weights (~1e-5) put cost differences below Gurobi's OptimalityTol and
    # its MIPs then prune the optimum. Everything below works in scaled
    # units; result() reports unscaled ones.
    scale = weighted_sum_scale(inst, weight1, weight2) if (weight1 or weight2) else 1.0
    w1, w2, b1, b2 = weight1 * scale, weight2 * scale, bound1_value, bound2_value
    st["scale"] = scale
    step = bpc._objective_step(inst, w1, w2)
    if seed_schedules is None:
        seed_schedules = bpc.heuristic_schedules(inst)

    ub, best = math.inf, None
    for a in seed_schedules:
        if len(a) == len(ti.jobs) and _within(inst, a, b1, b2):
            obj = _objective_of(inst, a, w1, w2)
            if obj < ub:
                ub, best = obj, dict(a)
    st["seed_obj"] = ub / scale

    def proves(L: float, u: float) -> bool:
        if not math.isfinite(u):
            return False
        margin = 1e-7 * max(1.0, abs(u))
        return L > u - step + margin if step > 0 else L >= u - margin

    def certified(L: float) -> bool:
        return proves(L, ub)

    def left() -> float:
        return math.inf if deadline is None else deadline - time.perf_counter()

    def result(status, how, L):
        st["time_s"] = time.perf_counter() - t0
        st["bound"] = L / scale
        if column_pool is not None:
            try:
                column_pool.update(int(a) for a, v in zip(master.arcs, master.arc_values().tolist())
                                   if v > 1e-9)
            except Exception:   # no LP solution yet
                pass
            if best is not None:
                column_pool.update(ti.arcs_of(best))
        return LPFirstResult(status, ub / scale if best is not None else math.inf, best,
                             L / scale, how, st)

    # in scaled units; _cg raises it (up to 1e6 more) while artificials stay
    big_m = 1e3 * max(1.0, float(np.max(w1 * ti.C1 + w2 * ti.C2)))
    master = _Master(ti, w1, w2, b1, b2, big_m, backend=opts.backend, threads=opts.lp_threads)
    seed = set(column_pool or ())
    for a in seed_schedules:
        seed.update(ti.arcs_of(a))
    seed.update(np.nonzero(~ti.capped)[0].tolist())
    master.add(sorted(seed))

    # --- column generation, stopped by the certificate when possible
    agg = None
    if opts.aggregate_lp:
        cand = ti.aggregated()
        if len(cand.jobs) <= 0.8 * len(ti.jobs):
            agg = cand
    if agg is not None:
        cg, master = _cg_aggregated(ti, agg, master, w1, w2, b1, b2, big_m, opts, deadline,
                                    certified, st)
    else:
        cg = _cg(ti, master, w1, w2, b1, b2, opts, deadline, None, certified, st)
    best_L, center, center_rc, rmp = cg.L, cg.center, cg.center_rc, cg.rmp
    if cg.status == "time":
        return result("feasible", "", best_L)
    if cg.status == "certified":
        st["columns"] = len(master.arcs)
        return result("optimal", "bound", best_L)
    st["columns"] = len(master.arcs)
    st["lp"] = rmp / scale

    def eligible_of(u: float) -> np.ndarray:
        """Placements a schedule below ``u`` may use (rc - min_j < u - L)."""
        _L, mins_ = _lagrangian(ti, center, center_rc, b1, b2)
        return np.nonzero(center_rc + center.sigma[ti.J] - mins_[ti.J]
                          < (u - best_L) + 1e-7 * max(1.0, abs(u)))[0]

    if master.artificial() > 1e-6:   # the LP optimum needs artificials: no schedule
        best = None
        return result("infeasible", "lp", best_L)

    # --- step 1b: primal heuristics from the LP (α-points, fix-and-solve, LNS)
    ub_at_rmp, early_rmp = INF, False   # the proof-first restricted-master MIP
    if opts.heuristics and math.isfinite(rmp) and not certified(best_L):
        th = time.perf_counter()
        stop = th + opts.heuristic_time if deadline is None else min(deadline, th + opts.heuristic_time)
        x = master.x()
        _L, mins = _lagrangian(ti, center, center_rc, b1, b2)
        gap_rc = center_rc + center.sigma[ti.J] - mins[ti.J]
        for alpha in (0.5, 0.25, 0.75, 0.0):
            a = _alpha_schedule(ti, x, alpha)
            if a is not None and _within(inst, a, b1, b2):
                obj = _objective_of(inst, a, w1, w2)
                if obj < ub:
                    ub, best = obj, a
        st["alpha_obj"] = ub / scale
        if certified(best_L):
            st["fix_obj"] = st["lns_obj"] = ub / scale
            st["heuristic_s"] = time.perf_counter() - th
            return result("optimal", "heuristic", best_L)
        # proof first: the restricted master as a MIP (cheap, and the
        # incumbent it gives shrinks the eligible set), then the eligible
        # MIP when that set is small
        # Neither the gap nor the eligible set says in advance whether this
        # MIP pays (pos_congestion's boxes: seed 86-254 % above the bound,
        # every placement eligible, and it finds the near-optimum in 1-2 s);
        # what it costs when it does not is its time cap, so keep that short.
        quick = opts.quick_proof
        if quick and opts.rmp_mip_time > 0 and left() > 0:
            cut = ub - step + 1e-7 * max(1.0, abs(ub)) if math.isfinite(ub) else INF
            tm = time.perf_counter()
            _s, obj, a = _mip(ti, master.arcs, w1, w2, b1, b2, cut,
                              min(opts.rmp_mip_time, opts.quick_rmp_time, left()), step,
                              opts.mip_threads, best, backend=opts.backend)
            st["quick_rmp_mip_s"] = time.perf_counter() - tm
            if a is not None and obj < ub and _within(inst, a, b1, b2):
                ub, best = obj, a
            ub_at_rmp, early_rmp = ub, True
            if certified(best_L):
                st["fix_obj"] = st["lns_obj"] = ub / scale
                st["heuristic_s"] = time.perf_counter() - th
                return result("optimal", "rmp_mip", best_L)
        if quick and math.isfinite(ub) and left() > 0:
            elig = eligible_of(ub)
            if len(elig) <= opts.quick_eligible:
                cut = ub - step + 1e-7 * max(1.0, abs(ub)) if step > 0 else ub - 1e-7 * max(1.0, abs(ub))
                tq = time.perf_counter()
                status, obj, a = _mip(ti, elig, w1, w2, b1, b2, cut,
                                      min(opts.quick_proof_time, left()), step, opts.mip_threads,
                                      backend=opts.backend)
                st["quick_mip_s"] = time.perf_counter() - tq
                st["quick_eligible"] = int(len(elig))
                if a is not None and obj < ub and _within(inst, a, b1, b2):
                    ub, best = obj, a
                if status in ("optimal", "none"):
                    st["heuristic_s"] = time.perf_counter() - th
                    st["eligible"] = int(len(elig))
                    return result("optimal", "quick_mip", best_L)
        if not certified(best_L):
            cut = ub - step + 1e-7 * max(1.0, abs(ub)) if math.isfinite(ub) else INF
            a = _fix_and_solve(ti, x, gap_rc, master.has, w1, w2, b1, b2, cut, stop, step,
                               opts, best)
            if a is not None and _within(inst, a, b1, b2):
                obj = _objective_of(inst, a, w1, w2)
                if obj < ub:
                    ub, best = obj, a
        st["fix_obj"] = ub / scale
        if best is not None and not certified(best_L):
            ub, best = _lns(ti, best, ub, gap_rc, master.has, w1, w2, b1, b2, step, stop, opts,
                            lambda u: proves(best_L, u), st)
        st["lns_obj"] = ub / scale
        st["heuristic_s"] = time.perf_counter() - th
        if certified(best_L):
            return result("optimal", "heuristic", best_L)

    # --- step 2: the restricted master as a MIP, with the heuristics'
    # incumbent as its cutoff (again, if the proof-first one ran with a worse one)
    if math.isfinite(rmp) and opts.rmp_mip_time > 0 and (not early_rmp or ub < ub_at_rmp):
        cut = ub - step + 1e-7 * max(1.0, abs(ub)) if math.isfinite(ub) else INF
        tm = time.perf_counter()
        status, obj, a = _mip(ti, master.arcs, w1, w2, b1, b2, cut,
                              min(opts.rmp_mip_time, left()), step, opts.mip_threads, best,
                              backend=opts.backend)
        st["rmp_mip_s"] = time.perf_counter() - tm
        if a is not None and obj < ub and _within(inst, a, b1, b2):
            ub, best = obj, a
        if certified(best_L):
            return result("optimal", "rmp_mip", best_L)
    if not math.isfinite(ub):
        if not opts.tree or left() <= 0:
            return result("feasible", "", best_L)
        if agg is not None:   # the placement tree needs the full master
            return solve_lpfirst(inst, weight1, weight2, bound1_value, bound2_value,
                                 time_limit=left() if deadline is not None else None,
                                 options=replace(opts, aggregate_lp=False), ti=ti,
                                 seed_schedules=seed_schedules, column_pool=column_pool,
                                 stats=st)
        # no incumbent: every placement is eligible, so search the tree
        tr = _Tree(ti, inst, w1, w2, b1, b2, big_m, opts, step, proves, st)
        ub, best, L_open, done = tr.run(ub, best, np.ones(ti.n_arcs, dtype=bool), master, cg,
                                        deadline)
        if done:
            if best is None:
                return result("infeasible", "tree", math.inf)
            return result("optimal", "tree", best_L)
        return result("feasible", "", max(best_L, L_open))

    # --- step 3: the exact MIP over the eligible placements
    eligible = eligible_of(ub)
    st["eligible"] = int(len(eligible))
    if agg is not None and opts.tree and len(eligible) > opts.max_eligible and left() > 0:
        return solve_lpfirst(inst, weight1, weight2, bound1_value, bound2_value,
                             time_limit=left() if deadline is not None else None,
                             options=replace(opts, aggregate_lp=False), ti=ti,
                             seed_schedules=[best, *seed_schedules], column_pool=column_pool,
                             stats=st)
    if opts.tree and len(eligible) > opts.max_eligible and left() > 0:
        # --- step 3': branch and price over placements (Phase 9)
        allowed = np.zeros(ti.n_arcs, dtype=bool)
        allowed[eligible] = True
        tr = _Tree(ti, inst, w1, w2, b1, b2, big_m, opts, step, proves, st)
        ub, best, L_open, done = tr.run(ub, best, allowed, master, cg, deadline)
        if done:
            return result("optimal", "tree", best_L)
        return result("feasible", "", max(best_L, L_open))
    if len(eligible) > opts.max_eligible or left() <= 0:
        return result("feasible", "", best_L)
    cut = ub - step + 1e-7 * max(1.0, abs(ub)) if step > 0 else ub - 1e-7 * max(1.0, abs(ub))
    tm = time.perf_counter()
    status, obj, a = _mip(ti, eligible, w1, w2, b1, b2, cut, left(), step, opts.mip_threads,
                          backend=opts.backend)
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
    rc[c] -= ti.Q[idx][c] * (pre[k[c], t[c] + p[c]] - pre[k[c], t[c]])
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


#: :func:`reference_points_lpfirst`'s default: HiGHS, no LNS (the proof steps
#: are cheaper than the sweeps on every measured instance) and the aggregated
#: master (identical jobs abound once the cloud placements are gone).
REFERENCE_OPTIONS = LPFirstOptions(backend="highs", lns_sweeps=0, aggregate_lp=True)


@dataclass
class ReferencePoints:
    """The weighted-sum reference points from :func:`reference_points_lpfirst`.

    ``refs`` — ``(f1_T, f2_T, f1_0, f2_0)``, as
    ``pareto.weighted_sum_reference``. ``certified`` — every value proven
    optimal. f1_T and f2_T always are (the call raises otherwise); f1_0 may
    be the best schedule found when ``require_certified=False``, an upper
    bound, which keeps the f1 <= f1_0 cut of ``model.solve_weighted_sum``
    valid. ``f1_0_bound`` — a proven lower bound on f1_0 (f1_0 itself when
    certified). ``ideal`` — the degenerate weighted sum's optimum, as
    ``pareto.weighted_sum_reference``'s second item. ``stats`` — per phase
    seconds and solver statistics."""
    refs: tuple[float, float, float, float]
    certified: bool
    f1_0_bound: float
    ideal: Solution | None = None
    stats: dict = field(default_factory=dict)


def reference_points_lpfirst(
    inst: Instance, *, min_cost_reference: bool = True, time_limit: float | None = None,
    options: LPFirstOptions | None = None, require_certified: bool = True,
    ea_seeds: bool = False,
) -> ReferencePoints:
    """``pareto.weighted_sum_reference`` on the LP-first engine, so with
    ``options.backend="highs"`` no Gurobi model is built.

    Phases: lex-min (f1, f2) gives (f1_T, f2_T); min f1 s.t. f2 <= 0 gives
    f1_0 (f2_0 = 0); when no schedule is free and ``min_cost_reference``,
    lex-min (f2, f1) gives (f1_0, f2_0) instead. With nonnegative data each
    phase runs on the placements its optima can use (:meth:`_TI.lex_f1_mask`,
    :meth:`_TI.free_mask`), which is exact and much smaller: on the f1_0
    phase every cloud placement goes.

    ``options`` defaults to :data:`REFERENCE_OPTIONS` (HiGHS).
    ``time_limit`` caps the whole call. ``ea_seeds`` adds an NSGA-II front to
    the seed schedules (:func:`fedhpc.bpc.heuristic_schedules`).
    """
    from .formulations import nonnegative_data
    from .model import weighted_sum_degenerate

    opts = options or REFERENCE_OPTIONS
    t0 = time.perf_counter()
    deadline = None if time_limit is None else t0 + time_limit
    stats: dict = {}

    def left() -> float | None:
        return None if deadline is None else max(0.0, deadline - time.perf_counter())

    ti = _TI(inst, bpc._prepare(inst))
    seeds = bpc.heuristic_schedules(inst, ea=ea_seeds)
    nonneg = nonnegative_data(inst)
    stats["setup_s"] = time.perf_counter() - t0

    def run(name, sub, w1, w2, b1=None, b2=None, extra=()):
        tp = time.perf_counter()
        own = [a for a in (*extra, *seeds) if sub.representable(a)]
        st: dict = {}
        res = solve_lpfirst(inst, w1, w2, b1, b2, time_limit=left(), options=opts, ti=sub,
                            seed_schedules=own, stats=st)
        if res.assignment is not None:
            seeds.insert(0, res.assignment)
        stats[name] = {"time_s": time.perf_counter() - tp, "status": res.status, "how": res.how,
                       "placements": sub.n_arcs, **{k: v for k, v in st.items()
                                                    if isinstance(v, (int, float, str))}}
        return res

    def proven(res, what):
        if res.status != "optimal":
            raise RuntimeError(f"lpfirst: {what} not proven optimal (status {res.status}); "
                               "raise the time limit")
        return _solution(inst, res)

    # Phases 1-2: lex-min (f1, f2)
    ti_T = ti.restrict(ti.lex_f1_mask()) if nonneg else ti
    s1 = proven(run("f1_T", ti_T, 1.0, 0.0), "min f1")
    s2 = proven(run("f2_T", ti_T, 0.0, 1.0, b1=s1.f1, extra=[s1.assignment]), "min f2 at f1_T")
    f1_T, f2_T = s1.f1, s2.f2
    stats["total_s"] = time.perf_counter() - t0
    if nonneg and f2_T <= 1e-8:
        return ReferencePoints((f1_T, f2_T, f1_T, 0.0), True, f1_T, s2, stats)

    # Phase 3: f1_0 = min f1 s.t. f2 <= 0
    s3, f2_0, bound, certified = None, 0.0, -math.inf, True
    zero_cost_infeasible = False
    if nonneg:
        try:
            ti_0 = ti.restrict(ti.free_mask())   # all free: the f2 <= 0 row is implied
            b2 = None
        except ValueError:   # some job has no free placement
            ti_0, zero_cost_infeasible = None, True
    else:
        ti_0, b2 = ti, 0.0
    if ti_0 is not None:
        r3 = run("f1_0", ti_0, 1.0, 0.0, b2=b2)
        if r3.status == "infeasible":
            zero_cost_infeasible = True
        elif r3.status == "optimal" or (r3.assignment is not None and not require_certified):
            s3 = _solution(inst, r3)
            certified = r3.status == "optimal"
            bound = s3.f1 if certified else r3.bound
        else:
            raise RuntimeError(f"lpfirst: min f1 at zero cost not proven optimal (best "
                               f"{r3.objective}, bound {r3.bound}); raise the time limit or "
                               "pass require_certified=False")
    if zero_cost_infeasible:
        if not min_cost_reference:
            raise RuntimeError("No zero-cost schedule exists; pass min_cost_reference=True.")
        r = run("f2_0", ti, 0.0, 1.0)
        s = proven(r, "min f2")
        s3 = proven(run("f1_0", ti, 1.0, 0.0, b2=s.f2, extra=[s.assignment]),
                    "min f1 at minimum cost")
        f2_0, bound = s3.f2, s3.f1
    f1_0 = s3.f1
    stats["total_s"] = time.perf_counter() - t0

    ideal = None
    if (zero_cost_infeasible or nonneg) and certified \
            and weighted_sum_degenerate(f1_T, f2_T, f1_0, f2_0):
        ideal = replace(s3, objective=s3.f2)
    return ReferencePoints((f1_T, f2_T, f1_0, f2_0), certified, bound, ideal, stats)


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
        """``secondary=None`` — the first phase only (status ``"optimal"``
        then means only the primary is proven; for
        pareto.parallel_epsilon_frontier). ``start`` — a Solution or a list
        of them, used as seeds (incumbent if feasible, columns either way)."""
        for x in (start if isinstance(start, (list, tuple)) else [start]):
            if x is not None and x.assignment and len(x.assignment) == len(inst.jobs):
                schedules.insert(0, dict(x.assignment))
        b1 = grid_floor(inst, "f1", bound_value) if bound == "f1" else None
        b2 = grid_floor(inst, "f2", bound_value) if bound == "f2" else None
        w = (1.0, 0.0) if primary == "f1" else (0.0, 1.0)
        first = solve(*w, b1, b2)
        if first.status == "infeasible":
            return _solution(inst, first)
        if first.status != "optimal":
            return _solution(inst, first, "feasible")
        sol1 = _solution(inst, first)
        if secondary is None:
            return sol1
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
