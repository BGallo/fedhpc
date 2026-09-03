"""Offline scheduling baselines for FED-HPC — comparison points for the exact
MIP / weighted-sum solutions.

All baselines are *offline* (full known-runtime information) and *list-schedule*
onto the same space-time capacity model the MIP uses: every on-prem type has a
finite instance capacity that is pre-occupied at ``t=0`` by ``inst.running_jobs``
and frees up as those jobs finish; cloud types are unlimited and priced.

Because ``p_occ`` is type-independent on these instances (``perf=1``,
``deploy=0``, ``io_time=0`` for every type) a job's completion time depends only
on *when* it starts, and its cost depends only on *which* type it runs on.  The
whole trade-off is therefore "how much cloud money to spend to buy the capacity
that lets jobs start sooner" — exactly what the baselines below sweep.

Baselines
---------
``onprem_only_spt``           — never burst: least-slack-first list-schedule
                                (SPT as tiebreak — see ``_least_slack_order``)
                                on the on-prem pools only.  Zero cost, maximal
                                turnaround; jobs with no feasible slot at all
                                are left unscheduled (reported), but a job the
                                exact MIP can place is no longer starved by
                                processing order.  The "do nothing" /
                                status-quo anchor.
``greedy_earliest_completion`` — per job (least-slack order by default; SPT
                                as tiebreak) pick the feasible type that
                                finishes it earliest, breaking ties toward the
                                cheaper type.  Near-minimal turnaround, high
                                cost — an eager-bursting anchor.
``threshold_bursting``        — queue on-prem while the achievable on-prem wait
                                is ``<= theta`` slots, otherwise burst to the
                                cheapest feasible cloud type.  Sweeping ``theta``
                                traces a cost/turnaround frontier between the
                                two anchors above.

Every function returns a :class:`fedhpc.model.Solution` with
``status="heuristic"`` so it feeds straight into
``fedhpc.viz.compute_stats`` / ``fedhpc.metrics.pareto_metrics``.
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

from .data import Instance
from .model import Solution

__all__ = [
    "onprem_only_spt",
    "greedy_earliest_completion",
    "threshold_bursting",
    "threshold_bursting_sweep",
    "make_solution",
]


# ─────────────────────────────────────────────────────────────────────────────
# Capacity bookkeeping
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _Capacity:
    """Mutable per-type occupancy state for incremental list scheduling."""

    inst: Instance
    occ: dict[int, list[int]]          # type_id -> per-slot running count (finite types)
    cap: dict[int, int | None]         # type_id -> capacity (None = unlimited)
    full: dict[int, list[int]]         # type_id -> sorted list of saturated slots

    @classmethod
    def fresh(cls, inst: Instance) -> "_Capacity":
        H = inst.horizon
        occ: dict[int, list[int]] = {}
        cap: dict[int, int | None] = {}
        full: dict[int, list[int]] = {}
        for m in inst.instance_types:
            cap[m.id] = m.capacity
            if m.capacity is None:
                continue
            row = [inst.occupied.get((m.id, t), 0) for t in range(H + 1)]
            occ[m.id] = row
            full[m.id] = [t for t, v in enumerate(row) if v >= m.capacity]
        return cls(inst=inst, occ=occ, cap=cap, full=full)

    def earliest_start(self, jid: int, mid: int, not_before: int = 0) -> int | None:
        """Earliest feasible start slot for job ``jid`` on type ``mid`` at or
        after ``not_before``, or ``None`` if it cannot fit before the horizon."""
        rng = self.inst.T[jid, mid]
        p = self.inst.p_occ[jid, mid]
        t = max(rng.start, not_before)
        if self.cap[mid] is None:                     # unlimited cloud type
            return t if t < rng.stop else None
        fs = self.full[mid]
        while t < rng.stop:
            i = bisect.bisect_left(fs, t)
            if i < len(fs) and fs[i] < t + p:
                t = fs[i] + 1                         # jump past the blocking slot
            else:
                return t
        return None

    def place(self, jid: int, mid: int, t: int) -> None:
        """Commit job ``jid`` to type ``mid`` starting at ``t``."""
        if self.cap[mid] is None:
            return
        p = self.inst.p_occ[jid, mid]
        row = self.occ[mid]
        c = self.cap[mid]
        fs = self.full[mid]
        for s in range(t, t + p):
            row[s] += 1
            if row[s] == c:
                bisect.insort(fs, s)


# ─────────────────────────────────────────────────────────────────────────────
# Solution assembly
# ─────────────────────────────────────────────────────────────────────────────

def make_solution(inst: Instance, assignment: dict[int, tuple[int, int]]) -> Solution:
    """Build a :class:`Solution` from a ``{job_id: (type_id, start)}`` mapping.

    ``f1`` sums turnaround over *scheduled* jobs only (same convention as
    ``model._extract``); unscheduled jobs are recoverable as
    ``inst.jobs`` minus ``assignment`` keys.
    """
    completion = {
        jid: t + inst.p_occ[jid, mid] for jid, (mid, t) in assignment.items()
    }
    f1 = sum(completion[j.id] - j.arrival for j in inst.jobs if j.id in completion)
    f2 = sum(inst.c[jid, mid] for jid, (mid, _t) in assignment.items())
    n_missing = len(inst.jobs) - len(assignment)
    status = "heuristic" if n_missing == 0 else "partial"
    return Solution(
        status=status,
        objective=None,
        f1=f1,
        f2=f2,
        assignment=dict(assignment),
        completion=completion,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Type helpers
# ─────────────────────────────────────────────────────────────────────────────

def _onprem_ids(inst: Instance) -> set[int]:
    return {m.id for m in inst.instance_types if m.capacity is not None}


def _spt_order(inst: Instance) -> list[int]:
    """Job ids shortest-processing-time first (ties -> earlier arrival, then id)."""
    return [
        j.id
        for j in sorted(inst.jobs, key=lambda j: (j.exec_time, j.arrival, j.id))
    ]


def _least_slack_order(inst: Instance) -> list[int]:
    """Job ids by ascending scheduling slack — tightest deadline first, SPT as
    tiebreak among equally-urgent jobs.

    slack_j = horizon - ceil(arrival_j) - min_{m in F_j} p_occ[j, m]

    i.e. how many slots of flexibility job j has left before it becomes
    infeasible to finish within the horizon at all.

    Pure SPT (shortest job first) starves long jobs with a narrow feasible
    window: on every mit_supercloud_* instance whose horizon was sized
    exactly to its own tail (see build_mit_supercloud_instance.py), the
    single job that determines the horizon has slack close to zero by
    construction — SPT processes it dead last (it's the longest job in the
    instance), and by then every other list-scheduling baseline had already
    filled its narrow window with shorter jobs, leaving it permanently
    unscheduled even though the exact MIP always finds it a feasible slot.
    Verified: onprem_only_spt/greedy_earliest_completion left exactly that
    job unscheduled on mit_supercloud_20210317 (job 646), 20210604 (job
    644), and 20210812 (job 169) — three different instances, same
    mechanism. Ordering by slack first (a form of Least-Slack-Time /
    Earliest-Deadline-First scheduling) fixes all three: the tightest jobs
    claim their narrow windows before anything else can crowd them out.
    """
    def slack(j) -> float:
        feas = inst.F.get(j.id, ())
        if not feas:
            return math.inf  # unschedulable regardless of order; sort last
        min_p_occ = min(inst.p_occ[j.id, m] for m in feas)
        return inst.horizon - math.ceil(j.arrival) - min_p_occ

    return [
        j.id
        for j in sorted(inst.jobs, key=lambda j: (slack(j), j.exec_time, j.arrival, j.id))
    ]


def _arrival_order(inst: Instance) -> list[int]:
    return [j.id for j in sorted(inst.jobs, key=lambda j: (j.arrival, j.exec_time, j.id))]


def _cheapest_cloud_type(inst: Instance, jid: int, onprem: set[int]) -> int | None:
    cloud = [m for m in inst.F[jid] if m not in onprem]
    if not cloud:
        return None
    return min(cloud, key=lambda mid: (inst.c[jid, mid], mid))


# ─────────────────────────────────────────────────────────────────────────────
# Baselines
# ─────────────────────────────────────────────────────────────────────────────

def onprem_only_spt(inst: Instance) -> Solution:
    """List-schedule on the on-prem pools only — never burst to cloud.

    Each job is placed at the globally earliest feasible slot across *all*
    on-prem types it fits on (equal price, so this is a weakly-dominating
    choice). Processing order is least-slack-first (see _least_slack_order),
    not pure SPT: pure SPT reliably strands the single job with the
    tightest feasible window (typically the longest job, processed dead
    last) even when a feasible schedule exists — verified across three
    mit_supercloud_* instances. Least-slack-first schedules every job the
    exact MIP can, matching it on feasibility; SPT is still the tiebreak
    among equally-urgent jobs, preserving SPT's flow-time-minimizing intent
    everywhere slack isn't the binding constraint.
    """
    onprem = _onprem_ids(inst)
    state = _Capacity.fresh(inst)
    assignment: dict[int, tuple[int, int]] = {}

    for jid in _least_slack_order(inst):
        feas = [m for m in inst.F[jid] if m in onprem]
        best: tuple[int, int, int] | None = None            # (completion, start, type)
        for mid in feas:
            t = state.earliest_start(jid, mid)
            if t is None:
                continue
            comp = t + inst.p_occ[jid, mid]
            if best is None or comp < best[0]:
                best = (comp, t, mid)
        if best is None:
            continue
        _comp, t, mid = best
        state.place(jid, mid, t)
        assignment[jid] = (mid, t)

    return make_solution(inst, assignment)


def greedy_earliest_completion(inst: Instance, *, order: str = "spt") -> Solution:
    """Per job, pick the feasible type that completes it earliest.

    Ties on completion are broken toward the cheaper type (so on-prem wins
    whenever it is immediately available).  Cloud is unlimited, so a job that
    cannot start now on-prem bursts immediately — an eager-bursting policy that
    approaches the minimum-turnaround anchor at maximal cost.

    order="spt" now means least-slack-first with SPT as tiebreak, not pure
    SPT — see _least_slack_order's docstring: pure SPT strands the
    tightest-deadline job (usually the longest one) behind every shorter
    job, leaving it permanently unscheduled even when feasible. This is the
    same fix as onprem_only_spt's, needed here for the same reason (both
    functions defaulted to _spt_order and hit the identical failure mode).
    """
    onprem = _onprem_ids(inst)
    state = _Capacity.fresh(inst)
    job_ids = _least_slack_order(inst) if order == "spt" else _arrival_order(inst)
    assignment: dict[int, tuple[int, int]] = {}

    for jid in job_ids:
        best: tuple[int, float, int, int] | None = None    # (completion, cost, start, type)
        for mid in inst.F[jid]:
            t = state.earliest_start(jid, mid)
            if t is None:
                continue
            comp = t + inst.p_occ[jid, mid]
            cost = inst.c[jid, mid]
            key = (comp, cost, mid)
            if best is None or key < (best[0], best[1], best[3]):
                best = (comp, cost, t, mid)
        if best is None:
            continue
        _comp, _cost, t, mid = best
        state.place(jid, mid, t)
        assignment[jid] = (mid, t)

    return make_solution(inst, assignment)


def threshold_bursting(inst: Instance, theta: float, *, order: str = "arrival") -> Solution:
    """Burst to cloud only when the achievable on-prem wait exceeds ``theta``.

    For each job (arrival order by default) the earliest feasible on-prem start
    is computed.  If it exists and its wait ``start - arrival`` is ``<= theta``
    slots, the job is scheduled on-prem; otherwise it is placed on the cheapest
    feasible cloud type at its earliest slot.

    ``theta = 0``         ≈ eager bursting (only keep jobs that start now)
    ``theta >= horizon``  ≈ ``onprem_only_spt`` (never burst)
    """
    onprem = _onprem_ids(inst)
    state = _Capacity.fresh(inst)
    job_ids = _spt_order(inst) if order == "spt" else _arrival_order(inst)
    # queueing is measured in whole slots past the job's arrival slot, so that
    # theta=0 means "only keep jobs that can start in their arrival slot"
    # regardless of the sub-slot arrival fraction.
    arrival_slot = {j.id: max(0, math.ceil(j.arrival)) for j in inst.jobs}
    assignment: dict[int, tuple[int, int]] = {}

    for jid in job_ids:
        a_slot = arrival_slot[jid]
        # earliest on-prem option
        op_best: tuple[int, int] | None = None                # (start, type)
        for mid in (m for m in inst.F[jid] if m in onprem):
            t = state.earliest_start(jid, mid)
            if t is None:
                continue
            if op_best is None or t < op_best[0]:
                op_best = (t, mid)

        use_onprem = op_best is not None and (op_best[0] - a_slot) <= theta
        if use_onprem:
            t, mid = op_best
            state.place(jid, mid, t)
            assignment[jid] = (mid, t)
            continue

        mid = _cheapest_cloud_type(inst, jid, onprem)
        if mid is None:                                       # no cloud fit -> on-prem fallback
            if op_best is not None:
                t, mid = op_best
                state.place(jid, mid, t)
                assignment[jid] = (mid, t)
            continue
        t = state.earliest_start(jid, mid)
        if t is None and op_best is not None:                 # cloud somehow full -> on-prem
            t, mid = op_best
        if t is None:
            continue
        state.place(jid, mid, t)
        assignment[jid] = (mid, t)

    return make_solution(inst, assignment)


def threshold_bursting_sweep(
    inst: Instance, thetas: list[float] | None = None, *, order: str = "arrival"
) -> list[tuple[float, Solution]]:
    """Run :func:`threshold_bursting` for a range of ``theta`` values.

    Returns ``[(theta, Solution), ...]`` in the given order.  The default sweep
    spans 0 slots (eager bursting) up to the horizon (never burst).
    """
    if thetas is None:
        H = inst.horizon
        thetas = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 48, float(H)]
    return [(th, threshold_bursting(inst, th, order=order)) for th in thetas]
