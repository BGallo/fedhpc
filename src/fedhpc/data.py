"""Instance data structures for FED-HPC (space-time network formulation).

Key design:
- InstanceType represents a homogeneous family of VMs (e.g. c7e, m6i, on-premises).
- F_j is computed from resource feasibility (CPU / mem / stor), not given as input.
- Two distinct processing times per (job, type) pair:
    p_bill  — billable time (no deploy penalty): used for monetary cost.
    p_occ   — occupation time (includes deploy): used for scheduling / time indexing.
- `kind` is a display-only label; the MIP does not use it.
- RunningJob represents a pre-scheduled job already running at t=0 that occupies
  on-prem capacity without being re-optimized.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path


class ScheduleError(ValueError):
    """A schedule violates the model (see :meth:`Instance.schedule_violations`).

    Raised where a solver hands back a schedule: an infeasible one would make
    its exact objective values, and any certificate built on them, meaningless.
    """



@dataclass
class Job:
    id: int
    arrival: float       # a_j
    exec_time: float     # e_j
    io_volume: float     # ι_j
    cpu: float           # q^cpu_j
    mem: float           # q^mem_j
    stor: float          # q^stor_j
    weight: float = 1.0  # w_j — turnaround weight in f1 (1.0 = plain total turnaround)
    # Must be integral (checked by Instance.build): with integer completions,
    # f1 values of two schedules then differ by an integer, which the exact
    # Pareto searches and BPC pruning rely on.


@dataclass
class InstanceType:
    id: int
    kind: str            # display label only — not used by the MIP
    perf: float          # P_m
    io_time: float       # τ^IO_m
    deploy: float        # π_m  (time penalty; generates no monetary cost)
    cost_io: float       # κ^IO_m
    cost_vm: float       # κ^VM_m
    cost_stor: float     # κ^S_m
    cpu: float           # C^cpu_m
    mem: float           # C^mem_m
    stor: float          # C^stor_m
    capacity: int | None = None  # max simultaneous instances (None = unlimited)
    fixed_cost: float = 0.0
    # U_m — one-off monetary charge added to f2 when at least one (new) job is
    # scheduled on this type (e.g. rental setup / activation fee). For a
    # capacity-1 type this is exactly a per-machine usage fee. Modelled in the
    # MIP by a binary z_m (see formulations.py); the C++ heuristics do not
    # support it.


@dataclass
class RunningJob:
    """A job already running at t=0 that pre-occupies on-prem capacity.

    RunningJobs are *not* re-optimized — they are fixed in the initial state.
    The formulations use them to reduce available capacity at t=0 and to model
    when those slots become free again as the jobs finish.

    Fields
    ------
    id       : identifier for display / debugging (e.g. original Slurm job ID).
    type_id  : which InstanceType it occupies (must be on-prem, i.e. finite capacity).
    end      : slot at which it finishes (exclusive); must satisfy end >= 1.
               Values beyond the planning horizon are clipped to ``horizon`` during
               ``Instance.build()``, so the slot is returned at the very last step.
    """
    id: int
    type_id: int
    end: int             # slot when the job finishes (exclusive); >= 1


@dataclass
class Instance:
    jobs: list[Job]
    instance_types: list[InstanceType]
    horizon: int         # H  (integer — discrete time steps 0..H)

    # Pre-assigned jobs already running at t=0 (optional; default: empty cluster).
    running_jobs: list[RunningJob] = field(default_factory=list)

    # Wall-clock duration of one slot, used for human-readable output only.
    # Read from _info.fedhpc_settings.time_scale_seconds when present; else 1.0.
    slot_size_seconds: float = 1.0

    # Monetary resolution δ (top-level "cost_resolution" in the JSON; default
    # 1e-4). build() rounds every c_jm and fixed cost U_m to a multiple of δ,
    # so f2 moves in whole steps of δ exactly as f1 moves in whole steps of 1
    # (integral weights × integer completions). The exact Pareto searches and
    # lexicographic certificates rely on both steps.
    cost_resolution: float = 1e-4

    # ---- Populated by build() for new (optimizable) jobs --------------------

    p_bill: dict[tuple[int, int], float] = field(default_factory=dict)
    # p^bill_jm = e_j/P_m + τ^IO_m · ι_j   (float, no deploy, used for cost)

    p_occ: dict[tuple[int, int], int] = field(default_factory=dict)
    # p^occ_jm = ceil(p^bill_jm + π_m)      (integer, with deploy, used for scheduling)

    c: dict[tuple[int, int], float] = field(default_factory=dict)
    c_units: dict[tuple[int, int], int] = field(default_factory=dict)
    # c_jm / δ — the exact integer cost; c_jm == c_units · δ
    # c^proc_jm = κ^IO_m·ι_j + κ^VM_m·p^bill_jm + κ^S_m·q^stor_j·p^bill_jm

    F: dict[int, list[int]] = field(default_factory=dict)
    # F_j = {m : q^cpu_j ≤ C^cpu_m, q^mem_j ≤ C^mem_m, q^stor_j ≤ C^stor_m}

    T: dict[tuple[int, int], range] = field(default_factory=dict)
    # T_jm = {t ∈ T : t ≥ a_j, t + p^occ_jm ≤ H}
    # Always a contiguous range (t + p_occ ≤ H is monotone in t for fixed p_occ),
    # stored as `range` rather than a materialized list so `in` membership tests
    # in the formulation builders are O(1) instead of O(len(T_jm)).

    # ---- Populated by build() for running jobs --------------------------------

    n_running: dict[int, int] = field(default_factory=dict)
    # {type_id: count of running jobs on that type at t=0}

    rj_arrivals: dict[tuple[int, int], int] = field(default_factory=dict)
    # {(type_id, slot): count of running jobs completing at that slot}
    # Used by SpaceTimeFormulation to inject ghost completions into flow equations.

    occupied: dict[tuple[int, int], int] = field(default_factory=dict)
    # {(type_id, t): count of running jobs occupying type during [t, t+1)}
    # Used by OccupancyFormulation to tighten the capacity upper bound on w[m,t].

    def build(self) -> None:
        fractional = [j.id for j in self.jobs if not float(j.weight).is_integer()]
        if fractional:
            raise ValueError(f"Job weights must be integral (jobs {fractional})")
        if not self.cost_resolution > 0:
            raise ValueError(f"cost_resolution must be positive, got {self.cost_resolution}")
        delta = self.cost_resolution
        # Fixed costs are rounded in place, so every consumer of m.fixed_cost
        # (formulations, BPC, reductions) sees the same grid value.
        for m in self.instance_types:
            m.fixed_cost = round(m.fixed_cost / delta) * delta

        type_map = {m.id: m for m in self.instance_types}

        for j in self.jobs:
            # (eq. 1) Resource-based feasibility
            self.F[j.id] = [
                m.id for m in self.instance_types
                if j.cpu <= m.cpu and j.mem <= m.mem and j.stor <= m.stor
            ]

            for m_id in self.F[j.id]:
                m = type_map[m_id]

                # (eq. 2) Billable time — no deploy penalty
                p_bill = j.exec_time / m.perf + m.io_time * j.io_volume
                self.p_bill[j.id, m_id] = p_bill

                # (eq. 3) Occupation time — includes deploy, ceiled to positive int
                self.p_occ[j.id, m_id] = max(1, math.ceil(p_bill + m.deploy))

                # (eq. 4) Monetary cost — uses p_bill (deploy is free),
                # rounded to the cost grid δ
                units = round((
                    m.cost_io * j.io_volume
                    + m.cost_vm * p_bill
                    + m.cost_stor * j.stor * p_bill
                ) / delta)
                self.c_units[j.id, m_id] = units
                self.c[j.id, m_id] = units * delta

                # (eq. 5) Feasible start times — uses p_occ.
                # t + p_occ ≤ horizon  ⟺  t ≤ horizon − p_occ, so the feasible
                # set is exactly the contiguous range [a_min, horizon − p_occ].
                # a_min is floored at 0: a job cannot be scheduled to start
                # before the current decision window even if its true arrival
                # (used unclamped in f1_expr for turnaround urgency) is
                # negative, i.e. it already waited before this window opened.
                # The flow-conservation constraints (eq. 19) only account for
                # capacity over v in [0, H]; a negative a_min would let a job
                # "start" at t < 0 without that slot ever being deducted from
                # any y[m,t]/occupancy, while still counting as a completion
                # (ghost arrival) at v = t + p_occ — fabricating free capacity.
                a_min = max(0, math.ceil(j.arrival))
                t_max_excl = self.horizon - self.p_occ[j.id, m_id] + 1
                self.T[j.id, m_id] = range(a_min, t_max_excl)

        infeasible = [j.id for j in self.jobs if not self.F[j.id]]
        if infeasible:
            raise ValueError(f"Jobs with no feasible instance type: {infeasible}")

        no_times = [
            (j.id, m_id)
            for j in self.jobs
            for m_id in self.F[j.id]
            if not self.T[j.id, m_id]
        ]
        if no_times:
            raise ValueError(f"Job-type pairs with no feasible start time: {no_times}")

        # ---- Running-job derived indices ------------------------------------
        # Each running job occupies exactly one capacity unit from slot 0 until
        # it finishes.  Jobs finishing beyond the horizon are treated as if they
        # finish at H (they permanently occupy a slot in the planning window).
        #
        # IMPORTANT: the total count per type must not exceed its capacity —
        # one entry is one capacity unit (e.g. one node; jobs sharing a node
        # must be merged into a single entry by the instance generator).
        # ``build()`` raises rather than silently dropping running jobs.
        capacity_map = {
            itype.id: itype.capacity
            for itype in self.instance_types
            if itype.capacity is not None
        }

        n_listed: dict[int, int] = {}
        for rj in self.running_jobs:
            if rj.type_id in capacity_map:
                n_listed[rj.type_id] = n_listed.get(rj.type_id, 0) + 1
        over = {mid: (n, capacity_map[mid]) for mid, n in n_listed.items()
                if n > capacity_map[mid]}
        if over:
            raise ValueError(
                "Running jobs exceed capacity (type: listed/capacity): "
                + ", ".join(f"{mid}: {n}/{cap}" for mid, (n, cap) in sorted(over.items()))
            )

        for rj in self.running_jobs:
            mid = rj.type_id
            cap = capacity_map.get(mid)

            # Silently skip if type is unknown or cloud (unlimited capacity)
            if cap is None:
                continue

            self.n_running[mid] = self.n_running.get(mid, 0) + 1

            # Clip end to horizon so the slot is returned within the window
            end_slot = min(rj.end, self.horizon)

            # Ghost completion: instance returns to idle pool at end_slot
            key = (mid, end_slot)
            self.rj_arrivals[key] = self.rj_arrivals.get(key, 0) + 1

            # Occupancy during [0, end_slot): counts towards per-slot capacity usage
            for t in range(end_slot):
                self.occupied[mid, t] = self.occupied.get((mid, t), 0) + 1

    # ---- Schedule feasibility ----------------------------------------------

    def schedule_violations(
        self,
        assignment: dict[int, tuple[int, int]],
        *,
        complete: bool = True,
    ) -> list[str]:
        """Every way the ``{job: (type, start)}`` map breaks the model, as
        messages (empty when feasible).

        Checks known job ids, an allowed type (m ∈ F_j), a start in T_jm
        (release and horizon), and, per finite-capacity type, that new jobs
        plus running jobs never exceed the capacity in any slot. With
        ``complete`` (the default) every job must be assigned.
        """
        out: list[str] = []
        known = {j.id for j in self.jobs}
        if complete:
            missing = sorted(known - assignment.keys())
            if missing:
                out.append(f"unassigned jobs {missing[:10]}")
        H = self.horizon
        cap = {m.id: m.capacity for m in self.instance_types if m.capacity is not None}
        # per finite type, a difference array of busy slots over [0, H)
        delta = {mid: [0] * (H + 1) for mid in cap}
        for jid, (mid, t) in assignment.items():
            if jid not in known:
                out.append(f"unknown job {jid}")
                continue
            if mid not in self.F[jid]:
                out.append(f"job {jid} on type {mid}, not in F_j {self.F[jid]}")
                continue
            if t not in self.T[jid, mid]:
                out.append(f"job {jid} on type {mid} starts at {t}, outside T_jm "
                           f"{self.T[jid, mid].start}..{self.T[jid, mid].stop - 1}")
                continue
            if mid in delta:
                delta[mid][t] += 1
                delta[mid][t + self.p_occ[jid, mid]] -= 1
        for mid, d in delta.items():
            busy = 0
            for t in range(H):
                busy += d[t]
                used = busy + self.occupied.get((mid, t), 0)
                if used > cap[mid]:
                    out.append(f"type {mid} uses {used} > capacity {cap[mid]} "
                               f"in slot {t}")
                    break
        return out

    def check_schedule(
        self,
        assignment: dict[int, tuple[int, int]],
        *,
        complete: bool = True,
    ) -> None:
        """Raise :class:`ScheduleError` if ``assignment`` is infeasible (see
        :meth:`schedule_violations`)."""
        violations = self.schedule_violations(assignment, complete=complete)
        if violations:
            raise ScheduleError("infeasible schedule: " + "; ".join(violations[:5]))

    # ---- Objective evaluation ----------------------------------------------

    @property
    def has_fixed_costs(self) -> bool:
        return any(m.fixed_cost for m in self.instance_types)

    def f1_of(self, completion: dict[int, int]) -> float:
        """Weighted turnaround Σ w_j (C_j − a_j) over the jobs in ``completion``."""
        return sum(
            j.weight * (completion[j.id] - j.arrival)
            for j in self.jobs if j.id in completion
        )

    def f1_level(self, f1: float) -> int:
        """The integer Σ w_j C_j behind an f1 value: f1 = Σ w_j C_j − Σ w_j a_j,
        so two schedules' f1 differ by exactly the difference of their levels."""
        return round(f1 + sum(j.weight * j.arrival for j in self.jobs))

    def f2_units(self, f2: float) -> int:
        """f2 in whole multiples of the cost resolution δ."""
        return round(f2 / self.cost_resolution)

    def f2_of(self, assignment: dict[int, tuple[int, int]]) -> float:
        """Monetary cost Σ c_jm + Σ_{m used} U_m of a ``{job: (type, start)}``
        map, summed exactly on the cost grid."""
        delta = self.cost_resolution
        fixed = {m.id: round(m.fixed_cost / delta) for m in self.instance_types}
        used = {mid for mid, _t in assignment.values()}
        return delta * (
            sum(self.c_units[jid, mid] for jid, (mid, _t) in assignment.items())
            + sum(fixed[mid] for mid in used)
        )

    @classmethod
    def from_dict(cls, data: dict) -> Instance:
        instance_types = [InstanceType(**td) for td in data["instance_types"]]
        jobs = [Job(**jd) for jd in data["jobs"]]
        running_jobs = [RunningJob(**rd) for rd in data.get("running_jobs", [])]

        # Infer slot_size_seconds from metadata when present.
        slot_size_seconds: float = 1.0
        info = data.get("_info", {})
        settings = info.get("fedhpc_settings", {})
        if "time_scale_seconds" in settings:
            slot_size_seconds = float(settings["time_scale_seconds"])

        inst = cls(
            jobs=jobs,
            instance_types=instance_types,
            horizon=int(data["horizon"]),
            running_jobs=running_jobs,
            slot_size_seconds=slot_size_seconds,
            cost_resolution=float(data.get("cost_resolution", 1e-4)),
        )
        inst.build()
        return inst

    @classmethod
    def from_file(cls, path: str | Path) -> Instance:
        with open(path) as f:
            return cls.from_dict(json.load(f))


REFERENCE_POINTS = Path(__file__).resolve().parents[2] / "data" / "reference_points.json"


def reference_points(instance_path: str | Path,
                     table: str | Path = REFERENCE_POINTS) -> tuple[float, float, float, float]:
    """Exact weighted-sum reference points ``(f1_T, f2_T, f1_0, f2_0)`` of an
    instance file, from ``table`` (``data/reference_points.json``).

    Raises ``KeyError`` when the instance has no entry and ``ValueError`` when
    the entry was computed on a different version of the file (its sha256
    differs) or with a different cost resolution — stale reference points
    silently mis-normalise the weighted sum.
    Recompute with ``scripts/compute_reference_points.py``.
    """
    path = Path(instance_path)
    with open(table) as f:
        entry = json.load(f)[path.stem]
    raw = path.read_bytes()
    resolution = float(json.loads(raw).get("cost_resolution", 1e-4))
    if entry["sha256"] != hashlib.sha256(raw).hexdigest():
        reason = "instance file changed"
    elif entry.get("cost_resolution") != resolution:
        reason = "cost resolution changed"
    else:
        return entry["f1_T"], entry["f2_T"], entry["f1_0"], entry["f2_0"]
    raise ValueError(
        f"reference points for {path.stem} are stale ({reason}); "
        "rerun scripts/compute_reference_points.py"
    )
