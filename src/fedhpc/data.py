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

import json
import math
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Job:
    id: int
    arrival: float       # a_j
    exec_time: float     # e_j
    io_volume: float     # ι_j
    cpu: float           # q^cpu_j
    mem: float           # q^mem_j
    stor: float          # q^stor_j


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
    budget: float        # B

    # Pre-assigned jobs already running at t=0 (optional; default: empty cluster).
    running_jobs: list[RunningJob] = field(default_factory=list)

    # Wall-clock duration of one slot, used for human-readable output only.
    # Read from _info.fedhpc_settings.time_scale_seconds when present; else 1.0.
    slot_size_seconds: float = 1.0

    # ---- Populated by build() for new (optimizable) jobs --------------------

    p_bill: dict[tuple[int, int], float] = field(default_factory=dict)
    # p^bill_jm = e_j/P_m + τ^IO_m · ι_j   (float, no deploy, used for cost)

    p_occ: dict[tuple[int, int], int] = field(default_factory=dict)
    # p^occ_jm = ceil(p^bill_jm + π_m)      (integer, with deploy, used for scheduling)

    c: dict[tuple[int, int], float] = field(default_factory=dict)
    # c^proc_jm = κ^IO_m·ι_j + κ^VM_m·p^bill_jm + κ^S_m·q^stor_j·p^bill_jm

    F: dict[int, list[int]] = field(default_factory=dict)
    # F_j = {m : q^cpu_j ≤ C^cpu_m, q^mem_j ≤ C^mem_m, q^stor_j ≤ C^stor_m}

    T: dict[tuple[int, int], list[int]] = field(default_factory=dict)
    # T_jm = {t ∈ T : t ≥ a_j, t + p^occ_jm ≤ H}

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

                # (eq. 4) Monetary cost — uses p_bill (deploy is free)
                self.c[j.id, m_id] = (
                    m.cost_io * j.io_volume
                    + m.cost_vm * p_bill
                    + m.cost_stor * j.stor * p_bill
                )

                # (eq. 5) Feasible start times — uses p_occ
                a_min = math.ceil(j.arrival)
                self.T[j.id, m_id] = [
                    t for t in range(a_min, self.horizon + 1)
                    if t + self.p_occ[j.id, m_id] <= self.horizon
                ]

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
        # IMPORTANT: the total count per type must not exceed its capacity.
        # Instance JSON generation should already cap the list; ``build()``
        # enforces this by ignoring excess entries and emitting a warning.
        capacity_map = {
            itype.id: itype.capacity
            for itype in self.instance_types
            if itype.capacity is not None
        }

        for rj in self.running_jobs:
            mid = rj.type_id
            cap = capacity_map.get(mid)

            # Silently skip if type is unknown or cloud (unlimited capacity)
            if cap is None:
                continue

            # Enforce capacity ceiling
            current = self.n_running.get(mid, 0)
            if current >= cap:
                import warnings
                warnings.warn(
                    f"Running jobs for type {mid} exceed capacity {cap}; "
                    f"excess entries are ignored.",
                    stacklevel=2,
                )
                continue

            self.n_running[mid] = current + 1

            # Clip end to horizon so the slot is returned within the window
            end_slot = min(rj.end, self.horizon)

            # Ghost completion: instance returns to idle pool at end_slot
            key = (mid, end_slot)
            self.rj_arrivals[key] = self.rj_arrivals.get(key, 0) + 1

            # Occupancy during [0, end_slot): counts towards per-slot capacity usage
            for t in range(end_slot):
                self.occupied[mid, t] = self.occupied.get((mid, t), 0) + 1

    @classmethod
    def from_dict(cls, data: dict) -> "Instance":
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
            budget=data["budget"],
            running_jobs=running_jobs,
            slot_size_seconds=slot_size_seconds,
        )
        inst.build()
        return inst

    @classmethod
    def from_file(cls, path: str | Path) -> "Instance":
        with open(path) as f:
            return cls.from_dict(json.load(f))
