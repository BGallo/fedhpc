"""Shared pytest fixtures for the FED-HPC test suite."""
import os

import pytest

from fedhpc.data import Instance, InstanceType, Job


@pytest.fixture(autouse=True)
def chdir_to_project_root(monkeypatch):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    monkeypatch.chdir(root)


# ---------------------------------------------------------------------------
# Known-answer fixture: 2 jobs × 2 types, exact 3-point Pareto front.
#
# Type 0  on-prem: perf=1, free,    capacity=1
# Type 1  cloud:   perf=2, cost=50/job (cost_vm=10, p_bill=5), unlimited
#
# Both jobs arrive at t=0, exec_time=10, same resource profile.
#
# Derived values:
#   p_occ[j, m0] = 10,  cost[j, m0] = 0
#   p_occ[j, m1] = 5,   cost[j, m1] = 50
#
# Pareto-optimal (f1, f2) points — analytically verified:
#   (10, 100)  both on cloud, start=0    →  turnaround 5+5, cost 50+50
#   (15,  50)  one cloud + one on-prem   →  turnaround 5+10, cost 50+0
#   (30,   0)  both on-prem sequential   →  turnaround 10+20 or 20+10, cost 0
#
# Any later start time raises f1 without changing f2 → dominated.
# No other (f1, f2) pair is Pareto-optimal.
# ---------------------------------------------------------------------------

@pytest.fixture
def known_pareto_inst() -> Instance:
    inst = Instance(
        jobs=[
            Job(id=0, arrival=0, exec_time=10, io_volume=0, cpu=2, mem=4, stor=20),
            Job(id=1, arrival=0, exec_time=10, io_volume=0, cpu=2, mem=4, stor=20),
        ],
        instance_types=[
            InstanceType(id=0, kind="on-prem", perf=1.0, io_time=0.0, deploy=0,
                         cost_io=0.0, cost_vm=0.0, cost_stor=0.0,
                         cpu=4, mem=8, stor=50, capacity=1),
            InstanceType(id=1, kind="cloud", perf=2.0, io_time=0.0, deploy=0,
                         cost_io=0.0, cost_vm=10.0, cost_stor=0.0,
                         cpu=4, mem=8, stor=50, capacity=None),
        ],
        horizon=30,
        budget=200,
    )
    inst.build()
    return inst


@pytest.fixture
def known_pareto_front() -> set[tuple[float, float]]:
    """The three Pareto-optimal (f1, f2) points for known_pareto_inst."""
    return {(10.0, 100.0), (15.0, 50.0), (30.0, 0.0)}


# ---------------------------------------------------------------------------
# Shared helpers (importable by test modules via explicit import)
# ---------------------------------------------------------------------------

def recompute_f1(inst: Instance, sol) -> float:
    """Re-derive f1 = Σ(completion - arrival) from the assignment."""
    return sum(sol.completion[j.id] - j.arrival for j in inst.jobs)


def recompute_f2(inst: Instance, sol) -> float:
    """Re-derive f2 = Σ c[j,m] from the assignment."""
    return sum(inst.c[jid, mid] for jid, (mid, _) in sol.assignment.items())


def capacity_violations(inst: Instance, sol) -> list[tuple]:
    """Return (type_id, slot, total_count, capacity) for every violated slot.

    Takes running-job pre-occupancy into account.
    """
    occ: dict[tuple[int, int], int] = {}
    for jid, (mid, t) in sol.assignment.items():
        for slot in range(t, t + inst.p_occ[jid, mid]):
            occ[(mid, slot)] = occ.get((mid, slot), 0) + 1

    viols = []
    for (mid, slot), cnt in occ.items():
        init = inst.occupied.get((mid, slot), 0)
        total = cnt + init
        cap = next(x for x in inst.instance_types if x.id == mid).capacity
        if cap is not None and total > cap:
            viols.append((mid, slot, total, cap))
    return viols


def unique_front_points(sols) -> set[tuple[float, float]]:
    """Return the set of distinct (f1, f2) values in a solution list."""
    return {(round(s.f1, 6), round(s.f2, 6)) for s in sols}


def is_non_dominated_set(sols) -> bool:
    """Return True iff no solution in *sols* is strictly dominated by another."""
    pts = [(s.f1, s.f2) for s in sols]
    for i, (f1i, f2i) in enumerate(pts):
        for j, (f1j, f2j) in enumerate(pts):
            if i == j:
                continue
            if f1j <= f1i and f2j <= f2i and (f1j < f1i or f2j < f2i):
                return False
    return True
