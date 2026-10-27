"""Shared pytest fixtures for the FED-HPC test suite."""
import os

import pytest

from fedhpc.data import Instance, InstanceType, Job, reference_points


@pytest.fixture(autouse=True)
def chdir_to_project_root(monkeypatch):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    monkeypatch.chdir(root)


@pytest.fixture
def weighted_fixed_cost_inst() -> Instance:
    """3 jobs x 2 types, with job weights != 1 and a fixed activation cost on
    the cloud type — exercises both formulation extras beyond MOFE §3 that
    data.py/formulations.py add (Job.weight and InstanceType.fixed_cost /
    add_fixed_cost_vars), which known_pareto_inst does not set (the trap
    instances below have weights but no fixed cost). Used by tests/test_bpc.py and tests/test_bpc_branch.py.
    """
    inst = Instance(
        jobs=[
            Job(id=0, arrival=0, exec_time=6, io_volume=0, cpu=2, mem=4, stor=20, weight=1.0),
            Job(id=1, arrival=0, exec_time=6, io_volume=0, cpu=2, mem=4, stor=20, weight=3.0),
            Job(id=2, arrival=2, exec_time=4, io_volume=0, cpu=2, mem=4, stor=20, weight=2.0),
        ],
        instance_types=[
            InstanceType(id=0, kind="on-prem", perf=1.0, io_time=0.0, deploy=0,
                         cost_io=0.0, cost_vm=0.0, cost_stor=0.0,
                         cpu=4, mem=8, stor=50, capacity=1),
            InstanceType(id=1, kind="cloud", perf=1.0, io_time=0.0, deploy=0,
                         cost_io=0.0, cost_vm=2.0, cost_stor=0.0,
                         cpu=4, mem=8, stor=50, capacity=None, fixed_cost=15.0),
        ],
        horizon=25,
    )
    inst.build()
    return inst


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
    )
    inst.build()
    return inst


@pytest.fixture
def known_pareto_front() -> set[tuple[float, float]]:
    """The three Pareto-optimal (f1, f2) points for known_pareto_inst."""
    return {(10.0, 100.0), (15.0, 50.0), (30.0, 0.0)}


# ---------------------------------------------------------------------------
# Trap instances: small, but with a non-obvious optimum.
#
# Four types: a free on-prem pool of 2 (cpu 32) and a free, faster on-prem
# "fat" node of 1 (cpu 64, perf 1.25) — both partly busy with running jobs at
# t = 0 — plus a slow cheap cloud type (deploy 1, io_time 0.5) and a fast
# pricey one (perf 2, deploy 2, io_time 0.25). Jobs have staggered fractional
# arrivals, bimodal runtimes, integer weights 1-3, some I/O (which slows
# cloud), and "fat" jobs (cpu 48) that only the fat node or cloud can run.
#
# Picked from a seeded search over that family for the widest gap between
# the exact optimum and every list-scheduling baseline (greedy earliest
# completion, threshold bursting, on-prem-only least-slack): the optimum
# hinges on which jobs wait for the fat node or a running job's release,
# not on size. tests/test_trap_instances.py pins those gaps.
#
#   trap_small   7 jobs, H=34: front of 10 points; baselines >= 9.7% above
#                min f1 and >= 24.7% above min f1 at zero cost.
#   trap_medium 12 jobs, H=43: front of 27 points; >= 14.2% / >= 81.2%.
#
# In both, the running jobs bind: dropping them lowers both anchors.
#
# Stress tests use data/pos_congestion_known_runtime_offline_10min.json
# (3340 jobs, 1525 running) instead — see pos_congestion_inst.
# ---------------------------------------------------------------------------

def _trap(horizon, cost_slow, cost_fast, jobs, running) -> Instance:
    """``jobs``: (arrival, exec_time, io_volume, fat, weight) per job id;
    ``running``: (type_id, end) per running job."""
    def itype(id, kind, perf, io_time, deploy, cost_io, cost_vm, cpu, mem, capacity):
        return dict(id=id, kind=kind, perf=perf, io_time=io_time, deploy=deploy,
                    cost_io=cost_io, cost_vm=cost_vm, cost_stor=0.0,
                    cpu=cpu, mem=mem, stor=1000, capacity=capacity)
    return Instance.from_dict(dict(
        horizon=horizon,
        instance_types=[
            itype(0, "on-prem", 1.0, 0.0, 0.0, 0.0, 0.0, 32, 128, 2),
            itype(1, "on-prem", 1.25, 0.0, 0.0, 0.0, 0.0, 64, 512, 1),
            itype(2, "cloud", 1.0, 0.5, 1.0, 0.05, cost_slow, 64, 512, None),
            itype(3, "cloud", 2.0, 0.25, 2.0, 0.05, cost_fast, 64, 512, None),
        ],
        jobs=[
            dict(id=j, arrival=a, exec_time=e, io_volume=io,
                 cpu=48 if fat else 16, mem=256 if fat else 64, stor=10, weight=w)
            for j, (a, e, io, fat, w) in enumerate(jobs)
        ],
        running_jobs=[dict(id=100 + k, type_id=m, end=end)
                      for k, (m, end) in enumerate(running)],
    ))


def trap_small() -> Instance:
    return _trap(34, 0.214, 0.979, [
        (8.91, 1.83, 0, False, 3.0), (6.87, 3.83, 6, False, 2.0),
        (4.14, 13.38, 6, True, 1.0), (8.15, 3.58, 2, False, 2.0),
        (0.56, 3.04, 2, True, 1.0), (6.54, 1.2, 0, False, 3.0),
        (3.04, 8.05, 6, False, 2.0),
    ], [(0, 3), (1, 4)])


def trap_medium() -> Instance:
    return _trap(43, 0.23, 1.027, [
        (4.24, 3.38, 2, True, 1.0), (2.78, 3.13, 0, True, 1.0),
        (3.58, 2.84, 6, False, 1.0), (5.49, 3.16, 2, False, 1.0),
        (1.16, 13.0, 0, False, 2.0), (7.04, 12.12, 0, False, 1.0),
        (6.59, 3.12, 2, False, 2.0), (4.48, 3.58, 0, False, 2.0),
        (0.36, 2.18, 6, False, 3.0), (3.98, 1.17, 0, False, 1.0),
        (5.0, 1.15, 6, False, 3.0), (5.11, 3.35, 6, False, 1.0),
    ], [(0, 4), (1, 4)])


TRAP_INSTANCES = {"trap_small": trap_small, "trap_medium": trap_medium}


@pytest.fixture(name="trap_small")
def trap_small_fixture() -> Instance:
    return trap_small()


@pytest.fixture(name="trap_medium")
def trap_medium_fixture() -> Instance:
    return trap_medium()


@pytest.fixture(params=sorted(TRAP_INSTANCES))
def trap_inst(request) -> Instance:
    """Each trap instance in turn."""
    return TRAP_INSTANCES[request.param]()


POS_CONGESTION = "data/pos_congestion_known_runtime_offline_10min.json"
# Exact weighted-sum reference points (data/reference_points.json; f2_0 = 0).
POS_CONGESTION_REFS = dict(zip(("f1_T", "f2_T", "f1_0"), reference_points(POS_CONGESTION)))


@pytest.fixture(scope="session")
def pos_congestion_inst() -> Instance:
    """The stress instance: 3340 jobs, 16 types (4 on-prem + 12 on-demand
    cloud), H=288, 1525 running jobs."""
    return Instance.from_file(POS_CONGESTION)


# ---------------------------------------------------------------------------
# Shared helpers (importable by test modules via explicit import)
# ---------------------------------------------------------------------------

def recompute_f1(inst: Instance, sol) -> float:
    """Re-derive f1 = Σ w_j (completion - arrival) from the assignment."""
    return sum(j.weight * (sol.completion[j.id] - j.arrival) for j in inst.jobs)


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
