"""Tests for the offline scheduling baselines (baselines.py).

Coverage
────────
Validity        Every baseline returns a Solution whose assignment respects
                F_j feasibility, T_jm start windows, and per-type capacity
                (running-job pre-occupancy included) — checked on the real
                fedhpc 10-min instance (the faster of the two real traces).
KnownAnswer     On the 2-job / 2-type analytic fixture the baselines land on
                the known extreme points.
ThresholdSweep  Cost is (weakly) monotone decreasing in θ and the sweep spans
                from a burst end to a zero-cost never-burst end.
"""
from pathlib import Path

import pytest
from conftest import capacity_violations

from fedhpc.baselines import (
    greedy_earliest_completion,
    onprem_only_spt,
    threshold_bursting,
    threshold_bursting_sweep,
)
from fedhpc.data import Instance

# Faster of the two real instances (964 jobs vs 3340); load+build ≈ 20 ms.
_REAL = "data/fedhpc_known_runtime_offline_10min.json"


@pytest.fixture(scope="module")
def real_inst() -> Instance:
    path = Path(_REAL)
    if not path.exists():
        pytest.skip(f"{_REAL} not present")
    return Instance.from_file(path)


def _assert_valid(inst: Instance, sol) -> None:
    assert not capacity_violations(inst, sol), "capacity violated"
    for jid, (mid, t) in sol.assignment.items():
        assert mid in inst.F[jid], f"job {jid} on infeasible type {mid}"
        rng = inst.T[jid, mid]
        assert rng.start <= t < rng.stop, f"job {jid} start {t} outside {rng}"
        assert sol.completion[jid] == t + inst.p_occ[jid, mid]
    f1 = sum(sol.completion[j.id] - j.arrival for j in inst.jobs if j.id in sol.completion)
    f2 = sum(inst.c[jid, mid] for jid, (mid, _t) in sol.assignment.items())
    assert sol.f1 == pytest.approx(f1)
    assert sol.f2 == pytest.approx(f2)


class TestValidity:
    """Run every baseline against the real fedhpc instance and check feasibility."""

    def test_onprem_only(self, real_inst):
        sol = onprem_only_spt(real_inst)
        _assert_valid(real_inst, sol)
        assert sol.f2 == 0.0                       # never bursts
        assert sol.assignment                      # schedules something

    def test_greedy(self, real_inst):
        sol = greedy_earliest_completion(real_inst)
        _assert_valid(real_inst, sol)
        assert len(sol.assignment) == len(real_inst.jobs)   # cloud is unlimited

    @pytest.mark.parametrize("theta", [0, 2, 8, 32, 10_000])
    def test_threshold(self, real_inst, theta):
        _assert_valid(real_inst, threshold_bursting(real_inst, theta))

    def test_greedy_reaches_turnaround_optimum(self, real_inst):
        """p_occ is type-independent on this instance, so earliest-completion for
        every job = start ASAP; greedy must reach the same f1 as never-wait."""
        greedy = greedy_earliest_completion(real_inst)
        burst0 = threshold_bursting(real_inst, 0)
        assert greedy.f1 == pytest.approx(burst0.f1)

    def test_onprem_only_beats_never_burst_threshold_on_f1(self, real_inst):
        """SPT + global pool balancing should schedule at least as tightly as the
        arrival-order never-burst threshold policy."""
        onprem = onprem_only_spt(real_inst)
        never = threshold_bursting(real_inst, real_inst.horizon)
        assert onprem.f2 == never.f2 == 0.0
        assert onprem.f1 <= never.f1


class TestThresholdSweep:
    def test_cost_monotone_in_theta(self, real_inst):
        sweep = threshold_bursting_sweep(real_inst)
        costs = [s.f2 for _th, s in sweep]
        assert costs == sorted(costs, reverse=True), "cost should not rise with θ"

    def test_sweep_spans_both_extremes(self, real_inst):
        sweep = threshold_bursting_sweep(real_inst)
        f2s = [s.f2 for _th, s in sweep]
        assert min(f2s) == 0.0
        assert max(f2s) > 0.0
        for _th, s in sweep:
            _assert_valid(real_inst, s)


class TestKnownAnswer:
    """known_pareto_inst: 2 jobs, type0 on-prem cap1 free (perf 1), type1 cloud
    $50/job (perf 2). Pareto points (f1, f2): (10,100), (15,50), (30,0)."""

    def test_onprem_only_is_zero_cost_extreme(self, known_pareto_inst):
        sol = onprem_only_spt(known_pareto_inst)
        assert (sol.f1, sol.f2) == (30.0, 0.0)

    def test_greedy_picks_earliest_completion_even_when_costly(self, known_pareto_inst):
        # cloud (perf 2) finishes each job at t=5 vs t=10 on-prem, so earliest-
        # completion sends BOTH to cloud — the (f1=10, f2=100) turnaround extreme.
        sol = greedy_earliest_completion(known_pareto_inst)
        assert (sol.f1, sol.f2) == (10.0, 100.0)

    def test_threshold_extremes(self, known_pareto_inst):
        burst = threshold_bursting(known_pareto_inst, 0)
        never = threshold_bursting(known_pareto_inst, known_pareto_inst.horizon)
        assert burst.f2 >= never.f2
        assert never.f2 == 0.0
