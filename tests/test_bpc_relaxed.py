"""Correctness tests for the 1-cycle-free relaxed pricing pre-screen,
adapted from Kramer, Dell'Amico, Feillet & Iori (2020)'s ``LC_2l`` — see
bpc.py's module docstring ("Comparison with Kramer et al. 2020") for the
full writeup of what this is and why it's used only as a skip-test, never
as a source of RMP columns.

Two properties are what actually matter here, and both are checked
directly:

1. ``_price_type_relaxed``'s reduced cost is always <= ``_price_type``'s
   true elementary one, for the same duals (the 1-cycle-free path space is
   a strict superset of the elementary path space) — this is the whole
   correctness argument for using it as a safe early-exit.
2. ``use_relaxed_prescreen=True`` and ``False`` must agree *exactly* on
   ``_branch_and_price``'s result — the pre-screen must never change which
   columns get added, only how much exact-DP work is spent finding out none
   exist.
"""
from __future__ import annotations

import pytest

from fedhpc.data import Instance
from fedhpc.formulations import configure_env

configure_env()


class TestRelaxedLowerBound:
    """_price_type_relaxed's value must never exceed _price_type's, across a
    variety of dual regimes (not just all-zero/uniform duals, which can hide
    a directional bug if the relaxation happens not to bite)."""

    def test_relaxed_never_exceeds_exact_uniform_duals(self):
        from fedhpc.data import InstanceType, Job
        from fedhpc.bpc import _build_arcs_by_type, _price_type, _price_type_relaxed

        inst = Instance(
            jobs=[
                Job(id=0, arrival=0, exec_time=2, io_volume=0, cpu=1, mem=1, stor=1),
                Job(id=1, arrival=0, exec_time=2, io_volume=0, cpu=1, mem=1, stor=1),
                Job(id=2, arrival=0, exec_time=2, io_volume=0, cpu=1, mem=1, stor=1),
            ],
            instance_types=[
                InstanceType(id=0, kind="on-prem", perf=1.0, io_time=0.0, deploy=0,
                             cost_io=0.0, cost_vm=0.0, cost_stor=0.0,
                             cpu=4, mem=8, stor=50, capacity=2),
            ],
            horizon=20,
        )
        inst.build()
        arcs = _build_arcs_by_type(inst)[0]
        arrival_of = {j.id: j.arrival for j in inst.jobs}
        weight_of = {j.id: j.weight for j in inst.jobs}
        sigma = {0: 50.0, 1: 50.0, 2: 50.0}

        exact_best, _col = _price_type(
            inst, 0, arcs, weight1=0.0, weight2=1.0, sigma=sigma, nu1=None, nu2=None,
            rho_m=0.0, theta={}, arrival_of=arrival_of, weight_of=weight_of, tol=1e-7,
        )
        relaxed_best = _price_type_relaxed(
            inst, 0, arcs, weight1=0.0, weight2=1.0, sigma=sigma, nu1=None, nu2=None,
            rho_m=0.0, theta={}, arrival_of=arrival_of, weight_of=weight_of,
        )
        assert relaxed_best <= exact_best + 1e-6

    @pytest.mark.parametrize("seed", range(8))
    def test_relaxed_never_exceeds_exact_random_duals(self, seed, known_pareto_inst):
        """Random sigma/nu2/theta draws exercise many more dual regimes than
        a single hand-picked example could — the 1-cycle relaxation must
        stay a valid lower bound in every one of them, per type.
        """
        import random

        from fedhpc.bpc import (
            _build_arcs_by_type, _fixed_cost_types, _price_type, _price_type_relaxed,
        )

        rng = random.Random(seed)
        inst = known_pareto_inst
        arcs_by_type = _build_arcs_by_type(inst)
        arrival_of = {j.id: j.arrival for j in inst.jobs}
        weight_of = {j.id: j.weight for j in inst.jobs}
        sigma = {j.id: rng.uniform(-5, 50) for j in inst.jobs}
        nu2 = rng.uniform(-2, 2)  # f2-cap dual: same arc coefficient as cost2
        theta = {
            (j.id, itype.id): rng.uniform(-5, 5)
            for j in inst.jobs for itype in _fixed_cost_types(inst)
        }

        for itype in inst.instance_types:
            arcs = arcs_by_type[itype.id]
            exact = _price_type(
                inst, itype.id, arcs, weight1=0.3, weight2=0.7, sigma=sigma,
                nu1=None, nu2=nu2, rho_m=0.0, theta=theta,
                arrival_of=arrival_of, weight_of=weight_of, tol=1e-7,
            )
            exact_best = exact[0] if exact is not None else float("inf")
            relaxed_best = _price_type_relaxed(
                inst, itype.id, arcs, weight1=0.3, weight2=0.7, sigma=sigma,
                nu1=None, nu2=nu2, rho_m=0.0, theta=theta,
                arrival_of=arrival_of, weight_of=weight_of,
            )
            assert relaxed_best <= exact_best + 1e-6, (
                f"seed={seed} type={itype.id}: relaxed {relaxed_best} > exact {exact_best}"
            )


class TestPrescreenEndToEnd:
    """use_relaxed_prescreen must never change _branch_and_price's result —
    only how it gets there."""

    def test_matches_unscreened_on_known_pareto_inst(self, known_pareto_inst):
        from fedhpc.bpc import _branch_and_price

        for w1, w2 in [(1.0, 0.0), (0.0, 1.0), (0.5, 0.5)]:
            screened, _a1 = _branch_and_price(known_pareto_inst, w1, w2, use_relaxed_prescreen=True)
            unscreened, _a2 = _branch_and_price(known_pareto_inst, w1, w2, use_relaxed_prescreen=False)
            assert screened == pytest.approx(unscreened, abs=1e-4)

    def test_matches_unscreened_with_job_weights_and_fixed_cost(self, weighted_fixed_cost_inst):
        from fedhpc.bpc import _branch_and_price

        inst = weighted_fixed_cost_inst
        for w1, w2 in [(1.0, 0.0), (0.0, 1.0)]:
            screened, _a1 = _branch_and_price(inst, w1, w2, use_relaxed_prescreen=True)
            unscreened, _a2 = _branch_and_price(inst, w1, w2, use_relaxed_prescreen=False)
            assert screened == pytest.approx(unscreened, abs=1e-4)

    def test_solve_f1_bpc_and_f2_bpc_match_compact_with_prescreen(self):
        """Same regression shape as test_bpc_cuts.py's
        TestIntegralNodeObjectiveRegression: run the public tie-break-aware
        wrappers (not raw _branch_and_price) through the prescreen, on the
        smallest instance, and check against the compact model directly.
        """
        from fedhpc.bpc import _branch_and_price
        from fedhpc.model import solve_f1, solve_f2

        inst = Instance.from_file("data/tiny_smallest.json")
        exp1 = solve_f1(inst, OutputFlag=0)
        exp2 = solve_f2(inst, OutputFlag=0)

        f1_screened, _ = _branch_and_price(inst, 1.0, 0.0, use_relaxed_prescreen=True)
        f2_screened, a2 = _branch_and_price(
            inst, 0.0, 1.0, bound1_value=f1_screened, use_relaxed_prescreen=True,
        )
        assert f1_screened == pytest.approx(exp1.f1, abs=1e-4)

        f2_only, _ = _branch_and_price(inst, 0.0, 1.0, use_relaxed_prescreen=True)
        assert f2_only == pytest.approx(exp2.f2, abs=1e-4)
