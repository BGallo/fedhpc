"""Phase 5 correctness tests: rank-1 subset-row (SR-1) cuts.

Covers the cut mechanism directly (separation, pricing's dual-trigger
handling) as well as the two real bugs found while validating it — see
bpc.py's module docstring for the full incident writeups:

1. A pruning bug, *not actually caused by cuts* but only exposed once the
   root's accumulated column pool (rather than a fresh heuristic seed) was
   reused to seed the tree: an "integral" node's objective was taken
   straight from ``mdl.ObjVal`` (the RMP's raw LP objective), which can
   diverge from what the *clean* extracted assignment actually costs (the
   same residual-mass phenomenon as Phase 4's bound-row bug, this time
   showing up in the value used for B&B pruning) — a falsely-optimistic
   value for one integral node silently pruned away the true optimum
   elsewhere in the tree.
2. Two jobs-can-still-help-despite-looking-unattractive-alone bugs in the
   cost1-irrelevant pricing prefilter (same root cause as same_pairs' own
   version of this bug in Phase 3, now also needed for cut-participating
   jobs).
"""
from __future__ import annotations

import pytest

from conftest import trap_small
from fedhpc.data import Instance
from fedhpc.formulations import configure_env
from fedhpc.model import solve_f1, solve_f2

configure_env()


class _FakeVar:
    def __init__(self, x: float) -> None:
        self.X = x


class _FakeState:
    """Minimal stand-in for _RmpState exposing only what _separate_sr_cuts
    reads (columns + vars) — avoids needing a real Gurobi RMP for a pure
    unit test of the separation logic.
    """
    def __init__(self, columns, lambdas):
        self.columns = columns
        self.vars = [_FakeVar(x) for x in lambdas]


class TestSeparateSrCuts:
    def test_finds_violated_triple(self):
        from fedhpc.bpc import Column, _separate_sr_cuts

        # Classic SR-1 violation shape: 3 columns, each pairwise-covering 2
        # of {0,1,2}, each with lambda=0.6 -> sum over count>=2 = 1.8 > 1.
        state = _FakeState(
            columns=[
                Column(type_id=0, jobs=((0, 0), (1, 5)), cost1=0.0, cost2=0.0),
                Column(type_id=0, jobs=((0, 0), (2, 5)), cost1=0.0, cost2=0.0),
                Column(type_id=0, jobs=((1, 0), (2, 5)), cost1=0.0, cost2=0.0),
            ],
            lambdas=[0.6, 0.6, 0.6],
        )
        cuts = _separate_sr_cuts(state, tol=1e-6, max_new=10)
        assert frozenset({0, 1, 2}) in cuts

    def test_no_violation_when_satisfied(self):
        from fedhpc.bpc import Column, _separate_sr_cuts

        state = _FakeState(
            columns=[
                Column(type_id=0, jobs=((0, 0), (1, 5)), cost1=0.0, cost2=0.0),
            ],
            lambdas=[0.5],
        )
        cuts = _separate_sr_cuts(state, tol=1e-6, max_new=10)
        assert cuts == []

    def test_respects_max_new(self):
        from fedhpc.bpc import Column, _separate_sr_cuts

        # 4 jobs, every pair co-occurring heavily -> multiple candidate
        # triples violated; max_new must cap the returned list.
        cols = []
        lambdas = []
        jobs = [0, 1, 2, 3]
        for i in range(len(jobs)):
            for j in range(i + 1, len(jobs)):
                cols.append(Column(type_id=0, jobs=((jobs[i], 0), (jobs[j], 5)),
                                    cost1=0.0, cost2=0.0))
                lambdas.append(0.6)
        state = _FakeState(cols, lambdas)
        cuts = _separate_sr_cuts(state, tol=1e-6, max_new=2)
        assert len(cuts) <= 2


class TestPriceTypeCutDual:
    def test_cut_dual_applied_once_at_trigger(self):
        """A column with 2+ of a cut's triple should have the cut's dual
        subtracted from its reduced cost exactly once — verified by
        comparing against the same scenario with no cut active.
        """
        from fedhpc.data import InstanceType, Job
        from fedhpc.bpc import _build_arcs_by_type, _price_type

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
        # each job individually attractive enough that the best path uses
        # all 3 (0,1,2) absent any cut.
        sigma = {0: 50.0, 1: 50.0, 2: 50.0}

        no_cut = _price_type(
            inst, 0, arcs, weight1=0.0, weight2=1.0, sigma=sigma, nu1=None, nu2=None,
            rho_m=0.0, theta={}, arrival_of=arrival_of, weight_of=weight_of, tol=1e-7,
        )
        assert no_cut is not None
        rc_no_cut, col_no_cut = no_cut
        used_no_cut = {jid for jid, _t in col_no_cut.jobs}
        assert used_no_cut == {0, 1, 2}, f"expected all 3 jobs without a cut, got {used_no_cut}"

        cut_dual = 5.0
        with_cut = _price_type(
            inst, 0, arcs, weight1=0.0, weight2=1.0, sigma=sigma, nu1=None, nu2=None,
            rho_m=0.0, theta={}, arrival_of=arrival_of, weight_of=weight_of, tol=1e-7,
            cut_duals={frozenset({0, 1, 2}): cut_dual},
        )
        assert with_cut is not None
        rc_with_cut, _col_with_cut = with_cut
        # the cut credit applies exactly once (triggered when the 2nd of the
        # 3 jobs joins), so the reduced cost should differ by exactly
        # cut_dual, not 0 (never triggered) or 2*cut_dual (double-counted).
        assert rc_with_cut == pytest.approx(rc_no_cut - cut_dual, abs=1e-6)


class TestCutsEndToEnd:
    """Cuts must never change the optimal weighted-sum objective — only how
    fast/robustly it's found. use_cuts=True and use_cuts=False must agree on
    ``weight1*f1 + weight2*f2`` exactly.

    Deliberately *not* comparing f1/f2 (or the assignment) individually:
    whenever one weight is 0, that objective is entirely unconstrained by
    the solve, so ties in the optimized objective can legitimately resolve
    to different assignments — and hence a different value of the *other*,
    unweighted objective — between cuts on/off, same as between any two runs
    that explore the B&B tree in a different order. (This is exactly why
    ``solve_f1_bpc``/``solve_f2_bpc`` add their own explicit tie-break bound
    row on the other objective — raw ``_branch_and_price`` alone makes no
    such guarantee, cuts or not.) An earlier version of this test asserted
    f1 *and* f2 matched and failed on ``weighted_fixed_cost_inst`` at
    ``w1=1,w2=0`` — both runs found f1=32.0 (correct, tied), just via
    different alternate-optimal assignments with different f2 (47 vs 39) —
    a test bug, not a cuts bug.
    """

    def test_matches_uncut_on_known_pareto_inst(self, known_pareto_inst):
        from fedhpc.bpc import _branch_and_price

        for w1, w2 in [(1.0, 0.0), (0.0, 1.0), (0.5, 0.5)]:
            cut, a_cut = _branch_and_price(known_pareto_inst, w1, w2, use_cuts=True)
            nocut, a_nocut = _branch_and_price(known_pareto_inst, w1, w2, use_cuts=False)
            assert a_cut is not None and a_nocut is not None
            assert cut == pytest.approx(nocut, abs=1e-4)

    def test_matches_uncut_with_job_weights_and_fixed_cost(self, weighted_fixed_cost_inst):
        from fedhpc.bpc import _branch_and_price

        inst = weighted_fixed_cost_inst
        for w1, w2 in [(1.0, 0.0), (0.0, 1.0)]:
            cut, a_cut = _branch_and_price(inst, w1, w2, use_cuts=True)
            nocut, a_nocut = _branch_and_price(inst, w1, w2, use_cuts=False)
            assert a_cut is not None and a_nocut is not None
            assert cut == pytest.approx(nocut, abs=1e-4)


class TestIntegralNodeObjectiveRegression:
    """Regression test for the pruning bug described in this module's
    docstring: an integral node's reported objective must match what its
    *own extracted assignment* truly costs, not a possibly-optimistic raw
    mdl.ObjVal — otherwise it can falsely look unbeatable and prune away the
    genuine optimum found later in the search. A solve_f1_bpc tie-break
    phase first exposed this; both lex directions are checked on trap_small.
    """

    def test_solve_f1_bpc_tie_break_finds_true_minimum(self):
        from fedhpc.bpc import solve_f1_bpc

        inst = trap_small()
        expected = solve_f1(inst, OutputFlag=0)
        got = solve_f1_bpc(inst)
        assert got.f1 == pytest.approx(expected.f1, abs=1e-4)
        assert got.f2 == pytest.approx(expected.f2, abs=1e-4)

    def test_solve_f2_bpc_tie_break_finds_true_minimum(self):
        from fedhpc.bpc import solve_f2_bpc

        inst = trap_small()
        expected = solve_f2(inst, OutputFlag=0)
        got = solve_f2_bpc(inst)
        assert got.f1 == pytest.approx(expected.f1, abs=1e-4)
        assert got.f2 == pytest.approx(expected.f2, abs=1e-4)
