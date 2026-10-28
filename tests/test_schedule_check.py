"""Runtime schedule validation (Instance.schedule_violations / check_schedule)
and where solvers apply it, plus hybrid_frontier's certified points."""
import pytest
from gurobipy import GRB

from fedhpc.data import ScheduleError
from fedhpc.formulations import SpaceTimeFormulation
from fedhpc.model import _extract, solve_f1
from fedhpc.moea import _to_solutions
from fedhpc.pareto import hybrid_frontier, true_pareto_frontier

from conftest import trap_small


@pytest.fixture(scope="module")
def inst():
    return trap_small()


@pytest.fixture(scope="module")
def optimal(inst):
    return solve_f1(inst).assignment


def _fat(inst, fat):
    """Job ids that do (fat) or do not fit on on-prem type 0."""
    return [j.id for j in inst.jobs if (0 not in inst.F[j.id]) == fat]


class TestScheduleViolations:
    def test_solver_schedule_is_feasible(self, inst, optimal):
        assert inst.schedule_violations(optimal) == []
        inst.check_schedule(optimal)

    def test_unassigned_job(self, inst, optimal):
        partial = dict(optimal)
        dropped = partial.popitem()[0]
        assert any(f"unassigned jobs [{dropped}]" in v
                   for v in inst.schedule_violations(partial))
        assert inst.schedule_violations(partial, complete=False) == []

    def test_unknown_job(self, inst, optimal):
        bad = {**optimal, 999: optimal[inst.jobs[0].id]}
        assert any("unknown job 999" in v for v in inst.schedule_violations(bad))

    def test_type_outside_F(self, inst, optimal):
        jid = _fat(inst, True)[0]
        bad = {**optimal, jid: (0, inst.T[jid, 1].start)}
        assert any("not in F_j" in v for v in inst.schedule_violations(bad))

    @pytest.mark.parametrize("where", ["before_release", "past_horizon"])
    def test_start_outside_T(self, inst, optimal, where):
        jid = max(inst.jobs, key=lambda j: j.arrival).id
        T = inst.T[jid, 2]
        t = T.start - 1 if where == "before_release" else T.stop
        bad = {**optimal, jid: (2, t)}
        assert any("outside T_jm" in v for v in inst.schedule_violations(bad))

    def test_capacity_overflow(self, inst, optimal):
        # three thin jobs at once on type 0 (capacity 2), after the running job
        bad = dict(optimal)
        for jid in _fat(inst, False)[:3]:
            bad[jid] = (0, 20)
        assert any("capacity 2" in v for v in inst.schedule_violations(bad))

    def test_running_job_occupancy_counts(self, inst, optimal):
        # type 1 has capacity 1 and a running job until slot 4
        jid = min((j for j in inst.jobs if 1 in inst.F[j.id]),
                  key=lambda j: inst.T[j.id, 1].start).id
        t = inst.T[jid, 1].start
        assert t < 4
        bad = {k: v for k, v in optimal.items() if v[0] != 1}
        bad[jid] = (1, t)
        assert any("type 1 uses 2 > capacity 1" in v
                   for v in inst.schedule_violations(bad))
        bad[jid] = (1, 4)   # starts as the running job ends
        assert not any("capacity" in v for v in inst.schedule_violations(bad))

    def test_check_schedule_raises(self, inst, optimal):
        partial = dict(optimal)
        partial.popitem()
        with pytest.raises(ScheduleError, match="unassigned"):
            inst.check_schedule(partial)


class TestSolverOutputsAreChecked:
    def test_extract_rejects_a_job_started_twice(self, inst):
        """cover_ge (Σx >= 1) lets a job start twice; _extract must refuse it."""
        fmt = SpaceTimeFormulation(cover_ge=True, collapse_unbounded=False,
                                   pareto_unbounded=False, rejection_deadlines=False,
                                   lex_deadlines=False)
        mdl, vars_ = fmt.build(inst)
        x = vars_["x"]
        jid = inst.jobs[0].id
        starts = [k for k in x if k[0] == jid and k[1] == 2][:2]
        for k in starts:
            x[k].LB = 1.0
        mdl.setObjective(fmt.f1_expr(inst, x), GRB.MINIMIZE)
        mdl.optimize()
        assert mdl.Status == GRB.OPTIMAL
        with pytest.raises(ScheduleError, match="started more than once"):
            _extract(inst, mdl, vars_)

    def test_ea_output_is_checked(self, inst):
        bad = [(2, inst.T[j.id, 2].start) for j in inst.jobs]
        jid = inst.jobs[0].id
        bad[0] = (2, inst.T[jid, 2].stop)          # past the horizon
        with pytest.raises(ScheduleError, match="outside T_jm"):
            _to_solutions(inst, [(bad, 0.0, 0.0)])


class TestHybridFrontier:
    def test_points_are_certified_pareto_points(self, inst):
        true_front = {(round(s.f1, 6), round(s.f2, 6))
                      for s in true_pareto_frontier(inst)}
        front = hybrid_frontier(inst, pop_size=40, n_gen=30, seed=1)
        assert front
        for s in front:
            assert s.status == "optimal"
            assert (round(s.f1, 6), round(s.f2, 6)) in true_front
            inst.check_schedule(s.assignment)

    def test_loose_mip_gap_never_mislabels(self, inst):
        """An explicit MIPGap stops Gurobi early, but "optimal" still comes
        only from the certificate (the dual bound within one step), so every
        such point is on the true front."""
        true_front = {(round(s.f1, 6), round(s.f2, 6))
                      for s in true_pareto_frontier(inst)}
        front = hybrid_frontier(inst, pop_size=40, n_gen=30, seed=1, mip_gap=0.5)
        assert front
        for s in front:
            assert s.status in ("optimal", "feasible")
            if s.status == "optimal":
                assert (round(s.f1, 6), round(s.f2, 6)) in true_front
