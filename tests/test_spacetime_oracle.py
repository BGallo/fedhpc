"""Brute-force oracle for the SpaceTimeFormulation dominance reductions.

Every feasible schedule of a tiny instance is enumerated directly — each job
on every (type, start) it may take, capacity checked slot by slot against
running jobs — with no flow network, no Gurobi and none of the reductions'
assumptions. The resulting set of (f1, f2) outcomes gives the exact lex
optima, weighted optima and Pareto front, which the reduced model must hit.
"""
import math
import random

import gurobipy as gp
import pytest
from gurobipy import GRB

from fedhpc import formulations
from fedhpc.data import Instance, InstanceType, Job, RunningJob
from fedhpc.formulations import LEX_F1, SpaceTimeFormulation
from fedhpc.model import solve_f1, solve_f2, solve_true_pareto_step

formulations.configure_env()

PLAIN = dict(collapse_unbounded=False, pareto_unbounded=False,
             rejection_deadlines=False, lex_deadlines=False)
FLOW_VARIANTS = [dict(), dict(normal_patterns=True), dict(compressed_flow=True)]


def _type(id, perf, deploy, cost_vm, capacity, fixed_cost=0.0):
    big = 1e9
    return InstanceType(id=id, kind="on-prem" if cost_vm == 0 else "cloud",
                        perf=perf, io_time=0.0, deploy=deploy, cost_io=0.0,
                        cost_vm=cost_vm, cost_stor=0.0, cpu=big, mem=big, stor=big,
                        capacity=capacity, fixed_cost=fixed_cost)


def _instance(seed: int, budget: float = 1e9, n_jobs: int = 4) -> Instance:
    """Two capacitated on-prem types, one on-prem type sized exactly at the
    unbounded threshold, and two or three cloud types with deliberate ties."""
    rng = random.Random(seed)
    fixed = rng.random() < 0.5
    tie = rng.random() < 0.5
    types = [
        _type(0, 1.0, 0.0, 0.0, 1),
        _type(1, 1.0, 0.0, 0.0, 2),
        # capacity - running = n_jobs: unbounded, but only just; slow
        _type(2, 0.5, 0.0, 0.0, n_jobs + 1),
        _type(3, 3.0, 0.0, 1.0, None),
        _type(4, 1.5, 0.0, 0.4, None, fixed_cost=1.5 if fixed else 0.0),
    ]
    if tie:  # same speed and cost as type 4, without its fixed cost
        types.append(_type(5, 1.5, 0.0, 0.4, None))
    jobs = [Job(id=j, arrival=round(rng.uniform(0, 1.5), 2),
                exec_time=float(rng.randint(1, 4)), io_volume=0.0, cpu=0.0,
                mem=0.0, stor=0.0, weight=float(rng.choice((0, 1, 1, 2, 3))))
            for j in range(n_jobs)]
    running = [RunningJob(id=100, type_id=1, end=rng.randint(1, 4)),
               RunningJob(id=101, type_id=2, end=rng.randint(1, 4))]
    inst = Instance(jobs=jobs, instance_types=types, horizon=11, budget=budget,
                    running_jobs=running)
    inst.build()
    return inst


def _enumerate(inst: Instance) -> dict[float, float]:
    """``{f1: min f2}`` over every feasible schedule within the budget."""
    H = inst.horizon
    cap = {m.id: m.capacity for m in inst.instance_types if m.capacity is not None}
    fixed = {m.id: m.fixed_cost for m in inst.instance_types}
    use = {m: [inst.occupied.get((m, t), 0) for t in range(H)] for m in cap}
    options = [
        [(m, t, t + inst.p_occ[j.id, m],
          j.weight * (t + inst.p_occ[j.id, m] - j.arrival), inst.c[j.id, m])
         for m in inst.F[j.id] for t in inst.T[j.id, m]]
        for j in inst.jobs
    ]
    best: dict[float, float] = {}
    used: list[int] = []

    def dfs(k: int, f1: float, f2: float) -> None:
        if k == len(options):
            total = f2 + sum(fixed[m] for m in set(used))
            if total <= inst.budget + 1e-9:
                key = round(f1, 9)
                best[key] = min(best.get(key, math.inf), total)
            return
        for m, t, end, df1, df2 in options[k]:
            if m in cap:
                slots = use[m]
                if any(slots[u] >= cap[m] for u in range(t, end)):
                    continue
                for u in range(t, end):
                    slots[u] += 1
            used.append(m)
            dfs(k + 1, f1 + df1, f2 + df2)
            used.pop()
            if m in cap:
                for u in range(t, end):
                    slots[u] -= 1

    dfs(0, 0.0, 0.0)
    return best


def _pareto(outcomes: dict[float, float]) -> list[tuple[float, float]]:
    front, best_f2 = [], math.inf
    for f1 in sorted(outcomes):
        if outcomes[f1] < best_f2 - 1e-9:
            front.append((f1, outcomes[f1]))
            best_f2 = outcomes[f1]
    return front


def _weighted_value(inst, fmt, a, b, f1_cap=None) -> float | None:
    mdl, v = fmt.build(inst, objective=(a, b))
    x = v["x"]
    if f1_cap is not None:
        mdl.addConstr(fmt.f1_expr(inst, x) <= f1_cap + 1e-9)
    mdl.setObjective(a * fmt.f1_expr(inst, x) + b * fmt.f2_expr(inst, x), GRB.MINIMIZE)
    mdl.setParam("MIPGap", 0)
    mdl.optimize()
    return mdl.ObjVal if mdl.Status == GRB.OPTIMAL else None


WEIGHTS = [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 4.0), (4.0, 1.0), (1.0, 20.0)]


@pytest.fixture(scope="module", params=range(12))
def case(request):
    inst = _instance(request.param)
    return inst, _enumerate(inst)


def test_oracle_matches_plain_model(case):
    """Sanity check of the oracle itself against the unreduced formulation."""
    inst, outcomes = case
    f1_star = min(outcomes)
    sol = solve_f1(inst, formulation=SpaceTimeFormulation(**PLAIN), MIPGap=0)
    assert (sol.f1, sol.f2) == pytest.approx((f1_star, outcomes[round(f1_star, 9)]))


@pytest.mark.parametrize("opts", FLOW_VARIANTS)
def test_lex_optima(case, opts):
    inst, outcomes = case
    f1_star = min(outcomes)
    sol = solve_f1(inst, formulation=SpaceTimeFormulation(**opts), MIPGap=0)
    assert (sol.f1, sol.f2) == pytest.approx((f1_star, outcomes[f1_star]))
    f2_star = min(outcomes.values())
    sol = solve_f2(inst, formulation=SpaceTimeFormulation(**opts), MIPGap=0)
    assert sol.f2 == pytest.approx(f2_star)
    assert sol.f1 == pytest.approx(min(f1 for f1, f2 in outcomes.items()
                                       if f2 <= f2_star + 1e-9))


@pytest.mark.parametrize("opts", FLOW_VARIANTS)
def test_weighted_optima(case, opts):
    inst, outcomes = case
    caps = [None, sorted(outcomes)[len(outcomes) // 3]]   # second one binds
    for a, b in WEIGHTS:
        for cap in caps:
            want = min(a * f1 + b * f2 for f1, f2 in outcomes.items()
                       if cap is None or f1 <= cap + 1e-9)
            got = _weighted_value(inst, SpaceTimeFormulation(**opts), a, b, cap)
            assert got == pytest.approx(want, abs=1e-7), (a, b, cap)


def test_pareto_front(case):
    inst, outcomes = case
    for f1, f2 in _pareto(outcomes):
        sol = solve_true_pareto_step(inst, f1, formulation=SpaceTimeFormulation(),
                                     MIPGap=0)
        assert (sol.f1, sol.f2) == pytest.approx((f1, f2))


def test_lex_under_f1_cap(case):
    """Lex-min (f1, f2) restricted to f1 <= cap, the pareto box-search shape."""
    inst, outcomes = case
    for cap in sorted(outcomes)[:3]:
        fmt = SpaceTimeFormulation()
        mdl, v = fmt.build(inst, objective=LEX_F1)
        f1e, f2e = fmt.f1_expr(inst, v["x"]), fmt.f2_expr(inst, v["x"])
        mdl.addConstr(f1e <= cap + 1e-9)
        mdl.setParam("MIPGap", 0)
        mdl.setObjective(f1e, GRB.MINIMIZE)
        mdl.optimize()
        mdl.addConstr(f1e <= mdl.ObjVal + 1e-6)
        mdl.setObjective(f2e, GRB.MINIMIZE)
        mdl.optimize()
        f1_star = min(outcomes)
        assert mdl.ObjVal == pytest.approx(outcomes[f1_star])


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("budget", [0.0, 2.0])
def test_other_budgets(seed, budget):
    """B = 0 (reductions on, costly options filtered) and a binding finite
    budget (reductions off): lex optimum must still match the oracle."""
    inst = _instance(seed, budget=budget)
    outcomes = _enumerate(inst)
    sol = solve_f1(inst, formulation=SpaceTimeFormulation(), MIPGap=0)
    if not outcomes:
        assert sol.status == "infeasible"
        return
    f1_star = min(outcomes)
    assert (sol.f1, sol.f2) == pytest.approx((f1_star, outcomes[f1_star]))
