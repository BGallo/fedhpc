"""ILS-SP engine (C++ ``_ilssp``) and driver on Rm||TWCT+TMUC."""
import itertools
import random

import pytest

from fedhpc import _ilssp  # type: ignore[attr-defined]
from fedhpc import ilssp
from fedhpc import rm_benchmark as rm

from test_rm_benchmark import M2N10_1  # published optimum 145220


def _random_instance(n, m, seed, big_u=False):
    rng = random.Random(seed)
    p = [[rng.randint(1, 100) for _ in range(m)] for _ in range(n)]
    w = [rng.randint(1, 100) for _ in range(n)]
    U = [rng.randint(5000, 60000) if big_u else rng.randint(100, 2000) for _ in range(m)]
    L = [rng.randint(1, 60) for _ in range(m)]
    return p, w, U, L


def _brute(data):
    n, m = len(data[1]), len(data[2])
    return min(rm.evaluate(data, list(a)) for a in itertools.product(range(m), repeat=n))


def test_evaluate_matches_python():
    data = _random_instance(30, 4, 1)
    eng = _ilssp.IlsSp(*data)
    rng = random.Random(0)
    for _ in range(50):
        a = [rng.randrange(4) for _ in range(30)]
        assert eng.evaluate(a) == rm.evaluate(data, a)


@pytest.mark.parametrize("big_u", [False, True])
def test_incremental_costs_stay_exact(big_u):
    data = _random_instance(60, 5, 2, big_u)
    eng = _ilssp.IlsSp(*data, seed=3, p_machine_kick=0.5)
    eng.construct()
    assert eng.check()
    for _ in range(5):
        eng.run(max_iters=20, time_limit=10.0)
        assert eng.check()
    assert eng.evaluate(eng.best_assignment()) == eng.best_cost


def test_published_optimum():
    data = rm.parse(M2N10_1)
    res = ilssp.solve(data, time_limit=2.0, target=145220)
    assert res.cost == 145220
    assert rm.evaluate(data, res.assignment) == res.cost


@pytest.mark.parametrize("seed", range(4))
def test_brute_force_small(seed):
    data = _random_instance(8, 3, 10 + seed, big_u=seed % 2 == 1)
    opt = _brute(data)
    res = ilssp.solve(data, time_limit=2.0, target=opt, seed=seed)
    assert res.cost == opt


def test_set_solution_polishes():
    data = _random_instance(40, 4, 5)
    eng = _ilssp.IlsSp(*data)
    bad = [0] * 40
    c = eng.set_solution(bad)
    assert c <= eng.evaluate(bad)
    assert eng.best_cost == c


def test_sp_driver_runs():
    data = _random_instance(80, 5, 7, big_u=True)
    res = ilssp.solve(data, time_limit=2.0, sp_interval=0.3)
    assert res.sp_calls > 0 and res.pool_size > 0
    assert rm.evaluate(data, res.assignment) == res.cost
