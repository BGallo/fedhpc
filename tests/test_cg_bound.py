"""Column-generation / Lagrangian lower bound for Rm||TWCT+TMUC."""
import itertools
import random

import pytest

from fedhpc import _rmprice  # type: ignore[attr-defined]
from fedhpc import cg_bound, ilssp
from fedhpc import rm_benchmark as rm

from test_ilssp import _random_instance
from test_rm_benchmark import M2N10_1  # optimum 145220, root LP 142258.5


def _brute_subset(data, k, beta, H):
    """min over subsets S (load ≤ H) of c(k,S) − β(S), with the empty set = 0."""
    p, w, U, L = data
    n = len(w)
    best = 0.0
    for r in range(1, n + 1):
        for S in itertools.combinations(range(n), r):
            if sum(p[j][k] for j in S) > H:
                continue
            js = sorted(S, key=lambda j: (p[j][k] / w[j], j))
            t, c = 0, U[k]
            for j in js:
                t += p[j][k]
                c += w[j] * t
            c += L[k] * t
            best = min(best, c - sum(beta[j] for j in S))
    return best


@pytest.mark.parametrize("seed", range(3))
def test_pricing_dp_matches_enumeration(seed):
    data = _random_instance(9, 3, 40 + seed)
    rng = random.Random(seed)
    beta = [rng.uniform(0, 8000) for _ in range(9)]
    H = [150, 250, 10_000]
    mins, cols = _rmprice.Pricer(*data).price(beta, [0.0] * 3, H, 3)
    for k in range(3):
        assert mins[k] == pytest.approx(min(0.0, _brute_subset(data, k, beta, H[k])), abs=1e-6)
    for k, jobs, cost, rc in cols:
        assert sum(data[0][j][k] for j in jobs) <= H[k]
        assert rc == pytest.approx(cost - sum(beta[j] for j in jobs), abs=1e-6)
        assert rc < 0


def test_root_lp_matches_paper():
    data = rm.parse(M2N10_1)
    res = ilssp.solve(data, 1.0, target=145220)
    b = cg_bound.lp_bound(data, res.cost, res.columns, subgradient_iters=0, smoothing=0.0)
    assert b.lp_optimal
    assert b.lagrangian == pytest.approx(142258.5, abs=1e-4)
    assert b.lb == 142259 and not b.proven_optimal


@pytest.mark.parametrize("seed", range(4))
def test_bound_is_valid(seed):
    data = _random_instance(8, 3, 60 + seed, big_u=seed % 2 == 1)
    n, m = len(data[1]), len(data[2])
    opt = min(rm.evaluate(data, list(a)) for a in itertools.product(range(m), repeat=n))
    res = ilssp.solve(data, 1.0, target=opt, seed=seed)
    b = cg_bound.lp_bound(data, res.cost, res.columns, incumbent=res.assignment)
    assert b.lb <= opt
    assert b.subgradient_lb <= opt + 1e-6


@pytest.mark.parametrize("seed", range(3))
def test_enumeration_is_complete(seed):
    """enumerate() returns exactly the subsets within the threshold."""
    data = _random_instance(10, 2, 80 + seed)
    p, w, U, L = data
    rng = random.Random(seed)
    beta = [rng.uniform(0, 6000) for _ in range(10)]
    pr = _rmprice.Pricer(*data)
    for k, H in [(0, 200), (1, 10_000)]:
        vals = {}
        for r in range(1, 11):
            for S in itertools.combinations(range(10), r):
                if sum(p[j][k] for j in S) <= H:
                    vals[S] = pr.cost(k, list(S)) - sum(beta[j] for j in S)
        thr = sorted(vals.values())[min(40, len(vals) - 1)]
        cols, complete = pr.enumerate(k, beta, H, thr)
        assert complete
        got = {tuple(js): rc for js, _, rc in cols}
        want = {S for S, v in vals.items() if v <= thr + 1e-6}
        assert set(got) == want
        for S, rc in got.items():
            assert rc == pytest.approx(vals[S], abs=1e-6)
