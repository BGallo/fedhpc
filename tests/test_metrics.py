"""Pareto indicators (metrics.py): reference-based comparison and the
self-normalised description of a single front."""
import pytest

from fedhpc.metrics import compare_fronts, pareto_metrics
from fedhpc.model import Solution

TRUE = [(10.0, 100.0), (15.0, 50.0), (30.0, 0.0)]


def _sols(pts):
    return [Solution("x", None, a, b, {}, {}) for a, b in pts]


def test_identical_front_scores_perfectly():
    m = compare_fronts(TRUE, TRUE)
    assert m["exact"] == 3 and m["dominated"] == m["dominates"] == 0
    assert m["igd"] == m["igd_plus"] == m["gd"] == m["eps_plus"] == 0.0
    assert m["hvr"] == pytest.approx(1.0)


def test_dominated_front_is_penalised():
    worse = [(12.0, 100.0), (18.0, 60.0), (30.0, 5.0)]
    m = compare_fronts(worse, TRUE)
    assert m["exact"] == 0 and m["dominated"] == 3 and m["dominates"] == 0
    assert m["igd_plus"] > 0 and m["eps_plus"] > 0 and m["hvr"] < 1


def test_point_beating_the_reference_is_flagged():
    assert compare_fronts([(10.0, 40.0)], TRUE)["dominates"] >= 1


def test_self_normalised_indicators_cannot_rank_fronts():
    """Two 2-point fronts, one on the true front and one far worse, are
    indistinguishable when each is normalised by its own extremes; against
    the reference the bad one is clearly worse."""
    good = [(10.0, 100.0), (30.0, 0.0)]
    bad = [(100.0, 1000.0), (300.0, 500.0)]
    g, b = pareto_metrics(_sols(good)), pareto_metrics(_sols(bad))
    assert g["igd"] is None and "vs_reference" not in g
    assert g["hypervolume"] == pytest.approx(b["hypervolume"])
    g_ref = pareto_metrics(_sols(good), reference=TRUE)
    b_ref = pareto_metrics(_sols(bad), reference=TRUE)
    assert b_ref["hypervolume"] < g_ref["hypervolume"]
    assert b_ref["vs_reference"]["hvr"] < g_ref["vs_reference"]["hvr"] <= 1.0
    assert b_ref["igd"] > g_ref["igd"] > 0


def test_dominated_solutions_are_dropped():
    m = pareto_metrics(_sols(TRUE + [(20.0, 60.0)]))
    assert m["cardinality"] == 3


def test_hypervolume_ignores_dominated_points():
    with_dom = compare_fronts(TRUE + [(20.0, 60.0)], TRUE)
    assert with_dom["hvr"] == pytest.approx(1.0) and with_dom["dominated"] == 1
