"""Pareto quality indicators for FED-HPC solution sets.

Two uses, kept apart on purpose:

* :func:`compare_fronts` scores a front against a *reference* front (the
  exact one, or the union of every method's points), normalised by the
  reference's ideal and nadir — the only setting in which indicators of
  different methods are comparable.
* :func:`pareto_metrics` describes one front. Without ``reference`` it is
  normalised by the front's own ideal/nadir, which says something about its
  shape but nothing about its quality: a poor front spanning a narrow range
  scores as well as the true one. Pass ``reference`` to compare methods.
"""
from __future__ import annotations

import numpy as np

from .model import Solution

_REF = np.array([1.1, 1.1])       # reference point for HV (normalised space)
_N_WEIGHTS = 101                    # weight-vector count for R2


def _hv2d(pts: np.ndarray, ref: np.ndarray) -> float:
    """2-D hypervolume via sweep-line (minimisation sense).

    Dominated points and points beyond ``ref`` add nothing: each strip is
    bounded by the best f2 seen so far, and only points inside the box count.
    """
    s = pts[(pts[:, 0] < ref[0]) & (pts[:, 1] < ref[1])]
    s = s[np.lexsort((s[:, 1], s[:, 0]))]
    hv, best_f2 = 0.0, ref[1]
    for i in range(len(s)):
        best_f2 = min(best_f2, s[i, 1])
        f1_next = ref[0] if i == len(s) - 1 else s[i + 1, 0]
        hv += (f1_next - s[i, 0]) * (ref[1] - best_f2)
    return hv


def _points(front) -> np.ndarray:
    """(f1, f2) rows of the solutions (or (f1, f2) pairs) that have both."""
    rows = [(s.f1, s.f2) if hasattr(s, "f1") else tuple(s) for s in front]
    return np.array([r for r in rows if r[0] is not None and r[1] is not None],
                    dtype=float).reshape(-1, 2)


def _nondominated_mask(pts: np.ndarray) -> np.ndarray:
    """True for the rows of ``pts`` no other row strictly dominates."""
    keep = np.ones(len(pts), dtype=bool)
    for i, p in enumerate(pts):
        keep[i] = not np.any(np.all(pts <= p, axis=1) & np.any(pts < p, axis=1))
    return keep


def _normaliser(ref_pts: np.ndarray, ideal=None, nadir=None):
    ideal = ref_pts.min(axis=0) if ideal is None else np.asarray(ideal, dtype=float)
    nadir = ref_pts.max(axis=0) if nadir is None else np.asarray(nadir, dtype=float)
    span = np.where(nadir - ideal > 1e-10, nadir - ideal, 1.0)
    return ideal, span


def compare_fronts(front, reference, *, ideal=None, nadir=None,
                   atol: float = 1e-6) -> dict:
    """Score ``front`` against ``reference`` (Solutions or (f1, f2) pairs).

    Both are normalised by ``ideal``/``nadir`` — by default the reference's
    own — so scores of different fronts against one reference are comparable.
    Dominated reference points are dropped first; ``front`` is kept as given
    (a dominated point counts against it in ``dominated``).

    Returns
    -------
    card        # points in ``front``
    exact       # front points equal (within ``atol``) to a reference point
    dominated   # front points strictly dominated by a reference point
    dominates   # front points strictly dominating a reference point (> 0 means
                the reference is not the true front)
    igd         mean distance reference → nearest front point (↓)
    igd_plus    IGD+ — distance counting only the objectives in which the front
                point is worse; weakly Pareto-compliant (↓)
    gd          mean distance front → nearest reference point (↓)
    eps_plus    additive ε: smallest shift making the front weakly dominate
                every reference point (↓, 0 = covers it)
    hvr         HV(front) / HV(reference), ref point (1.1, 1.1) (↑, 1 = matches)
    """
    fr = _points(front)
    rf = _points(reference)
    rf = rf[_nondominated_mask(rf)]
    if len(rf) == 0:
        raise ValueError("reference front is empty")
    lo, span = _normaliser(rf, ideal, nadir)
    f, r = (fr - lo) / span, (rf - lo) / span
    if len(f) == 0:
        inf = float("inf")
        return dict(card=0, exact=0, dominated=0, dominates=0, igd=inf,
                    igd_plus=inf, gd=inf, eps_plus=inf, hvr=0.0)
    diff = f[None, :, :] - r[:, None, :]            # [ref, front, obj]
    dist = np.linalg.norm(diff, axis=2)
    worse = np.linalg.norm(np.maximum(diff, 0.0), axis=2)
    dominated = sum(
        bool(np.any(np.all(rf <= p, axis=1) & np.any(rf < p, axis=1))) for p in fr)
    dominates = sum(
        bool(np.any(np.all(fr <= q, axis=1) & np.any(fr < q, axis=1))) for q in rf)
    hv_ref = _hv2d(r, _REF)
    return dict(
        card=len(fr),
        exact=int(sum(np.any(np.all(np.abs(rf - p) <= atol, axis=1)) for p in fr)),
        dominated=int(dominated),
        dominates=int(dominates),
        igd=float(dist.min(axis=1).mean()),
        igd_plus=float(worse.min(axis=1).mean()),
        gd=float(dist.min(axis=0).mean()),
        eps_plus=float(diff.max(axis=2).min(axis=1).max()),
        hvr=float(_hv2d(f, _REF) / hv_ref) if hv_ref > 0 else float("nan"),
    )


def pareto_metrics(solutions: list[Solution], reference=None) -> dict:
    """Compute Pareto quality indicators for a solution set.

    Dominated solutions are dropped first. Without ``reference`` every
    indicator is normalised with the front's own ideal/nadir: it describes
    the front's shape only and must not be used to compare methods (see the
    module docstring). With ``reference`` (Solutions or (f1, f2) pairs) the
    normalisation is the reference's, ``igd`` is measured against it and
    ``vs_reference`` holds :func:`compare_fronts`.

    Returns
    -------
    cardinality         int             — # valid non-dominated solutions
    hypervolume         float           — normalised HV (ref=(1.1,1.1)); ↑ better
    igd                 float|None      — IGD vs ``reference`` (None without one); ↓ better
    r2                  float           — R2 Tchebycheff (101 weights); ↓ better
    spread              float           — Δ uniformity metric (0=perfect); ↓ better
    coverage            float           — HV / ref-box area ∈ [0,1]; ↑ better
    knee_point          Solution|None   — max ⊥ distance from extreme-point line
    regret              dict
        min             float           — minimax Chebyshev distance to ideal
        max             float           — worst-case Chebyshev distance
        minimax_solution Solution|None  — solution achieving minimum regret
    vs_reference        dict            — only with ``reference``: compare_fronts
    """
    valid = [s for s in solutions if s.f1 is not None and s.f2 is not None]
    if valid:
        mask = _nondominated_mask(_points(valid))
        valid = [s for s, k in zip(valid, mask) if k]
    n = len(valid)
    if n == 0:
        return dict(
            cardinality=0, hypervolume=0.0, igd=None, r2=0.0,
            spread=0.0, coverage=0.0, knee_point=None,
            regret=dict(min=0.0, max=0.0, minimax_solution=None),
        )

    pts = _points(valid)
    ref_pts = pts if reference is None else _points(reference)
    lo, scale = _normaliser(ref_pts[_nondominated_mask(ref_pts)])
    norm = (pts - lo) / scale

    # ── Hypervolume ───────────────────────────────────────────────────────────
    hv = _hv2d(norm, _REF)

    # ── IGD: only meaningful against a real reference front ───────────────────
    vs_reference = None if reference is None else compare_fronts(valid, reference)
    igd = None if vs_reference is None else vs_reference["igd"]

    # ── R2 indicator (Tchebycheff scalarisation, _N_WEIGHTS weight vectors) ───
    lambdas = np.linspace(0.0, 1.0, _N_WEIGHTS)
    r2_total = 0.0
    for lam in lambdas:
        w = np.array([lam, 1.0 - lam])
        r2_total += np.max(w * norm, axis=1).min()
    r2 = float(r2_total / _N_WEIGHTS)

    # ── Spread (Δ uniformity metric) ──────────────────────────────────────────
    # Δ = Σ|d_i − d̄| / ((n−1) · d̄)   where d_i = dist between consecutive pts.
    # Front endpoints are the single-objective extremes, so d_f = d_l = 0.
    if n < 2:
        spread = 0.0
    else:
        sp_order = np.argsort(norm[:, 0])
        sp = norm[sp_order]
        d = np.linalg.norm(np.diff(sp, axis=0), axis=1)
        d_bar = d.mean()
        spread = 0.0 if d_bar < 1e-12 else float(np.abs(d - d_bar).sum() / ((n - 1) * d_bar))

    # ── Coverage (fraction of reference box dominated by the front) ───────────
    coverage = float(hv / (_REF[0] * _REF[1]))

    # ── Knee point (max perpendicular distance from line joining extremes) ─────
    if n == 1:
        knee_point: Solution | None = valid[0]
    else:
        ord2 = np.argsort(norm[:, 0])
        sp2 = norm[ord2]
        p1, p2 = sp2[0], sp2[-1]
        line = p2 - p1
        ln = float(np.linalg.norm(line))
        if ln < 1e-10:
            knee_point = valid[0]
        else:
            perp = [
                abs((p1[0] - pt[0]) * line[1] - (p1[1] - pt[1]) * line[0]) / ln
                for pt in sp2
            ]
            knee_point = valid[int(ord2[int(np.argmax(perp))])]

    # ── Regret (minimax Chebyshev distance to ideal in normalised space) ───────
    cheby = np.max(norm, axis=1)
    minimax_sol: Solution | None = valid[int(np.argmin(cheby))]
    regret = dict(
        min=float(cheby.min()),
        max=float(cheby.max()),
        minimax_solution=minimax_sol,
    )

    out = dict(
        cardinality=n,
        hypervolume=float(hv),
        igd=igd,
        r2=r2,
        spread=spread,
        coverage=coverage,
        knee_point=knee_point,
        regret=regret,
    )
    if vs_reference is not None:
        out["vs_reference"] = vs_reference
    return out


def metrics_to_serialisable(metrics: dict, ranked: list[Solution]) -> dict:
    """Convert a metrics dict to a JSON-serialisable form.

    Solution objects are replaced by their 1-based rank in *ranked*
    (sorted by f1 ascending, as produced by _print_pareto).
    """
    def _rank(sol: Solution | None) -> int | None:
        if sol is None:
            return None
        for i, s in enumerate(ranked):
            if s is sol:
                return i + 1
        return None

    return dict(
        cardinality=metrics["cardinality"],
        hypervolume=metrics["hypervolume"],
        igd=metrics["igd"],
        r2=metrics["r2"],
        spread=metrics["spread"],
        coverage=metrics["coverage"],
        knee_point_rank=_rank(metrics["knee_point"]),
        regret=dict(
            min=metrics["regret"]["min"],
            max=metrics["regret"]["max"],
            minimax_solution_rank=_rank(metrics["regret"]["minimax_solution"]),
        ),
        **({"vs_reference": metrics["vs_reference"]} if "vs_reference" in metrics else {}),
    )
