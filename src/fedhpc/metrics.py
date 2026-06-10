"""Pareto quality indicators for FED-HPC solution sets."""
from __future__ import annotations

import numpy as np

from .model import Solution

_REF = np.array([1.1, 1.1])       # reference point for HV (normalised space)
_N_REF_FRONT = 200                  # uniform diagonal reference front size for IGD
_N_WEIGHTS = 101                    # weight-vector count for R2


def _hv2d(pts: np.ndarray, ref: np.ndarray) -> float:
    """2-D hypervolume via sweep-line (minimisation sense)."""
    order = np.argsort(pts[:, 0])
    s = pts[order]
    hv = 0.0
    for i in range(len(s)):
        f1_next = ref[0] if i == len(s) - 1 else s[i + 1, 0]
        hv += (f1_next - s[i, 0]) * (ref[1] - s[i, 1])
    return max(hv, 0.0)


def pareto_metrics(solutions: list[Solution]) -> dict:
    """Compute Pareto quality indicators for a solution set.

    All indicators are normalised with the front's own ideal/nadir so they
    are dimensionless and comparable across instances.

    Returns
    -------
    cardinality         int             — # valid non-dominated solutions
    hypervolume         float           — normalised HV (ref=(1.1,1.1)); ↑ better
    igd                 float           — IGD vs uniform diagonal reference; ↓ better
    r2                  float           — R2 Tchebycheff (101 weights); ↓ better
    spread              float           — Δ uniformity metric (0=perfect); ↓ better
    coverage            float           — HV / ref-box area ∈ [0,1]; ↑ better
    knee_point          Solution|None   — max ⊥ distance from extreme-point line
    regret              dict
        min             float           — minimax Chebyshev distance to ideal
        max             float           — worst-case Chebyshev distance
        minimax_solution Solution|None  — solution achieving minimum regret
    """
    valid = [s for s in solutions if s.f1 is not None and s.f2 is not None]
    n = len(valid)
    if n == 0:
        return dict(
            cardinality=0, hypervolume=0.0, igd=0.0, r2=0.0,
            spread=0.0, coverage=0.0, knee_point=None,
            regret=dict(min=0.0, max=0.0, minimax_solution=None),
        )

    pts = np.array([(s.f1, s.f2) for s in valid], dtype=float)

    ideal = pts.min(axis=0)
    nadir = pts.max(axis=0)
    scale = np.where(nadir - ideal > 1e-10, nadir - ideal, 1.0)
    norm = (pts - ideal) / scale

    # ── Hypervolume ───────────────────────────────────────────────────────────
    hv = _hv2d(norm, _REF)

    # ── IGD vs uniform diagonal reference front ───────────────────────────────
    # Reference: _N_REF_FRONT uniformly spaced points on (0,1)→(1,0).
    t_vals = np.linspace(0.0, 1.0, _N_REF_FRONT)
    ref_front = np.column_stack([t_vals, 1.0 - t_vals])
    igd = float(np.mean([np.linalg.norm(norm - r, axis=1).min() for r in ref_front]))

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

    return dict(
        cardinality=n,
        hypervolume=float(hv),
        igd=igd,
        r2=r2,
        spread=spread,
        coverage=coverage,
        knee_point=knee_point,
        regret=regret,
    )


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
    )
