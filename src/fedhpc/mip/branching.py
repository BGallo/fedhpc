"""Variable selection: reliability pseudocost branching (Achterberg, Koch &
Martin 2005).

A pseudocost is the objective gain per unit of change seen when a variable
was branched on, kept per direction. A candidate x_j = v (fractional part f)
scores

    score = max(f psi_j^-, eps) * max((1 - f) psi_j^+, eps)       (product rule)

A variable branched on fewer than ``reliability`` times in a direction is
unreliable there: the best few such candidates (by the pseudocost estimate,
uninitialised directions taking the average) are strong-branched - both
children solved with the dual simplex under an iteration limit - and their
gains enter the pseudocosts. A strong branch that finds a child infeasible
tells the node to tighten that variable's bound instead of branching.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lp import INFEASIBLE, OPTIMAL, LpRelaxation


@dataclass
class BranchDecision:
    var: int
    value: float
    fix_down: bool = False     # strong branching proved the down child infeasible
    fix_up: bool = False
    down_bound: float | None = None
    up_bound: float | None = None


class Pseudocosts:
    def __init__(self, n: int, reliability: int = 4):
        self.sum = np.zeros((2, n))
        self.cnt = np.zeros((2, n))
        self.reliability = reliability

    def update(self, j: int, up: bool, gain: float, change: float) -> None:
        if change <= 1e-9 or not np.isfinite(gain):
            return
        d = int(up)
        self.sum[d, j] += max(gain, 0.0) / change
        self.cnt[d, j] += 1

    def mean(self, d: int) -> float:
        tot = self.cnt[d].sum()
        return float(self.sum[d].sum() / tot) if tot > 0 else 1.0

    def value(self, d: int, cols: np.ndarray) -> np.ndarray:
        c = self.cnt[d, cols]
        return np.where(c > 0, self.sum[d, cols] / np.maximum(c, 1), self.mean(d))

    def score(self, cols: np.ndarray, f: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        down = f * self.value(0, cols)
        up = (1 - f) * self.value(1, cols)
        return np.maximum(down, eps) * np.maximum(up, eps)

    def unreliable(self, cols: np.ndarray) -> np.ndarray:
        return np.minimum(self.cnt[0, cols], self.cnt[1, cols]) < self.reliability


def select_branch(pc: Pseudocosts, lp: LpRelaxation, x: np.ndarray, cands: np.ndarray,
                  lb: np.ndarray, ub: np.ndarray, obj: float, cutoff: float,
                  max_strong: int = 8, iteration_limit: int = 200, lookahead: int = 4) -> BranchDecision:
    f = x[cands] - np.floor(x[cands])
    score = pc.score(cands, f)
    order = np.argsort(-score, kind="stable")
    unrel = pc.unreliable(cands)
    to_test = [k for k in order if unrel[k]][:max_strong] if max_strong > 0 else []
    if not to_test:
        k = int(order[0])
        return BranchDecision(int(cands[k]), float(x[cands[k]]))
    basis = lp.get_state()
    best_k, best_s, no_better = int(order[0]), -1.0, 0
    best_bounds = (None, None)
    for k in to_test:
        j = int(cands[k])
        v = float(x[j])
        gains, exact = [], []
        for up in (False, True):
            tl, tu = lb.copy(), ub.copy()
            if up:
                tl[j] = np.ceil(v)
            else:
                tu[j] = np.floor(v)
            lp.set_bounds(tl, tu)
            st = lp.solve(iteration_limit)
            if st == INFEASIBLE:
                g, ex = np.inf, True
            elif st == OPTIMAL:
                g, ex = max(lp.objective - obj, 0.0), True
            else:
                # an iteration limit: the objective guides scores and
                # pseudocosts only (it is no proven bound)
                g, ex = max(lp.objective - obj, 0.0) if np.isfinite(lp.objective) else 0.0, False
            gains.append(g)
            exact.append(ex)
            lp.set_bounds(lb, ub)
            lp.set_basis(basis)
        fj = v - np.floor(v)
        if np.isfinite(gains[0]):
            pc.update(j, False, gains[0], fj)
        if np.isfinite(gains[1]):
            pc.update(j, True, gains[1], 1 - fj)
        dead = [exact[d] and (not np.isfinite(gains[d]) or obj + gains[d] >= cutoff) for d in (0, 1)]
        if dead[0] or dead[1]:
            return BranchDecision(j, v, fix_down=dead[0], fix_up=dead[1])
        gains = [g if np.isfinite(g) else 0.0 for g in gains]
        s = max(gains[0], 1e-6) * max(gains[1], 1e-6)
        if s > best_s:
            best_s, best_k, no_better = s, k, 0
            best_bounds = (obj + gains[0], obj + gains[1])
        else:
            no_better += 1
            if no_better >= lookahead:
                break
    j = int(cands[best_k])
    return BranchDecision(j, float(x[j]), down_bound=best_bounds[0], up_bound=best_bounds[1])
