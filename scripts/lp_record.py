"""Record the LP operations of lpfirst's column-generation masters, for
replaying the identical LP sequence into any solver (scripts/lp_replay.py).

Every fedlp master the run creates is logged: its rows, each column batch,
cost changes, crash hints and each solve's objective. The masters run on
fedlp (``LPFirstOptions(lp_backend="fedlp")``); the objectives recorded are
fedlp's, and a replay checks every solver against them.

Usage:
  uv run python scripts/lp_record.py pos_congestion_known_runtime_offline_10min.json
  uv run python scripts/lp_record.py fedhpc_known_runtime_offline_10min.json --runs bench

--runs anchor   the first anchor only: lexmin (f2, f1)
--runs bench    the five bench solves: lexmin (f1, f2), (f2, f1) and three
                epsilon-constrained solves at 25/50/75 % of the f1 range
Output: results/lp_replay/<instance stem>_<runs>.pkl (or --out).
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from fedhpc import formulations, lp_backend, lpfirst
from fedhpc.data import Instance

ROOT = Path(__file__).resolve().parent.parent


def record(inst: Instance, runs: str) -> list:
    """Run lpfirst on inst with fedlp masters; return one log per master."""
    logs: list = []
    B = lp_backend.FedlpBackend
    saved = (B._ensure, B.add_cols, B.change_costs, B.crash_hint, B.solve_lp)
    o_ensure, o_add, o_cost, o_hint, o_solve = saved

    def _ensure(self):
        if self.lp is None:
            self._log = [("init", list(self._sense), list(self._rhs))]
            logs.append(self._log)
        o_ensure(self)

    def add_cols(self, cost, ub, integer, starts, index, value):
        if len(cost):
            self._ensure()
            self._log.append(("add", np.array(cost, float), np.array(ub, float),
                              np.array(starts, np.int32), np.array(index, np.int32),
                              np.array(value, float)))
        o_add(self, cost, ub, integer, starts, index, value)

    def change_costs(self, idx, cost):
        self._ensure()
        self._log.append(("costs", np.array(idx, np.int64), np.array(cost, float)))
        o_cost(self, idx, cost)

    def crash_hint(self, cols):
        self._ensure()
        self._log.append(("hint", [int(c) for c in cols]))
        o_hint(self, cols)

    def solve_lp(self):
        r = o_solve(self)
        self._log.append(("solve", self.lp.objective if r == "optimal" else None))
        return r

    B._ensure, B.add_cols, B.change_costs, B.crash_hint, B.solve_lp = (
        _ensure, add_cols, change_costs, crash_hint, solve_lp)
    try:
        lb, _ = lpfirst.make_lpfirst_lexmin_bounded(
            inst, options=lpfirst.LPFirstOptions(lp_backend="fedlp"))
        if runs == "anchor":
            lb("f2", "f1")
        else:
            a = lb("f1", "f2")
            b = lb("f2", "f1")
            for q in (0.25, 0.5, 0.75):
                lb("f2", "f1", "f1", a.f1 + q * (b.f1 - a.f1))
    finally:
        B._ensure, B.add_cols, B.change_costs, B.crash_hint, B.solve_lp = saved
    return logs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("instance", help="instance file name under data/, or a path")
    ap.add_argument("--runs", choices=["anchor", "bench"], default="anchor")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    path = Path(args.instance)
    if not path.exists():
        path = ROOT / "data" / args.instance
    formulations.configure_env()
    logs = record(Instance.from_file(path), args.runs)
    out = args.out or ROOT / "results" / "lp_replay" / f"{path.stem}_{args.runs}.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(logs, f)
    print(f"{out}: {len(logs)} masters, solves {[sum(op[0] == 'solve' for op in L) for L in logs]}, "
          f"rows {[len(L[0][1]) for L in logs]}")


if __name__ == "__main__":
    main()
