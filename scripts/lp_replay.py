"""Replay a recorded lpfirst LP sequence (scripts/lp_record.py) into fedlp,
HiGHS or Gurobi: the same rows, column batches, cost changes and solves,
so solvers and fedlp options are compared on identical LPs.

Every solve's objective is checked against the recording (relative 1e-7);
the summary line counts mismatches as "bad".

Usage:
  uv run python scripts/lp_replay.py results/lp_replay/X_anchor.pkl fedlp [opt=value ...]
  uv run python scripts/lp_replay.py X.pkl highs [--strategy 4]    # 1 dual, 4 primal
  uv run python scripts/lp_replay.py X.pkl gurobi [--method 0]     # 0 primal, 1 dual
Options:
  -v          one line per solve (iterations, time, fedlp's dual counters)
  --timers    fedlp's phase timers and dual-row sizes (a -DFEDLP_PROFILE build)
  FEDLP_DIR   environment variable: import _fedlp from that directory (a
              standalone build) instead of fedhpc's
Time is the solver's own: fedlp's stats()["seconds"], Gurobi's Runtime,
HiGHS's run() wall time. fedlp's crash hints are replayed; the others
ignore them.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np


def parse_opt(lp, kv: str) -> None:
    k, v = kv.split("=", 1)
    cur = getattr(lp.options, k)
    if isinstance(cur, bool):
        val = v.lower() in ("1", "true", "yes")
    else:
        val = type(cur)(json.loads(v))
    setattr(lp.options, k, val)


class Fedlp:
    def __init__(self, sense, rhs, opts):
        if os.environ.get("FEDLP_DIR"):
            sys.path.insert(0, os.environ["FEDLP_DIR"])
            import _fedlp
        else:
            from fedhpc import _fedlp
        self.lp = _fedlp.Solver(sense, rhs)
        for kv in opts:
            parse_opt(self.lp, kv)
        self.agg: dict = {}

    def add(self, c, ub, st, ix, val):
        self.lp.add_columns(c, np.zeros(len(c)), ub, st, ix, val)

    def costs(self, idx, cost):
        self.lp.set_costs(idx.tolist(), cost.tolist())

    def hint(self, cols):
        self.lp.set_crash_hint(cols)

    def solve(self):
        self.lp.reset_stats()
        s = self.lp.solve()
        st = self.lp.stats()
        for k, v in st.items():
            self.agg[k] = self.agg.get(k, 0) + v
        extra = {k: st[k] for k in ("dual_iterations", "dual_flips", "bfrt_flips", "refactors")}
        return st["seconds"], st["iterations"], (self.lp.objective if s == 0 else None), extra


class Highs:
    def __init__(self, sense, rhs, strategy):
        import highspy
        self.hs = highspy
        h = self.h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("presolve", "off")
        if strategy is not None:
            h.setOptionValue("simplex_strategy", strategy)
        inf = highspy.kHighsInf
        lo = np.array([b if s in "=>" else -inf for s, b in zip(sense, rhs)], float)
        hi = np.array([b if s in "=<" else inf for s, b in zip(sense, rhs)], float)
        h.addRows(len(rhs), lo, hi, 0, np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0))

    def add(self, c, ub, st, ix, val):
        self.h.addCols(len(c), c, np.zeros(len(c)), np.minimum(ub, self.hs.kHighsInf),
                       len(ix), st, ix, val)

    def costs(self, idx, cost):
        self.h.changeColsCost(len(idx), idx.astype(np.int32), cost)

    def hint(self, cols):
        pass

    def solve(self):
        t = time.perf_counter()
        self.h.run()
        t = time.perf_counter() - t
        info = self.h.getInfo()
        return t, info.simplex_iteration_count, info.objective_function_value, {}


class Gurobi:
    def __init__(self, sense, rhs, method):
        import gurobipy as gp
        from fedhpc import formulations
        formulations.configure_env()
        self.gp = gp
        g = self.g = gp.Model(env=formulations._shared_env)
        g.Params.OutputFlag = 0
        if method is not None:
            g.Params.Method = method
        self.rows = [g.addLConstr(gp.LinExpr(), s, b) for s, b in zip(sense, rhs)]
        g.update()
        self.vars: list = []

    def add(self, c, ub, st, ix, val):
        ends = list(st[1:]) + [len(ix)]
        for k in range(len(c)):
            a, b = st[k], ends[k]
            col = self.gp.Column(val[a:b].tolist(), [self.rows[i] for i in ix[a:b]])
            self.vars.append(self.g.addVar(lb=0.0, ub=ub[k], obj=c[k], column=col))

    def costs(self, idx, cost):
        for i, c in zip(idx, cost):
            self.vars[i].Obj = c

    def hint(self, cols):
        pass

    def solve(self):
        self.g.optimize()
        obj = self.g.ObjVal if self.g.Status == 2 else None
        return self.g.Runtime, int(self.g.IterCount), obj, {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("log", help="a pickle from scripts/lp_record.py")
    ap.add_argument("solver", choices=["fedlp", "highs", "gurobi"])
    ap.add_argument("opts", nargs="*", help="fedlp options as name=value")
    ap.add_argument("--strategy", type=int, default=None, help="HiGHS simplex_strategy")
    ap.add_argument("--method", type=int, default=None, help="Gurobi Method")
    ap.add_argument("-v", action="store_true")
    ap.add_argument("--timers", action="store_true")
    args = ap.parse_args()
    with open(args.log, "rb") as f:
        logs = pickle.load(f)
    tot_t = 0.0
    tot_it = bad = n = 0
    agg: dict = {}
    for log in logs:
        _, sense, rhs = log[0]
        if args.solver == "fedlp":
            S = Fedlp(sense, rhs, args.opts)
        elif args.solver == "highs":
            S = Highs(sense, rhs, args.strategy)
        else:
            S = Gurobi(sense, rhs, args.method)
        for op in log[1:]:
            if op[0] == "add":
                S.add(*op[1:])
            elif op[0] == "costs":
                S.costs(*op[1:])
            elif op[0] == "hint":
                S.hint(op[1])
            else:
                n += 1
                t, it, obj, extra = S.solve()
                ref = op[1]
                ok = (obj is None) == (ref is None) and (
                    ref is None or abs(obj - ref) <= 1e-7 * max(1.0, abs(ref)))
                bad += not ok
                tot_t += t
                tot_it += it
                if args.v:
                    print(f"   solve {n}: its {it:6d} {t:7.3f}s obj {obj}"
                          f"{'' if ok else f'  MISMATCH ref={ref}'} {extra or ''}")
        if args.solver == "fedlp":
            for k, v in S.agg.items():
                agg[k] = agg.get(k, 0) + v
    label = " ".join(args.opts) or (f"strategy={args.strategy}" if args.strategy is not None
                                    else f"method={args.method}" if args.method is not None else "")
    print(f"{os.path.basename(args.log):28s} {args.solver:6s} {label:36s} its {tot_it:7d} "
          f"time {tot_t:7.2f}s us/it {1e6 * tot_t / max(1, tot_it):6.1f} bad {bad}/{n}")
    if args.timers and agg:
        print({k: round(v, 3) for k, v in agg.items()
               if k.startswith("t_") and v or k.startswith("d_")})


if __name__ == "__main__":
    main()
