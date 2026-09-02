"""Gurobi hyperparameter ablation for the FED-HPC weighted-sum solve.

For each instance and a fixed λ, solve the normalised weighted-sum scalar
    (λ/(f1_0−f1_T))·f1 + ((1−λ)/f2_T)·f2      s.t.  f1 ≤ f1_0
with NO warm start, once per Gurobi-parameter configuration (each config
changes one setting from Gurobi's defaults). Records wall-clock breakdown
(build / presolve / root-LP / B&B), node count, and the proven objective so
every run can be checked to land on the same optimum.

The model is built once per instance and reused across configs via
``model.reset()`` (params are re-applied each time), so the ~90 s pos_congestion
build is paid once, not per config.

Output (under --out-dir, default results/lambda_eval/ablation/):
  <stem>_lam<λ>_ablation.json   per-config records + parsed Gurobi log fields
  <stem>_lam<λ>_ablation.csv    flat table
  <stem>_lam<λ>_<config>.gurobi.log
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

from gurobipy import GRB

from fedhpc.data import Instance
from fedhpc.formulations import SpaceTimeFormulation, configure_env
from fedhpc.model import _extract

sys.stdout.reconfigure(line_buffering=True)

_KNOWN_REF = {
    "pos_congestion_known_runtime_offline_10min": (44163.67, 648.526, 51251.67),
    "fedhpc_known_runtime_offline_10min": (11151.04, 380.11, 16577.04),
}

# (name, {param: value})  — each changes ONE setting from Gurobi defaults.
CONFIGS: list[tuple[str, dict]] = [
    ("default",              {}),
    ("method_primal_simplex", {"Method": 0}),
    ("method_dual_simplex",   {"Method": 1}),
    ("method_barrier",        {"Method": 2}),
    ("method_concurrent",     {"Method": 3}),
    ("crossover_off",         {"Crossover": 0}),
    ("barrier_no_crossover",  {"Method": 2, "Crossover": 0}),
    ("presolve_off",          {"Presolve": 0}),
    ("presolve_aggressive",   {"Presolve": 2}),
    ("threads_1",             {"Threads": 1}),
    ("threads_4",             {"Threads": 4}),
    ("cuts_off",              {"Cuts": 0}),
    ("heuristics_off",        {"Heuristics": 0.0}),
    ("mipfocus_bound",        {"MIPFocus": 3}),
    # combined candidates — not part of the 1-at-a-time sweep; run explicitly
    # with --configs once the sweep points at a winner.
    ("presolve_off+dual_simplex",   {"Presolve": 0, "Method": 1}),
    ("presolve_off+primal_simplex", {"Presolve": 0, "Method": 0}),
]


_NUM = r"(-?\d[\d.eE+-]*)"   # must start with a digit (Gurobi prints a bare '-' for "n/a")


def _f(s: str) -> float | None:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _parse_log(txt: str) -> dict:
    out: dict = {}
    ps = re.findall(r"presolve time = (\d+)s", txt)
    if ps:
        out["presolve_s"] = float(ps[-1])
    m = re.search(r"Presolve time: ([\d.]+)s", txt)
    if m:
        out["presolve_s"] = float(m.group(1))
    m = re.search(rf"Root relaxation: objective {_NUM}, (\d+) iterations, ([\d.]+) seconds", txt)
    if m:
        out["root_obj"] = _f(m.group(1))
        out["root_iters"] = int(m.group(2))
        out["root_lp_s"] = float(m.group(3))
    m = re.search(r"Explored (\d+) nodes \((\d+) simplex iterations\) in ([\d.]+) seconds", txt)
    if m:
        out["n_nodes"] = int(m.group(1))
        out["total_simplex_iters"] = int(m.group(2))
        out["explored_s"] = float(m.group(3))
    m = re.search(r"Barrier performed (\d+) iterations", txt)
    if m:
        out["barrier_iters"] = int(m.group(1))
    m = re.search(rf"Best objective {_NUM}, best bound {_NUM}", txt)
    if m:
        out["log_best_obj"] = _f(m.group(1))
        out["log_best_bound"] = _f(m.group(2))
    out["no_incumbent_at_limit"] = "Best objective -," in txt
    return out


def run_instance(path: str, lam: float, args) -> None:
    stem = Path(path).stem
    inst = Instance.from_file(path)
    f1_T, f2_T, f1_0 = _KNOWN_REF[stem]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*96}\n{stem}  λ={lam}  —  {len(inst.jobs)} jobs, "
          f"{len(inst.instance_types)} types, H={inst.horizon}\n{'='*96}")

    fmt = SpaceTimeFormulation()
    t0 = time.perf_counter()
    mdl, vars_ = fmt.build(inst)
    build_s = time.perf_counter() - t0
    x = vars_["x"]
    f1e, f2e = fmt.f1_expr(inst, x), fmt.f2_expr(inst, x)
    mdl.addConstr(f1e <= f1_0, name="pareto_region")
    degenerate = f1_0 <= f1_T + 1e-8 or f2_T <= 1e-8
    if degenerate:
        mdl.setObjective(f2e, GRB.MINIMIZE)
    else:
        mdl.setObjective((lam / (f1_0 - f1_T)) * f1e + ((1 - lam) / f2_T) * f2e, GRB.MINIMIZE)
    mdl.update()
    print(f"  model built in {build_s:.0f}s  "
          f"({mdl.NumVars:,} vars, {mdl.NumConstrs:,} constrs, "
          f"{mdl.NumBinVars:,} binaries)")

    configs = CONFIGS if not args.configs else [c for c in CONFIGS if c[0] in args.configs]
    records = []
    for name, params in configs:
        log_path = out / f"{stem}_lam{lam}_{name}.gurobi.log"
        if log_path.exists():
            log_path.unlink()

        mdl.reset(1)
        # restore every tunable we might touch to its Gurobi default, so a
        # setting from a previous config never leaks into this one
        _DEFAULTS = {"Method": -1, "Crossover": -1, "Presolve": -1, "Cuts": -1,
                     "MIPFocus": 0, "NumericFocus": 0, "Heuristics": 0.05,
                     "Threads": 0}
        for k, v in _DEFAULTS.items():
            mdl.setParam(k, v)
        mdl.setParam("OutputFlag", 1)
        mdl.setParam("LogToConsole", 0)
        mdl.setParam("LogFile", str(log_path))
        mdl.setParam("TimeLimit", args.time_limit)
        mdl.setParam("MIPGap", args.mip_gap)
        for k, v in params.items():
            mdl.setParam(k, v)

        t1 = time.perf_counter()
        mdl.optimize()
        solve_s = time.perf_counter() - t1
        sol = _extract(inst, mdl, vars_)
        parsed = _parse_log(log_path.read_text(errors="ignore"))
        rec = dict(
            config=name, params=params,
            build_s=build_s, solve_s=solve_s,
            status=int(mdl.Status), sol_count=int(mdl.SolCount),
            **parsed,
        )
        if mdl.SolCount > 0:
            rec.update(
                f1=sol.f1, f2=sol.f2, obj=float(mdl.ObjVal),
                bound=float(mdl.ObjBound), mip_gap=float(mdl.MIPGap),
            )
        records.append(rec)

        flag = "" if mdl.SolCount and float(mdl.MIPGap) <= args.mip_gap * 1.5 else "  ⚠ not proven"
        print(f"  {name:24}  solve {solve_s:6.0f}s  "
              f"root_lp {parsed.get('root_lp_s', float('nan')):6.1f}s  "
              f"nodes {parsed.get('n_nodes', '?'):>3}  "
              f"obj {rec.get('obj', float('nan')):.6f}{flag}")

    # merge with any existing records (so a follow-up --configs run adds rows
    # instead of clobbering the full sweep); new records win on name collision.
    jpath = out / f"{stem}_lam{lam}_ablation.json"
    merged = {r["config"]: r for r in records}
    if jpath.exists():
        try:
            prev = json.loads(jpath.read_text())
            for r in prev.get("records", []):
                merged.setdefault(r["config"], r)
        except Exception:  # noqa: BLE001
            pass
    records = sorted(merged.values(), key=lambda r: r["solve_s"])
    jpath.write_text(json.dumps(
        dict(instance=stem, lam=lam,
             reference_points=dict(f1_T=f1_T, f2_T=f2_T, f1_0=f1_0),
             build_s=build_s, time_limit=args.time_limit, records=records),
        indent=2, default=float,
    ))
    keys: list[str] = []
    for r in records:
        for k in r:
            if k not in keys and k != "params":
                keys.append(k)
    with open(out / f"{stem}_lam{lam}_ablation.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)
    print(f"  → {out / f'{stem}_lam{lam}_ablation.json'}")
    print(f"  → {out / f'{stem}_lam{lam}_ablation.csv'}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("instances", nargs="*", default=[
        "data/fedhpc_known_runtime_offline_10min.json",
        "data/pos_congestion_known_runtime_offline_10min.json",
    ])
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--time-limit", type=float, default=1200.0)
    p.add_argument("--mip-gap", type=float, default=1e-4)
    p.add_argument("--configs", nargs="*", default=None,
                   help="subset of config names to run (default: all)")
    p.add_argument("--out-dir", default="results/lambda_eval/ablation")
    args = p.parse_args()

    configure_env(verbose=False)
    for path in args.instances:
        run_instance(path, args.lam, args)


if __name__ == "__main__":
    main()
