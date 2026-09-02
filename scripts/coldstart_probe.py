"""Cold-start probe: can Gurobi solve the pos_congestion weighted-sum WITHOUT a
metaheuristic warm start?

Mirrors scripts/lambda_eval.py's `_gurobi_weighted` exactly, but passes NO MIP
start. Captures the full Gurobi log (incumbent / dual-bound trajectory) and
reports time-to-first-incumbent, final incumbent, bound, and gap.

Usage:
  uv run python scripts/coldstart_probe.py [instance.json] \
      --lams 0.5 --time-limit 3600 [--mip-gap 1e-4] [--presolve -1] [--mip-focus 0]
"""
from __future__ import annotations

import argparse
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


def _parse_log(path: Path) -> list[dict]:
    """Pull (time_s, incumbent, bound, gap%) rows out of a Gurobi MIP log."""
    rows: list[dict] = []
    # Gurobi MIP node line: "  <nodes> <...> <incumbent> <bestbd> <gap>% <...> <time>s"
    pat = re.compile(
        r"^[H\*\s]?\s*\d+\s+\d+.*?\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([\d.]+)%\s+[-\d.eE+]+\s+(\d+)s\s*$"
    )
    for line in path.read_text(errors="ignore").splitlines():
        m = pat.match(line)
        if not m:
            continue
        try:
            inc, bnd, gap, t = float(m[1]), float(m[2]), float(m[3]), int(m[4])
        except ValueError:
            continue
        rows.append(dict(t=t, incumbent=inc, bound=bnd, gap_pct=gap))
    return rows


def probe(inst, lam, f1_T, f2_T, f1_0, *, time_limit, mip_gap, log_path, extra_params):
    fmt = SpaceTimeFormulation()
    t0 = time.perf_counter()
    mdl, vars_ = fmt.build(inst)
    build_s = time.perf_counter() - t0
    mdl.setParam("OutputFlag", 1)
    mdl.setParam("LogFile", str(log_path))
    mdl.setParam("LogToConsole", 0)
    mdl.setParam("TimeLimit", time_limit)
    mdl.setParam("MIPGap", mip_gap)
    for k, v in extra_params.items():
        mdl.setParam(k, v)
    x = vars_["x"]
    f1e, f2e = fmt.f1_expr(inst, x), fmt.f2_expr(inst, x)
    mdl.addConstr(f1e <= f1_0, name="pareto_region")
    degenerate = f1_0 <= f1_T + 1e-8 or f2_T <= 1e-8
    if degenerate:
        mdl.setObjective(f2e, GRB.MINIMIZE)
    else:
        mdl.setObjective(
            (lam / (f1_0 - f1_T)) * f1e + ((1 - lam) / f2_T) * f2e, GRB.MINIMIZE
        )

    # NO set_mip_start — this is the whole point.
    t1 = time.perf_counter()
    mdl.optimize()
    solve_s = time.perf_counter() - t1
    sol = _extract(inst, mdl, vars_, build_time=build_s, solve_time=solve_s)

    info = dict(
        lam=lam, degenerate=degenerate,
        build_s=build_s, solve_s=solve_s,
        status=int(mdl.Status), sol_count=int(mdl.SolCount),
        n_bin=int(mdl.NumBinVars), n_var=int(mdl.NumVars), n_constr=int(mdl.NumConstrs),
        presolve_removed=None,
    )
    if mdl.SolCount > 0:
        info.update(
            f1=sol.f1, f2=sol.f2,
            obj=float(mdl.ObjVal), bound=float(mdl.ObjBound), mip_gap=float(mdl.MIPGap),
        )
    else:
        try:
            info["bound"] = float(mdl.ObjBound)
        except Exception:  # noqa: BLE001
            info["bound"] = None
    return sol, info


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("instance", nargs="?",
                   default="data/pos_congestion_known_runtime_offline_10min.json")
    p.add_argument("--lams", type=lambda s: [float(x) for x in s.split(",")], default=[0.5])
    p.add_argument("--time-limit", type=float, default=3600.0)
    p.add_argument("--mip-gap", type=float, default=1e-4)
    p.add_argument("--presolve", type=int, default=None, help="Gurobi Presolve (-1 auto).")
    p.add_argument("--mip-focus", type=int, default=None, help="Gurobi MIPFocus (0..3).")
    p.add_argument("--heuristics", type=float, default=None, help="Gurobi Heuristics fraction.")
    p.add_argument("--out-dir", default="results/lambda_eval/coldstart")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    extra_params: dict = {}
    if args.presolve is not None:
        extra_params["Presolve"] = args.presolve
    if args.mip_focus is not None:
        extra_params["MIPFocus"] = args.mip_focus
    if args.heuristics is not None:
        extra_params["Heuristics"] = args.heuristics

    configure_env(verbose=False)
    stem = Path(args.instance).stem
    inst = Instance.from_file(args.instance)
    f1_T, f2_T, f1_0 = _KNOWN_REF[stem]

    print(f"{stem} — {len(inst.jobs)} jobs, {len(inst.instance_types)} types, H={inst.horizon}")
    print(f"reference points: f1_T={f1_T} f2_T={f2_T} f1_0={f1_0}")
    print(f"extra Gurobi params: {extra_params or '(defaults)'}")
    print(f"time limit: {args.time_limit:g}s   NO warm start\n")

    results = []
    for lam in args.lams:
        log_path = out / f"{stem}_lam{lam}_coldstart.gurobi.log"
        if log_path.exists():
            log_path.unlink()
        print(f"=== lambda = {lam} ===")
        t0 = time.perf_counter()
        sol, info = probe(
            inst, lam, f1_T, f2_T, f1_0,
            time_limit=args.time_limit, mip_gap=args.mip_gap,
            log_path=log_path, extra_params=extra_params,
        )
        wall = time.perf_counter() - t0
        traj = _parse_log(log_path)
        first_inc = traj[0] if traj else None
        info["wall_s"] = wall
        info["first_incumbent"] = first_inc
        info["trajectory"] = traj
        results.append(info)

        print(f"  build {info['build_s']:.0f}s  solve {info['solve_s']:.0f}s  "
              f"({info['n_bin']:,} binaries, {info['n_constr']:,} constrs)")
        if first_inc:
            print(f"  first incumbent: t={first_inc['t']}s  obj={first_inc['incumbent']:.4g}  "
                  f"gap={first_inc['gap_pct']:.1f}%")
        else:
            print(f"  first incumbent: NONE")
        if info["sol_count"] > 0:
            print(f"  final: f1={info['f1']:.1f} f2={info['f2']:.2f}  "
                  f"obj={info['obj']:.5f}  bound={info['bound']:.5f}  gap={info['mip_gap']*100:.2f}%")
        else:
            print(f"  final: no incumbent  bound={info.get('bound')}")
        print(f"  log: {log_path}\n")

    # Merge into a combined summary keyed by lambda so repeated runs at
    # different lambdas accumulate instead of overwriting.
    combined_path = out / f"{stem}_coldstart_summary.json"
    combined = {"instance": stem,
                "reference_points": dict(f1_T=f1_T, f2_T=f2_T, f1_0=f1_0),
                "runs": {}}
    if combined_path.exists():
        try:
            prev = json.loads(combined_path.read_text())
            combined["runs"] = prev.get("runs", {})
        except Exception:  # noqa: BLE001
            pass
    for info in results:
        combined["runs"][str(info["lam"])] = dict(
            info, extra_params=extra_params, time_limit=args.time_limit
        )
    # keep the legacy `results` list too (featured = lowest lambda run this call)
    combined["results"] = results
    combined_path.write_text(json.dumps(combined, indent=2, default=float))
    print(f"→ {combined_path}")


if __name__ == "__main__":
    main()
