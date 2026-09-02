"""Precision sweep: how does the FED-HPC weighted-sum solve behave when a
tighter optimality gap is demanded?

The cold-start / ablation work showed the LP relaxation is almost integral and
the solve closes in 1 branch-and-bound node at the default MIPGap of 1e-4. This
script pushes MIPGap down (1e-4 → 1e-6 → 1e-8 → 1e-9 → 0) and also tries a
"hardened numerics" variant (NumericFocus 3 + tightened Int/Opt/Feas tolerances),
recording for each run:

  - solve time, root-LP time, branch-and-bound node count
  - the gap actually achieved and whether optimality was proved
  - the recovered (f1, f2) — does the schedule stabilise as the gap tightens?
  - any Gurobi numerical warnings emitted

Uses the ablation-recommended fast config as the base
(Presolve=0, Method=1, Heuristics=0), cold (no warm start), λ configurable.

Output: results/lambda_eval/precision/<stem>_lam<λ>_precision.{json,csv}
        + per-run Gurobi logs.
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

# base config: the ablation winner. NOTE Heuristics stays at the default (0.05):
# the ablation showed Heuristics=0 leaves pos_congestion with NO incumbent at all
# (the feasible point comes from Gurobi rounding the near-integral LP, not from
# branching), so turning it off is a trap on the big instance.
_BASE = {"Presolve": 0, "Method": 1}

# (name, MIPGap, extra params)
# NOTE: the hardened run tightens IntFeasTol to 1e-9, which forces branch-and-bound
# on this near-integral model. On the 25M-var instance the B&B tree exhausted
# 125 GB RAM and was OOM-killed — NodeFileStart spills the tree to disk at 24 GB
# so a re-run degrades to slow-but-alive instead of dying.
def _runs(gaps: list[float]) -> list[tuple[str, float, dict]]:
    out = [(f"gap_{g:g}", g, {}) for g in gaps]
    out.append(("gap_0_hardened", 0.0, {
        "NumericFocus": 3, "IntFeasTol": 1e-9,
        "OptimalityTol": 1e-9, "FeasibilityTol": 1e-9,
        "NodeFileStart": 24.0,
    }))
    return out


_NUM = r"(-?\d[\d.eE+-]*)"   # digit-anchored (Gurobi prints a bare '-' for "n/a")


def _f(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _parse_log(txt: str) -> dict:
    o: dict = {}
    m = re.search(rf"Root relaxation: objective {_NUM}, (\d+) iterations, ([\d.]+) seconds", txt)
    if m:
        o["root_obj"], o["root_iters"], o["root_lp_s"] = _f(m[1]), int(m[2]), float(m[3])
    m = re.search(r"Explored (\d+) nodes \((\d+) simplex iterations\) in ([\d.]+) seconds", txt)
    if m:
        o["n_nodes"], o["total_simplex_iters"], o["explored_s"] = int(m[1]), int(m[2]), float(m[3])
    m = re.search(rf"Best objective {_NUM}, best bound {_NUM}, gap ([\d.eE+%-]+)", txt)
    if m:
        o["log_best_obj"], o["log_best_bound"] = _f(m[1]), _f(m[2])
        o["log_gap_str"] = m[3]
    o["no_incumbent_at_limit"] = "Best objective -," in txt
    warns = sorted(set(re.findall(r"Warning: (.+)", txt)))
    o["warnings"] = warns
    o["numerical_trouble"] = any(
        k in txt.lower() for k in ("numerical trouble", "numerical issues",
                                   "ill-conditioned", "suboptimal termination",
                                   "unscaled")
    )
    return o


def run_instance(path: str, lam: float, args) -> None:
    stem = Path(path).stem
    inst = Instance.from_file(path)
    f1_T, f2_T, f1_0 = _KNOWN_REF[stem]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*96}\n{stem}  λ={lam}   base config {_BASE}\n{'='*96}")

    fmt = SpaceTimeFormulation()
    t0 = time.perf_counter()
    mdl, vars_ = fmt.build(inst)
    build_s = time.perf_counter() - t0
    x = vars_["x"]
    f1e, f2e = fmt.f1_expr(inst, x), fmt.f2_expr(inst, x)
    mdl.addConstr(f1e <= f1_0, name="pareto_region")
    mdl.setObjective((lam / (f1_0 - f1_T)) * f1e + ((1 - lam) / f2_T) * f2e, GRB.MINIMIZE)
    mdl.update()
    print(f"  build {build_s:.0f}s  ({mdl.NumVars:,} vars, {mdl.NumBinVars:,} binaries)")

    records = []
    for name, gap, extra in _runs(args.gaps):
        log_path = out / f"{stem}_lam{lam}_{name}.gurobi.log"
        if log_path.exists():
            log_path.unlink()

        mdl.reset(1)
        for k, v in {"Presolve": -1, "Method": -1, "Heuristics": 0.05,
                     "MIPFocus": 0, "NumericFocus": 0, "IntFeasTol": 1e-5,
                     "OptimalityTol": 1e-6, "FeasibilityTol": 1e-6,
                     "MIPGapAbs": 1e-10}.items():
            mdl.setParam(k, v)
        for k, v in _BASE.items():
            mdl.setParam(k, v)
        for k, v in extra.items():
            mdl.setParam(k, v)
        mdl.setParam("MIPGap", gap)
        mdl.setParam("MIPGapAbs", 0.0)
        mdl.setParam("TimeLimit", args.time_limit)
        mdl.setParam("OutputFlag", 1)
        mdl.setParam("LogToConsole", 0)
        mdl.setParam("LogFile", str(log_path))

        t1 = time.perf_counter()
        mdl.optimize()
        solve_s = time.perf_counter() - t1
        sol = _extract(inst, mdl, vars_)
        parsed = _parse_log(log_path.read_text(errors="ignore"))

        rec = dict(name=name, mip_gap_target=gap, extra=extra,
                   build_s=build_s, solve_s=solve_s,
                   status=int(mdl.Status), sol_count=int(mdl.SolCount), **parsed)
        if mdl.SolCount > 0:
            rec.update(f1=sol.f1, f2=sol.f2, obj=float(mdl.ObjVal),
                       bound=float(mdl.ObjBound), mip_gap_achieved=float(mdl.MIPGap))
        proved = mdl.Status == GRB.OPTIMAL
        rec["proved_optimal"] = proved
        records.append(rec)

        tag = "OPT" if proved else f"status {mdl.Status}"
        warn = "  ⚠NUMERICAL" if parsed.get("numerical_trouble") else ""
        print(f"  {name:20}  {solve_s:7.0f}s  nodes {parsed.get('n_nodes','?'):>4}  "
              f"gap→{rec.get('mip_gap_achieved', float('nan')):.2e}  "
              f"f1={rec.get('f1', float('nan')):.2f} f2={rec.get('f2', float('nan')):.4f}  "
              f"[{tag}]{warn}")

    # did the schedule stabilise?
    f1s = [round(r["f1"], 2) for r in records if r.get("f1") is not None]
    f2s = [round(r["f2"], 4) for r in records if r.get("f2") is not None]
    print(f"  distinct (f1,f2) across precision levels: "
          f"{len(set(zip(f1s, f2s)))}  "
          f"(f1 spread {max(f1s)-min(f1s):.2f}, f2 spread {max(f2s)-min(f2s):.4f})")

    (out / f"{stem}_lam{lam}_precision.json").write_text(json.dumps(
        dict(instance=stem, lam=lam, base_config=_BASE,
             reference_points=dict(f1_T=f1_T, f2_T=f2_T, f1_0=f1_0),
             build_s=build_s, time_limit=args.time_limit, records=records),
        indent=2, default=float,
    ))
    keys: list[str] = []
    for r in records:
        for k in r:
            if k not in keys and k not in ("extra", "warnings"):
                keys.append(k)
    with open(out / f"{stem}_lam{lam}_precision.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)
    print(f"  → {out / f'{stem}_lam{lam}_precision.json'}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("instances", nargs="*", default=[
        "data/fedhpc_known_runtime_offline_10min.json",
        "data/pos_congestion_known_runtime_offline_10min.json",
    ])
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--gaps", type=lambda s: [float(x) for x in s.split(",")],
                   default=[1e-4, 1e-6, 1e-8, 1e-9, 0.0])
    p.add_argument("--time-limit", type=float, default=1200.0)
    p.add_argument("--out-dir", default="results/lambda_eval/precision")
    args = p.parse_args()

    configure_env(verbose=False)
    for path in args.instances:
        run_instance(path, args.lam, args)


if __name__ == "__main__":
    main()
