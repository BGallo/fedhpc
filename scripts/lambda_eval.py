"""Evaluate the Gurobi weighted-sum solution at several lambdas against the
offline baselines, on the two 10-minute instances.

For each instance and each lambda in --lams:
  1. compute (or load) the four weighted-sum reference points;
  2. warm-start Gurobi from the memetic metaheuristic (fedhpc.weighted_solve)
     and solve the normalised scalar
     (lam/(f1_0-f1_T))*(f1-f1_T) + ((1-lam)/(f2_T-f2_0))*(f2-f2_0)
     s.t. f1 <= f1_0, f2 <= f2_T (lam = 1: lex-min (f1, f2)) with a per-solve
     TimeLimit; record the incumbent + MIPGap;
  3. compute the full KPI block (fedhpc.viz.compute_stats) for the incumbent.

Then the instance-level baselines (fedhpc.baselines):
  onprem_only_spt, greedy_earliest_completion, threshold_bursting sweep.

Outputs (under --out-dir, default results/lambda_eval/):
  <stem>_solutions.json   full per-run records incl. compute_stats
  <stem>_summary.csv      one row per (method, lam/theta) with headline KPIs
  and prints the summary tables to stdout.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from gurobipy import GRB

from fedhpc.baselines import (
    greedy_earliest_completion,
    onprem_only_spt,
    threshold_bursting_sweep,
)
from fedhpc.data import Instance, reference_points
from fedhpc.formulations import SpaceTimeFormulation, configure_env
from fedhpc.model import (
    Solution,
    _extract,
    set_mip_start,
    solve_weighted_sum,
    weighted_sum_degenerate,
    weighted_sum_scale,
    weighted_sum_terms,
)
from fedhpc.moea import heuristic_weighted_sum_reference, weighted_solve
from fedhpc.pareto import weighted_sum_reference
from fedhpc.viz import compute_stats

sys.stdout.reconfigure(line_buffering=True)

def _known_reference_points(path) -> tuple[float, float, float, float] | None:
    """(f1_T, f2_T, f1_0, f2_0) from data/reference_points.json, or None when
    the instance has no entry or its entry is stale (then they are recomputed)."""
    try:
        return reference_points(path)
    except KeyError:
        return None
    except ValueError as e:
        print(f"  {e}")
        return None


def _gurobi_weighted(inst, lam, f1_T, f2_T, f1_0, f2_0, *, time_limit, mip_gap, hint):
    """Solve model.solve_weighted_sum's scalar with a warm start; return
    (Solution, mip_gap, obj_bound, wall_s, degenerate)."""
    degenerate = weighted_sum_degenerate(f1_T, f2_T, f1_0, f2_0)
    if lam == 1.0 and not degenerate:
        # solve_weighted_sum's lex-min (f1, f2) path; certified, so gap 0
        t0 = time.perf_counter()
        sol = solve_weighted_sum(inst, lam, f1_T, f2_T, f1_0, hint=hint, f2_0=f2_0,
                                 OutputFlag=0, TimeLimit=time_limit)
        return sol, 0.0, sol.objective, time.perf_counter() - t0, degenerate
    fmt = SpaceTimeFormulation()
    scale = 1.0
    if not degenerate:
        w1, w2, c = weighted_sum_terms(lam, f1_T, f2_T, f1_0, f2_0)
        scale = weighted_sum_scale(inst, w1, w2)
    weights = None if degenerate else (w1, w2)
    t0 = time.perf_counter()
    mdl, vars_ = fmt.build(inst, objective=weights)
    build_s = time.perf_counter() - t0
    mdl.setParam("OutputFlag", 0)
    mdl.setParam("TimeLimit", time_limit)
    mdl.setParam("MIPGap", mip_gap)
    x = vars_["x"]
    if hint is not None:
        set_mip_start(inst, x, hint)
    f1e, f2e = fmt.f1_expr(inst, x), fmt.f2_expr(inst, x)
    mdl.addConstr(fmt.bound_constr(inst, x, "f1", f1_0), name="pareto_region")
    if degenerate:
        mdl.setObjective(f2e, GRB.MINIMIZE)
    else:
        mdl.addConstr(fmt.bound_constr(inst, x, "f2", f2_T), name="pareto_region_f2")
        mdl.setObjective(scale * (w1 * f1e + w2 * f2e + c), GRB.MINIMIZE)
    t1 = time.perf_counter()
    mdl.optimize()
    solve_s = time.perf_counter() - t1
    sol = _extract(inst, mdl, vars_, build_time=build_s, solve_time=solve_s)
    try:
        mgap = float(mdl.MIPGap)
    except Exception:  # noqa: BLE001
        mgap = float("nan")
    try:
        bound = float(mdl.ObjBound) / scale
    except Exception:  # noqa: BLE001
        bound = float("nan")
    return sol, mgap, bound, build_s + solve_s, degenerate


def _scalar_g(lam, f1, f2, f1_T, f2_T, f1_0, f2_0, degenerate):
    """The normalised ĝ Gurobi minimises (constant included)."""
    if degenerate:
        return f2
    w1, w2, c = weighted_sum_terms(lam, f1_T, f2_T, f1_0, f2_0)
    return w1 * f1 + w2 * f2 + c


def _kpi_row(name, key, sol: Solution, inst: Instance, extra: dict) -> dict:
    """Flatten compute_stats into one summary row (wall-clock seconds)."""
    ss = inst.slot_size_seconds
    if sol.assignment:
        st = compute_stats(sol, inst)
        sysd = st["system"]
        bsd_sorted = sorted(r["bounded_slowdown"] for r in st["per_job"])
        p95 = bsd_sorted[min(len(bsd_sorted) - 1, int(0.95 * len(bsd_sorted)))]
        makespan = max(r["end"] for r in st["per_job"]) * ss
        row = dict(
            method=name, key=key,
            n_scheduled=st["n_scheduled"], n_unscheduled=st["n_total"] - st["n_scheduled"],
            f1_turnaround_slots=sol.f1, f2_cost=sol.f2,
            total_turnaround_s=st["turnaround"]["total"] * ss,
            avg_turnaround_s=st["turnaround"]["avg"] * ss,
            avg_wait_s=st["wait_time"]["avg"] * ss,
            max_wait_s=st["wait_time"]["max"] * ss,
            avg_run_s=st["run_time"]["avg"] * ss,
            avg_bounded_slowdown=st["bounded_slowdown"]["avg"],
            p95_bounded_slowdown=p95,
            max_bounded_slowdown=st["bounded_slowdown"]["max"],
            makespan_s=makespan,
            onprem_jobs=sysd["onprem_jobs"], cloud_jobs=sysd["cloud_jobs"],
            cloud_cost=sysd["cloud_cost"],
            onprem_util_pct=sysd["onprem_util_pct"],
        )
    else:
        row = dict(method=name, key=key, n_scheduled=0)
    row.update(extra)
    return row


def _print_table(rows: list[dict]) -> None:
    cols = [
        ("method", 26, "s"), ("key", 8, "s"),
        ("f1_turnaround_slots", 14, ".1f"), ("f2_cost", 11, ".2f"),
        ("avg_turnaround_s", 15, ".1f"), ("avg_wait_s", 12, ".1f"),
        ("avg_bounded_slowdown", 12, ".3f"), ("p95_bounded_slowdown", 12, ".2f"),
        ("cloud_jobs", 8, "d"), ("n_unscheduled", 8, "d"),
        ("mip_gap_pct", 8, ".2f"), ("wall_s", 8, ".1f"),
    ]
    hdr = "  ".join(f"{c:>{w}}" for c, w, _ in cols)
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        cells = []
        for c, w, f in cols:
            v = r.get(c)
            if v is None:
                cells.append(f"{'-':>{w}}")
            elif f == "s":
                cells.append(f"{str(v):>{w}}")
            else:
                try:
                    cells.append(f"{v:>{w}{f}}")
                except (ValueError, TypeError):
                    cells.append(f"{str(v):>{w}}")
        print("  ".join(cells))


def run_instance(path: str, args) -> None:
    stem = Path(path).stem
    inst = Instance.from_file(path)
    print(f"\n{'='*100}")
    print(f"{stem}  —  {len(inst.jobs)} jobs, {len(inst.running_jobs)} running, "
          f"{len(inst.instance_types)} types, H={inst.horizon}, slot={inst.slot_size_seconds:g}s")
    print(f"{'='*100}")

    # ── reference points ────────────────────────────────────────────────────
    if args.ref_points:
        f1_T, f2_T, f1_0, *f2_0 = (float(v) for v in args.ref_points.split(","))
        f2_0 = f2_0[0] if f2_0 else 0.0
        ref_src = "cli"
    elif not args.recompute_ref and (known := _known_reference_points(path)) is not None:
        f1_T, f2_T, f1_0, f2_0 = known
        ref_src = "known-exact"
    else:
        t0 = time.perf_counter()
        try:
            (f1_T, f2_T, f1_0, f2_0), _ = weighted_sum_reference(
                inst, min_cost_reference=True, OutputFlag=0, TimeLimit=args.ref_time_limit
            )
            ref_src = "gurobi"
        except Exception as e:  # noqa: BLE001
            print(f"  exact reference points failed ({type(e).__name__}: {e})")
            f1_T, f2_T, f1_0, f2_0 = heuristic_weighted_sum_reference(inst)
            ref_src = "heuristic"
        print(f"  reference-point solve: {time.perf_counter()-t0:.1f}s")
    print(f"  reference points [{ref_src}]: f1_T={f1_T:.2f}  f2_T={f2_T:.2f}  "
          f"f1_0={f1_0:.2f}  f2_0={f2_0:.2f}")

    records: list[dict] = []
    rows: list[dict] = []

    # ── Gurobi weighted-sum at each lambda ──────────────────────────────────
    for lam in args.lams:
        print(f"\n  --- lambda = {lam} ---")
        t0 = time.perf_counter()
        he = weighted_solve(
            inst, lam, f1_T=f1_T, f2_T=f2_T, f1_0=f1_0, f2_0=f2_0,
            pop_size=args.heur_pop, n_gen=args.heur_gen,
        )
        t_he = time.perf_counter() - t0
        print(f"    metaheuristic warm start: f1={he.f1:.1f} f2={he.f2:.2f}  ({t_he:.1f}s)")

        sol, mgap, bound, wall, degen = _gurobi_weighted(
            inst, lam, f1_T, f2_T, f1_0, f2_0,
            time_limit=args.time_limit, mip_gap=args.mip_gap, hint=he,
        )
        if not sol.assignment:
            print(f"    Gurobi: no incumbent in {args.time_limit:.0f}s — using warm start")
            sol = he
            mgap = float("nan")
        g = _scalar_g(lam, sol.f1, sol.f2, f1_T, f2_T, f1_0, f2_0, degen)
        g_he = _scalar_g(lam, he.f1, he.f2, f1_T, f2_T, f1_0, f2_0, degen)
        print(f"    Gurobi incumbent: f1={sol.f1:.1f} f2={sol.f2:.2f}  "
              f"g={g:.5f}  MIPgap={mgap*100:.2f}%  ({wall:.1f}s)")

        extra = dict(
            lam=lam, scalar_g=g, scalar_g_heur=g_he,
            heur_gap_pct=100.0 * (g_he - g) / abs(g) if abs(g) > 1e-12 else float("nan"),
            mip_gap_pct=mgap * 100.0, obj_bound=bound, wall_s=wall,
            ref_f1_T=f1_T, ref_f2_T=f2_T, ref_f1_0=f1_0, ref_src=ref_src,
        )
        row = _kpi_row(f"gurobi_weighted", f"λ={lam}", sol, inst, extra)
        rows.append(row)
        records.append(dict(
            method="gurobi_weighted", lam=lam, degenerate=degen,
            f1=sol.f1, f2=sol.f2, status=sol.status, mip_gap=mgap, obj_bound=bound,
            scalar_g=g, warm_start=dict(f1=he.f1, f2=he.f2, scalar_g=g_he),
            stats=compute_stats(sol, inst) if sol.assignment else None,
            summary_row=row,
        ))

    # ── baselines ──────────────────────────────────────────────────────────
    print(f"\n  --- baselines ---")
    t0 = time.perf_counter()
    b_onprem = onprem_only_spt(inst)
    b_greedy = greedy_earliest_completion(inst, order="spt")
    sweep = threshold_bursting_sweep(inst)
    print(f"    baselines computed in {time.perf_counter()-t0:.1f}s")

    for name, key, sol in (
        ("onprem_only_spt", "θ=∞", b_onprem),
        ("greedy_earliest_completion", "greedy", b_greedy),
        *[("threshold_bursting", f"θ={th:g}", s) for th, s in sweep],
    ):
        extra = dict(theta=key)
        row = _kpi_row(name, key, sol, inst, extra)
        rows.append(row)
        records.append(dict(
            method=name, key=key, f1=sol.f1, f2=sol.f2, status=sol.status,
            stats=compute_stats(sol, inst) if sol.assignment else None,
            summary_row=row,
        ))

    # ── output ─────────────────────────────────────────────────────────────
    print()
    _print_table(rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}_solutions.json").write_text(json.dumps(
        dict(instance=stem, path=path,
             n_jobs=len(inst.jobs), n_running=len(inst.running_jobs),
             horizon=inst.horizon, slot_size_seconds=inst.slot_size_seconds,
             reference_points=dict(f1_T=f1_T, f2_T=f2_T, f1_0=f1_0, f2_0=f2_0, source=ref_src),
             records=records),
        indent=2, default=float,
    ))
    all_keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in all_keys:
                all_keys.append(k)
    with open(out_dir / f"{stem}_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=all_keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\n  → {out_dir/f'{stem}_solutions.json'}")
    print(f"  → {out_dir/f'{stem}_summary.csv'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("instances", nargs="*", default=[
        "data/fedhpc_known_runtime_offline_10min.json",
        "data/pos_congestion_known_runtime_offline_10min.json",
    ])
    p.add_argument("--lams", type=lambda s: [float(x) for x in s.split(",")],
                   default=[0.25, 0.5, 0.75])
    p.add_argument("--time-limit", type=float, default=900.0,
                   help="Gurobi TimeLimit per weighted-sum solve (s).")
    p.add_argument("--mip-gap", type=float, default=1e-4)
    p.add_argument("--ref-time-limit", type=float, default=300.0,
                   help="Gurobi TimeLimit per reference-point phase (s).")
    p.add_argument("--ref-points", default=None, help="f1_T,f2_T,f1_0[,f2_0] (skip solve).")
    p.add_argument("--recompute-ref", action="store_true",
                   help="Ignore the known-exact reference-point table.")
    p.add_argument("--heur-pop", type=int, default=32)
    p.add_argument("--heur-gen", type=int, default=60)
    p.add_argument("--out-dir", default="results/lambda_eval")
    args = p.parse_args()

    configure_env(verbose=False)
    for path in args.instances:
        run_instance(path, args)


if __name__ == "__main__":
    main()
