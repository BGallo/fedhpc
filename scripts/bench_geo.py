"""Benchmark the two-centre federation instances (fedhpc.geo_gen): the
suggested experiment of docs/geo_distributed_transfer.md §3 plus the
federation comparisons the review asks for (critique §2.2, §3.1, §3.4, §4.2;
improvement_plan A4).

Per instance, one JSON line in results/geo_bench/<run>.jsonl:
  size      — SpaceTime model columns / rows / non-zeros and build time
              (default reductions), and the plain model's size
  root      — LP relaxation of min f1 (z_LP) against the certified f1_T:
              gap_f1 = (f1_T − z_LP) / f1_T
  lex_f1/2  — Gurobi lex-min (f1, f2) and (f2, f1): values, status, time
  zero_cost — whether every job has a free option (critique §3.6) and f2_0
  per λ     — Gurobi weighted sum (status, ĝ, time, root-free), the dispatch
              heuristic with the exact reference points ("benchmark mode")
              and with heuristic ones, no Gurobi ("deployment mode", time
              includes estimating them), and the best threshold baseline,
              each scored on the same exact ĝ
  baselines — on-prem-only, earliest-completion greedy, Slurm-federation-
              inspired earliest-start, threshold sweep
  placement — per solution: jobs at their origin centre / at the other centre /
              in the cloud, staging slots paid, f1 and f2

    uv run python scripts/bench_geo.py --set core [--workers 4 --threads 6]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

LAMS = (0.25, 0.5, 0.75)
# The paper's GA (results/generated_bench/dispatch_study/tuning/final.spec, F_gamid)
GAMID = dict(flags=12352, shortlist=48, pop_size=16, n_gen=120, adapt=215)
GA_SEEDS = (42, 1, 7)
THETAS = (0, 1, 2, 3, 6, 12, 24, 48, 96)


def instance_set(name: str) -> list[str]:
    out = []
    if name in ("core", "all"):
        for w in ("fedhpc", "pos"):
            for n in (50, 150):
                for wan in (0, 10, 1):                                # transfer effect, paired
                    out.append(f"G_{w}_n{n}_lH_iM_w{wan}_fedC")
                for mode in ("isoC", "isoN", "fedN"):                 # value of federating
                    out.append(f"G_{w}_n{n}_lH_iM_w1_{mode}")
                out.append(f"G_{w}_n{n}_lH_iM_w0_fedN")               # no-cloud control
    if name in ("io", "all"):
        for w in ("fedhpc", "pos"):
            for wan in (0, 1):
                out.append(f"G_{w}_n50_lH_iH_w{wan}_fedC")            # heavy results
    if name == "smoke":
        out = ["G_fedhpc_n50_lH_iM_w1_fedC", "G_fedhpc_n50_lH_iM_w1_isoN"]
    return out


def g_hat(lam, f1, f2, ref):
    f1_T, f2_T, f1_0, f2_0 = ref
    if f1_0 <= f1_T + 1e-8 or f2_T - f2_0 <= 1e-8:
        return None
    return lam * (f1 - f1_T) / (f1_0 - f1_T) + (1 - lam) * (f2 - f2_0) / (f2_T - f2_0)


def placement(inst, sol) -> dict:
    site = {m.id: m.site for m in inst.instance_types}
    origin = {j.id: j.origin for j in inst.jobs}
    home = other = cloud = 0
    stage = 0.0
    for jid, (mid, _t) in sol.assignment.items():
        s = site[mid]
        if s == origin[jid]:
            home += 1
        elif s in ("S1", "S2"):
            other += 1
        else:
            cloud += 1
        stage += inst.io_t[jid, mid]
    return dict(home=home, other=other, cloud=cloud, unplaced=len(inst.jobs) - len(sol.assignment),
                stage_slots=round(stage, 3), f1=sol.f1, f2=sol.f2)


def run_one(name: str, threads: int, time_limit: float, phase2_presolve: int | None = 0) -> dict:
    import gurobipy as gp
    from gurobipy import GRB

    from fedhpc import baselines
    from fedhpc import geo_gen as gg
    from fedhpc.formulations import SpaceTimeFormulation
    from fedhpc.model import solve_f1, solve_f2, solve_weighted_sum
    from fedhpc.moea import (heuristic_weighted_sum_reference, weighted_solve_dispatch,
                             weighted_solve_memetic)
    from fedhpc.pareto import _zero_cost_possible

    t = time.perf_counter()
    spec = gg.GeoSpec.parse(name)
    raw = gg.generate(spec)
    from fedhpc.data import Instance
    inst = Instance.from_dict(raw)
    rec = dict(raw["_info"]["spec"], io_info=raw["_info"]["io"],
               n_jobs=len(inst.jobs), gen_s=time.perf_counter() - t)
    grb = dict(OutputFlag=0, Threads=threads, TimeLimit=time_limit, MIPGap=0)

    # ---- model size and root relaxation (critique §3.1) ---------------------
    size = {}
    for label, fmt in (("reduced", SpaceTimeFormulation()),
                       ("plain", SpaceTimeFormulation(collapse_unbounded=False,
                                                      pareto_unbounded=False,
                                                      rejection_deadlines=False,
                                                      lex_deadlines=False))):
        t = time.perf_counter()
        mdl, v = fmt.build(inst)
        mdl.update()
        size[label] = dict(cols=mdl.NumVars, rows=mdl.NumConstrs, nz=mdl.NumNZs,
                           build_s=time.perf_counter() - t)
        if label == "reduced":
            mdl.setObjective(fmt.f1_expr(inst, v["x"]), GRB.MINIMIZE)
            mdl.update()
            lp = mdl.relax()
            lp.Params.OutputFlag = 0
            lp.Params.Threads = threads
            t = time.perf_counter(); lp.optimize()
            size["root_lp_f1"] = lp.ObjVal if lp.Status == GRB.OPTIMAL else None
            size["root_lp_s"] = time.perf_counter() - t
            lp.dispose()
        mdl.dispose()
    rec["size"] = size

    # ---- lexicographic anchors ---------------------------------------------
    # lex phase 2 only: its dense primary-fix row can stall presolve (bench_generated.py)
    p2 = None if phase2_presolve is None else dict(Presolve=phase2_presolve)
    rec["phase2_presolve"] = phase2_presolve
    t = time.perf_counter(); s1 = solve_f1(inst, phase2_params=p2, **grb)
    rec["lex_f1"] = dict(status=s1.status, f1=s1.f1, f2=s1.f2, t=time.perf_counter() - t)
    if s1.f1 is None:                      # no schedule finishes every job in the horizon
        rec["infeasible"] = True
        b = baselines.earliest_start_federation(inst)
        rec["baselines"] = dict(fed_start=dict(placement(inst, b)))
        return rec
    t = time.perf_counter(); s2 = solve_f2(inst, hint=None, phase2_params=p2, **grb)
    rec["lex_f2"] = dict(status=s2.status, f1=s2.f1, f2=s2.f2, t=time.perf_counter() - t)
    for s in (s1, s2):
        inst.check_schedule(s.assignment)
    ref = (s1.f1, s1.f2, s2.f1, s2.f2)
    rec["zero_cost"] = dict(possible=_zero_cost_possible(inst), f2_0=s2.f2)
    lp = size.get("root_lp_f1")
    rec["root_gap_f1"] = None if lp is None or not s1.f1 else (s1.f1 - lp) / s1.f1
    rec["placement"] = dict(lex_f1=placement(inst, s1), lex_f2=placement(inst, s2))

    # ---- baselines ----------------------------------------------------------
    base = {}
    for key, fn in (("onprem", baselines.onprem_only_spt),
                    ("greedy", baselines.greedy_earliest_completion),
                    ("fed_start", baselines.earliest_start_federation)):
        t = time.perf_counter(); b = fn(inst)
        base[key] = dict(placement(inst, b), t=time.perf_counter() - t)
    sweep = []
    for th in THETAS:
        b = baselines.threshold_bursting(inst, th)
        sweep.append(dict(theta=th, **placement(inst, b)))
    base["threshold"] = sweep
    rec["baselines"] = base

    # ---- weighted sums ------------------------------------------------------
    degenerate = g_hat(0.5, s1.f1, s1.f2, ref) is None
    rec["degenerate"] = degenerate
    if not degenerate:   # root LP of the λ = 0.5 scalar ĝ (no Pareto-region rows)
        from fedhpc.model import weighted_sum_terms
        w1, w2, c0 = weighted_sum_terms(0.5, *ref)
        fmt = SpaceTimeFormulation()
        mdl, v = fmt.build(inst)
        mdl.setObjective(w1 * fmt.f1_expr(inst, v["x"]) + w2 * fmt.f2_expr(inst, v["x"]) + c0,
                         GRB.MINIMIZE)
        mdl.update()
        lp = mdl.relax(); lp.Params.OutputFlag = 0; lp.Params.Threads = threads
        lp.optimize()
        rec["root_lp_g05"] = lp.ObjVal if lp.Status == GRB.OPTIMAL else None
        lp.dispose(); mdl.dispose()
    if not degenerate:   # deployment mode: reference points without Gurobi, once per instance
        t = time.perf_counter()
        href = heuristic_weighted_sum_reference(inst, n_threads=threads)
        rec["heur_ref"] = dict(ref=href, t=time.perf_counter() - t)
    rec["lams"] = {}
    lams = (1.0,) if degenerate else LAMS       # no trade-off: min f1 is the whole story
    for lam in lams:
        r = {}
        if not degenerate:
            t = time.perf_counter()
            sw = solve_weighted_sum(inst, lam, ref[0], ref[1], ref[2], f2_0=ref[3], **grb)
            inst.check_schedule(sw.assignment)
            r["gurobi"] = dict(status=sw.status, g=g_hat(lam, sw.f1, sw.f2, ref),
                               t=time.perf_counter() - t, **placement(inst, sw))
        t = time.perf_counter()
        hd = weighted_solve_dispatch(inst, lam, f1_T=ref[0], f2_T=ref[1], f1_0=ref[2],
                                     f2_0=ref[3], n_threads=threads)
        inst.check_schedule(hd.assignment)
        r["dispatch"] = dict(g=g_hat(lam, hd.f1, hd.f2, ref), t=time.perf_counter() - t,
                             **placement(inst, hd))
        ga = []
        for seed in GA_SEEDS:                                         # benchmark mode
            t = time.perf_counter()
            hg = weighted_solve_memetic(inst, lam, f1_T=ref[0], f2_T=ref[1], f1_0=ref[2],
                                        f2_0=ref[3], seed=seed, n_threads=threads, **GAMID)
            inst.check_schedule(hg.assignment)
            ga.append(dict(seed=seed, g=g_hat(lam, hg.f1, hg.f2, ref), t=time.perf_counter() - t,
                           **placement(inst, hg)))
        r["gamid"] = ga
        if not degenerate:
            h1, h2, h3, h4 = rec["heur_ref"]["ref"]
            t = time.perf_counter()
            hdd = weighted_solve_dispatch(inst, lam, f1_T=h1, f2_T=h2, f1_0=h3, f2_0=h4,
                                          n_threads=threads)
            inst.check_schedule(hdd.assignment)
            r["dispatch_deploy"] = dict(g=g_hat(lam, hdd.f1, hdd.f2, ref),
                                        t=time.perf_counter() - t, **placement(inst, hdd))
            t = time.perf_counter()
            hgd = weighted_solve_memetic(inst, lam, f1_T=h1, f2_T=h2, f1_0=h3, f2_0=h4,
                                         seed=GA_SEEDS[0], n_threads=threads, **GAMID)
            inst.check_schedule(hgd.assignment)
            r["gamid_deploy"] = dict(g=g_hat(lam, hgd.f1, hgd.f2, ref),
                                     t=time.perf_counter() - t, **placement(inst, hgd))
            best = min((p for p in sweep if p["unplaced"] == 0),
                       key=lambda p: g_hat(lam, p["f1"], p["f2"], ref))
            r["threshold"] = dict(best, g=g_hat(lam, best["f1"], best["f2"], ref))
            fs = base["fed_start"]
            if fs["unplaced"] == 0:
                r["fed_start_g"] = g_hat(lam, fs["f1"], fs["f2"], ref)
        rec["lams"][str(lam)] = r
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--set", default=None, help="core / io / all / smoke")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--time-limit", type=float, default=3600)
    ap.add_argument("--phase2-presolve", type=int, default=0,
                    help="Gurobi Presolve for lex phase 2 only (-1 = Gurobi default)")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "geo_bench" / "bench.jsonl")
    args = ap.parse_args()
    names = list(args.names) + (instance_set(args.set) if args.set else [])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        done = {json.loads(l)["name"] for l in args.out.read_text().splitlines() if l.strip()}
    todo = [n for n in dict.fromkeys(names) if n not in done]
    todo.sort(key=lambda n: ("_pos_" not in n, not n.endswith("fedC"), n))   # slowest first
    print(f"{len(todo)} to run ({len(done)} already in {args.out.name})", flush=True)
    with ProcessPoolExecutor(args.workers) as ex, open(args.out, "a") as f:
        p2 = None if args.phase2_presolve < 0 else args.phase2_presolve
        futs = {ex.submit(run_one, n, args.threads, args.time_limit, p2): n for n in todo}
        for k, fut in enumerate(as_completed(futs), 1):
            n = futs[fut]
            try:
                rec = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"[{k}/{len(todo)}] {n} FAILED {type(e).__name__}: {e}", flush=True)
                continue
            f.write(json.dumps(rec) + "\n"); f.flush()
            if rec.get("infeasible"):
                print(f"[{k}/{len(todo)}] {n} INFEASIBLE", flush=True); continue
            ts = rec["lex_f1"]["t"] + rec["lex_f2"]["t"] + sum(
                r.get("gurobi", {}).get("t", 0) for r in rec["lams"].values())
            print(f"[{k}/{len(todo)}] {n} gurobi {ts:.0f}s", flush=True)


if __name__ == "__main__":
    main()
