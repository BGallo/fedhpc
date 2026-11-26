"""Fixed benchmark suite for the branch-price-and-cut engine (``fedhpc.bpc``).

Every BPC change is accepted or rejected by running this suite before and
after it and comparing the two runs::

    uv run python scripts/bpc_bench.py run --label baseline
    uv run python scripts/bpc_bench.py run --label phase1
    uv run python scripts/bpc_bench.py compare baseline phase1

Suite
-----
- ``trap_small`` / ``trap_medium``: the 7- and 12-job trap instances of
  ``tests/conftest.py`` (running jobs bind, baselines are far off).
- ``real{n}_s{seed}``: ``n`` consecutive-by-arrival jobs of the 964-job
  fedhpc trace, from a seeded offset, arrivals shifted to start at 0.
  On-prem capacities and running jobs are scaled by ``n / 964`` (at least
  one instance per type), so the slice keeps the trace's congestion ratio
  instead of having a cluster hundreds of times larger than its workload.
- ``rm_M{m}N{n}_{i}``: Rm||TWCT+TMUC instances (Chen et al., EJOR 2024)
  through ``rm_benchmark.to_instance``: capacity-1 machines with fixed costs
  and unrelated processing times. The objective is f1 + f2.

Tasks
-----
- ``root_f1`` / ``root_f2``: root column generation for (1, 0) / (0, 1);
  the bound is compared with the compact LP relaxation.
- ``lex_f1`` / ``lex_f2``: both lex-min phases (each with ``--time-limit``).
- ``ws_sum`` (Rm only): min f1 + f2.
- ``pareto`` (trap instances only): ``true_pareto_frontier(engine="bpc")``
  under ``--pareto-budget`` seconds, scored against the exact front.

References (compact Gurobi model, certified) are cached in
``results/bpc_bench/reference.json``; each run appends to
``results/bpc_bench/<label>.jsonl``.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from gurobipy import GRB  # noqa: E402

from fedhpc import bpc, pareto  # noqa: E402
from fedhpc import rm_benchmark as rm  # noqa: E402
from fedhpc.data import Instance  # noqa: E402
from fedhpc.formulations import SpaceTimeFormulation, configure_env  # noqa: E402
from fedhpc.model import solve_f1, solve_f2  # noqa: E402

OUT = ROOT / "results" / "bpc_bench"
TRACE = ROOT / "data" / "fedhpc_known_runtime_offline_10min.json"
RM_ZIP = ROOT / "Rm_(TWCT+TMUC)" / "Benchmarks.zip"
RM_CSV = ROOT / "Rm_(TWCT+TMUC)" / "fedhpc_results.csv"


# ---------------------------------------------------------------------------
# Instances
# ---------------------------------------------------------------------------

def real_slice(n: int, seed: int) -> Instance:
    data = json.loads(TRACE.read_text())
    jobs = sorted(data["jobs"], key=lambda j: (j["arrival"], j["id"]))
    rng = random.Random(seed)
    start = rng.randrange(len(jobs) - n)
    picked = jobs[start:start + n]
    a0 = int(min(j["arrival"] for j in picked))
    scale = n / len(jobs)
    types, running = [], []
    by_type: dict[int, list[dict]] = {}
    for rj in data.get("running_jobs", []):
        by_type.setdefault(rj["type_id"], []).append(rj)
    for t in data["instance_types"]:
        t = dict(t)
        if t.get("capacity") is not None:
            cap = max(1, round(t["capacity"] * scale))
            pool = sorted(by_type.get(t["id"], []), key=lambda r: r["id"])
            k = min(cap, round(len(pool) * scale))
            running += rng.sample(pool, k) if k else []
            t["capacity"] = cap
        types.append(t)
    d = dict(data)
    d["instance_types"] = types
    d["jobs"] = [dict(j, arrival=j["arrival"] - a0) for j in picked]
    d["running_jobs"] = running
    return Instance.from_dict(d)


def _rm_horizons() -> dict[tuple[str, str], int]:
    with RM_CSV.open() as fh:
        return {(r["class"], r["id"]): int(r["horizon"]) for r in csv.DictReader(fh)}


def _rm_reference(m: int, n: int, i: int, cls: str = "V1_F1") -> dict:
    """Chen et al.'s LP bound and optimum (``fedhpc_results.csv``), used for
    the hard tier instead of a compact-model solve."""
    with RM_CSV.open() as fh:
        for r in csv.DictReader(fh):
            if (r["class"], r["id"]) == (cls, f"M{m}N{n}_{i}"):
                return {"ws_sum": [float(r["ref_ofv"]), float(r["ref_lb"]), "paper"]}
    raise KeyError((m, n, i))


def rm_instance(m: int, n: int, i: int, cls: str = "V1_F1") -> Instance:
    with zipfile.ZipFile(RM_ZIP) as z:
        text = z.read(f"Benchmarks/{cls}/M{m}_N{n}_{cls}_{i}.txt").decode()
    horizon = _rm_horizons()[cls, f"M{m}N{n}_{i}"]
    return rm.to_instance(rm.parse(text), horizon)


HARD_RM = ((2, 30, 1), (5, 50, 1), (8, 40, 1), (8, 40, 2), (10, 50, 1))
# Rm instances whose root bound has a gap (0.3-3 %): the subset-row cut A/B
GAP_RM = ((2, 10, 1), (2, 10, 3), (2, 10, 9), (2, 20, 3), (2, 20, 4), (2, 20, 9),
          (5, 25, 1), (5, 25, 6), (8, 40, 1))


def suite(tier: str = "base") -> dict[str, tuple[callable, list[str]]]:
    """``base``: the suite above. ``hard``: larger instances that the base
    tier no longer separates — Rm M2N30 .. M10N50 (f1 + f2, scored against
    the paper's optimum) and trap_medium's Pareto front. (Real slices of 32
    and 48 jobs solve in under a second, so they are left out.) ``gap``:
    the Rm instances with a 0.3-3 % root gap (``GAP_RM``), for cut A/Bs
    such as ``--pricing sr_cuts=8,sr_per_round=4``."""
    out: dict[str, tuple[callable, list[str]]] = {}
    if tier == "gap":
        for m, n, i in GAP_RM:
            out[f"rm_M{m}N{n}_{i}"] = (lambda m=m, n=n, i=i: rm_instance(m, n, i), ["ws_sum"])
        return out
    if tier == "hard":
        for m, n, i in HARD_RM:
            out[f"rm_M{m}N{n}_{i}"] = (lambda m=m, n=n, i=i: rm_instance(m, n, i), ["ws_sum"])
        from conftest import TRAP_INSTANCES
        out["trap_medium"] = (TRAP_INSTANCES["trap_medium"], ["pareto"])
        return out
    from conftest import TRAP_INSTANCES
    lex = ["root_f1", "root_f2", "lex_f1", "lex_f2"]
    for name, fn in sorted(TRAP_INSTANCES.items()):
        out[name] = (fn, lex + ["pareto"])
    for n in (8, 12, 16, 24):
        for seed in (0, 1):
            out[f"real{n}_s{seed}"] = (lambda n=n, seed=seed: real_slice(n, seed), lex)
    for m, n, i in ((2, 10, 1), (2, 10, 2), (2, 10, 3), (2, 20, 1), (5, 25, 1)):
        out[f"rm_M{m}N{n}_{i}"] = (lambda m=m, n=n, i=i: rm_instance(m, n, i),
                                   ["root_f1", "ws_sum"])
    return out


# ---------------------------------------------------------------------------
# References (compact Gurobi model)
# ---------------------------------------------------------------------------

def compact_lp(inst: Instance, w1: float, w2: float) -> float:
    fmt = SpaceTimeFormulation()
    mdl, vars_ = fmt.build(inst)
    for v in mdl.getVars():
        v.vtype = GRB.CONTINUOUS
    x = vars_["x"]
    mdl.setObjective(w1 * fmt.f1_expr(inst, x) + w2 * fmt.f2_expr(inst, x), GRB.MINIMIZE)
    mdl.optimize()
    return mdl.ObjVal


def compute_reference(name: str, inst: Instance, tasks: list[str]) -> dict:
    ref: dict = {"n_jobs": len(inst.jobs), "n_running": len(inst.running_jobs)}
    if "root_f1" in tasks:
        ref["lp_f1"] = compact_lp(inst, 1.0, 0.0)
    if "root_f2" in tasks:
        ref["lp_f2"] = compact_lp(inst, 0.0, 1.0)
    if "lex_f1" in tasks:
        s = solve_f1(inst)
        ref["lex_f1"] = [s.f1, s.f2, s.status]
    if "lex_f2" in tasks:
        s = solve_f2(inst)
        ref["lex_f2"] = [s.f1, s.f2, s.status]
    if "ws_sum" in tasks:
        s, bound = rm.solve_total_cost(inst, MIPGap=0)
        ref["ws_sum"] = [s.f1 + s.f2, bound, s.status]
    if "pareto" in tasks:
        front = pareto.true_pareto_frontier(inst)
        ref["pareto"] = sorted({(round(s.f1, 6), round(s.f2, 6)) for s in front
                                if s.status == "optimal"})
    return ref


def load_references(names: list[str], s: dict) -> dict:
    path = OUT / "reference.json"
    refs = json.loads(path.read_text()) if path.exists() else {}
    changed = False
    for name in names:
        if name not in refs:
            fn, tasks = s[name]
            print(f"[ref] {name}", flush=True)
            if name.startswith("rm_") and tasks == ["ws_sum"]:
                m, n, i = map(int, name[4:].replace("N", " ").replace("_", " ").split())
                refs[name] = _rm_reference(m, n, i)
            else:
                refs[name] = compute_reference(name, fn(), tasks)
            changed = True
    if changed:
        OUT.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(refs, indent=1))
    return refs


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

def _bnp(inst, w1, w2, tl, **kw):
    st: dict = {}
    t0 = time.perf_counter()
    try:
        obj, a = bpc._branch_and_price(inst, w1, w2, time_limit=tl, stats=st, **kw)
        status = "optimal" if a is not None else "infeasible"
    except bpc.BPCTimeLimit as e:
        obj, a, status = e.objective, e.assignment, "timeout"
        st["open_bound"] = e.bound
    except RuntimeError as e:
        obj, a, status = float("inf"), None, f"error: {e}"
    st["time_s"] = time.perf_counter() - t0
    return obj, a, status, st


def run_task(inst: Instance, task: str, ref: dict, tl: float, budget: float) -> dict:
    rec: dict = {"task": task}
    if task in ("root_f1", "root_f2"):
        w = (1.0, 0.0) if task == "root_f1" else (0.0, 1.0)
        t0 = time.perf_counter()
        r = bpc.solve_root_relaxation(inst, *w)
        rec.update(time_s=time.perf_counter() - t0, bound=r.objective,
                   iters=r.iterations, columns=r.n_columns,
                   ref=ref.get("lp_" + task[-2:]))
        rec["ok"] = rec["ref"] is None or rec["bound"] >= rec["ref"] - 1e-4 * max(1, abs(rec["ref"]))
        return rec
    if task in ("lex_f1", "lex_f2"):
        prim = task[-2:]
        w = (1.0, 0.0) if prim == "f1" else (0.0, 1.0)
        obj, a, st1, s1 = _bnp(inst, *w, tl)
        rec.update(phase1=s1, status1=st1)
        if st1 != "optimal":
            rec.update(status=st1, time_s=s1["time_s"], ok=None)
            return rec
        kw = {"bound1_value": obj} if prim == "f1" else {"bound2_value": obj}
        w2 = (0.0, 1.0) if prim == "f1" else (1.0, 0.0)
        _obj2, a2, st2, s2 = _bnp(inst, *w2, tl, **kw)
        rec.update(phase2=s2, status=st2, time_s=s1["time_s"] + s2["time_s"])
        sched = a2 if a2 is not None else a
        comp = {j: t + inst.p_occ[j, m] for j, (m, t) in sched.items()}
        rec["f1"], rec["f2"] = inst.f1_of(comp), inst.f2_of(sched)
        r1, r2, _ = ref[task]
        rec["ref"] = [r1, r2]
        rec["ok"] = (st2 == "optimal" and abs(rec["f1"] - r1) < 1e-6
                     and abs(rec["f2"] - r2) < 1e-6) if st2 == "optimal" else None
        return rec
    if task == "ws_sum":
        obj, a, status, s = _bnp(inst, 1.0, 1.0, tl)
        rec.update(phase1=s, status=status, time_s=s["time_s"], obj=obj, ref=ref["ws_sum"][0])
        if status != "optimal":
            rec["ok"] = None
        elif ref["ws_sum"][2] == "paper":   # the paper's value may not be optimal
            rec["ok"] = obj <= rec["ref"] + 1e-4
        else:
            rec["ok"] = abs(obj - rec["ref"]) < 1e-4
        return rec
    if task == "pareto":
        t0 = time.perf_counter()
        front = pareto.true_pareto_frontier(inst, engine="bpc", time_budget=budget,
                                            per_solve_time_limit=tl)
        pts = sorted({(round(s.f1, 6), round(s.f2, 6)) for s in front if s.status == "optimal"})
        refpts = {tuple(p) for p in ref["pareto"]}
        rec.update(time_s=time.perf_counter() - t0, n_points=len(pts),
                   n_ref=len(refpts), hits=len(refpts & set(pts)),
                   wrong=len(set(pts) - refpts))
        rec["ok"] = rec["wrong"] == 0
        rec["status"] = "complete" if rec["hits"] == len(refpts) else "partial"
        return rec
    raise ValueError(task)


def _pricing(spec: str | None) -> bpc.PricingOptions:
    """``"ng_size=8,smoothing=0.5"`` -> PricingOptions (``None``/``none``
    values become None)."""
    kw = {}
    for part in filter(None, (spec or "").split(",")):
        k, v = part.split("=")
        if v.lower() == "none":
            kw[k] = None
        elif k in ("engine", "master"):
            kw[k] = v
        else:
            kw[k] = float(v) if "." in v else int(v)
    return bpc.PricingOptions(**kw)


def cmd_run(args) -> None:
    configure_env()
    bpc.DEFAULT_PRICING = _pricing(args.pricing)
    for part in filter(None, (args.tree or "").split(",")):
        k, v = part.split("=")
        bpc.TREE_DEFAULTS[k] = int(v) if v.isdigit() else v
    print(f"[pricing] {bpc.DEFAULT_PRICING}  [tree] {bpc.TREE_DEFAULTS}", flush=True)
    s = suite(args.suite)
    names = [n for n in s if not args.only or any(o in n for o in args.only)]
    refs = load_references(names, s)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{args.label}.jsonl"
    if out.exists() and not args.append:
        out.unlink()
    for name in names:
        fn, tasks = s[name]
        inst = fn()
        for task in tasks:
            if args.tasks and task not in args.tasks:
                continue
            rec = {"label": args.label, "instance": name, "pricing": args.pricing,
                   "tree": args.tree, **run_task(
                inst, task, refs[name], args.time_limit, args.pareto_budget)}
            with out.open("a") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
            print(_fmt(rec), flush=True)


def _fmt(r: dict) -> str:
    bits = [f"{r['instance']:14s} {r['task']:8s} {r.get('status', ''):9s}",
            f"{r['time_s']:8.2f}s"]
    if "bound" in r:
        bits.append(f"bound={r['bound']:.4f} ref={r['ref']} it={r['iters']} cols={r['columns']}")
    for ph in ("phase1", "phase2"):
        if ph in r:
            p = r[ph]
            bits.append(f"{ph}: nodes={p.get('nodes')} root={p.get('root_bound', float('nan')):.3f}"
                        f" price={p.get('pricing_s', 0):.1f}s lp={p.get('lp_s', 0):.1f}s"
                        f" it={p.get('cg_iters', 0)} lbstop={p.get('lb_stops', 0)}"
                        f" lbprune={p.get('lb_pruned', 0)} ng={p.get('ng_rounds', 0)}"
                        f" fixed={p.get('root_fixed', 0)} bound={p.get('open_bound', float('nan')):.2f}")
    if "n_points" in r:
        bits.append(f"pts={r['n_points']} hits={r['hits']}/{r['n_ref']} wrong={r['wrong']}")
    bits.append(f"ok={r.get('ok')}")
    return "  ".join(bits)


def cmd_compare(args) -> None:
    runs = []
    for label in args.labels:
        rows = [json.loads(ln) for ln in (OUT / f"{label}.jsonl").read_text().splitlines()]
        runs.append({(r["instance"], r["task"]): r for r in rows})
    keys = sorted(set().union(*runs), key=lambda k: (k[0], k[1]))
    hdr = f"{'instance':14s} {'task':8s}" + "".join(
        f" | {lab[:22]:>22s}" for lab in args.labels)
    print(hdr)
    print("-" * len(hdr))
    totals = [0.0] * len(runs)
    for k in keys:
        cells = []
        for i, run in enumerate(runs):
            r = run.get(k)
            if r is None:
                cells.append(f"{'-':>22s}")
                continue
            totals[i] += r["time_s"]
            st = r.get("status", "")
            tag = {"optimal": "", "complete": ""}.get(st, st[:7])
            nodes = sum(r[p].get("nodes", 0) for p in ("phase1", "phase2") if p in r)
            extra = f"n={nodes}" if nodes else (f"{r['hits']}/{r['n_ref']}" if "hits" in r else "")
            if r.get("ok") is False:
                tag = "WRONG"
            cells.append(f"{r['time_s']:8.2f}s {extra:>6s} {tag:>6s}")
        print(f"{k[0]:14s} {k[1]:8s}" + "".join(f" | {c}" for c in cells))
    print("-" * len(hdr))
    print(f"{'total':23s}" + "".join(f" | {t:8.1f}s{'':13s}" for t in totals))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--label", required=True)
    r.add_argument("--only", nargs="*", help="substrings of instance names")
    r.add_argument("--tasks", nargs="*")
    r.add_argument("--time-limit", type=float, default=60.0, help="seconds per BPC phase")
    r.add_argument("--pareto-budget", type=float, default=300.0)
    r.add_argument("--append", action="store_true")
    r.add_argument("--suite", choices=["base", "hard", "gap"], default="base")
    r.add_argument("--pricing", help="PricingOptions as k=v,... (e.g. ng_size=none)")
    r.add_argument("--tree", help="bpc.TREE_DEFAULTS as k=v,... (e.g. node_selection=dfs)")
    c = sub.add_parser("compare")
    c.add_argument("labels", nargs="+")
    args = ap.parse_args()
    {"run": cmd_run, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
