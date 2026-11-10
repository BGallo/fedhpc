"""Validate the current weighted_solve_dispatch defaults on every instance of
results/generated_bench/bench_io_noreduc.jsonl (the 9 dispatch-study
instances selected them; the other 99 are held out): gap to the exact g* and
wall time, beside the bench's own dispatch run (the previous defaults).

Output: results/generated_bench/dispatch_study/validate[_TAG].jsonl. --kw
passes extra weighted_solve_dispatch keyword arguments (JSON) to compare
configurations on the same cells. --trivial runs only the trivial instances
(f2^T = 0, skipped otherwise; their gap is None, g* being 0). --solver
memetic runs weighted_solve_memetic instead (--kw then takes its keyword
arguments); --seed sets either solver's seed; --names restricts the instances,
--split tune|test to one side of dispatch_study/split.json. Each cell also
records its anytime trace: (seconds, gap %) at every improvement of the best
pen_g, the C++ trace shifted by the Python-side setup time.

    uv run python scripts/dispatch_validate.py [--workers 3] [--threads 4] [--tag T --kw '{"K": 256}'] [--trivial]
        [--solver dispatch|memetic] [--seed 42] [--names A_pos_n100_lE_iM ...] [--split tune|test]
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
from dispatch_diagnose import BENCH, OUT, _g  # noqa: E402

LAMS = ("0.25", "0.5", "0.75")


def _trace(st, wall, gap_of):
    """Best-so-far (s, gap %) points from a solver's (ms, pen_g) trace; the
    C++ clock starts after the Python setup, so its points are shifted by
    wall - total_ms. Thinned to at most ~120 points (all improvements kept
    while few)."""
    off = max(0.0, wall - st["total_ms"] / 1e3)
    pts, best = [], float("inf")
    for ms, pg in st.get("trace", []):
        if pg < best - 1e-12:
            best = pg
            pts.append((round(off + ms / 1e3, 4), round(gap_of(pg), 4)))
    if len(pts) > 120:
        keep = sorted({round(i * (len(pts) - 1) / 119) for i in range(120)})
        pts = [pts[i] for i in keep]
    return pts


def run_one(rec, threads, kw, trivial=False, solver="dispatch", seed=42):
    from fedhpc import instance_gen as gen
    from fedhpc.data import Instance
    from fedhpc.moea import weighted_solve_dispatch, weighted_solve_memetic

    solve = weighted_solve_memetic if solver == "memetic" else weighted_solve_dispatch

    inst = Instance.from_file(gen.DATA / "generated" / f"{rec['name']}.json")
    ref = (rec["lex_f1"]["f1"], rec["lex_f1"]["f2"], rec["lex_f2"]["f1"])
    out = dict(name=rec["name"], cls=rec["cls"], workload=rec["workload"], count=rec["count"],
               load=rec["load"], io=rec["io"], trivial=rec["lex_f1"]["f2"] == 0, lams={})
    for lam in LAMS:
        if out["trivial"] and not trivial:
            continue
        L = float(lam)
        gstar = rec["lams"][lam]["gurobi"]["g"]
        old = rec["lams"][lam]["dispatch"]
        t = time.perf_counter()
        sol, st = solve(inst, L, f1_T=ref[0], f2_T=ref[1], f1_0=ref[2], seed=seed,
                        n_threads=threads, return_stats=True, **kw)
        dt = time.perf_counter() - t
        inst.check_schedule(sol.assignment)
        # degenerate (trivial) instance: g = f2, as in bench_generated._g
        g = sol.f2 if out["trivial"] else _g(L, sol.f1, sol.f2, ref)
        rel = (lambda x: 100 * (x - gstar) / gstar) if gstar > 1e-12 else (lambda x: None)
        # pen_g = w1 f1 + w2 f2 (+ penalties); g = pen_g - w1 f1_T (f2_0 = 0)
        w1 = L / (ref[2] - ref[0])
        trace = [] if out["trivial"] else _trace(st, dt, lambda pg: rel(pg - w1 * ref[0]))
        out["lams"][lam] = dict(g=g, gstar=gstar, gap=rel(g), t=dt,
                                old_g=old["g"], old_gap=rel(old["g"]), old_t=old["t"], trace=trace,
                                stats={k: v for k, v in st.items() if isinstance(v, (int, float))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", default="")
    ap.add_argument("--kw", default="{}")
    ap.add_argument("--trivial", action="store_true")
    ap.add_argument("--solver", choices=("dispatch", "memetic"), default="dispatch")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--names", nargs="*")
    ap.add_argument("--split", choices=("tune", "test"))
    args = ap.parse_args()
    kw = json.loads(args.kw)
    args.out = OUT / f"validate{'_' + args.tag if args.tag else ''}.jsonl"
    bench = [json.loads(l) for l in BENCH.read_text().splitlines()]
    if args.split:
        args.names = json.loads((OUT / "split.json").read_text())[args.split]
    done = set()
    if args.out.exists():
        done = {json.loads(l)["name"] for l in args.out.read_text().splitlines()}
    todo = [r for r in bench if r["name"] not in done
            and (not args.trivial or r["lex_f1"]["f2"] == 0)
            and (not args.names or r["name"] in args.names)]
    print(f"{len(todo)} instances", flush=True)
    with ProcessPoolExecutor(args.workers) as ex, open(args.out, "a") as f:
        futs = {ex.submit(run_one, r, args.threads, kw, args.trivial, args.solver, args.seed): r["name"] for r in todo}
        for k, fut in enumerate(as_completed(futs), 1):
            try:
                rec = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"[{k}/{len(todo)}] {futs[fut]} FAILED {type(e).__name__}: {e}", flush=True)
                continue
            f.write(json.dumps(rec) + "\n"); f.flush()
            print(f"[{k}/{len(todo)}] {rec['name']} " + " ".join(
                f"{l}:{x['old_gap'] or 0:.1f}->{x['gap'] or 0:.1f}%/{x['old_t']:.0f}->{x['t']:.0f}s"
                for l, x in rec["lams"].items()), flush=True)


if __name__ == "__main__":
    main()
