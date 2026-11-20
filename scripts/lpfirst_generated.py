"""lpfirst on the 108 generated instances, against the compact Gurobi results
of ``results/generated_bench/bench_io_noreduc.jsonl`` (BPC plan, Phase 6).

    uv run python scripts/lpfirst_generated.py run --label p6 --workers 3 --threads 4
    uv run python scripts/lpfirst_generated.py run --label p6 B_pos_n100_lE_iM
    uv run python scripts/lpfirst_generated.py summary p6

Per instance: lex (f1, f2), lex (f2, f1) and, on non-trivial instances, the
weighted sums at λ = 0.25 / 0.5 / 0.75 with Gurobi's own reference points
(f1_T, f2_T, f1_0) and the same Pareto box (f1 <= f1_0, f2 <= f2_T), so ĝ is
comparable. Every phase gets ``--time-limit`` seconds (Gurobi had 600).

A solve is ``ok`` when both prove the same value (lex: f1 and f2; weighted:
ĝ within 1e-6 relative), or lpfirst proves what Gurobi did not. ``better``
marks a proof Gurobi lacked. Results: ``results/lpfirst_bench/gen_<label>.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "lpfirst_bench"
REF = ROOT / "results" / "generated_bench" / "bench_io_noreduc.jsonl"
LAMS = (0.25, 0.5, 0.75)


def _phase(inst, ti, w1, w2, b1, b2, seeds, opts, limit):
    from fedhpc import lpfirst
    st: dict = {}
    t = time.perf_counter()
    res = lpfirst.solve_lpfirst(inst, w1, w2, b1, b2, time_limit=limit, options=opts, ti=ti,
                                seed_schedules=seeds, stats=st)
    st["wall_s"] = time.perf_counter() - t
    st["how"] = res.how
    return res, {k: v for k, v in st.items() if isinstance(v, (int, float, str))}


def _f(inst, a):
    completion = {jid: t + inst.p_occ[jid, mid] for jid, (mid, t) in a.items()}
    return inst.f1_of(completion), inst.f2_of(a)


def run_one(name: str, ref: dict, threads: int, limit: float, opts_kw: dict) -> dict:
    from fedhpc import bpc, instance_gen, lpfirst
    from fedhpc.formulations import configure_env
    from fedhpc.model import weighted_sum_degenerate, weighted_sum_terms
    configure_env()
    opts = lpfirst.LPFirstOptions(mip_threads=threads, lp_threads=threads, **opts_kw)
    t0 = time.perf_counter()
    inst = instance_gen.build(name)
    ti = lpfirst._TI(inst, bpc._prepare(inst))
    seeds = lpfirst.seed_schedules(inst, ti) if hasattr(lpfirst, "seed_schedules") \
        else bpc.heuristic_schedules(inst)
    rec = {"name": name, "build_s": time.perf_counter() - t0, "solves": {}}

    def lex(primary):
        w = (1.0, 0.0) if primary == "f1" else (0.0, 1.0)
        t = time.perf_counter()
        r1, s1 = _phase(inst, ti, *w, None, None, seeds, opts, limit)
        out = {"phase1": s1}
        if r1.status == "optimal":
            f1, f2 = _f(inst, r1.assignment)
            b1, b2 = (f1, None) if primary == "f1" else (None, f2)
            r2, s2 = _phase(inst, ti, *w[::-1], b1, b2, [r1.assignment, *seeds], opts, limit)
            out["phase2"] = s2
            a = r2.assignment or r1.assignment
            status = r2.status
        else:
            a, status = r1.assignment, "feasible" if r1.assignment else "none"
        if a:
            seeds.insert(0, a)
            out["f1"], out["f2"] = _f(inst, a)
        out["status"] = status
        out["t"] = time.perf_counter() - t
        g = ref[f"lex_{primary}"]
        out["gurobi"] = {"status": g["status"], "f1": g["f1"], "f2": g["f2"], "t": g["t"]}
        both = status == g["status"] == "optimal"
        same = a is not None and abs(out["f1"] - g["f1"]) < 1e-4 and abs(out["f2"] - g["f2"]) < 1e-4
        out["ok"] = (both and same) or (status == "optimal" and g["status"] != "optimal") \
            or (status != "optimal")
        out["better"] = status == "optimal" and g["status"] != "optimal"
        out["agree"] = same if both else None
        return out

    rec["solves"]["lex_f1"] = lex("f1")
    rec["solves"]["lex_f2"] = lex("f2")
    f1_T, f2_T, f1_0 = ref["lex_f1"]["f1"], ref["lex_f1"]["f2"], ref["lex_f2"]["f1"]
    if not weighted_sum_degenerate(f1_T, f2_T, f1_0, 0.0):
        for lam in LAMS:
            w1, w2, c = weighted_sum_terms(lam, f1_T, f2_T, f1_0, 0.0)
            t = time.perf_counter()
            r, s = _phase(inst, ti, w1, w2, f1_0, f2_T, seeds, opts, limit)
            out = {"phase1": s, "status": r.status, "t": time.perf_counter() - t}
            if r.assignment:
                seeds.insert(0, r.assignment)
                out["f1"], out["f2"] = _f(inst, r.assignment)
                out["g"] = w1 * out["f1"] + w2 * out["f2"] + c
            out["bound_g"] = r.bound + c
            g = ref["lams"][str(lam)]["gurobi"]
            out["gurobi"] = {k: g[k] for k in ("status", "f1", "f2", "g", "t")}
            both = r.status == g["status"] == "optimal"
            same = "g" in out and abs(out["g"] - g["g"]) <= 1e-6 * max(1.0, abs(g["g"]))
            out["ok"] = (both and same) or r.status != "optimal" or g["status"] != "optimal"
            if r.status == "optimal" and g["status"] != "optimal":
                out["ok"] = "g" in out and out["g"] <= g["g"] + 1e-6
            out["better"] = r.status == "optimal" and g["status"] != "optimal"
            out["agree"] = same if both else None
            rec["solves"][f"ws_{lam}"] = out
    rec["t"] = time.perf_counter() - t0
    return rec


def run(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"gen_{args.label}.jsonl"
    refs = {(r := json.loads(l))["name"]: r for l in REF.read_text().splitlines() if l.strip()}
    names = args.names or sorted(refs)
    done = set()
    if out.exists():
        done = {json.loads(l)["name"] for l in out.read_text().splitlines() if l.strip()}
    todo = [n for n in names if n not in done]
    todo.sort(key=lambda n: (("pos" not in n), not n.startswith("B"), n))
    opts_kw = json.loads(args.options) if args.options else {}
    print(f"{len(todo)} to run ({len(done)} already in {out.name})", flush=True)
    with ProcessPoolExecutor(args.workers) as ex, out.open("a") as f:
        futs = {ex.submit(run_one, n, refs[n], args.threads, args.time_limit, opts_kw): n
                for n in todo}
        for k, fut in enumerate(as_completed(futs), 1):
            n = futs[fut]
            try:
                rec = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"[{k}/{len(todo)}] {n} FAILED {type(e).__name__}: {e}", flush=True)
                continue
            rec["label"] = args.label
            f.write(json.dumps(rec) + "\n")
            f.flush()
            s = rec["solves"]
            print(f"[{k}/{len(todo)}] {n} {rec['t']:.0f}s "
                  + " ".join(f"{key}:{v['status'][:3]}{'' if v['ok'] else '!DIFF'}"
                             f"{'+' if v['better'] else ''}" for key, v in s.items()), flush=True)


def summary(args) -> None:
    for label in args.labels:
        rows = [json.loads(l) for l in (OUT / f"gen_{label}.jsonl").read_text().splitlines()
                if l.strip()]
        n = proven = better = diff = 0
        lp_t = gu_t = 0.0
        unproven, slow = [], []
        for r in rows:
            for key, v in r["solves"].items():
                n += 1
                lp_t += v["t"]
                gu_t += v["gurobi"]["t"]
                proven += v["status"] == "optimal"
                better += bool(v["better"])
                if not v["ok"]:
                    diff += 1
                    print(f"  DIFF {r['name']} {key}: lp {v.get('f1')},{v.get('f2')} "
                          f"gurobi {v['gurobi']['f1']},{v['gurobi']['f2']}")
                if v["status"] != "optimal":
                    unproven.append(f"{r['name']}:{key}")
                slow.append((v["t"], f"{r['name']}:{key}", v["gurobi"]["t"], v["gurobi"]["status"]))
        print(f"{label}: {len(rows)} instances, {n} solves; lpfirst proved {proven} "
              f"({better} that Gurobi did not) in {lp_t:.0f} s; Gurobi {gu_t:.0f} s; "
              f"disagreements {diff}")
        if unproven:
            print("  unproven:", " ".join(unproven))
        print("  slowest lpfirst solves (lp s, gurobi s):")
        for t, key, gt, gs in sorted(slow, reverse=True)[:args.top]:
            print(f"    {key:32s} {t:7.1f}  {gt:7.1f} {gs}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("names", nargs="*")
    r.add_argument("--label", required=True)
    r.add_argument("--workers", type=int, default=3)
    r.add_argument("--threads", type=int, default=4)
    r.add_argument("--time-limit", type=float, default=600.0, help="seconds per phase")
    r.add_argument("--options", default="", help="LPFirstOptions overrides as JSON")
    s = sub.add_parser("summary")
    s.add_argument("labels", nargs="+")
    s.add_argument("--top", type=int, default=15)
    args = ap.parse_args()
    run(args) if args.cmd == "run" else summary(args)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    main()
