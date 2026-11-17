"""Benchmark the LP-first engine (``fedhpc.lpfirst``) against the compact
Gurobi model on lex-min and Pareto-box solves.

    uv run python scripts/lpfirst_bench.py run --label lpfirst
    uv run python scripts/lpfirst_bench.py run --label big --suite big --gurobi-limit 1800

Each instance gets five solves through the engines' ``lexmin_bounded``
closures:
- the two anchors, lex (f1, f2) and lex (f2, f1);
- min f2 s.t. ``f1 <= f1_A + q (f1_B - f1_A)`` for q = 0.25, 0.5, 0.75,
  where A and B are the anchors.

A solve is ``ok`` when both engines prove the same (f1, f2), or when
Gurobi did not finish but lpfirst proved a point. Results are appended to
``results/lpfirst_bench/<label>.jsonl``.

Suites:
- ``base``: 100-, 200- and 400-job slices of the 964-job trace
  (``bpc_bench.real_slice``).
- ``big``: the full trace and pos_congestion (3340 jobs).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import bpc_bench  # noqa: E402

from fedhpc import lpfirst, pareto  # noqa: E402
from fedhpc.data import Instance  # noqa: E402
from fedhpc.formulations import SpaceTimeFormulation, configure_env  # noqa: E402

OUT = ROOT / "results" / "lpfirst_bench"


def suite(name: str) -> dict:
    if name == "big":
        return {
            "fedhpc964": lambda: Instance.from_file(ROOT / "data" / "fedhpc_known_runtime_offline_10min.json"),
            "pos_congestion": lambda: Instance.from_file(
                ROOT / "data" / "pos_congestion_known_runtime_offline_10min.json"),
        }
    return {f"real{n}_s{s}": (lambda n=n, s=s: bpc_bench.real_slice(n, s))
            for n, s in ((100, 0), (100, 1), (200, 0), (200, 1), (400, 0))}


def run(args) -> None:
    configure_env()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{args.label}.jsonl"
    for name, make in suite(args.suite).items():
        inst = make()
        t0 = time.perf_counter()
        lp, lp_counters = lpfirst.make_lpfirst_lexmin_bounded(inst, args.lp_limit, ea_seeds=False)
        lp_build = time.perf_counter() - t0
        gu = None
        if args.gurobi_limit > 0:
            gu, _ = pareto._make_lexmin_bounded(inst, SpaceTimeFormulation(), {"IntFeasTol": 1e-9},
                                                args.gurobi_limit, 1)

        def both(p, s, bk=None, bv=None):
            t = time.perf_counter()
            a = lp(p, s, bk, bv)
            ta = time.perf_counter() - t
            rec = {"label": args.label, "instance": name, "solve": f"{p}|{bk}<={bv}",
                   "lp_status": a.status, "lp": [a.f1, a.f2], "lp_s": ta, "lp_build_s": lp_build}
            if gu is not None:
                t = time.perf_counter()
                b = gu(p, s, bk, bv)
                rec.update(gu_status=b.status, gu=[b.f1, b.f2], gu_s=time.perf_counter() - t)
                same = (a.status == b.status == "optimal" and abs(a.f1 - b.f1) < 1e-4
                        and abs(a.f2 - b.f2) < 1e-4)
                rec["ok"] = same or (b.status != "optimal" and a.status == "optimal")
            print(json.dumps(rec), flush=True)
            with out.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            return a

        A = both("f1", "f2")
        B = both("f2", "f1")
        if A.f1 is None or B.f1 is None:
            continue
        for q in (0.25, 0.5, 0.75):
            both("f2", "f1", "f1", A.f1 + q * (B.f1 - A.f1))


def summary(args) -> None:
    for label in args.labels:
        rows = [json.loads(line) for line in (OUT / f"{label}.jsonl").open()]
        lp_t = sum(r["lp_s"] for r in rows)
        gu_t = sum(r.get("gu_s", 0.0) for r in rows)
        proven = sum(r["lp_status"] == "optimal" for r in rows)
        bad = [r for r in rows if r.get("ok") is False]
        print(f"{label}: {len(rows)} solves, lpfirst proved {proven} in {lp_t:.1f} s, "
              f"gurobi {gu_t:.1f} s, disagreements {len(bad)}")
        for r in rows:
            print(f"  {r['instance']:15s} {r['solve']:28s} lp {r['lp_status']:8s} {r['lp_s']:7.2f}s"
                  + (f"  gurobi {r['gu_status']:8s} {r['gu_s']:7.1f}s  {'ok' if r['ok'] else 'DIFF'}"
                     if "gu_s" in r else ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--label", required=True)
    r.add_argument("--suite", default="base", choices=["base", "big"])
    r.add_argument("--lp-limit", type=float, default=600.0)
    r.add_argument("--gurobi-limit", type=float, default=600.0, help="0 skips Gurobi")
    s = sub.add_parser("summary")
    s.add_argument("labels", nargs="+")
    args = ap.parse_args()
    run(args) if args.cmd == "run" else summary(args)


if __name__ == "__main__":
    main()
