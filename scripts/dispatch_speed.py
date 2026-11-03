"""Decoder speed ablation of weighted_solve_dispatch (current defaults) on the
dispatch-study sample: the LS decode engine arms, each one change from the
default, run one cell at a time (n_threads 4), best wall time of --reps.

  ref            reference decoder (DF_REF_DECODE)
  fast           bitset/event decoder, full decode per candidate (no restart, no pruning)
  fast_prune     … stop a candidate once its lower bound exceeds the incumbent
  restart        restart from the recorded incumbent, stop on convergence (default)
  restart_prune  … plus pruning

All arms decide identically, so g must agree (checked); only time differs.
Output: results/generated_bench/dispatch_study/speed.jsonl.

    uv run python scripts/dispatch_speed.py [--reps 2] [--arms a,b]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dispatch_diagnose import BENCH, LAMS, OUT, SAMPLE, _g  # noqa: E402

REF_DECODE, NO_RESTART, PRUNE = 16384, 32768, 65536
ARMS = {"ref": REF_DECODE, "fast": NO_RESTART, "fast_prune": NO_RESTART | PRUNE,
        "restart": 0, "restart_prune": PRUNE}


def main():
    from fedhpc import instance_gen as gen
    from fedhpc.data import Instance
    from fedhpc.moea import weighted_solve_dispatch

    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--out", type=Path, default=OUT / "speed.jsonl")
    args = ap.parse_args()
    default_flags = weighted_solve_dispatch.__kwdefaults__["flags"]
    bench = {r["name"]: r for r in map(json.loads, BENCH.read_text().splitlines())}
    with open(args.out, "a") as f:
        for name in SAMPLE:
            rec = bench[name]
            inst = Instance.from_file(gen.DATA / "generated" / f"{name}.json")
            ref = (rec["lex_f1"]["f1"], rec["lex_f1"]["f2"], rec["lex_f2"]["f1"])
            for lam in LAMS:
                out = dict(name=name, lam=lam, arms={})
                for arm in args.arms.split(","):
                    best = None
                    for _ in range(args.reps):
                        t = time.perf_counter()
                        sol, st = weighted_solve_dispatch(
                            inst, float(lam), f1_T=ref[0], f2_T=ref[1], f1_0=ref[2],
                            flags=default_flags | ARMS[arm], n_threads=args.threads, return_stats=True)
                        dt = time.perf_counter() - t
                        if best is None or dt < best["t"]:
                            best = dict(t=dt, g=_g(float(lam), sol.f1, sol.f2, ref),
                                        stats={k: v for k, v in st.items() if isinstance(v, (int, float))})
                    out["arms"][arm] = best
                gs = {a: x["g"] for a, x in out["arms"].items()}
                out["same_g"] = max(gs.values()) - min(gs.values()) <= 1e-9 * max(1.0, abs(min(gs.values())))
                f.write(json.dumps(out) + "\n"); f.flush()
                print(f"{name} {lam} same_g={out['same_g']} " + " ".join(
                    f"{a}:{x['t']:.2f}s" for a, x in out["arms"].items()), flush=True)


if __name__ == "__main__":
    main()
