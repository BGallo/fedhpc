"""Run fedhpc's weighted memetic solver on the Rm||TWCT+TMUC benchmark set.

Benchmarks: ``Rm_(TWCT+TMUC)/{Benchmarks,Additional_benchmarks}.zip`` (unrelated
parallel machines, minimise  sum_j w_j C_j + sum_{k used} (U_k + L_k * load_k)).
Reference optima / best-known values come from the two
``Experimental_results_of_BaPwithHeuristic*.xlsx`` sheets.

Mapping onto fedhpc
-------------------
Each machine k becomes an instance type with capacity 1; every job arrives at 0.
The per-job slot tables handed to the C++ ``_ext.weighted`` memetic solver carry

    f1_contrib = w_j * (t + p_jk)      (weighted completion)
    cost       = L_k * p_jk            (variable machine-usage cost)

so ``w1 = w2 = 1`` minimises TWCT + variable TMUC exactly. fedhpc has no
fixed-charge term, so U_k is invisible to the search: it is added afterwards
for every machine the returned schedule uses. Each machine's job sequence is
then re-ordered by WSPT (Smith's rule — optimal for a fixed assignment with no
release dates), which can only lower the objective; both the raw and the
WSPT-sequenced values are reported.

Usage
-----
    uv run --with openpyxl python scripts/run_rm_twct_tmuc.py \
        [--max-n 150] [--classes V1_F1,V2_F2] [--out results.csv]
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import time
import zipfile
from pathlib import Path

from fedhpc import _moea as _ext  # type: ignore[attr-defined]
from fedhpc import rm_benchmark as rm

ROOT = Path(__file__).resolve().parents[1] / "Rm_(TWCT+TMUC)"
NAME_RE = re.compile(r"M(\d+)_N(\d+)_(V\d_F\d)_(\d+)\.txt$")


# ---------------------------------------------------------------------------
# Instance I/O
# ---------------------------------------------------------------------------

def load_instances() -> dict[tuple[str, str], str]:
    """{(class, "M{m}N{n}_{i}"): file text}, deduplicated across both zips."""
    out: dict[tuple[str, str], str] = {}
    for zname in ("Benchmarks.zip", "Additional_benchmarks.zip"):
        with zipfile.ZipFile(ROOT / zname) as zf:
            for name in zf.namelist():
                mt = NAME_RE.search(name)
                if not mt:
                    continue
                m, n, cls, i = mt.groups()
                key = (cls, f"M{m}N{n}_{i}")
                text = zf.read(name).decode()
                if key in out:
                    assert out[key] == text, f"conflicting copies of {key}"
                    continue
                out[key] = text
    return out


def load_reference() -> dict[tuple[str, str], tuple[float, float]]:
    """{(class, id): (LB, OFV)} from the BaP result spreadsheets."""
    import openpyxl

    ref: dict[tuple[str, str], tuple[float, float]] = {}
    files = [
        ("Experimental_results_of_BaPwithHeuristic_V1_F1(for additional instances).xlsx", None),
        ("Experimental_results_of_BaPwithHeuristic.xlsx", None),  # overrides duplicates
    ]
    for fname, _ in files:
        wb = openpyxl.load_workbook(ROOT / fname, data_only=True, read_only=True)
        for ws in wb.worksheets:
            cls = "V1_F1" if ws.title == "Sheet1" else ws.title.split("(")[0]
            for row in ws.iter_rows(min_row=2, values_only=True):
                if row[0] is None or row[2] is None:
                    continue
                lb, ofv = float(row[1]), float(row[2])
                # 1e12 is the sheet's "no incumbent found" placeholder
                ref[cls, str(row[0])] = (lb, ofv if ofv < 1e11 else float("nan"))
    return ref


# ---------------------------------------------------------------------------
# Objective / baselines
# ---------------------------------------------------------------------------

def greedy(p, w, U, L) -> list[int]:
    """Cheap constructive: jobs by best WSPT ratio, each appended to the machine
    with the smallest marginal cost (incl. U_k when opening it). Used only to
    size the time horizon and as a sanity baseline."""
    n, m = len(w), len(U)
    order = sorted(range(n), key=lambda j: min(p[j]) / w[j])
    load = [0] * m
    machine_of = [0] * n
    for j in order:
        best = min(
            range(m),
            key=lambda k: w[j] * (load[k] + p[j][k]) + L[k] * p[j][k]
            + (U[k] if load[k] == 0 else 0),
        )
        machine_of[j] = best
        load[best] += p[j][best]
    return machine_of


# ---------------------------------------------------------------------------
# fedhpc run
# ---------------------------------------------------------------------------

def run_fedhpc(p, w, U, L, *, horizon: int, pop_size: int, n_gen: int,
               seed: int, n_threads: int) -> tuple[list[int], float, float]:
    n, m = len(w), len(U)
    job_slots = []
    for j in range(n):
        sl = []
        for k in range(m):
            pk = p[j][k]
            c = float(L[k] * pk)
            for t in range(0, horizon - pk + 1):
                sl.append((k, t, pk, float(w[j] * (t + pk)), c))
        job_slots.append(sl)
    asgn, f1, f2, _g, _ls = _ext.weighted(
        n_jobs=n, budget=1e18, job_slots=job_slots,
        type_cap=[1] * m, type_risk=[0.0] * m, init_occ=[],
        w1=1.0, w2=1.0, f1_cap=1e18,
        pop_size=pop_size, n_gen=n_gen, seed=seed, n_threads=n_threads,
        ls_moves=8, restart_patience=6, shortlist=24, ablate=0,
        extra_seeds=[], xover_mode=2, mut_mode=0, decode_order=1, fbi_passes=0,
    )
    return [k for k, _t in asgn], f1, f2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", default="", help="comma list, e.g. V1_F1,V3_F2")
    ap.add_argument("--max-n", type=int, default=10**9)
    ap.add_argument("--only", default="", help="regex on instance id")
    ap.add_argument("--pop-size", type=int, default=32)
    ap.add_argument("--n-gen", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--horizon-factor", type=float, default=1.5)
    ap.add_argument("--out", default=str(ROOT / "fedhpc_results.csv"))
    args = ap.parse_args()

    insts = load_instances()
    ref = load_reference()
    classes = set(args.classes.split(",")) if args.classes else None
    only = re.compile(args.only) if args.only else None

    def sort_key(key):
        cls, iid = key
        m, n, i = map(int, re.match(r"M(\d+)N(\d+)_(\d+)", iid).groups())
        return (n * m, cls, m, n, i)

    keys = [k for k in sorted(insts, key=sort_key)
            if (classes is None or k[0] in classes)
            and int(re.match(r"M\d+N(\d+)", k[1]).group(1)) <= args.max_n
            and (only is None or only.search(k[1]))]

    fields = ["class", "id", "m", "n", "horizon", "ref_lb", "ref_ofv",
              "greedy", "fedhpc_raw", "fedhpc_wspt", "gap_pct", "gap_lb_pct", "machines_used",
              "time_s"]
    out_path = Path(args.out)
    done: set[tuple[str, str]] = set()
    if out_path.exists():
        with open(out_path) as f:
            done = {(r["class"], r["id"]) for r in csv.DictReader(f)}
    new_file = not out_path.exists()
    with open(out_path, "a", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        if new_file:
            wr.writeheader()
        for key in keys:
            if key in done:
                continue
            cls, iid = key
            data = rm.parse(insts[key])
            p, w, U, L = data
            n, m = len(w), len(U)

            g_mach = greedy(p, w, U, L)
            g_val = rm.evaluate(data, g_mach)
            g_load = max(sum(p[j][k] for j in range(n) if g_mach[j] == k) for k in range(m))
            horizon = max(math.ceil(args.horizon_factor * g_load),
                          max(max(r) for r in p))

            t0 = time.perf_counter()
            mach, f1, f2 = run_fedhpc(
                p, w, U, L, horizon=horizon, pop_size=args.pop_size,
                n_gen=args.n_gen, seed=args.seed, n_threads=args.threads)
            dt = time.perf_counter() - t0

            used = sorted(set(mach))
            # fedhpc's own schedule value (its start times may leave idle gaps)
            raw = f1 + f2 + sum(U[k] for k in used)
            best = rm.evaluate(data, mach)

            lb, ofv = ref.get(key, (float("nan"), float("nan")))
            gap = 100 * (best - ofv) / ofv if ofv == ofv else float("nan")
            gap_lb = 100 * (best - lb) / lb if lb == lb else float("nan")
            row = dict(**{"class": cls}, id=iid, m=m, n=n, horizon=horizon,
                       ref_lb=lb, ref_ofv=ofv, greedy=g_val, fedhpc_raw=raw,
                       fedhpc_wspt=best, gap_pct=round(gap, 4), gap_lb_pct=round(gap_lb, 4),
                       machines_used=len(used), time_s=round(dt, 3))
            wr.writerow(row)
            fh.flush()
            print(f"{cls} {iid:>12}  ref={ofv:>12.0f}  fedhpc={best:>12.0f} "
                  f"(raw {raw:>12.0f})  gap={gap:6.2f}%  gapLB={gap_lb:6.2f}%  used={len(used)}/{m}  {dt:.2f}s",
                  flush=True)


if __name__ == "__main__":
    main()
