"""
Ablation study harness comparing NSGA-II, NSGA-III, and MOEA/D on a FED-HPC instance.

All three algorithms run with identical budget (same pop_size × n_gen) over a
configurable list of random seeds.  Quality indicators are computed in a shared
normalised objective space derived from the union of all runs, so HV / IGD / R2
values are directly comparable across algorithms.

CLI:  fedhpc-ablation --instance FILE [options]
API:  fedhpc.ablation.run() / fedhpc.ablation.report()
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import NamedTuple

import numpy as np

from .data import Instance
from .model import Solution

_ALGORITHMS = ("nsga2", "nsga3", "moead")
_W          = 96       # table width
_REF        = np.array([1.1, 1.1])   # HV reference point in normalised space
_N_IGD      = 200      # points on diagonal reference front for IGD


# ── Quality indicators (cross-normalised) ────────────────────────────────────

def _hv2d(pts: np.ndarray, ref: np.ndarray) -> float:
    order = np.argsort(pts[:, 0])
    s     = pts[order]
    hv    = 0.0
    for i in range(len(s)):
        f1_next = ref[0] if i == len(s) - 1 else s[i + 1, 0]
        hv += (f1_next - s[i, 0]) * (ref[1] - s[i, 1])
    return max(hv, 0.0)


def _quality(
    solutions: list[Solution],
    ideal: np.ndarray,
    scale: np.ndarray,
) -> dict:
    """Quality indicators in the shared (globally normalised) objective space.

    ideal and scale are computed from the union of all algorithms' solutions so
    every run is measured on the same ruler.

    Returns
    -------
    cardinality : int    — # non-dominated feasible solutions
    hv          : float  — hypervolume (ref = (1.1, 1.1)); ↑ better
    igd         : float  — IGD vs uniform diagonal front;    ↓ better
    r2          : float  — R2 Tchebycheff (101 weights);     ↓ better
    spread      : float  — Δ uniformity;                     ↓ better (0 = perfect)
    coverage    : float  — HV / ref-box area ∈ [0,1];        ↑ better
    """
    valid = [s for s in solutions if s.f1 is not None and s.f2 is not None]
    n     = len(valid)
    if n == 0:
        return dict(cardinality=0, hv=0.0, igd=float("inf"),
                    r2=float("inf"), spread=float("inf"), coverage=0.0)

    pts  = np.array([(s.f1, s.f2) for s in valid], dtype=float)
    norm = (pts - ideal) / scale

    hv = _hv2d(norm, _REF)

    t_vals    = np.linspace(0.0, 1.0, _N_IGD)
    ref_front = np.column_stack([t_vals, 1.0 - t_vals])
    igd = float(np.mean([np.linalg.norm(norm - r, axis=1).min() for r in ref_front]))

    lambdas = np.linspace(0.0, 1.0, 101)
    r2 = float(
        sum(np.max(np.array([lam, 1.0 - lam]) * norm, axis=1).min()
            for lam in lambdas) / 101
    )

    if n < 2:
        spread = 0.0
    else:
        sp    = norm[np.argsort(norm[:, 0])]
        d     = np.linalg.norm(np.diff(sp, axis=0), axis=1)
        d_bar = d.mean()
        spread = 0.0 if d_bar < 1e-12 else float(np.abs(d - d_bar).sum() / ((n - 1) * d_bar))

    coverage = float(hv / (_REF[0] * _REF[1]))

    return dict(cardinality=n, hv=float(hv), igd=igd,
                r2=r2, spread=spread, coverage=coverage)


def _timing(prof: dict) -> dict:
    """Extract comparable timing fields from any algorithm's profile dict.

    All three algorithms share offspring_avg_ms.  The "selection" phase maps to:
      nsga2: combine_select_avg_ms   (NDS + crowding + survival)
      nsga3: combine_select_avg_ms   (NDS + normalise + niching)
      moead: replacement_avg_ms      (sequential neighbourhood replacement)

    The "rank" phase captures per-generation NDS overhead not in selection:
      nsga2: nds_avg_ms + crowding_avg_ms
      nsga3: rank_nds_avg_ms
      moead: 0
    """
    return dict(
        total_ms      = prof.get("total_ms",        0.0),
        init_ms       = prof.get("init_eval_ms",     0.0),
        offspring_avg = prof.get("offspring_avg_ms", 0.0),
        select_avg    = (prof.get("combine_select_avg_ms")
                         or prof.get("replacement_avg_ms", 0.0)),
        rank_avg      = (prof.get("rank_nds_avg_ms",  0.0)
                         + prof.get("nds_avg_ms",     0.0)
                         + prof.get("crowding_avg_ms",0.0)),
        extract_ms    = prof.get("extract_ms",       0.0),
    )


# ── Core runner ───────────────────────────────────────────────────────────────

class RunResult(NamedTuple):
    algo:      str
    seed:      int
    solutions: list[Solution]
    prof:      dict


def run(
    inst: Instance,
    *,
    pop_size:          int       = 200,
    n_divisions:       int       = 199,
    n_gen:             int       = 300,
    neighborhood_size: int       = 20,
    moead_max_replace: int       = 5,
    moead_archive_size: int      = 20,
    seeds:             list[int] | None = None,
    n_threads:         int       = 0,
    verbose:           bool      = False,
) -> list[RunResult]:
    """Run all three algorithms over all seeds.

    Parameters
    ----------
    inst              : built fedhpc Instance.
    pop_size          : population size (same for all algorithms).
                        For NSGA-III, n_divisions should equal pop_size − 1.
    n_divisions       : NSGA-III reference-point divisions; default pop_size − 1.
    n_gen             : generations per run.
    neighborhood_size : MOEA/D neighbourhood size |T|.
    moead_max_replace : MOEA/D bounded-replacement cap (nr); default 5
                        (A/B-benchmarked improvement). <=0 = original
                        unbounded behaviour. See moea.moead_frontier.
    moead_archive_size : MOEA/D elitist archive capacity; default 20
                        (A/B-benchmarked improvement). <=0 = disabled.
                        See moea.moead_frontier.
    seeds             : RNG seeds; default [42, 43, 44, 45, 46].
    n_threads         : OpenMP threads; 0 = all cores.
    verbose           : print per-run progress to stderr.

    Returns
    -------
    One RunResult per (algorithm, seed) combination.
    """
    if seeds is None:
        seeds = list(range(42, 47))

    try:
        from . import _moea as _ext
        from .moea import _init_occ, _job_slots, _to_solutions, _type_cap, _type_risk
    except ImportError as e:
        raise ImportError(f"fedhpc C++ extension not built: {e}") from e

    common = dict(
        n_jobs    = len(inst.jobs),
        budget    = inst.budget,
        job_slots = _job_slots(inst),
        type_cap  = _type_cap(inst),
        type_risk = _type_risk(inst),
        init_occ  = _init_occ(inst),
        n_gen     = n_gen,
        n_threads = n_threads,
    )

    results: list[RunResult] = []
    for algo in _ALGORITHMS:
        for seed in seeds:
            if verbose:
                tag = f"{algo.upper():6s}  seed={seed}"
                print(f"  {tag} … ", end="", file=sys.stderr, flush=True)

            if algo == "nsga2":
                raw, prof = _ext.nsga2(pop_size=pop_size, seed=seed, **common)
            elif algo == "nsga3":
                raw, prof = _ext.nsga3(
                    pop_size=pop_size, n_divisions=n_divisions, seed=seed, **common)
            else:  # moead
                raw, prof = _ext.moead(
                    n_weights=pop_size, neighborhood_size=neighborhood_size,
                    max_replace=moead_max_replace, archive_size=moead_archive_size,
                    seed=seed, **common)

            sols = _to_solutions(inst, raw)
            results.append(RunResult(algo=algo, seed=seed,
                                     solutions=sols, prof=dict(prof)))

            if verbose:
                print(f"{len(sols):3d} pts  {prof.get('total_ms', 0):.0f} ms",
                      file=sys.stderr, flush=True)

    return results


# ── Report ────────────────────────────────────────────────────────────────────

def _stat(vals: list[float]) -> tuple[float, float]:
    m = mean(vals)
    s = stdev(vals) if len(vals) > 1 else 0.0
    return m, s


def _mstd(m: float, s: float, w: int = 14, dec: int = 3) -> str:
    """Format mean ± std, fitting in w characters."""
    if s < 5e-4:
        return f"{m:{w}.{dec}f}"
    return f"{m:.{dec}f}±{s:.{dec}f}".rjust(w)


def _best_mark(algo: str, algo_means: dict[str, float], higher_better: bool) -> str:
    best = max(algo_means.values()) if higher_better else min(algo_means.values())
    return "*" if abs(algo_means[algo] - best) < 1e-9 else " "


def report(
    results:  list[RunResult],
    *,
    instance_name: str       = "",
    pop_size:      int       = 200,
    n_divisions:   int       = 199,
    n_gen:         int       = 300,
    neighborhood_size: int   = 20,
    moead_max_replace: int   = 5,
    moead_archive_size: int  = 20,
    out_dir:       Path | None = None,
    as_json:       bool      = False,
) -> None:
    """Print quality + timing tables and optionally save CSV / JSON.

    Quality metrics are computed in the shared objective space defined by the
    combined ideal/nadir across every algorithm and every seed, so HV / IGD /
    R2 values are directly comparable.
    """
    seeds = sorted({r.seed for r in results})
    n_seeds = len(seeds)

    # ── Cross-normalisation reference ─────────────────────────────────────────
    all_sols = [s for r in results for s in r.solutions
                if s.f1 is not None and s.f2 is not None]
    if not all_sols:
        print("No feasible solutions found across any run.", file=sys.stderr)
        return

    pts_all = np.array([(s.f1, s.f2) for s in all_sols])
    ideal   = pts_all.min(axis=0)
    nadir   = pts_all.max(axis=0)
    scale   = np.where(nadir - ideal > 1e-10, nadir - ideal, 1.0)

    # ── Aggregate per algorithm ───────────────────────────────────────────────
    agg: dict[str, dict] = {}
    raw_rows: list[dict] = []  # for CSV / JSON

    for algo in _ALGORITHMS:
        runs = [r for r in results if r.algo == algo]

        q_list = [_quality(r.solutions, ideal, scale) for r in runs]
        t_list = [_timing(r.prof)                     for r in runs]

        q_keys = ("cardinality", "hv", "igd", "r2", "spread", "coverage")
        t_keys = ("total_ms", "init_ms", "offspring_avg", "select_avg",
                  "rank_avg", "extract_ms")

        agg[algo] = {
            "quality": {k: _stat([q[k] for q in q_list]) for k in q_keys},
            "timing":  {k: _stat([t[k] for t in t_list]) for k in t_keys},
        }

        for r, q, t in zip(runs, q_list, t_list):
            raw_rows.append({"algo": algo, "seed": r.seed, **q, **t})

    # ── JSON output ───────────────────────────────────────────────────────────
    if as_json:
        out: dict = {}
        for algo in _ALGORITHMS:
            out[algo] = {
                "quality": {k: {"mean": m, "std": s}
                            for k, (m, s) in agg[algo]["quality"].items()},
                "timing":  {k: {"mean": m, "std": s}
                            for k, (m, s) in agg[algo]["timing"].items()},
            }
        print(json.dumps(out, indent=2))
        return

    # ── Console tables ────────────────────────────────────────────────────────
    sep  = "═" * _W
    thin = "─" * _W

    inst_tag = f"  {instance_name}" if instance_name else ""
    seed_tag = str(seeds) if n_seeds <= 6 else f"{seeds[0]}…{seeds[-1]} ({n_seeds} seeds)"

    print(f"\n{sep}")
    print(f"ABLATION STUDY{inst_tag}")
    print(f"  pop={pop_size}  gen={n_gen}  "
          f"nsga3_divs={n_divisions}  moead_T={neighborhood_size}  "
          f"moead_nr={moead_max_replace}  moead_archive={moead_archive_size}  "
          f"seeds={seed_tag}")
    print(sep)

    # ── Quality table ─────────────────────────────────────────────────────────
    q_cols = [
        ("Card",    "cardinality", True,  0),
        ("HV ↑",   "hv",          True,  3),
        ("IGD ↓",  "igd",         False, 3),
        ("R2 ↓",   "r2",          False, 3),
        ("Spread↓","spread",       False, 3),
        ("Cover↑", "coverage",    True,  3),
    ]

    print(f"\nQUALITY INDICATORS  "
          f"(mean ± std over {n_seeds} seed{'s' if n_seeds > 1 else ''}; "
          f"* = best per column; cross-normalised)")
    print(thin)

    col_w = 14
    hdr = f"  {'Algo':<8}" + "".join(f"{h:>{col_w}}" for h, *_ in q_cols)
    print(hdr)
    print(f"  {'─'*8}" + "─" * (col_w * len(q_cols)))

    for algo in _ALGORITHMS:
        row = f"  {algo.upper():<8}"
        for _, key, higher, dec in q_cols:
            m, s = agg[algo]["quality"][key]
            means = {a: agg[a]["quality"][key][0] for a in _ALGORITHMS}
            mark  = _best_mark(algo, means, higher)
            cell  = _mstd(m, s, col_w - 1, dec) + mark
            row  += cell
        print(row)

    print(thin)

    # ── Timing table ──────────────────────────────────────────────────────────
    t_cols = [
        ("Total(ms)",    "total_ms",      False, 1),
        ("Init(ms)",     "init_ms",       False, 1),
        ("Off/gen",      "offspring_avg", False, 2),
        ("Sel/gen",      "select_avg",    False, 2),
        ("Rank/gen",     "rank_avg",      False, 2),
        ("Extr(ms)",     "extract_ms",    False, 1),
    ]

    print(f"\nTIMING (ms, mean ± std over {n_seeds} seed{'s' if n_seeds > 1 else ''}; "
          f"Off/gen = offspring hot path; Sel/gen = selection step)")
    print(thin)

    hdr2 = f"  {'Algo':<8}" + "".join(f"{h:>{col_w}}" for h, *_ in t_cols)
    print(hdr2)
    print(f"  {'─'*8}" + "─" * (col_w * len(t_cols)))

    for algo in _ALGORITHMS:
        row = f"  {algo.upper():<8}"
        for _, key, higher, dec in t_cols:
            m, s = agg[algo]["timing"][key]
            means = {a: agg[a]["timing"][key][0] for a in _ALGORITHMS}
            mark  = _best_mark(algo, means, higher)
            cell  = _mstd(m, s, col_w - 1, dec) + mark
            row  += cell
        print(row)

    print(thin)
    print(f"\nNote: Rank/gen = {{'nsga2': nds+crowding/gen, 'nsga3': rank_nds/gen, 'moead': 0}}")
    print(sep)

    # ── CSV / JSON file output ────────────────────────────────────────────────
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        csv_path = out_dir / "ablation_raw.csv"
        csv_fields = ["algo", "seed", "cardinality", "hv", "igd", "r2",
                      "spread", "coverage", "total_ms", "init_ms",
                      "offspring_avg", "select_avg", "rank_avg", "extract_ms"]
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=csv_fields)
            w.writeheader()
            w.writerows(raw_rows)
        print(f"Raw data CSV     → {csv_path}", file=sys.stderr)

        summary_path = out_dir / "ablation_summary.json"
        summary: dict = {}
        for algo in _ALGORITHMS:
            summary[algo] = {
                "quality": {k: {"mean": m, "std": s}
                            for k, (m, s) in agg[algo]["quality"].items()},
                "timing":  {k: {"mean": m, "std": s}
                            for k, (m, s) in agg[algo]["timing"].items()},
            }
        with open(summary_path, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"Summary JSON     → {summary_path}", file=sys.stderr)


# ── CLI entry point ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fedhpc-ablation",
        description=(
            "Compare NSGA-II, NSGA-III, and MOEA/D on a FED-HPC instance.\n\n"
            "Runs all three algorithms with identical budget over multiple random\n"
            "seeds and reports cross-normalised quality metrics and timing tables."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--instance", required=True, metavar="FILE",
                   help="Path to instance JSON file.")
    p.add_argument("--pop-size", type=int, default=200, metavar="N",
                   help="Population size for all algorithms (default: 200).")
    p.add_argument("--n-divisions", type=int, default=None, metavar="P",
                   help=(
                       "NSGA-III reference-point divisions (default: pop_size − 1). "
                       "Should equal pop_size − 1 for best coverage."
                   ))
    p.add_argument("--n-gen", type=int, default=300, metavar="N",
                   help="Generations per run (default: 300).")
    p.add_argument("--neighborhood-size", type=int, default=20, metavar="T",
                   help="MOEA/D neighbourhood size |T| (default: 20).")
    p.add_argument("--moead-max-replace", type=int, default=5, metavar="NR",
                   help=(
                       "MOEA/D bounded-replacement cap (nr): max neighbours a "
                       "child may overwrite per generation. Default 5 "
                       "(A/B-benchmarked improvement); <=0 = original "
                       "unbounded behaviour."
                   ))
    p.add_argument("--moead-archive-size", type=int, default=20, metavar="N",
                   help=(
                       "MOEA/D elitist archive capacity, merged into the "
                       "returned front. Default 20 (A/B-benchmarked "
                       "improvement); <=0 = disabled."
                   ))
    p.add_argument("--seeds", type=int, nargs="+", default=None, metavar="S",
                   help=(
                       "RNG seeds (space-separated). "
                       "Default: 42 43 44 45 46 (5 seeds)."
                   ))
    p.add_argument("--n-threads", type=int, default=0, metavar="N",
                   help="OpenMP thread count; 0 = all cores (default).")
    p.add_argument("--output-dir", default=None, metavar="DIR",
                   help="Directory for ablation_raw.csv and ablation_summary.json.")
    p.add_argument("--json", action="store_true",
                   help="Print results as JSON instead of tables.")
    p.add_argument("--verbose", action="store_true",
                   help="Print per-run progress to stderr.")
    return p


def main(argv: list[str] | None = None) -> None:
    args   = build_parser().parse_args(argv)
    seeds  = args.seeds or list(range(42, 47))
    n_div  = args.n_divisions if args.n_divisions is not None else args.pop_size - 1

    try:
        inst = Instance.from_file(args.instance)
    except Exception as e:
        print(f"Error loading instance: {e}", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(
            f"Ablation: {args.instance}  pop={args.pop_size}  gen={args.n_gen}"
            f"  divs={n_div}  T={args.neighborhood_size}"
            f"  seeds={seeds}"
            + (f"  threads={args.n_threads}" if args.n_threads else "  threads=all"),
            file=sys.stderr,
        )

    try:
        results = run(
            inst,
            pop_size          = args.pop_size,
            n_divisions       = n_div,
            n_gen             = args.n_gen,
            neighborhood_size = args.neighborhood_size,
            moead_max_replace = args.moead_max_replace,
            moead_archive_size = args.moead_archive_size,
            seeds             = seeds,
            n_threads         = args.n_threads,
            verbose           = args.verbose,
        )
    except ImportError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    report(
        results,
        instance_name     = Path(args.instance).name,
        pop_size          = args.pop_size,
        n_divisions       = n_div,
        n_gen             = args.n_gen,
        neighborhood_size = args.neighborhood_size,
        moead_max_replace = args.moead_max_replace,
        moead_archive_size = args.moead_archive_size,
        out_dir           = Path(args.output_dir) if args.output_dir else None,
        as_json           = args.json,
    )


if __name__ == "__main__":
    main()
