"""Ablation of the weighted-sum decoder / encoding changes vs the old algorithm.

Variants (all minimise the identical normalised scalar g(lam), so the number is
pure optimisation quality on one fixed objective):

    base     memetic GA, SPT list-scheduling decoder            (old algorithm)
    A        + Extract-from-Preempt re-decode (SRPT order)      decode_order=1
    D        + forward-backward improvement                     fbi_passes=3
    A+D      both
    brkga    random-key + serial-SGS encoding (option B)        use_delay=0
    brkga+C  random-key + serial-SGS + delay keys (option C)    use_delay=1

Reported gap%:
    * fedhpc_known_runtime_offline_10min — vs the proven Gurobi optimum g
      (hard-coded from scripts/weighted_heur_vs_gurobi.py runs).
    * other instances — vs `base` (i.e. the improvement over the old algorithm),
      and vs the best proven point in the map checkpoint when one is given.

Usage:  uv run python scripts/ablation_decoder.py [instance.json ...]
env:    LAMS=0.25,0.5,0.75  SEEDS=42,1  POP=32  NGEN=60  BRKGA_POP=48 BRKGA_NGEN=100
        ONLY=base,A,D,A+D,brkga,brkga+C
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from fedhpc.data import Instance
from fedhpc.formulations import configure_env
from fedhpc.moea import (
    heuristic_weighted_reference_points,
    weighted_solve,
    weighted_solve_brkga,
)

sys.stdout.reconfigure(line_buffering=True)

INSTANCES = sys.argv[1:] or [
    "data/fedhpc_known_runtime_offline_10min.json",
    "data/pos_congestion_known_runtime_offline_10min.json",
]
LAMS = [float(x) for x in os.environ.get("LAMS", "0.25,0.5,0.75").split(",")]
SEEDS = [int(x) for x in os.environ.get("SEEDS", "42,1").split(",")]
POP = int(os.environ.get("POP", "32"))
NGEN = int(os.environ.get("NGEN", "60"))
BRKGA_POP = int(os.environ.get("BRKGA_POP", "48"))
BRKGA_NGEN = int(os.environ.get("BRKGA_NGEN", "100"))
ONLY = set(os.environ.get("ONLY", "base,A,D,A+D,brkga,brkga+C").split(","))

FEDHPC_REF = dict(f1_T=11151.0, f2_T=361.96, f1_0=16577.0)
FEDHPC_GUR_G = {0.25: 0.7401, 0.50: 1.3895, 0.75: 1.7729}

# proven-front map checkpoints, for a soft anchor on the congested instances
CHECKPOINTS = {
    "pos_congestion_known_runtime_offline_10min":
        ["pareto_runs/pos_congestion_10min_map_checkpoint.json"],
}

REF_CACHE = os.path.join(os.path.dirname(__file__), "..",
                         "pareto_runs", "ab_decoder_refs.json")


def _cache() -> dict:
    try:
        with open(REF_CACHE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save(c: dict) -> None:
    with open(REF_CACHE, "w") as f:
        json.dump(c, f, indent=2)


def run_variant(name, inst, lam, refs, seed):
    f1_T, f2_T, f1_0 = refs
    kw = dict(f1_T=f1_T, f2_T=f2_T, f1_0=f1_0, seed=seed)
    if name == "base":
        return weighted_solve(inst, lam, pop_size=POP, n_gen=NGEN, **kw)
    if name == "A":
        return weighted_solve(inst, lam, pop_size=POP, n_gen=NGEN, decode_order=1, **kw)
    if name == "D":
        return weighted_solve(inst, lam, pop_size=POP, n_gen=NGEN, fbi_passes=3, **kw)
    if name == "A+D":
        return weighted_solve(inst, lam, pop_size=POP, n_gen=NGEN,
                              decode_order=1, fbi_passes=3, **kw)
    if name == "brkga":
        return weighted_solve_brkga(inst, lam, pop_size=BRKGA_POP, n_gen=BRKGA_NGEN,
                                    use_delay=False, **kw)
    if name == "brkga+C":
        return weighted_solve_brkga(inst, lam, pop_size=BRKGA_POP, n_gen=BRKGA_NGEN,
                                    use_delay=True, **kw)
    raise ValueError(name)


def anchor_g(inst_name, inst, lam, refs):
    f1_T, f2_T, f1_0 = refs
    deg = f1_0 <= f1_T + 1e-8 or f2_T <= 1e-8
    if inst_name.startswith("fedhpc_known_runtime_offline_10min"):
        return FEDHPC_GUR_G[lam], "gurobi-proven"
    for cp in CHECKPOINTS.get(inst_name, []):
        try:
            from fedhpc.pareto import _filter_dominated, _solution_from_dict
            with open(cp) as f:
                sols = _filter_dominated([_solution_from_dict(s)
                                          for s in json.load(f)["solutions"]])
            gs = [(f2 if deg else (lam / (f1_0 - f1_T)) * f1 + ((1 - lam) / f2_T) * f2)
                  for f1, f2 in ((s.f1, s.f2) for s in sols)]
            if gs:
                return min(gs), f"proven-front({len(gs)}pt)"
        except Exception:  # noqa: BLE001
            pass
    return None, None


def main():
    configure_env(verbose=False)
    cache = _cache()
    variants = [v for v in ["base", "A", "D", "A+D", "brkga", "brkga+C"] if v in ONLY]
    print(f"variants={variants}  seeds={SEEDS} lams={LAMS}  "
          f"memetic pop/gen={POP}/{NGEN}  brkga pop/gen={BRKGA_POP}/{BRKGA_NGEN}\n")

    grand = {v: [] for v in variants}       # gap% vs base
    grand_anchor = {v: [] for v in variants}  # gap% vs anchor

    for path in INSTANCES:
        name = path.split("/")[-1].removesuffix(".json")
        inst = Instance.from_file(path)
        if name.startswith("fedhpc_known_runtime_offline_10min"):
            refs = (FEDHPC_REF["f1_T"], FEDHPC_REF["f2_T"], FEDHPC_REF["f1_0"])
            refsrc = "gurobi-proven"
        elif f"{name}::refs" in cache:
            d = cache[f"{name}::refs"]
            refs, refsrc = (d["f1_T"], d["f2_T"], d["f1_0"]), "heuristic(cached)"
        else:
            t = time.time()
            refs = heuristic_weighted_reference_points(inst)
            cache[f"{name}::refs"] = dict(f1_T=refs[0], f2_T=refs[1], f1_0=refs[2])
            _save(cache)
            refsrc = f"heuristic({time.time()-t:.0f}s)"
        deg = refs[2] <= refs[0] + 1e-8 or refs[1] <= 1e-8

        print(f"{'='*104}\n{name}  —  {len(inst.jobs)} jobs, "
              f"{len(inst.instance_types)} types   refs[{refsrc}] "
              f"f1_T={refs[0]:.0f} f2_T={refs[1]:.2f} f1_0={refs[2]:.0f}"
              f"{'  [DEGENERATE]' if deg else ''}\n{'='*104}")
        print(f"  {'lam':>5} {'variant':>8} {'f1':>10} {'f2':>9} {'g':>10} "
              f"{'vs base':>9} {'vs anchor':>10} {'t(s)':>7}")

        for lam in LAMS:
            ga, asrc = anchor_g(name, inst, lam, refs)
            per_variant_g = {}
            for v in variants:
                gs, f1s, f2s, ts = [], [], [], []
                for sd in SEEDS:
                    t = time.time()
                    s = run_variant(v, inst, lam, refs, sd)
                    ts.append(time.time() - t)
                    g = (s.f2 if deg else
                         (lam / (refs[2] - refs[0])) * s.f1 + ((1 - lam) / refs[1]) * s.f2)
                    gs.append(g); f1s.append(s.f1); f2s.append(s.f2)
                per_variant_g[v] = dict(g=float(np.mean(gs)), f1=float(np.mean(f1s)),
                                        f2=float(np.mean(f2s)), t=float(np.mean(ts)),
                                        gstd=float(np.std(gs)))
            gbase = per_variant_g["base"]["g"]
            for v in variants:
                r = per_variant_g[v]
                vb = 100 * (r["g"] - gbase) / abs(gbase)
                va = 100 * (r["g"] - ga) / abs(ga) if ga else float("nan")
                grand[v].append(vb)
                if ga:
                    grand_anchor[v].append(va)
                print(f"  {lam:>5.2f} {v:>8} {r['f1']:>10.1f} {r['f2']:>9.2f} "
                      f"{r['g']:>10.4f} {vb:>+8.2f}% "
                      f"{(f'{va:+.2f}%' if ga else '—'):>10} {r['t']:>7.1f}")
            if ga:
                print(f"  {'':>5} {'anchor':>8} {'':>10} {'':>9} {ga:>10.4f} "
                      f"{'':>9} {asrc:>10}")
            print()

    print(f"{'='*104}\nGRAND MEANS  (across all instance x lam x seed)\n{'='*104}")
    for v in variants:
        mb = float(np.mean(grand[v])) if grand[v] else float("nan")
        ma = float(np.mean(grand_anchor[v])) if grand_anchor[v] else float("nan")
        print(f"  {v:>8}   vs base {mb:>+7.2f}%    vs anchor {ma:>+7.2f}%")


if __name__ == "__main__":
    main()
