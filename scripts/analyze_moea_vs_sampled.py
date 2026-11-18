"""Summarise results/generated_bench/moea_vs_sampled/results.jsonl
(from compare_moea_vs_sampled_front.py) by algorithm and instance group.

usage: analyze_moea_vs_sampled.py OUT_DIR
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np

out = sys.argv[1]
rows = [json.loads(l) for l in open(os.path.join(out, "results.jsonl"))]
ALGOS = list(dict.fromkeys(r["algo"] for r in rows))


def tag(name):  # A_fedhpc_n100_lF_iM -> parts
    c, w, n, l, i = name.split("_")
    return dict(cls=c, wl=w, n=n[1:], load=l[1:], io=i[1:])


def summary(group_fn, label):
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        g = group_fn(r)
        if g is not None:
            groups[g][r["algo"]].append(r)
    for metric, f in [("dominated %", lambda rs: 100 * np.mean([r["dominated"] / max(r["card"], 1) for r in rs])),
                      ("exact hits", lambda rs: np.mean([r["exact"] for r in rs])),
                      ("IGD+", lambda rs: np.mean([r["igd_plus"] for r in rs])),
                      ("eps+", lambda rs: np.mean([r["eps_plus"] for r in rs])),
                      ("f1 anchor miss (abs)", lambda rs: np.mean([r["f1_miss_abs"] for r in rs])),
                      ("median cost gap @ sample (% f2 span)",
                       lambda rs: 100 * np.mean([np.median([g for g in r["cost_gaps"] if np.isfinite(g)])
                                                 for r in rs])),
                      ("time s", lambda rs: np.mean([r["t"] for r in rs]))]:
        print(f"\n#### {metric} — by {label}\n")
        print("| " + label + " | " + " | ".join(ALGOS) + " |")
        print("|---|" + "---:|" * len(ALGOS))
        for g in sorted(groups):
            vals = [f(groups[g][a]) if groups[g][a] else float("nan") for a in ALGOS]
            print(f"| {g} | " + " | ".join(f"{v:.3f}" if abs(v) < 100 else f"{v:.1f}" for v in vals) + " |")


print(f"{len(rows)} runs, {len({r['instance'] for r in rows})} instances")
print("any EA point dominating an exact point:", sum(r["dominates"] for r in rows))
summary(lambda r: "all", "all")
summary(lambda r: f"{tag(r['instance'])['wl']} l{tag(r['instance'])['load']}" if tag(r['instance'])['io'] == "M" else None, "workload·load (io M)")
summary(lambda r: f"{tag(r['instance'])['wl']} n{tag(r['instance'])['n']}" if tag(r['instance'])['io'] == "M" else None, "workload·nodes (io M)")
summary(lambda r: f"class {tag(r['instance'])['cls']}", "class")
summary(lambda r: f"{tag(r['instance'])['cls']} io{tag(r['instance'])['io']}"
        if tag(r['instance'])['wl'] == "fedhpc" and tag(r['instance'])['n'] == "100" and tag(r['instance'])['load'] == "H" else None,
        "io sweep (fedhpc n100 lH)")

# per-instance best algorithm by IGD+
print("\n#### per instance: best mean IGD+ / its dominated %\n")
inst = defaultdict(lambda: defaultdict(list))
for r in rows:
    inst[r["instance"]][r["algo"]].append(r)
print("| instance | n_ref | best algo | IGD+ | dom % | eps+ |")
print("|---|---:|---|---:|---:|---:|")
for name in sorted(inst):
    best = min(inst[name], key=lambda a: np.mean([r["igd_plus"] for r in inst[name][a]]))
    rs = inst[name][best]
    print(f"| {name} | {rs[0]['n_ref']} | {best} | {np.mean([r['igd_plus'] for r in rs]):.3f} | "
          f"{100*np.mean([r['dominated']/r['card'] for r in rs]):.0f} | {np.mean([r['eps_plus'] for r in rs]):.3f} |")
