"""Plot the proven-optimal Pareto frontier for
data/fedhpc_known_runtime_offline_10min.json from the pareto_runs checkpoints.

  uv run python scripts/plot_true_frontier.py [--ea] [-o OUT.png]

--ea   overlay the NSGA-II / MOEA-D heuristic fronts (pop 400 x 2000 gen)
-o     output path (default: pareto_runs/true_frontier.png)
"""
from __future__ import annotations

import argparse
import itertools
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fedhpc.data import Instance
from fedhpc.metrics import pareto_metrics
from fedhpc.pareto import _filter_dominated, _solution_from_dict

CKPTS = [
    "pareto_runs/fedhpc_10min_map_checkpoint.json",
    "pareto_runs/fedhpc_10min_gap_checkpoint.json",
]
INSTANCE = "data/fedhpc_known_runtime_offline_10min.json"

EXACT = "#0f766e"
HEUR = "#b4530a"
MUTED = "#8a9a97"
INK = "#16201f"


def load_front():
    sols = []
    for cp in CKPTS:
        with open(cp) as f:
            sols += [_solution_from_dict(s) for s in json.load(f)["solutions"]]
    front = _filter_dominated(sols)
    seen, uniq = set(), []
    for s in sorted(front, key=lambda s: s.f1):
        k = (round(s.f1, 3), round(s.f2, 3))
        if k not in seen:
            seen.add(k)
            uniq.append(s)
    return uniq


def open_boxes():
    spans = []
    for cp in CKPTS:
        with open(cp) as f:
            boxes = json.load(f)["boxes"]
        for l, r in boxes:
            spans.append((l["f1"], r["f1"]))
    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ea", action="store_true")
    ap.add_argument("-o", default="pareto_runs/true_frontier.png")
    args = ap.parse_args()

    inst = Instance.from_file(INSTANCE)
    front = load_front()
    f1 = np.array([s.f1 for s in front])
    f2 = np.array([s.f2 for s in front])
    m = pareto_metrics(front)
    knee = m["knee_point"]

    slot_h = inst.slot_size_seconds / 3600.0

    plt.rcParams.update({
        "font.family": "monospace",
        "font.size": 9,
        "axes.edgecolor": "#b9c4c2",
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(9.2, 5.8), dpi=150)

    obx = open_boxes()
    # unmapped gaps: shade the wide spans between consecutive proven points
    # (narrow gaps in the dense cluster are omitted for legibility). Skipped in
    # --ea mode, where the story is the heuristic gap, not the mapping gap.
    if not args.ea:
        sf1 = np.sort(f1)
        shaded_label = False
        for a, b in itertools.pairwise(sf1):
            if b - a < 250:
                continue
            ax.axvspan(a, b, color=HEUR, alpha=0.09, lw=0,
                       label="unmapped gap (≥ 250 slots wide)" if not shaded_label else None)
            shaded_label = True

    if args.ea:
        try:
            with open(
                "/tmp/claude-1000/-home-bernardo-Documentos-programas-fedhpc/"
                "68e50666-adb2-411a-b4da-7d7b152ccc62/scratchpad/frontier_data.json"
            ) as f:
                ea = json.load(f)
        except FileNotFoundError:
            from fedhpc.formulations import configure_env
            from fedhpc.moea import moead_frontier, nsga2_frontier
            configure_env(verbose=False)
            ea = {
                "nsga2": [(s.f1, s.f2) for s in nsga2_frontier(inst, pop_size=400, n_gen=2000, seed=42)],
                "moead": [(s.f1, s.f2) for s in moead_frontier(inst, n_weights=400, n_gen=2000, seed=42)],
            }
        for key, col, lab in [("nsga2", HEUR, "NSGA-II  (pop 400 x 2000)"),
                              ("moead", MUTED, "MOEA/D   (pop 400 x 2000)")]:
            p = np.array(sorted(ea[key]))
            ax.plot(p[:, 0], p[:, 1], "-", color=col, lw=1.0, alpha=0.55, zorder=2)
            ax.scatter(p[:, 0], p[:, 1], s=6, color=col, alpha=0.55, zorder=2, label=lab)

    order = np.argsort(f1)
    ax.plot(f1[order], f2[order], "-", color=EXACT, lw=1.1, alpha=0.85, zorder=3)
    ax.scatter(f1, f2, s=26, color=EXACT, zorder=4, edgecolor="white", linewidth=0.5,
               label=f"proven-optimal front  ({len(front)} pts, MIPGap 1e-9)")

    ax.scatter([knee.f1], [knee.f2], s=130, facecolor="none", edgecolor=INK,
               linewidth=1.6, zorder=5)
    ax.annotate(f"knee  ·  f1={knee.f1:.0f}, f2=${knee.f2:.0f}",
                (knee.f1, knee.f2), textcoords="offset points", xytext=(46, 20),
                fontsize=8.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.7,
                                connectionstyle="arc3,rad=-0.2"))

    for s, tag, dx, dy in [(front[int(np.argmin(f1))], "min turnaround", 10, -4),
                           (front[int(np.argmin(f2))], "min cost", -12, 12)]:
        ax.annotate(f"{tag}\nf1={s.f1:.0f}  f2=${s.f2:.0f}",
                    (s.f1, s.f2), textcoords="offset points", xytext=(dx, dy),
                    fontsize=8, color=EXACT, ha="right" if dx < 0 else "left")

    ax.set_xlabel("f1  —  total turnaround  (600 s slots)")
    ax.set_ylabel("f2  —  total cost  (USD)")
    fig.suptitle("FED-HPC proven Pareto frontier  ·  fedhpc_known_runtime_offline_10min  "
                 "(964 jobs, 29 types)", fontsize=10, y=0.99)

    secx = ax.secondary_xaxis("top", functions=(lambda v: v * slot_h, lambda v: v / slot_h))
    secx.set_xlabel("total turnaround  (hours)", fontsize=8, labelpad=3)

    txt = (f"HV {m['hypervolume']:.3f}   coverage {m['coverage']:.3f}   "
           f"R2 {m['r2']:.3f}   spread {m['spread']:.2f}\n"
           f"f1 in [{f1.min():.0f}, {f1.max():.0f}]   f2 in [0, {f2.max():.1f}]   "
           f"(front still incomplete: {len(obx)} open boxes)")
    ax.text(0.98, 0.96, txt, transform=ax.transAxes, ha="right", va="top",
            fontsize=7.5, color="#4b5a58",
            bbox=dict(boxstyle="round,pad=0.5", fc="#f2f4f3", ec="#d3dbd9"))

    ax.legend(loc="lower left", frameon=True, framealpha=0.95, fontsize=8,
              edgecolor="#d3dbd9")
    ax.grid(True, color="#e3e8e7", lw=0.6)
    ax.margins(x=0.03, y=0.06)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.o, bbox_inches="tight")
    print(f"wrote {args.o}  ({len(front)} points)")


if __name__ == "__main__":
    main()
