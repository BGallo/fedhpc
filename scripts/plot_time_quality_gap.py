"""Single combined chart: time to solution, objective quality, and the
heuristic's gap vs Gurobi's proven optimum -- all (instance, lambda) pairs in
one graph, as a dumbbell plot (2px neutral connector, endpoints colored by
method). Reads the logs written by scripts/gurobi_convergence.py.

Palette: validated categorical slots 1 (blue, Gurobi) and 2 (orange,
heuristic) from the dataviz skill's reference palette -- passes lightness,
chroma, CVD-separation and contrast checks for a 2-series categorical pair.

  uv run python scripts/plot_time_quality_gap.py [-o OUT.png]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN_DIR = Path("pareto_runs/convergence")

GUROBI = "#2a78d6"    # validated categorical slot 1 (blue)
HEUR = "#eb6834"      # validated categorical slot 2 (orange)
CONNECTOR = "#b9c4c2"  # neutral, one-step-off-surface gray
INK = "#16201f"
TEXT_MUTED = "#5a6664"

INSTANCE_LABEL = {
    "fedhpc_known_runtime_offline_10min.json": "moderate  (964 jobs)",
    "pos_congestion_known_runtime_offline_10min.json": "dense / congested  (3340 jobs)",
}


def load_runs():
    runs = []
    for p in sorted(IN_DIR.glob("*.json")):
        runs.append(json.loads(p.read_text()))
    runs.sort(key=lambda d: (d["instance"], d["lam"]))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", default="pareto_runs/time_quality_gap.png")
    args = ap.parse_args()

    runs = load_runs()
    if not runs:
        print(f"no logs found in {IN_DIR}")
        return

    plt.rcParams.update({
        "font.family": "monospace",
        "font.size": 9.5,
        "axes.edgecolor": "#b9c4c2",
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(10.5, 7.2), dpi=150)

    # label offsets tuned per point to avoid collisions between the two
    # instance clusters and the six dumbbells
    offsets = {
        (0.25, "fedhpc_known_runtime_offline_10min.json"): (10, 14),
        (0.5,  "fedhpc_known_runtime_offline_10min.json"): (10, -22),
        (0.75, "fedhpc_known_runtime_offline_10min.json"): (10, 10),
        (0.25, "pos_congestion_known_runtime_offline_10min.json"): (10, -22),
        (0.5,  "pos_congestion_known_runtime_offline_10min.json"): (10, -22),
        (0.75, "pos_congestion_known_runtime_offline_10min.json"): (10, 10),
    }

    for run in runs:
        gur, he, lam, inst = run["gurobi"], run["heuristic"], run["lam"], run["instance"]
        x = [he["time"], gur["wall"]]
        y = [he["g"], gur["g"]]

        ax.plot(x, y, color=CONNECTOR, lw=2, solid_capstyle="round", zorder=2)
        ax.scatter([gur["wall"]], [gur["g"]], s=100, color=GUROBI, zorder=4,
                   edgecolor="white", linewidth=2)
        ax.scatter([he["time"]], [he["g"]], s=100, color=HEUR, zorder=4,
                   edgecolor="white", linewidth=2)

        gap_pct = 100.0 * (he["g"] - gur["g"]) / gur["g"]
        speedup = gur["wall"] / he["time"]
        dx, dy = offsets[(lam, inst)]
        ax.annotate(
            f"λ={lam}   +{gap_pct:.1f}% gap   {speedup:.1f}× faster",
            xy=(he["time"], he["g"]), xytext=(dx, dy),
            textcoords="offset points", fontsize=8, color=TEXT_MUTED,
            ha="left", va="center",
        )

    # cluster labels, placed once per instance near its point cloud
    ax.text(0.02, 0.97, INSTANCE_LABEL["fedhpc_known_runtime_offline_10min.json"],
             transform=ax.transAxes, fontsize=9, color=INK, ha="left", va="top",
             fontweight="bold")
    ax.text(0.98, 0.34, INSTANCE_LABEL["pos_congestion_known_runtime_offline_10min.json"],
             transform=ax.transAxes, fontsize=9, color=INK, ha="right", va="top",
             fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlabel("time to solution (s, log scale)")
    ax.set_ylabel("objective  g(λ)  —  lower is better")
    ax.grid(True, color="#e3e8e7", lw=0.8, which="major")
    ax.margins(x=0.15, y=0.12)

    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=GUROBI, markersize=9,
                    markeredgecolor="white", markeredgewidth=1.5,
                    label="Gurobi (proven optimal)"),
        plt.Line2D([], [], marker="o", linestyle="", color=HEUR, markersize=9,
                    markeredgecolor="white", markeredgewidth=1.5,
                    label="mono-objective heuristic"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=True, framealpha=0.95,
              fontsize=8.5, edgecolor="#d3dbd9")

    fig.suptitle("Time to solution vs objective quality  ·  Gurobi vs mono-objective heuristic  ·  all λ",
                 fontsize=11.5, y=0.99)
    ax.set_title("connector = one λ; label = heuristic's gap and speedup vs Gurobi's proven optimum",
                 fontsize=8.5, color=TEXT_MUTED, pad=10)

    fig.tight_layout()
    fig.savefig(args.o, bbox_inches="tight")
    print(f"wrote {args.o}")


if __name__ == "__main__":
    main()
