"""Summary scatter: time-to-solution (x) vs objective quality (y) for Gurobi
(proven optimal, uncapped) vs the mono-objective memetic heuristic, across all
lambda values, reading the logs written by scripts/gurobi_convergence.py.

  uv run python scripts/plot_time_vs_quality.py [-o OUT.png]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN_DIR = Path("pareto_runs/convergence")

EXACT = "#0f766e"   # Gurobi
HEUR = "#b4530a"    # heuristic
INK = "#16201f"
MUTED = "#8a9a97"

LAM_MARKERS = {0.25: "o", 0.5: "s", 0.75: "^"}


def load_logs():
    by_instance: dict[str, list[dict]] = {}
    for p in sorted(IN_DIR.glob("*.json")):
        d = json.loads(p.read_text())
        by_instance.setdefault(d["instance"], []).append(d)
    for insts in by_instance.values():
        insts.sort(key=lambda d: d["lam"])
    return by_instance


def plot_instance(ax, name: str, runs: list[dict]):
    for run in runs:
        lam = run["lam"]
        gur = run["gurobi"]
        he = run["heuristic"]
        marker = LAM_MARKERS.get(lam, "d")

        ax.plot([he["time"], gur["wall"]], [he["g"], gur["g"]],
                color=MUTED, lw=0.9, ls=":", zorder=2)

        ax.scatter([gur["wall"]], [gur["g"]], s=90, marker=marker, color=EXACT,
                   zorder=4, edgecolor="white", linewidth=0.6)
        ax.scatter([he["time"]], [he["g"]], s=90, marker=marker, color=HEUR,
                   zorder=4, edgecolor="white", linewidth=0.6)

        gap_pct = 100.0 * (he["g"] - gur["g"]) / gur["g"]
        speedup = gur["wall"] / he["time"]
        ax.annotate(f"λ={lam}\n+{gap_pct:.1f}%, {speedup:.1f}x faster",
                    (he["time"], he["g"]), textcoords="offset points",
                    xytext=(8, -14), fontsize=7.3, color=INK, ha="left")

    ax.set_xscale("log")
    ax.set_xlabel("time to solution (s, log scale)")
    ax.grid(True, color="#e3e8e7", lw=0.6)
    ax.set_title(name.replace(".json", ""), fontsize=9.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", default="pareto_runs/time_vs_quality.png")
    args = ap.parse_args()

    plt.rcParams.update({
        "font.family": "monospace",
        "font.size": 9,
        "axes.edgecolor": "#b9c4c2",
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    by_instance = load_logs()
    if not by_instance:
        print(f"no logs found in {IN_DIR}")
        return

    names = list(by_instance.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(6.3 * len(names), 5.2), dpi=150)
    if len(names) == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        plot_instance(ax, name, by_instance[name])
    axes[0].set_ylabel("objective  g(λ)  —  lower is better")

    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=EXACT, markersize=8,
                    label="Gurobi (proven optimal)"),
        plt.Line2D([], [], marker="o", linestyle="", color=HEUR, markersize=8,
                    label="mono-objective heuristic"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=True,
               framealpha=0.95, fontsize=8.5, edgecolor="#d3dbd9",
               bbox_to_anchor=(0.5, 1.04))

    fig.suptitle("Time to solution vs objective quality  ·  Gurobi vs mono-objective heuristic",
                 fontsize=11, y=1.13)
    fig.tight_layout()
    fig.savefig(args.o, bbox_inches="tight")
    print(f"wrote {args.o}")


if __name__ == "__main__":
    main()
