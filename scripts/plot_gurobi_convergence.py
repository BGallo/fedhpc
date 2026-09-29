"""Plot Gurobi's incumbent/bound convergence (objective vs wall time) against
the mono-objective memetic heuristic's single (time, objective) point, for
every (instance, lambda) log written by scripts/gurobi_convergence.py.

  uv run python scripts/plot_gurobi_convergence.py [-o OUT_DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN_DIR = Path("pareto_runs/convergence")

EXACT = "#0f766e"   # Gurobi incumbent
BOUND = "#8a9a97"   # Gurobi dual bound
HEUR = "#b4530a"    # mono-objective heuristic
INK = "#16201f"

NO_INCUMBENT = 1e98  # Gurobi's MIP_OBJBST sentinel before any incumbent is found


def load_logs():
    by_instance: dict[str, list[dict]] = {}
    for p in sorted(IN_DIR.glob("*.json")):
        d = json.loads(p.read_text())
        by_instance.setdefault(d["instance"], []).append(d)
    for insts in by_instance.values():
        insts.sort(key=lambda d: d["lam"])
    return by_instance


def plot_instance(name: str, runs: list[dict], out_path: Path):
    fig, axes = plt.subplots(1, len(runs), figsize=(4.6 * len(runs), 4.6), dpi=150)
    if len(runs) == 1:
        axes = [axes]

    plt.rcParams.update({
        "font.family": "monospace",
        "font.size": 9,
        "axes.edgecolor": "#b9c4c2",
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    for ax, run in zip(axes, runs):
        lam = run["lam"]
        gur = run["gurobi"]
        he = run["heuristic"]
        log = gur["log"]

        t_inc = [row[0] for row in log if row[1] is not None and row[1] < NO_INCUMBENT]
        y_inc = [row[1] for row in log if row[1] is not None and row[1] < NO_INCUMBENT]
        # MIP_OBJBND reads exactly 0.0 before Gurobi has computed any real bound
        # (root relaxation not yet done) -- drop that leading placeholder run.
        bnd_rows = [row for row in log if row[2] is not None]
        first_real = next((i for i, row in enumerate(bnd_rows) if row[2] != 0.0), None)
        bnd_rows = bnd_rows[first_real:] if first_real is not None else []
        t_bnd = [row[0] for row in bnd_rows]
        y_bnd = [row[2] for row in bnd_rows]

        if t_inc:
            ax.step(t_inc, y_inc, where="post", color=EXACT, lw=1.4, zorder=4,
                     label="Gurobi incumbent")
            ax.scatter(t_inc, y_inc, s=14, color=EXACT, zorder=5)
        if t_bnd:
            ax.step(t_bnd, y_bnd, where="post", color=BOUND, lw=1.1, ls="--",
                     zorder=3, label="Gurobi dual bound")

        ax.scatter([he["time"]], [he["g"]], s=90, marker="*", color=HEUR,
                   zorder=6, edgecolor="white", linewidth=0.5,
                   label=f"heuristic  (g={he['g']:.4f}, t={he['time']:.1f}s)")
        ax.axhline(he["g"], color=HEUR, lw=0.7, alpha=0.4, ls=":", zorder=1)

        gap = gur["mip_gap"]
        gap_str = f"{gap*100:.2f}%" if gap == gap else "n/a"
        status = "proven optimal" if gap < 1e-5 else f"MIPGap {gap_str} at cutoff"
        ax.set_title(f"λ={lam}\n{status}  ·  {gur['wall']:.0f}s wall",
                     fontsize=9)
        ax.set_xlabel("wall time (s)")
        if ax is axes[0]:
            ax.set_ylabel("objective  g(λ)")
        ax.set_xscale("log")
        ax.grid(True, color="#e3e8e7", lw=0.6)
        ax.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=7,
                   edgecolor="#d3dbd9")

    fig.suptitle(f"Gurobi convergence vs mono-objective heuristic  ·  {name}",
                 fontsize=10.5, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", default="pareto_runs")
    args = ap.parse_args()
    out_dir = Path(args.o)

    by_instance = load_logs()
    if not by_instance:
        print(f"no logs found in {IN_DIR}")
        return
    for name, runs in by_instance.items():
        stem = name.replace(".json", "")
        plot_instance(name, runs, out_dir / f"gurobi_convergence_{stem}.png")


if __name__ == "__main__":
    main()
