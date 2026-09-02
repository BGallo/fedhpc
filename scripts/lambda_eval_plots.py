"""Render cost/turnaround frontier plots from lambda_eval.py output.

Reads results/lambda_eval/<stem>_solutions.json and writes
results/lambda_eval/<stem>_frontier.png — the exact/heuristic anchors, the
threshold-bursting sweep curve, and the Gurobi weighted-sum points at each
lambda, in (f1 turnaround, f2 cost) space.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results/lambda_eval")


def _load(stem: str) -> dict:
    return json.loads((OUT / f"{stem}_solutions.json").read_text())


def plot(stem: str) -> Path:
    d = _load(stem)
    recs = d["records"]
    ss = d["slot_size_seconds"]

    gur = [(r["lam"], r["f1"], r["f2"], r.get("mip_gap")) for r in recs if r["method"] == "gurobi_weighted"]
    thr = sorted(
        ((float(r["key"].split("=")[1]), r["f1"], r["f2"]) for r in recs if r["method"] == "threshold_bursting"),
        key=lambda x: x[0],
    )
    onp = next((r for r in recs if r["method"] == "onprem_only_spt"), None)
    grd = next((r for r in recs if r["method"] == "greedy_earliest_completion"), None)
    rp = d["reference_points"]

    fig, ax = plt.subplots(figsize=(8.5, 6))

    # turnaround axis in hours (f1 is in slot units)
    hx = ss / 3600.0
    if thr:
        xs = [t[1] * hx for t in thr]
        ys = [t[2] for t in thr]
        ax.plot(xs, ys, "-o", color="#888", ms=4, lw=1.2, label="threshold bursting sweep", zorder=2)
        for th, f1, f2 in thr:
            if th in (0, 1, 2, 4, 8) or th == thr[-1][0]:
                ax.annotate(f"θ={th:g}", (f1 * hx, f2), fontsize=7, color="#555",
                            xytext=(3, 3), textcoords="offset points")

    if grd:
        ax.scatter([grd["f1"] * hx], [grd["f2"]], marker="s", s=90, color="#F57C00",
                   edgecolor="black", zorder=4, label="greedy earliest-completion")
    if onp:
        ax.scatter([onp["f1"] * hx], [onp["f2"]], marker="D", s=90, color="#1976D2",
                   edgecolor="black", zorder=4, label="on-prem only (SPT)")

    for lam, f1, f2, gap in sorted(gur):
        ax.scatter([f1 * hx], [f2], marker="*", s=260, color="#C62828",
                   edgecolor="black", zorder=5)
        tag = f"  λ={lam}" + (f" (gap {gap*100:.1f}%)" if gap and gap == gap and gap > 1e-4 else "")
        ax.annotate(tag, (f1 * hx, f2), fontsize=9, fontweight="bold", color="#C62828",
                    xytext=(6, 0), textcoords="offset points", va="center")
    ax.scatter([], [], marker="*", s=200, color="#C62828", edgecolor="black",
               label="Gurobi weighted-sum")

    # anchor guides
    ax.axvline(rp["f1_T"] * hx, ls="--", color="green", alpha=0.5, lw=1)
    ax.text(rp["f1_T"] * hx, ax.get_ylim()[1], " f1_T (min turnaround)", rotation=90,
            va="top", ha="left", fontsize=7, color="green")
    ax.axvline(rp["f1_0"] * hx, ls="--", color="purple", alpha=0.5, lw=1)
    ax.text(rp["f1_0"] * hx, ax.get_ylim()[1], " f1_0 (min turnaround @ $0)", rotation=90,
            va="top", ha="left", fontsize=7, color="purple")

    ax.set_xlabel("f1 — total turnaround (hours, Σ over scheduled jobs)")
    ax.set_ylabel("f2 — total monetary cost ($)")
    ax.set_title(f"{stem}\ncost / turnaround trade-off — Gurobi vs offline baselines")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = OUT / f"{stem}_frontier.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> None:
    stems = sys.argv[1:] or [
        "fedhpc_known_runtime_offline_10min",
        "pos_congestion_known_runtime_offline_10min",
    ]
    for s in stems:
        if (OUT / f"{s}_solutions.json").exists():
            print(f"  → {plot(s)}")
        else:
            print(f"  (skip {s}: no solutions.json)")


if __name__ == "__main__":
    main()
