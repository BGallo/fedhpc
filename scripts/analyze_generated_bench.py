"""Analyse results/generated_bench/bench.jsonl (scripts/bench_generated.py):
figures and a markdown summary in results/generated_bench/.

    uv run python scripts/analyze_generated_bench.py [bench.jsonl]
"""
from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "generated_bench" / "bench.jsonl"
OUT = SRC.parent

# reference palette (dataviz skill), first three slots: validated all-pairs
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SOLVERS = [("gurobi", "Gurobi (exact)", BLUE), ("dispatch", "Dispatch heuristic", ORANGE),
           ("threshold", "Threshold bursting", AQUA)]
GROUPS = [("A", "fedhpc"), ("A", "pos"), ("B", "fedhpc"), ("B", "pos")]
LOADS = "FHE"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb", "savefig.dpi": 160, "legend.frameon": False,
})


def load(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def gurobi_total(r):
    return r["lex_f1"]["t"] + r["lex_f2"]["t"] + sum(v["gurobi"]["t"] for v in r["lams"].values())


def load_frac(r):
    """Share of on-prem nodes busy at t = 0."""
    return r["n_running"] / r["onprem_nodes"]


def trivial(r):
    """Zero cost already reaches min turnaround: no trade-off to optimise."""
    return r["lex_f1"]["f2"] <= 1e-9


def gaps(r, solver):
    """Per λ: g(solver) − g(Gurobi) in normalised-objective units (0..~1)."""
    return [v[solver]["g"] - v["gurobi"]["g"] for v in r["lams"].values()]


def label(ax, text):
    ax.set_title(text, loc="left", fontsize=9.5, color=INK)


# ── figures ──────────────────────────────────────────────────────────────────

def fig_time_breakdown(rows):
    """Where the wall time goes, per class × workload (log scale)."""
    parts = [("lex_f1", "Exact min turnaround (lex f1)"),
             ("lex_f2", "Exact min cost (lex f2)"),
             ("ws_gurobi", "Exact weighted sum, per λ"),
             ("ws_dispatch", "Dispatch heuristic, per λ"),
             ("baselines", "Threshold sweep (9 θ)")]
    colors = [BLUE, BLUE, BLUE, ORANGE, AQUA]
    hatches = [None, "//", "..", None, None]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.6), sharey=True)
    for ax, (c, w) in zip(axes, GROUPS):
        sub = [r for r in rows if r["cls"] == c and r["workload"] == w]
        if not sub:
            ax.set_visible(False)
            continue
        data = [
            [r["lex_f1"]["t"] for r in sub],
            [r["lex_f2"]["t"] for r in sub],
            [v["gurobi"]["t"] for r in sub for v in r["lams"].values()],
            [v["dispatch"]["t"] for r in sub for v in r["lams"].values()],
            [r["baselines"]["threshold"]["t"] for r in sub],
        ]
        bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=True,
                        flierprops=dict(marker="o", markersize=2.5, alpha=0.5, markeredgewidth=0))
        for patch, col, h in zip(bp["boxes"], colors, hatches):
            patch.set(facecolor=col + "55", edgecolor=col, linewidth=1, hatch=h)
        for k, med in enumerate(bp["medians"]):
            med.set(color=colors[k], linewidth=2)
        for el in ("whiskers", "caps"):
            for ln in bp[el]:
                ln.set(color=INK2, linewidth=0.8)
        ax.set_yscale("log")
        ax.set_xticks(range(1, 6), ["lex f1", "lex f2", "WS\nGurobi", "WS\ndispatch", "thresh."])
        label(ax, f"Class {c} · {w}  (n={len(sub)})")
    axes[0].set_ylabel("wall time per solve (s, log)")
    fig.suptitle("Time to solution by component — the exact min-cost anchor dominates",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_time_breakdown.png")
    plt.close(fig)


def fig_quality_vs_time(rows):
    """Per (instance, λ): extra normalised objective over the exact optimum vs wall time."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, w in zip(axes, ("fedhpc", "pos")):
        sub = [r for r in rows if r["workload"] == w and not trivial(r)]
        if not sub:
            continue
        for key, name, col in SOLVERS:
            xs, ys = [], []
            for r in sub:
                for v in r["lams"].values():
                    xs.append(max(v[key]["t"], 1e-3))
                    ys.append(max(v[key]["g"] - v["gurobi"]["g"], 0) + 1e-4)
            ax.scatter(xs, ys, s=14, color=col, alpha=0.6, edgecolors="#fcfcfb",
                       linewidths=0.5, label=name)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("wall time (s, log)")
        label(ax, f"Workload {w} — {len(sub)} non-trivial instances × 3 λ")
    axes[0].set_ylabel("g(λ) − g*(λ)  (normalised, +1e-4, log)")
    axes[0].legend(loc="upper right")
    fig.suptitle("Quality vs time: Gurobi is exact (bottom row) and usually finishes before the "
                 "dispatch heuristic", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_quality_vs_time.png")
    plt.close(fig)


def fig_load_effects(rows):
    """Cost of min turnaround, turnaround at zero cost and exact solve time vs
    the share of on-prem nodes busy at t = 0."""
    busy = ("on-prem nodes busy at t = 0 (%)", lambda r: 100 * load_frac(r))
    width = ("trade-off width f1^0 − f1^T (slots, log)",
             lambda r: max(r["lex_f2"]["f1"] - r["lex_f1"]["f1"], 1))
    metrics = [
        (busy, lambda r: r["lex_f1"]["f2"], "f2^T — cost of the fastest\nschedule (USD)", False),
        (busy, lambda r: 100 * (r["lex_f2"]["f1"] / r["lex_f1"]["f1"] - 1),
         "f1^0 / f1^T − 1 — turnaround penalty\nof a free schedule (%)", False),
        (width, gurobi_total, "total exact (Gurobi) wall time\nvs trade-off width (s, log)", True),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4))
    for row_ax, w in zip(axes, ("fedhpc", "pos")):
        for ax, ((xlab, xf), fn, title, logy) in zip(row_ax, metrics):
            for c, col in (("A", BLUE), ("B", ORANGE)):
                sub = [r for r in rows if r["workload"] == w and r["cls"] == c]
                ax.scatter([xf(r) for r in sub], [fn(r) for r in sub], s=14,
                           color=col, alpha=0.65, edgecolors="#fcfcfb", linewidths=0.5,
                           label=f"Class {c}")
            if logy and ax.has_data():
                ax.set_yscale("log"); ax.set_xscale("log")
            ax.set_xlabel(xlab)
            label(ax, f"{w} · {title}")
    axes[0][0].legend(loc="upper left")
    fig.suptitle("Initial load sets the cost of speed; exact solve time follows the width of the "
                 "trade-off",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_load_effects.png")
    plt.close(fig)


def fig_class_a_grid(rows):
    """Class A heatmaps: f2^T and lex-f2 time over (count, load) of both types."""
    for w in ("fedhpc", "pos"):
        sub = {(tuple(r["counts"]), r["loads"]): r for r in rows
               if r["cls"] == "A" and r["workload"] == w}
        if not sub:
            continue
        counts = sorted({k[0][0] for k in sub})
        keys = [(n, l) for n in counts for l in LOADS]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        for ax, (fn, title, fmt, cmap) in zip(axes, [
                (lambda r: r["lex_f1"]["f2"], "f2^T — cost of the fastest schedule (USD)", "{:.0f}", "Blues"),
                (lambda r: r["lex_f2"]["t"], "exact min-cost solve time (s)", "{:.0f}", "Oranges")]):
            M = np.full((len(keys), len(keys)), np.nan)
            for i, (n0, l0) in enumerate(keys):
                for j, (n1, l1) in enumerate(keys):
                    r = sub.get(((n0, n1), l0 + l1))
                    if r:
                        M[i, j] = fn(r)
            im = ax.imshow(M, cmap=cmap, aspect="auto")
            ax.grid(False)
            tick = [f"{n}{l}" for n, l in keys]
            ax.set_xticks(range(len(keys)), tick, rotation=90, fontsize=7)
            ax.set_yticks(range(len(keys)), tick, fontsize=7)
            ax.set_xlabel("c64 type: count + load (F/H/E)")
            ax.set_ylabel("c40 type: count + load")
            vmax = np.nanmax(M)
            for i in range(len(keys)):
                for j in range(len(keys)):
                    if not np.isnan(M[i, j]):
                        ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=5.5,
                                color="#fcfcfb" if M[i, j] > 0.55 * vmax else INK)
            label(ax, title)
            fig.colorbar(im, ax=ax, shrink=0.8)
        fig.suptitle(f"Class A · {w}: every count × load combination",
                     x=0.01, ha="left", fontsize=11, color=INK)
        fig.tight_layout()
        fig.savefig(OUT / f"fig4_classA_grid_{w}.png")
        plt.close(fig)


def fig_heuristic_gap_by_lambda(rows):
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    lams = sorted(rows[0]["lams"], key=float)
    pos = 0
    ticks, names = [], []
    for c, w in GROUPS:
        sub = [r for r in rows if r["cls"] == c and r["workload"] == w and not trivial(r)]
        for k, lam in enumerate(lams):
            for off, (key, _, col) in zip((-0.18, 0.18), SOLVERS[1:]):
                vals = [100 * (r["lams"][lam][key]["g"] - r["lams"][lam]["gurobi"]["g"])
                        / max(r["lams"][lam]["gurobi"]["g"], 1e-9) for r in sub]
                if vals:
                    b = ax.boxplot([vals], positions=[pos + off], widths=0.3, patch_artist=True,
                                   showfliers=False)
                    b["boxes"][0].set(facecolor=col + "55", edgecolor=col)
                    b["medians"][0].set(color=col, linewidth=2)
            ticks.append(pos); names.append(f"{c}·{w}\nλ={lam}")
            pos += 1
        pos += 0.6
    ax.set_xticks(ticks, names, fontsize=6.5)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylabel("gap to exact optimum (% of g*, symlog)")
    ax.plot([], [], color=ORANGE, lw=6, alpha=0.5, label="Dispatch heuristic")
    ax.plot([], [], color=AQUA, lw=6, alpha=0.5, label="Threshold bursting (best θ)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8)
    label(ax, "Heuristic gap to the exact weighted-sum optimum, by λ")
    fig.tight_layout()
    fig.savefig(OUT / "fig5_gap_by_lambda.png")
    plt.close(fig)


# ── summary ──────────────────────────────────────────────────────────────────

def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else float("nan")


def summary(rows):
    out = ["# Generated-instance benchmark", "",
           f"{len(rows)} instances; Gurobi Threads=4, TimeLimit 600 s, MIPGap 0; "
           "dispatch heuristic n_threads=4; 4 instances in parallel.", ""]
    statuses = defaultdict(int)
    for r in rows:
        statuses[r["lex_f1"]["status"]] += 1
        statuses[r["lex_f2"]["status"]] += 1
        for v in r["lams"].values():
            statuses[v["gurobi"]["status"]] += 1
    out += [f"Gurobi solve statuses: {dict(statuses)}", ""]

    out += ["## Time (s): median [p90] per solve", "",
            "| class·workload | n | trivial | lex f1 | lex f2 | WS Gurobi | WS dispatch | exact total | dispatch total |",
            "|---|---|---|---|---|---|---|---|---|"]
    for c, w in GROUPS:
        sub = [r for r in rows if r["cls"] == c and r["workload"] == w]
        if not sub:
            continue
        cell = lambda xs: f"{st.median(xs):.1f} [{q(xs, .9):.1f}]"  # noqa: E731
        out.append("| " + " | ".join([
            f"{c}·{w}", str(len(sub)), str(sum(map(trivial, sub))),
            cell([r["lex_f1"]["t"] for r in sub]), cell([r["lex_f2"]["t"] for r in sub]),
            cell([v["gurobi"]["t"] for r in sub for v in r["lams"].values()]),
            cell([v["dispatch"]["t"] for r in sub for v in r["lams"].values()]),
            cell([gurobi_total(r) for r in sub]),
            cell([sum(v["dispatch"]["t"] for v in r["lams"].values()) for r in sub]),
        ]) + " |")

    out += ["", "## Weighted-sum quality: gap to exact g* (non-trivial instances)", "",
            "| class·workload | λ | dispatch median gap % | dispatch p90 % | dispatch optimal (≤1e-6) | threshold median % | WS Gurobi faster than dispatch |",
            "|---|---|---|---|---|---|---|"]
    for c, w in GROUPS:
        sub = [r for r in rows if r["cls"] == c and r["workload"] == w and not trivial(r)]
        if not sub:
            continue
        for lam in sorted(sub[0]["lams"], key=float):
            rel = lambda key: [100 * (r["lams"][lam][key]["g"] - r["lams"][lam]["gurobi"]["g"])  # noqa: E731
                               / max(r["lams"][lam]["gurobi"]["g"], 1e-9) for r in sub]
            d = rel("dispatch")
            hit = sum(r["lams"][lam]["dispatch"]["g"] - r["lams"][lam]["gurobi"]["g"] <= 1e-6 for r in sub)
            faster = sum(r["lams"][lam]["gurobi"]["t"] < r["lams"][lam]["dispatch"]["t"] for r in sub)
            out.append(f"| {c}·{w} | {lam} | {st.median(d):.2f} | {q(d, .9):.2f} | {hit}/{len(sub)} | "
                       f"{st.median(rel('threshold')):.1f} | {faster}/{len(sub)} |")

    out += ["", "## Anchors vs initial load (median over instances)", "",
            "| class·workload | load | n | f2^T (USD) | f1^0/f1^T − 1 (%) | lex f2 time (s) | exact total (s) |",
            "|---|---|---|---|---|---|---|"]
    for c, w in GROUPS:
        for name, lo, hi in (("≤25 %", -1, .25), ("25–60 %", .25, .6), (">60 %", .6, 2)):
            sub = [r for r in rows if r["cls"] == c and r["workload"] == w and lo < load_frac(r) <= hi]
            if sub:
                out.append(f"| {c}·{w} | {name} | {len(sub)} | {st.median(r['lex_f1']['f2'] for r in sub):.0f} | "
                           f"{st.median(100 * (r['lex_f2']['f1'] / r['lex_f1']['f1'] - 1) for r in sub):.1f} | "
                           f"{st.median(r['lex_f2']['t'] for r in sub):.1f} | {st.median(gurobi_total(r) for r in sub):.1f} |")

    # correlation of exact time with load and size
    out += ["", "## What drives exact solve time (Spearman ρ with lex-f2 time)", ""]
    def rank(xs):
        o = sorted(range(len(xs)), key=lambda i: xs[i]); rk = [0] * len(xs)
        for k, i in enumerate(o):
            rk[i] = k
        return rk
    for w in ("fedhpc", "pos"):
        sub = [r for r in rows if r["workload"] == w]
        if len(sub) < 3:
            continue
        t = rank([r["lex_f2"]["t"] for r in sub])
        for nm, fn in (("busy share at t=0", load_frac), ("on-prem nodes", lambda r: r["onprem_nodes"]),
                       ("running jobs", lambda r: r["n_running"]),
                       ("f1^0 − f1^T", lambda r: r["lex_f2"]["f1"] - r["lex_f1"]["f1"])):
            x = rank([fn(r) for r in sub])
            rho = float(np.corrcoef(x, t)[0, 1])
            out.append(f"- {w}: {nm}: ρ = {rho:+.2f}")
    (OUT / "summary.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


def main():
    rows = load(SRC)
    fig_time_breakdown(rows)
    fig_quality_vs_time(rows)
    fig_load_effects(rows)
    fig_class_a_grid(rows)
    fig_heuristic_gap_by_lambda(rows)
    summary(rows)


if __name__ == "__main__":
    main()
