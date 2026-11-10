"""Compare strategies on the generated instances: by default every non-trivial
instance × λ (tune + test split of dispatch_study/split.json, 87 × 3 = 261 cells;
trivial instances have g* = 0, so no % gap), or with --cells test only the held-out
test split (65 × 3 = 195 cells).

Strategies: Gurobi (bench g*, its wall time), the dispatch_study final arms
(validate_F_<arm>_s{42,1,7}.jsonl, seed-mean gap and time per cell), and the
bench baselines threshold bursting (ONE θ for every cell: the swept θ with the
lowest mean gap over the cells shown) and greedy earliest completion (its f1/f2 re-normalised with the instance's lex anchors).

Gap = 100·(g − g*)/g*. Figures (PDF + 600-dpi PNG, sized for a double-column
paper: COL / FULL inches wide, fonts at print size) go to dispatch_study/final/figures/.

    uv run python scripts/plot_strategy_comparison.py [--heur gamid ...] [--out DIR]

--heur keeps only the listed heuristic arms (colours and markers stay fixed
per strategy); --out defaults to figures/ for all arms, figures_<arms>/ otherwise.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GB = ROOT / "results" / "generated_bench"
D = GB / "dispatch_study"
OUT = D / "final" / "figures"
SEEDS = (42, 1, 7)
LAMS = ("0.25", "0.5", "0.75")
GROUPS = [("A", "fedhpc"), ("A", "pos"), ("B", "fedhpc"), ("B", "pos")]
WNAME = {"fedhpc": "W1", "pos": "W2"}  # display names used in the paper

INK, INK2, GRID, SURF = "#000000", "#3d3c3a", "#e4e3df", "#ffffff"
# paper geometry (double-column, e.g. IEEEtran): one column / full text width, inches
COL, FULL = 3.5, 7.16
# key, label, colour (reference categorical palette, fixed order), marker
STRATS = [
    ("gurobi", "Gurobi (exact)", "#2a78d6", "o"),
    ("ils", "ILS (default)", "#eb6834", "s"),
    ("ilslong", "ILS-long", "#1baf7a", "D"),
    ("gafast", "GA-fast", "#eda100", "^"),
    ("gamid", "GA-mid", "#e87ba4", "v"),
    ("gaqual", "GA-quality", "#008300", "P"),
    ("threshold", "Threshold", "#6250d6", "X"),  # label set once θ is chosen
    ("greedy", "Greedy", "#e34948", "*"),
]
HEUR = ["ils", "ilslong", "gafast", "gamid", "gaqual"]

_ap = argparse.ArgumentParser()
_ap.add_argument("--heur", nargs="+", choices=HEUR, default=HEUR)
_ap.add_argument("--out", type=Path)
_ap.add_argument("--baselines", nargs="*", choices=("threshold", "greedy"),
                 default=["threshold", "greedy"])
_ap.add_argument("--cells", choices=("all", "test"), default="all",
                 help="all = every non-trivial instance × λ (tune + test split); test = test only")
ARGS = _ap.parse_args()
if ARGS.heur != HEUR:
    HEUR = [h for h in HEUR if h in ARGS.heur]
    STRATS = [x for x in STRATS if x[0] not in {"ils", "ilslong", "gafast", "gamid", "gaqual"}
              or x[0] in HEUR]
    OUT = OUT.with_name("figures_" + "_".join(HEUR))
STRATS = [x for x in STRATS if x[0] not in ("threshold", "greedy") or x[0] in ARGS.baselines]
if [h for h in HEUR if h.startswith("ga")] == ["gamid"]:   # the only GA shown: call it "GA"
    STRATS = [(k, "GA" if k == "gamid" else lab, c, m) for k, lab, c, m in STRATS]
if ARGS.cells == "test":
    OUT = OUT.with_name(OUT.name + "_test")
OUT = ARGS.out or OUT
# validate_<tag>_s<seed>.jsonl files per arm: the final runs cover the test split; the tune
# split has the same configuration under a tuning tag (GA-mid only so far)
TUNE_TAGS = {"gamid": {42: "T_sl48p16g120", 1: "T_gamid", 7: "T_gamid"}}

plt.rcParams.update({
    # sizes are final print sizes: figures are saved at COL / FULL width, so do not rescale
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.4,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF,
    "axes.facecolor": SURF, "savefig.dpi": 600, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "legend.frameon": False, "legend.handletextpad": 0.4,
    "legend.columnspacing": 1.0, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.titlepad": 3, "axes.labelpad": 2,
})


def g_of(lam, f1, f2, ref):
    f1_T, f2_T, f1_0 = ref
    if f1_0 <= f1_T + 1e-9 or f2_T <= 1e-9:
        return f2
    return lam * (f1 - f1_T) / (f1_0 - f1_T) + (1 - lam) * f2 / f2_T


def load_cells():
    split = json.loads((D / "split.json").read_text())
    names = set(split["test"]) | (set(split["tune"]) if ARGS.cells == "all" else set())
    cells = {}
    for line in (GB / "bench_io_noreduc.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["name"] not in names:
            continue
        ref = (r["lex_f1"]["f1"], r["lex_f1"]["f2"], r["lex_f2"]["f1"])
        gr = r["baselines"]["greedy"]
        for lam in LAMS:
            v = r["lams"][lam]
            gs = v["gurobi"]["g"]
            gap = lambda g: 100 * (g - gs) / max(gs, 1e-9)  # noqa: E731
            assert abs(g_of(float(lam), v["threshold"]["f1"], v["threshold"]["f2"], ref)
                       - v["threshold"]["g"]) < 1e-9
            cells[(r["name"], lam)] = dict(
                cls=r["cls"], workload=r["workload"], lam=lam, gstar=gs,
                gurobi=dict(gap=0.0, t=v["gurobi"]["t"]),
                thr={p["theta"]: gap(g_of(float(lam), p["f1"], p["f2"], ref))
                     for p in r["baselines"]["threshold"]["points"]},
                thr_t=r["baselines"]["threshold"]["t"] / len(r["baselines"]["threshold"]["points"]),
                greedy=dict(gap=gap(g_of(float(lam), gr["f1"], gr["f2"], ref)), t=gr["t"]))
    # one global θ: the lowest mean gap over all cells (every λ and workload)
    thetas = set.intersection(*(set(x["thr"]) for x in cells.values()))
    best = min(thetas, key=lambda th: st.mean(x["thr"][th] for x in cells.values()))
    for x in cells.values():
        x["threshold"] = dict(gap=x["thr"][best], t=x["thr_t"])
    i = next(i for i, s in enumerate(STRATS) if s[0] == "threshold")
    STRATS[i] = ("threshold", f"Threshold (θ = {best * 10 / 60:g} h)",
                 *STRATS[i][2:])
    print(f"global θ = {best:g} slots; mean gap per θ: "
          + ", ".join(f"{th:g}: {st.mean(x['thr'][th] for x in cells.values()):.1f}"
                      for th in sorted(thetas)))
    for arm in HEUR:
        acc = defaultdict(list)
        for s in SEEDS:
            files = [D / f"validate_F_{arm}_s{s}.jsonl"]
            if ARGS.cells == "all":
                assert arm in TUNE_TAGS, f"{arm}: no tune-split runs; use --cells test"
                files.append(D / f"validate_{TUNE_TAGS[arm][s]}_s{s}.jsonl")
            for line in (l for f in files for l in f.read_text().splitlines()):
                r = json.loads(line)
                for lam, x in r["lams"].items():
                    if x["gap"] is not None:
                        assert abs(x["gstar"] - cells[(r["name"], lam)]["gstar"]) < 1e-12
                        acc[(r["name"], lam)].append((x["gap"], x["t"]))
        assert set(acc) == set(cells), arm
        for k, xs in acc.items():
            assert len(xs) == len(SEEDS), (arm, k)
            cells[k][arm] = dict(gap=st.mean(a for a, _ in xs), t=st.mean(b for _, b in xs))
    return list(cells.values())


def title(ax, text):
    ax.set_title(text, loc="left", color=INK)


def strat(k):
    return next(x for x in STRATS if x[0] == k)


def short(k, tick=False):
    """Display name: drop the parenthetical, keep θ for the threshold (two lines on ticks)."""
    lab = strat(k)[1]
    if k == "threshold":
        return lab.replace(" (", "\n(") if tick else lab
    return lab.split(" (")[0]


def legend_under(fig, axes, handles, labels, **kw):
    """Legend just below the lowest tick label / axis label of the given axes."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    y0 = min(ax.get_tightbbox(r).y0 for ax in axes) / fig.bbox.height
    fig.legend(handles, labels, loc="upper center", ncol=len(handles),
               bbox_to_anchor=(0.5, y0 - 0.01), **kw)


PLAIN = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}")
    plt.close(fig)


def fig_gap_box(cells):
    """Gap distribution per strategy and class·workload (symlog), full text width."""
    keys = [k for k, *_ in STRATS if k != "gurobi"]
    fig, axes = plt.subplots(1, 4, figsize=(FULL, 2.15 if len(keys) <= 3 else 2.45),
                             sharey=True)
    for ax, (c, w) in zip(axes, GROUPS):
        sub = [x for x in cells if x["cls"] == c and x["workload"] == w]
        for i, k in enumerate(keys, 1):
            col = strat(k)[2]
            bp = ax.boxplot([[x[k]["gap"] for x in sub]], positions=[i], widths=0.6,
                            patch_artist=True, whis=(5, 95),
                            flierprops=dict(marker="o", markersize=1.8, markeredgewidth=0,
                                            markerfacecolor=col, alpha=0.8))
            bp["boxes"][0].set(facecolor=col + "40", edgecolor=col, linewidth=0.8)
            bp["medians"][0].set(color=col, linewidth=1.4)
            for el in ("whiskers", "caps"):
                for ln in bp[el]:
                    ln.set(color=INK2, linewidth=0.6)
            ax.plot(i, st.mean(x[k]["gap"] for x in sub), marker="D", markersize=2.6,
                    color=INK, linestyle="", zorder=4)
        ax.set_yscale("symlog", linthresh=1)
        top = max(x[k]["gap"] for x in cells for k in keys)
        tk = [t for t in (0, 1, 3, 10, 30, 100, 300, 1000, 3000) if t <= 3 * top]
        ax.set_yticks(tk, [f"{t:g}" for t in tk])
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlim(0.4, len(keys) + 0.6)
        if len(keys) <= 3:
            ax.set_xticks(range(1, len(keys) + 1), [short(k, tick=True) for k in keys])
        else:
            ax.set_xticks(range(1, len(keys) + 1),
                          [short(k).replace("Threshold (θ = ", "Thresh. (") for k in keys],
                          rotation=45, ha="right", rotation_mode="anchor")
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="x", visible=False)
        title(ax, f"Class {c} · {WNAME[w]} ({len(sub)} cells)")
    axes[0].set_ylabel("gap to optimum (%)")
    fig.tight_layout(w_pad=0.6)
    save(fig, "strategy_gap_by_group")


def fig_tradeoff(cells):
    """Mean gap vs median time per cell, one point per strategy, per workload (full width)."""
    shown = [x for x in STRATS if x[0] not in ("greedy", "threshold")]
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 1.95))
    panels = [("all", cells), ("fedhpc", [x for x in cells if x["workload"] == "fedhpc"]),
              ("pos", [x for x in cells if x["workload"] == "pos"])]
    for ax, (name, sub) in zip(axes, panels):
        for k, lab, col, m in shown:
            t = st.median(x[k]["t"] for x in sub)
            g = st.mean(x[k]["gap"] for x in sub)
            ax.plot(t, g, marker=m, markersize=5.5, color=col, linestyle="",
                    markeredgecolor=SURF, markeredgewidth=0.6, zorder=3)
            if len(HEUR) == 1:
                right = k == "gurobi"
                ax.annotate(f"{short(k)}\n{g:.1f} %, {t:.1f} s", (t, g),
                            xytext=(-6 if right else 6, 0), textcoords="offset points",
                            ha="right" if right else "left", va="center",
                            fontsize=6.5, color=INK, linespacing=1.1)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(PLAIN)
        ax.margins(x=0.3 if len(HEUR) == 1 else 0.1, y=0.15)
        ax.set_xlabel("median time per cell (s)")
        title(ax, f"{'All workloads' if name == 'all' else 'Workload ' + WNAME[name]}"
                  f" ({len(sub)} cells)")
    axes[0].set_ylabel("mean gap to optimum (%)")
    if len(HEUR) > 1:
        hs = [plt.Line2D([], [], marker=m, linestyle="", color=c, markersize=5,
                         markeredgecolor=SURF, markeredgewidth=0.5, label=short(k))
              for k, _, c, m in shown]
        fig.tight_layout(w_pad=0.8)
        legend_under(fig, axes, hs, [h.get_label() for h in hs])
    else:
        fig.tight_layout(w_pad=0.8)
    save(fig, "strategy_tradeoff")


def fig_by_lambda(cells):
    """Mean gap per λ (grouped bars): one column wide for ≤ 2 strategies, else full width."""
    keys = HEUR + [b for b in ("threshold",) if b in ARGS.baselines]
    narrow = len(keys) <= 2
    fig, axes = plt.subplots(1, 2, figsize=(COL if narrow else FULL, 1.9), sharey=False)
    for ax, w in zip(axes, ("fedhpc", "pos")):
        sub = [x for x in cells if x["workload"] == w]
        bw = 0.8 / len(keys)
        for i, k in enumerate(keys):
            ys = [st.mean(x[k]["gap"] for x in sub if x["lam"] == lam) for lam in LAMS]
            xs = [j + (i - (len(keys) - 1) / 2) * bw for j in range(len(LAMS))]
            ax.bar(xs, ys, width=bw * 0.9, color=strat(k)[2], edgecolor=SURF,
                   linewidth=0.4, label=short(k))
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(PLAIN)
        ax.set_xticks(range(len(LAMS)), [f"$\\lambda$={lam}" for lam in LAMS])
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="x", visible=False)
        title(ax, f"Workload {WNAME[w]} ({len(sub) // 3} cells/$\\lambda$)")
    axes[0].set_ylabel("mean gap to optimum (%)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.tight_layout(w_pad=0.5)
    legend_under(fig, axes, h, lab, handlelength=1.2)
    save(fig, "strategy_gap_by_lambda")


def fig_quality_scatter(cells):
    """One point per (instance, λ): time vs gap, log-log, per workload (style of the bench's
    fig2_quality_vs_time). Gaps below FLOOR (incl. Gurobi's 0) sit on the bottom row."""
    FLOOR = 1e-3
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.35), sharey=True)
    order = [k for k, *_ in STRATS if k != "gurobi"] + ["gurobi"]   # Gurobi drawn last
    for ax, w in zip(axes, ("fedhpc", "pos")):
        sub = [x for x in cells if x["workload"] == w]
        for k in order:
            _, lab, col, m = strat(k)
            ax.scatter([max(x[k]["t"], 1e-3) for x in sub],
                       [max(x[k]["gap"], FLOOR) for x in sub],
                       s=16 if m == "*" else 9, marker=m, color=col, alpha=0.75, edgecolors=SURF,
                       linewidths=0.25, zorder=3, label=short(k), clip_on=False)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(PLAIN)
        ax.set_yticks([FLOOR, 1e-2, 1e-1, 1, 10, 100, 1000],
                      ["≈0", "0.01", "0.1", "1", "10", "100", "1000"])
        ax.set_ylim(FLOOR / 1.6, 5000)
        ax.set_xlabel("wall time per cell (s)")
        title(ax, f"Workload {WNAME[w]} ({len(sub) // 3} instances × 3 $\\lambda$)")
    axes[0].set_ylabel("gap to optimum (%)")
    fig.tight_layout(w_pad=0.8)
    keep = [k for k, *_ in STRATS]          # legend in the fixed strategy order
    hs = [plt.Line2D([], [], marker=strat(k)[3], linestyle="", color=strat(k)[2],
                     markersize=5 if strat(k)[3] == "*" else 4, markeredgecolor=SURF, markeredgewidth=0.3)
          for k in keep]
    legend_under(fig, axes, hs, [short(k) for k in keep])
    save(fig, "strategy_quality_vs_time")


def table(cells):
    rows = ["| strategy | mean gap | median | p90 | max | cells ≤ 1 % | median t/cell (s) |",
            "|---|---|---|---|---|---|---|"]
    for k, lab, *_ in STRATS:
        gs = sorted(x[k]["gap"] for x in cells)
        rows.append(f"| {lab} | {st.mean(gs):.2f} | {st.median(gs):.2f} | "
                    f"{gs[int(0.9 * len(gs))]:.1f} | {gs[-1]:.1f} | "
                    f"{100 * sum(g <= 1 for g in gs) / len(gs):.0f} % | "
                    f"{st.median(x[k]['t'] for x in cells):.2f} |")
    (OUT / "strategy_table.md").write_text(
        f"{len(cells)} cells ({ARGS.cells}); heuristics = seed-mean over seeds {SEEDS}; "
        "generated by scripts/plot_strategy_comparison.py\n\n" + "\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    cs = load_cells()
    fig_gap_box(cs)
    fig_tradeoff(cs)
    fig_by_lambda(cs)
    fig_quality_scatter(cs)
    table(cs)
