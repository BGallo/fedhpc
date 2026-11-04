"""Analyse a scripts/bench_generated.py run: figures and a markdown summary in
results/generated_bench/<run>/.

The run's settings (command, Gurobi parameters, workers, hardware, code
state) are read from the sidecar ``<run>.setup.json`` next to the jsonl and
copied into the summary; without it the summary says so.

--dispatch replaces the run's own dispatch results (g, t) by those of
scripts/dispatch_validate.py records (same instances, reference points and
λ), e.g. a later heuristic version; the summary names the files and the
configuration they were run with (--dispatch-cfg).

    uv run python scripts/analyze_generated_bench.py [results/generated_bench/bench_io_noreduc.jsonl]
        [--dispatch V.jsonl [V2.jsonl ...] --dispatch-cfg "K=64, ..."]
"""
from __future__ import annotations

import argparse
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
_ap = argparse.ArgumentParser()
_ap.add_argument("src", nargs="?", type=Path,
                 default=ROOT / "results" / "generated_bench" / "bench_io_noreduc.jsonl")
_ap.add_argument("--dispatch", nargs="+", type=Path, default=[])
_ap.add_argument("--dispatch-cfg", default="")
ARGS = _ap.parse_args()
SRC = ARGS.src
SETUP = SRC.with_suffix(".setup.json")
OUT = SRC.parent / SRC.stem

# reference palette (dataviz skill), first three slots: validated all-pairs
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# ordered levels (count 100/300/500, io L/M/H): one hue, light → dark, plus markers
SEQ = ("#9cc3ef", "#4f8fdc", "#1b4f99")
MARKS = ("o", "s", "^")
SOLVERS = [("gurobi", "Gurobi (exact)", BLUE), ("dispatch", "Dispatch heuristic", ORANGE),
           ("threshold", "Threshold bursting", AQUA)]
GROUPS = [("A", "fedhpc"), ("A", "pos"), ("B", "fedhpc"), ("B", "pos")]
LOADS = "FHE"
LOAD_NAMES = {"F": "free", "H": "half", "E": "full"}
IOS = "LMH"
COUNTS = (100, 300, 500)

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb", "savefig.dpi": 160, "legend.frameon": False,
})


def load(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def group(rows, c, w, nontrivial=False):
    return [r for r in rows if r["cls"] == c and r["workload"] == w and not (nontrivial and trivial(r))]


def gurobi_total(r):
    return r["lex_f1"]["t"] + r["lex_f2"]["t"] + sum(v["gurobi"]["t"] for v in r["lams"].values())


def load_frac(r):
    """Share of on-prem nodes busy at t = 0."""
    return r["n_running"] / r["onprem_nodes"]


def trivial(r):
    """Zero cost already reaches min turnaround: no trade-off to optimise."""
    return r["lex_f1"]["f2"] <= 1e-9


def penalty(r):
    """Turnaround penalty of the best free schedule, f1^0 / f1^T − 1 (%)."""
    return 100 * (r["lex_f2"]["f1"] / r["lex_f1"]["f1"] - 1)


def rel_gap(r, lam, solver):
    """g(solver) above the exact g*(λ), in % of g*."""
    v = r["lams"][lam]
    return 100 * (v[solver]["g"] - v["gurobi"]["g"]) / max(v["gurobi"]["g"], 1e-9)


def label(ax, text):
    ax.set_title(text, loc="left", fontsize=9.5, color=INK)


def boxes(ax, data, positions, color, width=0.6, hatch=None):
    bp = ax.boxplot(data, positions=positions, widths=width, patch_artist=True, showfliers=True,
                    flierprops=dict(marker="o", markersize=3, alpha=0.7, markeredgewidth=0,
                                    markerfacecolor=color))
    for patch in bp["boxes"]:
        patch.set(facecolor=color + "55", edgecolor=color, linewidth=1, hatch=hatch)
    for med in bp["medians"]:
        med.set(color=color, linewidth=2)
    for el in ("whiskers", "caps"):
        for ln in bp[el]:
            ln.set(color=INK2, linewidth=0.8)
    return bp


# ── figures ──────────────────────────────────────────────────────────────────

def fig_time_breakdown(rows):
    """Where the wall time goes, per class × workload (log scale)."""
    colors = [BLUE, BLUE, BLUE, ORANGE, AQUA]
    hatches = [None, "//", "..", None, None]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.6), sharey=True)
    for ax, (c, w) in zip(axes, GROUPS):
        sub = group(rows, c, w)
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
        for k, (d, col, h) in enumerate(zip(data, colors, hatches), 1):
            boxes(ax, [d], [k], col, hatch=h)
        ax.axhline(600, color=INK2, linewidth=0.8, linestyle="--")
        ax.set_yscale("log")
        ax.set_xticks(range(1, 6), ["lex f1", "lex f2", "WS\nGurobi", "WS\ndispatch", "thresh."])
        label(ax, f"Class {c} · {w}  (n={len(sub)})")
    axes[0].set_ylabel("wall time per solve (s, log)")
    axes[-1].text(5.4, 600, "Gurobi\ntime limit", fontsize=7, color=INK2, va="bottom", ha="right")
    fig.suptitle("Time to solution by component", x=0.01, ha="left", fontsize=11, color=INK)
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
        ax.set_xlabel("wall time per weighted-sum solve (s, log)")
        label(ax, f"Workload {w} — {len(sub)} non-trivial instances × 3 λ")
    axes[0].set_ylabel("g(λ) − g*(λ)  (normalised, +1e-4, log)")
    axes[0].legend(loc="upper right")
    fig.suptitle("Quality vs time per (instance, λ); Gurobi sits on the bottom row by definition",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_quality_vs_time.png")
    plt.close(fig)


def fig_count_load(rows):
    """Count × load: anchors and exact time, median over the three io levels."""
    metrics = [
        (lambda r: r["lex_f1"]["f2"], "f2^T — cost of the fastest\nschedule (USD)", False),
        (penalty, "f1^0 / f1^T − 1 — turnaround\npenalty of a free schedule (%)", False),
        (gurobi_total, "exact (Gurobi) wall time,\nanchors + 3 λ (s, log)", True),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(12, 8), sharex=True)
    x = np.arange(len(LOADS))
    for col, (c, w) in enumerate(GROUPS):
        sub = group(rows, c, w)
        for row, (fn, title, logy) in enumerate(metrics):
            ax = axes[row][col]
            for n, shade, mk in zip(COUNTS, SEQ, MARKS):
                ys = [st.median(fn(r) for r in sub if r["count"] == n and r["load"] == l) for l in LOADS]
                ax.plot(x, ys, color=shade, marker=mk, markersize=5, linewidth=2,
                        label=f"{n} nodes / type")
            if logy:
                ax.set_yscale("log")
            if row == 0:
                label(ax, f"Class {c} · {w}")
            if col == 0:
                ax.set_ylabel(title)
            ax.set_xticks(x, [f"{l}\n{LOAD_NAMES[l]}" for l in LOADS])
    axes[0][0].legend(loc="upper left", fontsize=7.5)
    for ax in axes[-1]:
        ax.set_xlabel("load at t = 0")
    fig.suptitle("Node count and initial load: cost of speed, price of free, and exact solve time "
                 "(median over io L/M/H)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_count_load.png")
    plt.close(fig)


def _by_cell(rows):
    """{(cls, workload, count, load): {io: record}} — the io triples share jobs and capacity."""
    cells = defaultdict(dict)
    for r in rows:
        cells[(r["cls"], r["workload"], r["count"], r["load"])][r["io"]] = r
    return cells


def fig_io(rows):
    """io level: anchors relative to level L (paired cells) and the dispatch gap."""
    cells = _by_cell(rows)
    panels = [
        ("f2^T at level / f2^T at L", lambda g, io: g[io]["lex_f1"]["f2"] / g["L"]["lex_f1"]["f2"]),
        ("f1^T at level / f1^T at L", lambda g, io: g[io]["lex_f1"]["f1"] / g["L"]["lex_f1"]["f1"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))
    for ax, (title, fn) in zip(axes[:2], panels):
        for k, (c, w) in enumerate(GROUPS):
            gs = [g for key, g in cells.items() if key[:2] == (c, w) and not trivial(g["L"])]
            for j, (io, shade) in enumerate(zip(IOS, SEQ)):
                boxes(ax, [[fn(g, io) for g in gs]], [k + (j - 1) * 0.27], shade, width=0.22)
        ax.set_xticks(range(len(GROUPS)), [f"{c}·{w}" for c, w in GROUPS])
        ax.axhline(1, color=INK2, linewidth=0.8)
        label(ax, title)
    ax = axes[2]
    for k, (c, w) in enumerate(GROUPS):
        sub = group(rows, c, w, nontrivial=True)
        for j, (io, shade) in enumerate(zip(IOS, SEQ)):
            vals = [rel_gap(r, lam, "dispatch") for r in sub if r["io"] == io for lam in r["lams"]]
            if vals:
                boxes(ax, [vals], [k + (j - 1) * 0.27], shade, width=0.22)
    ax.set_xticks(range(len(GROUPS)), [f"{c}·{w}" for c, w in GROUPS])
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylim(bottom=0)
    label(ax, "dispatch gap to g* (% of g*, symlog, all λ)")
    for io, shade in zip(IOS, SEQ):
        axes[0].plot([], [], color=shade, lw=6, alpha=0.7, label=f"io {io}")
    axes[0].legend(loc="upper left", fontsize=7.5)
    fig.suptitle("Result-data level: what egress adds to the fastest schedule, and to the heuristic's gap "
                 "(non-trivial cells)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_io_effects.png")
    plt.close(fig)


def fig_heuristic_gap_by_lambda(rows):
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    lams = sorted(rows[0]["lams"], key=float)
    pos = 0
    ticks, names = [], []
    for c, w in GROUPS:
        sub = group(rows, c, w, nontrivial=True)
        for lam in lams:
            for off, (key, _, col) in zip((-0.18, 0.18), SOLVERS[1:]):
                vals = [rel_gap(r, lam, key) for r in sub]
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
    label(ax, "Heuristic gap to the exact weighted-sum optimum, by λ (non-trivial instances)")
    fig.tight_layout()
    fig.savefig(OUT / "fig5_gap_by_lambda.png")
    plt.close(fig)


# ── summary ──────────────────────────────────────────────────────────────────

def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else float("nan")


def override_dispatch(rows):
    """Replace each (instance, λ) dispatch g / t by the --dispatch records; all must be covered."""
    if not ARGS.dispatch:
        return
    new = {}
    for p in ARGS.dispatch:
        for v in load(p):
            for lam, x in v["lams"].items():
                new[(v["name"], lam)] = x
    for r in rows:
        for lam, v in r["lams"].items():
            x = new.pop((r["name"], lam))         # KeyError: a cell the files do not cover
            assert abs(x["gstar"] - v["gurobi"]["g"]) <= 1e-12 * max(1.0, abs(x["gstar"]))
            v["dispatch"] = dict(g=x["g"], t=x["t"])
    assert not new, f"{len(new)} --dispatch cells not in {SRC.name}"


def dispatch_lines():
    if not ARGS.dispatch:
        return []
    return ["## Dispatch results", "",
            "The dispatch columns are **not** this run's own dispatch results. They are replaced by "
            + ", ".join(f"`{p.relative_to(ROOT) if p.is_absolute() else p}`" for p in ARGS.dispatch)
            + " (`scripts/dispatch_validate.py`: the same instances, reference points and λ, "
            "3 instances in parallel × 4 threads, one seed)"
            + (f", run with {ARGS.dispatch_cfg}" if ARGS.dispatch_cfg else "") + ". "
            "The Dispatch line under Setup is the configuration of the original run.", ""]


def setup_lines():
    if not SETUP.exists():
        return [f"_No {SETUP.name}: run settings unknown._", ""]
    s = json.loads(SETUP.read_text())
    cfg, hw, sw, code = s["settings"], s["hardware"], s["software"], s["code"]
    return [
        "## Setup", "",
        "Full description: `EXPERIMENT_SETUP.md` in this folder.", "",
        f"- Command: `{s['command']}`",
        f"- {s['instances']} instances, {s['failures']} failures; {s['started']} → {s['finished']} "
        f"({s['wall_hours']} h wall).",
        f"- {cfg['workers']} instances in parallel, {cfg['threads_per_worker']} threads each; "
        f"Gurobi {cfg['gurobi']}; λ ∈ {cfg['lambdas']}; θ sweep {cfg['threshold_thetas_slots']} slots.",
        f"- Formulation: `{cfg['formulation']}`.",
        f"- Dispatch: `{cfg['dispatch']['function']}` with "
        + ", ".join(f"{k}={v}" for k, v in cfg["dispatch"].items() if k != "function") + ".",
        f"- Hardware: {hw['cpu']}, {hw['memory_gb']} GB RAM, pinning: {hw['cpu_pinning']}; "
        f"{hw['concurrent_load']}.",
        f"- Software: {sw['os']}; {sw['python']}; gurobipy {sw['gurobipy']}; numpy {sw['numpy']}.",
        f"- Code: fedhpc {code['base_commit'][:7]} + `{Path(code['uncommitted_patch'].split()[0]).name}` "
        f"(sha256 {code['uncommitted_patch_sha256'][:12]}…).",
        f"- {s['reproducibility_check']}.", "",
    ]


def summary(rows):
    out = [f"# Generated-instance benchmark — {SRC.stem}", ""] + setup_lines() + dispatch_lines()

    statuses = defaultdict(int)
    limited = []
    for r in rows:
        for k in ("lex_f1", "lex_f2"):
            statuses[r[k]["status"]] += 1
            if r[k]["status"] != "optimal":
                limited.append(f"{r['name']} {k}")
        for lam, v in r["lams"].items():
            statuses[v["gurobi"]["status"]] += 1
            if v["gurobi"]["status"] != "optimal":
                limited.append(f"{r['name']} λ={lam}")
    out += ["## Gurobi statuses", "",
            f"{dict(statuses)} over {len(rows) * (2 + len(rows[0]['lams']))} solves. "
            "`feasible` = time limit reached with an incumbent (not proven optimal):", ""]
    out += [f"- {x}" for x in limited] + [""]
    out += ["A non-optimal lex anchor shifts the normalisation (f1^T, f2^T or f1^0) for every "
            "method alike. A non-optimal weighted-sum incumbent is ≥ g*, so the heuristic gap "
            "reported for that (instance, λ) is a lower bound on the true one.", ""]

    triv = [r for r in rows if trivial(r)]
    out += ["## Trivial instances (f2^T = 0: the fastest schedule is free, front is one point)", "",
            f"{len(triv)} of {len(rows)}; Gurobi and dispatch reach g = 0 on "
            f"{sum(all(v['dispatch']['g'] <= 1e-9 for v in r['lams'].values()) for r in triv)} of them. "
            "Excluded from every quality statistic below.", ""]
    out += ["| class·workload | count | load | io levels |", "|---|---|---|---|"]
    tcells = defaultdict(list)
    for r in triv:
        tcells[(r["cls"], r["workload"], r["count"], r["load"])].append(r["io"])
    for (c, w, n, l), ios in sorted(tcells.items()):
        out.append(f"| {c}·{w} | {n} | {l} | {''.join(i for i in IOS if i in ios)} |")

    out += ["", "## Time (s): median [p90] per solve", "",
            "| class·workload | n | trivial | lex f1 | lex f2 | WS Gurobi | WS dispatch | exact total | dispatch total |",
            "|---|---|---|---|---|---|---|---|---|"]
    for c, w in GROUPS:
        sub = group(rows, c, w)
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

    out += ["", "## Weighted-sum quality: gap to exact g* (% of g*, non-trivial instances)", "",
            "| class·workload | n | λ | dispatch median | p90 | max | dispatch = g* (≤1e-6) | "
            "threshold median | WS Gurobi faster than dispatch |",
            "|---|---|---|---|---|---|---|---|---|"]
    for c, w in GROUPS:
        sub = group(rows, c, w, nontrivial=True)
        if not sub:
            continue
        for lam in sorted(sub[0]["lams"], key=float):
            d = [rel_gap(r, lam, "dispatch") for r in sub]
            hit = sum(r["lams"][lam]["dispatch"]["g"] - r["lams"][lam]["gurobi"]["g"] <= 1e-6 for r in sub)
            faster = sum(r["lams"][lam]["gurobi"]["t"] < r["lams"][lam]["dispatch"]["t"] for r in sub)
            out.append(f"| {c}·{w} | {len(sub)} | {lam} | {st.median(d):.1f} | {q(d, .9):.1f} | "
                       f"{max(d):.1f} | {hit}/{len(sub)} | "
                       f"{st.median(rel_gap(r, lam, 'threshold') for r in sub):.1f} | {faster}/{len(sub)} |")

    out += ["", "## Factor effects (median per cell; gap over non-trivial instances and all λ)", "",
            "| class·workload | factor | level | n | trivial | f2^T (USD) | f1^0/f1^T − 1 (%) | "
            "dispatch gap (%) | exact total (s) |",
            "|---|---|---|---|---|---|---|---|---|"]
    for c, w in GROUPS:
        sub = group(rows, c, w)
        for fac, levels in (("count", COUNTS), ("load", LOADS), ("io", IOS)):
            for lv in levels:
                s = [r for r in sub if r[fac] == lv]
                nt = [r for r in s if not trivial(r)]
                gap = [rel_gap(r, lam, "dispatch") for r in nt for lam in r["lams"]]
                out.append(f"| {c}·{w} | {fac} | {lv} | {len(s)} | {len(s) - len(nt)} | "
                           f"{st.median(r['lex_f1']['f2'] for r in s):.0f} | "
                           f"{st.median(map(penalty, s)):.1f} | "
                           f"{st.median(gap) if gap else float('nan'):.1f} | "
                           f"{st.median(map(gurobi_total, s)):.0f} |")

    cells = _by_cell(rows)
    out += ["", "## io level, paired over (class, workload, count, load) cells (non-trivial)", "",
            "| class·workload | cells | total ι at M / L | at H / L | f2^T M / L | f2^T H / L | "
            "f1^T H / L | f1^0 H / L |", "|---|---|---|---|---|---|---|---|"]
    for c, w in GROUPS:
        gs = [g for key, g in cells.items() if key[:2] == (c, w) and not trivial(g["L"])]
        if not gs:
            continue
        ratio = lambda io, fn: st.median(fn(g[io]) / fn(g["L"]) for g in gs)  # noqa: E731
        out.append(f"| {c}·{w} | {len(gs)} | {ratio('M', lambda r: r['io_gb']):.1f} | "
                   f"{ratio('H', lambda r: r['io_gb']):.1f} | "
                   f"{ratio('M', lambda r: r['lex_f1']['f2']):.2f} | {ratio('H', lambda r: r['lex_f1']['f2']):.2f} | "
                   f"{ratio('H', lambda r: r['lex_f1']['f1']):.3f} | {ratio('H', lambda r: r['lex_f2']['f1']):.3f} |")

    out += ["", "## What drives exact solve time (Spearman ρ with total Gurobi time)", ""]
    def rank(xs):
        o = sorted(range(len(xs)), key=lambda i: xs[i]); rk = [0] * len(xs)
        for k, i in enumerate(o):
            rk[i] = k
        return rk
    for w in ("fedhpc", "pos"):
        sub = [r for r in rows if r["workload"] == w]
        if len(sub) < 3:
            continue
        t = rank([gurobi_total(r) for r in sub])
        for nm, fn in (("nodes per type", lambda r: r["count"]), ("busy share at t=0", load_frac),
                       ("result volume ι (GB)", lambda r: r["io_gb"]),
                       ("f1^0 − f1^T", lambda r: r["lex_f2"]["f1"] - r["lex_f1"]["f1"])):
            x = rank([fn(r) for r in sub])
            rho = float(np.corrcoef(x, t)[0, 1])
            out.append(f"- {w}: {nm}: ρ = {rho:+.2f}")
    (OUT / "summary.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


def main():
    rows = load(SRC)
    override_dispatch(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    fig_time_breakdown(rows)
    fig_quality_vs_time(rows)
    fig_count_load(rows)
    fig_io(rows)
    fig_heuristic_gap_by_lambda(rows)
    summary(rows)


if __name__ == "__main__":
    main()
