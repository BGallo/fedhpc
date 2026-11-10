"""Final ILS vs GA (memetic) comparison on the held-out test cells, against
the bench's Gurobi weighted-sum solves (bench_io_noreduc.jsonl: final g* and
wall time t per cell; no incumbent history). Gurobi enters the anytime views as
a step: no solution before t, g* from t on. That is pessimistic for Gurobi
before t (its earlier incumbents were not logged), exact from t on. The bench
timings were measured in another session (same Threads 4).

Inputs: dispatch_study/validate_<arm>_s<seed>.jsonl for every arm × seed (with
anytime traces, see dispatch_validate.py) and the bench. Per cell the
bench Gurobi optimum g* is the reference (gap = 100·(g − g*)/g*).

Reports (markdown on stdout, figures in dispatch_study/final/):
  1. per arm: gap statistics per seed and across seeds, wall time;
  2. paired arm-vs-arm on the seed-mean gap per cell: wins/ties/losses
     (tie ±0.5 pp), exact sign-test p, bootstrap 95 % CI of the mean difference;
  3. anytime: gap of the best-so-far at fixed wall-clock budgets (heuristics keep
     their final schedule after stopping; Gurobi from its incumbents);
  4. time to target gap (≤ 10/5/2/1 % and optimal-equal ≤ 0.01 %);
  5. primal integral (Berthold 2013: primal gap |g−g*|/max(g, g*), 1 before the
     first solution) over [0, T] for T = 10, 60, 600 s, normalised by T;
  6. Gurobi-relative: each heuristic's final gap and time against Gurobi's time
     to optimal and its gap at the heuristic's own finishing time.

    uv run python scripts/ils_vs_ga_final.py --arms F_ils F_ilslong F_gafast F_gaqual --seeds 42 1 7
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parents[1] / "results" / "generated_bench" / "dispatch_study"
BUDGETS = (0.1, 0.3, 1, 3, 10, 30, 100, 600)
TARGETS = (10, 5, 2, 1, 0.01)


def load_arm(tag):
    out = {}
    for line in (D / f"validate_{tag}.jsonl").read_text().splitlines():
        r = json.loads(line)
        for lam, x in r["lams"].items():
            if x["gap"] is None:
                continue
            tr = [tuple(p) for p in x.get("trace", [])]
            if not tr or abs(tr[-1][1] - x["gap"]) > 1e-3:
                tr.append((x["t"], x["gap"]))         # final schedule (canon/polish rounding)
            out[(r["name"], lam)] = dict(gap=x["gap"], t=x["t"], trace=tr, gstar=x["gstar"],
                                         cls=r["cls"], workload=r["workload"], count=r["count"])
    return out


def load_gurobi(path):
    """Bench Gurobi per non-trivial cell: gap 0 at its own time t (step trace)."""
    out = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r["lex_f1"]["f2"] == 0:
            continue
        for lam, x in r["lams"].items():
            x = x["gurobi"]
            out[(r["name"], lam)] = dict(gap=0.0, t=x["t"], trace=[(x["t"], 0.0)], status=x["status"],
                                         gstar=x["g"], cls=r["cls"], workload=r["workload"],
                                         count=r["count"])
    return out


def gap_at(trace, t):
    """Best-so-far gap at wall time t (inf before the first solution)."""
    best = math.inf
    for s, g in trace:
        if s > t:
            break
        best = min(best, g)
    return best


def time_to(trace, target):
    for s, g in trace:
        if g <= target + 1e-9:
            return s
    return math.inf


def primal_integral(trace, gstar, T):
    """∫_0^T p(t) dt / T, p = |g − g*| / max(g, g*) (g from gap %), 1 before a solution."""
    acc, t_prev, p_prev = 0.0, 0.0, 1.0
    for s, gap in trace:
        if s >= T:
            break
        acc += p_prev * (s - t_prev)
        g = gstar * (1 + gap / 100)
        p = abs(g - gstar) / max(g, gstar, 1e-12)
        t_prev, p_prev = s, min(p_prev, p)
    acc += p_prev * (T - t_prev)
    return acc / T


def sign_test(w, l):
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def boot_ci(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(d)
    m = rng.choice(d, (n, len(d)), replace=True).mean(axis=1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def fmt(x, nd=1):
    return "–" if x is None or (isinstance(x, float) and math.isinf(x)) else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 7])
    ap.add_argument("--gurobi", default=str(D.parent / "bench_io_noreduc.jsonl"))
    ap.add_argument("--labels", nargs="*")
    a = ap.parse_args()
    labels = dict(zip(a.arms, a.labels or a.arms))
    runs = {arm: {s: load_arm(f"{arm}_s{s}") for s in a.seeds} for arm in a.arms}
    grb = load_gurobi(Path(a.gurobi))
    cells = sorted(set.intersection(*(set(r) for arm in runs.values() for r in arm.values())))
    if grb:
        cells = [c for c in cells if c in grb]
    print(f"{len(cells)} common cells, seeds {a.seeds}; Gurobi: bench solves (step at its time t)\n")

    # 1. per-arm statistics
    print("## 1. Final gap and time\n")
    print("| arm | mean gap (per seed) | mean ± sd over seeds | median | p90 | max | cells > 10 % | time/seed (s) | median t/cell (s) |")
    print("|---|---|---|---|---|---|---|---|---|")
    seedmean = {}
    for arm, rs in runs.items():
        per = [np.mean([rs[s][c]["gap"] for c in cells]) for s in a.seeds]
        allg = np.array([[rs[s][c]["gap"] for c in cells] for s in a.seeds])
        tt = [sum(rs[s][c]["t"] for c in cells) for s in a.seeds]
        seedmean[arm] = allg.mean(axis=0)
        flat = allg.ravel()
        print(f"| {labels[arm]} | {' / '.join(f'{p:.2f}' for p in per)} | {np.mean(per):.2f} ± {np.std(per, ddof=1) if len(per) > 1 else 0:.2f} | "
              f"{np.median(flat):.2f} | {np.percentile(flat, 90):.1f} | {flat.max():.1f} | "
              f"{(flat > 10).sum() / len(a.seeds):.1f} | {np.mean(tt):.0f} | "
              f"{np.median([rs[s][c]['t'] for s in a.seeds for c in cells]):.2f} |")
    if grb:
        g = np.array([grb[c]["gap"] for c in cells])
        print(f"| Gurobi (bench) | {g.mean():.3f} | | {np.median(g):.3f} | | {g.max():.2f} | | "
              f"{sum(grb[c]['t'] for c in cells):.0f} | {np.median([grb[c]['t'] for c in cells]):.1f} |")
        st = {}
        for c in cells:
            st[grb[c]["status"]] = st.get(grb[c]["status"], 0) + 1
        print(f"\nGurobi status: {st}")

    # 2. paired
    print("\n## 2. Paired comparison (seed-mean gap per cell; tie ±0.5 pp)\n")
    print("| A vs B | mean diff A−B (pp) | 95 % CI | W/T/L (A better/tie/worse) | sign-test p |")
    print("|---|---|---|---|---|")
    for i, x in enumerate(a.arms):
        for y in a.arms[i + 1:]:
            d = seedmean[x] - seedmean[y]
            w, l = int((d < -0.5).sum()), int((d > 0.5).sum())
            lo, hi = boot_ci(d)
            print(f"| {labels[x]} vs {labels[y]} | {d.mean():+.2f} | [{lo:+.2f}, {hi:+.2f}] | "
                  f"{w}/{len(d) - w - l}/{l} | {sign_test(w, l):.2g} |")

    # 3. anytime gap at budgets
    def curves(src_trace):
        return np.array([[gap_at(tr, b) for b in BUDGETS] for tr in src_trace])

    print("\n## 3. Gap of the best-so-far at a wall-clock budget (median over cells×seeds; % cells with a solution)\n")
    print("| method | " + " | ".join(f"{b:g} s" for b in BUDGETS) + " |")
    print("|---|" + "---|" * len(BUDGETS))
    anyt = {}
    for arm, rs in runs.items():
        anyt[arm] = curves([rs[s][c]["trace"] for s in a.seeds for c in cells])
    if grb:
        anyt["gurobi"] = curves([grb[c]["trace"] for c in cells])
    for m, M in anyt.items():
        row = []
        for k in range(len(BUDGETS)):
            v = M[:, k]
            ok = np.isfinite(v)
            row.append(f"{np.median(v):.2f} ({100 * ok.mean():.0f} %)" if ok.mean() >= 0.5
                       else f"– ({100 * ok.mean():.0f} %)")
        print(f"| {labels.get(m, 'Gurobi')} | " + " | ".join(row) + " |")
    print("\nMean gap at budget (cells without a solution excluded; n in brackets):\n")
    print("| method | " + " | ".join(f"{b:g} s" for b in BUDGETS) + " |")
    print("|---|" + "---|" * len(BUDGETS))
    for m, M in anyt.items():
        row = []
        for k in range(len(BUDGETS)):
            v = M[:, k][np.isfinite(M[:, k])]
            row.append(f"{v.mean():.2f} [{len(v)}]" if len(v) else "–")
        print(f"| {labels.get(m, 'Gurobi')} | " + " | ".join(row) + " |")

    # 4. time to target
    print("\n## 4. Time to reach a gap target (median s over cells×seeds; % of runs reaching it)\n")
    print("| method | " + " | ".join(f"≤ {t:g} %" for t in TARGETS) + " |")
    print("|---|" + "---|" * len(TARGETS))
    srcs = {arm: [rs[s][c]["trace"] for s in a.seeds for c in cells] for arm, rs in runs.items()}
    if grb:
        srcs["gurobi"] = [grb[c]["trace"] for c in cells]
    for m, trs in srcs.items():
        row = []
        for tg in TARGETS:
            ts = np.array([time_to(tr, tg) for tr in trs])
            ok = np.isfinite(ts)
            row.append(f"{np.median(ts[ok]):.2f} ({100 * ok.mean():.0f} %)" if ok.any() else "– (0 %)")
        print(f"| {labels.get(m, 'Gurobi')} | " + " | ".join(row) + " |")

    # 5. primal integral
    print("\n## 5. Primal integral / T (mean over cells×seeds; lower is better)\n")
    print("| method | T = 10 s | T = 60 s | T = 600 s |")
    print("|---|---|---|---|")
    pis = {}
    for m in srcs:
        if m == "gurobi":
            pts = [(grb[c]["trace"], grb[c]["gstar"]) for c in cells]
        else:
            pts = [(runs[m][s][c]["trace"], runs[m][s][c]["gstar"]) for s in a.seeds for c in cells]
        pis[m] = [np.mean([primal_integral(tr, gs, T) for tr, gs in pts]) for T in (10, 60, 600)]
        print(f"| {labels.get(m, 'Gurobi')} | " + " | ".join(f"{x:.4f}" for x in pis[m]) + " |")

    # 6. against Gurobi per cell
    if grb:
        print("\n## 6. Against Gurobi per cell (seed-mean)\n")
        print("| arm | heuristic t / Gurobi t (median) | Gurobi gap at the heuristic's finish (median) | "
              "heuristic better than Gurobi-at-same-time (% cells) |")
        print("|---|---|---|---|")
        for arm, rs in runs.items():
            ratio, gg, better = [], [], 0
            for c in cells:
                th = np.mean([rs[s][c]["t"] for s in a.seeds])
                gh = np.mean([rs[s][c]["gap"] for s in a.seeds])
                ratio.append(th / grb[c]["t"])
                gt = gap_at(grb[c]["trace"], th)
                gg.append(gt)
                better += gh < gt - 1e-9
            gg = np.array(gg)
            print(f"| {labels[arm]} | {np.median(ratio):.3f} | "
                  f"{fmt(float(np.median(gg)), 2) if np.isfinite(np.median(gg)) else 'no solution'} "
                  f"({100 * np.isfinite(gg).mean():.0f} % have one) | {100 * better / len(cells):.0f} |")

    # 7. by group
    print("\n## 7. Seed-mean gap by λ and class·workload\n")
    groups = sorted({(runs[a.arms[0]][a.seeds[0]][c]["cls"], runs[a.arms[0]][a.seeds[0]][c]["workload"]) for c in cells})
    lams = sorted({c[1] for c in cells}, key=float)
    print("| arm | " + " | ".join(f"λ {l}" for l in lams) + " | " + " | ".join(f"{x}·{y}" for x, y in groups) + " |")
    print("|---|" + "---|" * (len(lams) + len(groups)))
    for arm in a.arms:
        sm = dict(zip(cells, seedmean[arm]))
        r1 = [np.mean([sm[c] for c in cells if c[1] == l]) for l in lams]
        r2 = [np.mean([sm[c] for c in cells if (runs[arm][a.seeds[0]][c]["cls"], runs[arm][a.seeds[0]][c]["workload"]) == g]) for g in groups]
        print(f"| {labels[arm]} | " + " | ".join(f"{x:.2f}" for x in r1 + r2) + " |")

    # figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = D / "final"
    out.mkdir(exist_ok=True)
    ts = np.logspace(-2, np.log10(600), 120)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for m, trs in srcs.items():
        M = np.array([[gap_at(tr, t) for t in ts] for tr in trs])
        frac = np.isfinite(M).mean(axis=0)
        med = np.array([np.median(M[:, k]) if frac[k] >= 0.5 else np.nan for k in range(len(ts))])
        lab = labels.get(m, "Gurobi")
        ax[0].plot(ts, med, label=lab, lw=1.6, ls="--" if m == "gurobi" else "-")
        thr = 1.0
        ax[1].plot(ts, [(M[:, k] <= thr).mean() for k in range(len(ts))], label=lab, lw=1.6,
                   ls="--" if m == "gurobi" else "-")
    ax[0].set(xscale="log", yscale="symlog", xlabel="wall time (s)", ylabel="median gap to g* (%)",
              title="Best-so-far gap (median over cells × seeds)")
    ax[1].set(xscale="log", xlabel="wall time (s)", ylabel="fraction of runs with gap ≤ 1 %",
              title="Runs within 1 % of the Gurobi optimum")
    for x in ax:
        x.grid(alpha=0.3, which="both")
        x.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "anytime.png", dpi=140)
    print(f"\nFigure: {out / 'anytime.png'}")


if __name__ == "__main__":
    main()
