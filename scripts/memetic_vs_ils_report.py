"""Compare validate_<tag>.jsonl arms (dispatch_study) on their common cells:
gap to g* (mean / median / p90 / max, cells > 10 %), wins/ties/losses against
the first arm (tie = within 0.5 pp), total wall time, per-λ and per class·workload
mean gap, and, for memetic arms, how often the final incumbent used the EDF
priority (stats mu == -2).

    uv run python scripts/memetic_vs_ils_report.py ils_s42 mem_base_s42 mem_edf_s42 ...
"""
from __future__ import annotations

import json
import statistics as stt
import sys
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "results" / "generated_bench" / "dispatch_study"


def load(tag):
    out = {}
    for line in (D / f"validate_{tag}.jsonl").read_text().splitlines():
        r = json.loads(line)
        for lam, x in r["lams"].items():
            if x["gap"] is not None:
                out[(r["name"], lam)] = dict(x, cls=r["cls"], workload=r["workload"])
    return out


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]


def main(tags):
    arms = {t: load(t) for t in tags}
    cells = sorted(set.intersection(*(set(a) for a in arms.values())))
    ref = tags[0]
    print(f"{len(cells)} common cells\n")
    print("| arm | mean | median | p90 | max | >10 % | W/T/L vs " + ref + " | time (s) | EDF incumbent |")
    print("|---|---|---|---|---|---|---|---|---|")
    for t, a in arms.items():
        g = [a[c]["gap"] for c in cells]
        w = sum(a[c]["gap"] < arms[ref][c]["gap"] - 0.5 for c in cells)
        l = sum(a[c]["gap"] > arms[ref][c]["gap"] + 0.5 for c in cells)
        mu = [a[c]["stats"].get("mu") for c in cells]
        edf = f"{sum(m == -2.0 for m in mu)}/{len(cells)}" if "deadline" in a[cells[0]]["stats"] else "–"
        print(f"| {t} | {stt.mean(g):.1f} | {stt.median(g):.1f} | {pct(g, .9):.1f} | {max(g):.1f} | "
              f"{sum(x > 10 for x in g)} | {w}/{len(cells) - w - l}/{l} | "
              f"{sum(a[c]['t'] for c in cells):.0f} | {edf} |")
    lams = sorted({c[1] for c in cells}, key=float)
    groups = sorted({(arms[ref][c]["cls"], arms[ref][c]["workload"]) for c in cells})
    print("\nmean gap by λ and by class·workload\n")
    print("| arm | " + " | ".join(f"λ {l}" for l in lams) + " | "
          + " | ".join(f"{c}·{w}" for c, w in groups) + " |")
    print("|---|" + "---|" * (len(lams) + len(groups)))
    for t, a in arms.items():
        by_l = [stt.mean(a[c]["gap"] for c in cells if c[1] == l) for l in lams]
        by_g = [stt.mean(a[c]["gap"] for c in cells
                         if (a[c]["cls"], a[c]["workload"]) == grp) for grp in groups]
        print(f"| {t} | " + " | ".join(f"{x:.1f}" for x in by_l + by_g) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
