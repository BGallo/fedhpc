"""Summarise results/geo_bench/bench.jsonl (scripts/bench_geo.py) as Markdown
tables: results/geo_bench/REPORT.md.

    uv run python scripts/geo_report.py [--in results/geo_bench/bench.jsonl]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAMS = ("0.25", "0.5", "0.75")


def load(path: Path) -> dict[str, dict]:
    return {(r := json.loads(l))["name"]: r for l in path.read_text().splitlines() if l.strip()}


def gap(g, bks):
    return 100 * (g - bks) / bks if bks > 1e-12 else 0.0


def cert(r):
    """'' when both lex anchors are certified optimal, else a marker."""
    ok = all(r.get(k, {}).get("status") == "optimal" for k in ("lex_f1", "lex_f2"))
    return "" if ok else "†"


def gurobi_s(r):
    return r["lex_f1"]["t"] + r["lex_f2"]["t"] + sum(x.get("gurobi", {}).get("t", 0)
                                                     for x in r["lams"].values())


def fmt(x, nd=0):
    return "–" if x is None else f"{x:,.{nd}f}"


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, default=ROOT / "results/geo_bench/bench.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "results/geo_bench/REPORT.md")
    a = ap.parse_args()
    R = load(a.inp)
    F = {n: r for n, r in R.items() if not r.get("infeasible")}
    p2 = {r.get("phase2_presolve") for r in R.values()}
    md = ["# Two-centre federation benchmark (fedhpc.geo_gen)", "",
          f"{len(R)} instances from `{a.inp.name}` ({len(R) - len(F)} infeasible). Load H, "
          "output level M unless the name says `iH`. `n` = nodes per on-prem type per centre "
          "(2 centres × {c40, c64}, so n50/n150 = class A's n100/n300 in total capacity). "
          f"Gurobi: MIPGap 0, 6 threads, 3600 s per solve, lex phase-2 Presolve {p2}. "
          "† = a lex anchor is not certified optimal (time limit); values are incumbents.", ""]

    # 1. transfer effect -------------------------------------------------------
    rows = []
    for w in ("fedhpc", "pos"):
        for n in (50, 150):
            for wan in (0, 10, 1):
                r = F.get(f"G_{w}_n{n}_lH_iM_w{wan}_fedC")
                if not r:
                    continue
                sz = r["size"]["reduced"]
                p = r["lams"]["0.5"]["gurobi"]
                rg = r.get("root_lp_g05")
                rows.append([w + cert(r), n, "co-loc." if wan == 0 else f"{wan} Gb/s",
                             f"{sz['cols'] / 1e6:.2f}M", f"{r['size']['plain']['cols'] / 1e6:.2f}M",
                             fmt(sz["build_s"], 1), fmt(gurobi_s(r)),
                             f"{100 * abs(r['root_gap_f1'] or 0):.3f}",
                             "–" if rg is None or not p["g"] else f"{100 * abs(p['g'] - rg) / p['g']:.3f}",
                             fmt(r["lex_f1"]["f1"]), fmt(r["lex_f2"]["f1"]),
                             fmt(p["f1"]), fmt(p["f2"]),
                             f"{p['other']}/{p['cloud']}", fmt(p["stage_slots"])])
    md += ["## 1. Effect of inter-centre transfer (paired: same jobs, capacity, cloud)", "",
           "Rows of one (workload, n) differ only in the S1↔S2 link. `cols`: SpaceTime "
           "columns with / without the dominance reductions. Root gaps: LP relaxation vs "
           "certified optimum of min f1 and of the λ = 0.5 scalar ĝ. f1_T / f1_0: minimum "
           "turnaround with / without cloud spending (job-slots). λ=.5: the weighted-sum "
           "optimum's f1, cost ($), jobs at the *other* centre / in the cloud, and staging "
           "slots paid.", "",
           table(["wl", "n", "WAN", "cols", "cols plain", "build s", "Gurobi s", "root f1 %",
                  "root ĝ %", "f1_T", "f1_0", "λ=.5 f1", "λ=.5 $", "other/cloud", "stage"], rows),
           ""]

    # 2. value of federating ---------------------------------------------------
    rows = []
    for w in ("fedhpc", "pos"):
        for n in (50, 150):
            k = lambda s: R.get(f"G_{w}_n{n}_lH_iM_{s}")
            iso_n, fed_n, ctl_n = k("w1_isoN"), k("w1_fedN"), k("w0_fedN")
            if not (iso_n and fed_n):
                continue
            iso_inf = iso_n.get("infeasible")
            fcfs = iso_n["baselines"]["fed_start"]
            fs = fed_n["baselines"]["fed_start"]
            ga = fed_n["lams"]["1.0"]["gamid"]
            fs_txt = fmt(fs["f1"]) + (f" ({fs['unplaced']} unpl.)" if fs["unplaced"] else "")
            fc_txt = fmt(fcfs["f1"]) + (f" ({fcfs['unplaced']} unpl.)" if fcfs["unplaced"] else "")
            rows.append([w, n, "infeasible" if iso_inf else fmt(iso_n["lex_f1"]["f1"]) + cert(iso_n),
                         fc_txt, fs_txt, fmt(st.mean(x["f1"] for x in ga)),
                         fmt(fed_n["lex_f1"]["f1"]) + cert(fed_n),
                         fmt(ctl_n["lex_f1"]["f1"]) + cert(ctl_n) if ctl_n else "–",
                         "–" if iso_inf else
                         f"{100 * (1 - fed_n['lex_f1']['f1'] / iso_n['lex_f1']['f1']):.1f}"])
    md += ["## 2a. Value of federating — no cloud (1 Gb/s WAN), min turnaround f1", "",
           "`iso` = isolated centres, each job only at its origin. Heuristic policies: "
           "per-centre FCFS earliest start (isolated) and the Slurm-federation-inspired "
           "earliest-start policy over both centres (transfer-blind). Optimised: GA-mid "
           "(mean of 3 seeds) and the MILP over the federation; co-loc. = the same federation "
           "with free, instant transfer (a lower bound on what any transfer model allows).", "",
           table(["wl", "n", "iso MILP", "iso FCFS", "fed earliest-start", "fed GA-mid",
                  "fed MILP", "co-loc. MILP", "MILP fed vs iso %"], rows), ""]
    rows = []
    for w in ("fedhpc", "pos"):
        for n in (50, 150):
            iso_c, fed_c = F.get(f"G_{w}_n{n}_lH_iM_w1_isoC"), F.get(f"G_{w}_n{n}_lH_iM_w1_fedC")
            if not (iso_c and fed_c):
                continue
            row = [w, n]
            for r in (iso_c, fed_c):
                row += [fmt(r["lex_f1"]["f2"]) + cert(r), fmt(r["lex_f2"]["f1"]) + cert(r),
                        fmt(r["lex_f2"]["f2"])]
            for lam in LAMS:
                a_, b_ = iso_c["lams"][lam]["gurobi"], fed_c["lams"][lam]["gurobi"]
                row.append(f"{fmt(a_['f1'])}/{fmt(a_['f2'])} → {fmt(b_['f1'])}/{fmt(b_['f2'])}")
            rows.append(row)
    md += ["## 2b. Value of federating — with cloud (1 Gb/s WAN)", "",
           "Cost of the minimum-turnaround schedule (f2_T), the best turnaround and the cost of "
           "the cheapest schedule (f1_0, f2_0), isolated vs federated, and the weighted-sum "
           "optima as f1/$ (iso → fed). Isolated centres share one cloud account.", "",
           table(["wl", "n", "iso f2_T $", "iso f1_0", "iso f2_0 $", "fed f2_T $", "fed f1_0",
                  "fed f2_0 $", "λ .25", "λ .5", "λ .75"], rows), ""]

    # 3. heuristics vs exact -----------------------------------------------------
    rows, cells = [], []
    for name, r in sorted(F.items()):
        if r["degenerate"]:
            continue
        per = []
        for lam in LAMS:
            x = r["lams"][lam]
            gs = [s["g"] for s in x["gamid"]]
            bks = min([x["gurobi"]["g"], x["dispatch"]["g"], x["dispatch_deploy"]["g"],
                       x["gamid_deploy"]["g"], *gs])
            c = dict(ga=st.mean(gap(g, bks) for g in gs), ga_hit=gap(min(gs), bks) <= 1e-4,
                     ga_dep=gap(x["gamid_deploy"]["g"], bks), dis=gap(x["dispatch"]["g"], bks),
                     thr=gap(x["threshold"]["g"], bks), grb=gap(x["gurobi"]["g"], bks),
                     ga_t=st.mean(s["t"] for s in x["gamid"]), grb_t=x["gurobi"]["t"],
                     geo=("w0" not in name) and "fedC" in name, pos="_pos_" in name)
            per.append(c); cells.append(c)
        sl = lambda key, nd=1: " / ".join(f"{c[key]:.{nd}f}" for c in per)
        rows.append([name.removeprefix("G_") + cert(r), sl("ga"), sl("ga_dep"), sl("dis"),
                     sl("thr", 0), fmt(st.mean(c["grb_t"] for c in per), 1),
                     fmt(st.mean(c["ga_t"] for c in per), 1)])
    md += ["## 3. Heuristics vs Gurobi (gap % to the best known ĝ; λ = .25 / .5 / .75)", "",
           "GA-mid = the paper's memetic GA (pop 16, 120 generations; mean of seeds 42/1/7), "
           "given Gurobi's exact reference points (*benchmark mode*). *deploy* = GA-mid "
           "seed 42 on reference points it estimates itself (no Gurobi anywhere). Dispatch = "
           "the ILS heuristic. Threshold = the best threshold-bursting baseline. Times per λ.", "",
           table(["instance", "GA-mid", "GA-mid deploy", "dispatch ILS", "threshold",
                  "Gurobi s", "GA-mid s"], rows), ""]

    def summary(sel, label):
        cs = [c for c in cells if sel(c)]
        if not cs:
            return []
        return [f"| {label} | {len(cs)} | {st.mean(c['ga'] for c in cs):.2f} | "
                f"{st.median(c['ga'] for c in cs):.2f} | {max(c['ga'] for c in cs):.1f} | "
                f"{sum(c['ga_hit'] for c in cs)} | {st.mean(c['ga_dep'] for c in cs):.2f} | "
                f"{st.mean(c['dis'] for c in cs):.2f} | {st.mean(c['grb'] for c in cs):.3f} |"]
    md += ["| subset | cells | GA-mid mean % | median | max | GA = BKS | deploy mean | "
           "dispatch mean | Gurobi mean |", "|---|---|---|---|---|---|---|---|---|"]
    md += summary(lambda c: True, "all")
    md += summary(lambda c: not c["pos"], "W1 (fedhpc)")
    md += summary(lambda c: c["pos"], "W2 (pos)")
    md += summary(lambda c: c["geo"], "fed + cloud, WAN > 0")
    md += summary(lambda c: not c["geo"], "co-located control or iso")
    md += [""]

    # 4. end-to-end time: Gurobi pipeline vs Gurobi-free pipeline ------------------
    rows = []
    for name, r in sorted(F.items()):
        if r["degenerate"]:
            continue
        ga_dep = r["heur_ref"]["t"] + sum(r["lams"][l]["gamid_deploy"]["t"] for l in LAMS)
        rows.append([name.removeprefix("G_") + cert(r), fmt(r["lex_f1"]["t"] + r["lex_f2"]["t"]),
                     fmt(sum(r["lams"][l]["gurobi"]["t"] for l in LAMS)), fmt(gurobi_s(r)),
                     fmt(r["heur_ref"]["t"]), fmt(ga_dep)])
    md += ["## 4. End-to-end time for three operating points (critique §4.2)", "",
           "Gurobi: two lex anchors plus three weighted sums. Gurobi-free: heuristic "
           "reference points (once) plus GA-mid at three λ.", "",
           table(["instance", "Gurobi anchors s", "Gurobi 3×WS s", "Gurobi total s",
                  "heur. refs s", "Gurobi-free total s"], rows), ""]

    # 5. checks ------------------------------------------------------------------
    rows = []
    for n, r in sorted(R.items()):
        io = r["io_info"]
        if r.get("infeasible"):
            rows.append([n.removeprefix("G_"), "–", "infeasible", io["capped"],
                         fmt(io["in_gb"] / 1e3, 1), fmt(io["out_gb"] / 1e3, 1), io["pinned"]])
            continue
        rows.append([n.removeprefix("G_"), r["zero_cost"]["possible"],
                     fmt(r["zero_cost"]["f2_0"], 2) + cert(r), io["capped"],
                     fmt(io["in_gb"] / 1e3, 1), fmt(io["out_gb"] / 1e3, 1), io["pinned"]])
    md += ["## 5. Instance checks (critique §3.6)", "",
           "A free option for every job is necessary, not sufficient, for a free schedule: "
           "f2_0 > 0 means the free capacity cannot finish every job inside the horizon.", "",
           table(["instance", "free option per job", "f2_0 $", "volumes capped",
                  "in TB", "out TB", "origin-only jobs"], rows), ""]
    a.out.write_text("\n".join(md))
    print(a.out)


if __name__ == "__main__":
    main()
