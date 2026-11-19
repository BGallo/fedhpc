"""Analysis of results/kramer_fair/test.csv (see scripts/bench_kramer_fair.py)."""
import csv, collections, math, sys
from pathlib import Path
from scipy.stats import wilcoxon
D = Path(__file__).parent
rows = list(csv.DictReader(open(D / "test.csv")))
M = ["TI", "AF", "fed_st", "fed_af"]; C = ["default", "tuned", "tuned+ws"]; TL = 1800
by = collections.defaultdict(dict)
for r in rows:
    by[r["inst"]][(r["model"], r["config"])] = r
I = sorted(by)
def sg(ts, s=10): return math.exp(sum(math.log(t + s) for t in ts) / len(ts)) - s
def t(r): return float(r["solve_s"]) if r["status"] == "optimal" else 2 * TL
def gap(r): return (float(r["obj"]) - float(r["bound"])) / float(r["obj"]) * 100
opt = {i: min(float(r["obj"]) for r in by[i].values() if r["obj"]) for i in I}

print("== Table-3 layout, default params (12 inst/line): opt | gap_lp% | gap% | nodes | t(s) ==")
print(f"{'n':>3} {'m':>2} " + " | ".join(f"{m:^34}" for m in M))
tot = collections.defaultdict(list)
for n in ("20", "50", "100"):
    for mm in ("2", "3", "5", "10"):
        cells = []
        for m in M:
            rs = [by[i][(m, "default")] for i in I if by[i][(m, "default")]["n"] == n and by[i][(m, "default")]["m"] == mm]
            tot[m] += rs
            glp = sum((opt[r["inst"]] - float(r["lp"])) / opt[r["inst"]] * 100 for r in rs) / len(rs)
            cells.append(f"{sum(r['status']=='optimal' for r in rs):2} {glp:.3f} {sum(gap(r) for r in rs)/len(rs):.3f} "
                         f"{sum(float(r['nodes']) for r in rs)/len(rs):7.1f} {sum(float(r['solve_s']) for r in rs)/len(rs):6.1f}")
        print(f"{n:>3} {mm:>2} " + " | ".join(f"{c:^34}" for c in cells))
print("Sum/Avg " + " | ".join(
    f"{sum(r['status']=='optimal' for r in tot[m]):3} {sum(gap(r) for r in tot[m])/len(tot[m]):.4f} "
    f"{sum(float(r['nodes']) for r in tot[m])/len(tot[m]):7.1f} {sum(float(r['solve_s']) for r in tot[m])/len(tot[m]):6.1f}" for m in M))

print("\n== Overall per config (144 inst): solved, sgm time (PAR2, shift 10), mean build ==")
for m in M:
    print(f"{m:7} " + "   ".join(
        f"{c}: {sum(by[i][(m,c)]['status']=='optimal' for i in I)} / {sg([t(by[i][(m,c)]) for i in I]):5.1f}s"
        for c in C) + f"   build {sum(float(by[i][(m,'default')]['build_s']) for i in I)/len(I):.2f}s")

def paired(a, b, label, filt=lambda i: True):
    xs = [i for i in I if filt(i) and max(t(by[i][a]), t(by[i][b])) > 1.0]   # skip trivial (<1 s) pairs
    lr = [math.log(t(by[i][b]) + 1) - math.log(t(by[i][a]) + 1) for i in xs]
    ratio = math.exp(sum(lr) / len(lr))
    p = wilcoxon(lr).pvalue if any(lr) else 1.0
    wins = sum(x < 0 for x in lr)
    print(f"{label:<44} n={len(xs):3} geo ratio {b[0]}/{a[0]} = {ratio:5.3f}  {b[0]} faster on {wins:3}/{len(xs)}  Wilcoxon p={p:.3g}")
    return lr

print("\n== Noise baseline (identical params, different run) ==")
noise = paired(("AF", "default"), ("AF", "tuned"), "AF default vs AF tuned (same params)")
print(f"   per-instance |log ratio| median {sorted(abs(x) for x in noise)[len(noise)//2]:.3f}")
print("\n== Model comparisons, default params ==")
for a, b in [("TI", "AF"), ("fed_st", "fed_af"), ("AF", "fed_af"), ("AF", "fed_st"), ("TI", "fed_st")]:
    paired((a, "default"), (b, "default"), f"{a} -> {b}")
print("\n== Model comparisons, each at its tuned params ==")
for a, b in [("TI", "AF"), ("fed_st", "fed_af"), ("AF", "fed_af"), ("AF", "fed_st")]:
    paired((a, "tuned"), (b, "tuned"), f"{a} -> {b}")
print("\n== Tuning effect (default -> tuned) and warm start (tuned -> tuned+ws) ==")
for m in M:
    if m != "AF":
        paired((m, "default"), (m, "tuned"), f"{m} tuning")
    paired((m, "tuned"), (m, "tuned+ws"), f"{m} warm start")
print("\n== Hard subset n=100 only, default ==")
for a, b in [("AF", "fed_af"), ("AF", "fed_st"), ("TI", "AF")]:
    paired((a, "default"), (b, "default"), f"{a} -> {b} (n=100)", lambda i: by[i][(a,'default')]['n'] == '100')

print("\n== Root behaviour, default (solved runs) ==")
for m in M:
    rs = [by[i][(m, "default")] for i in I if by[i][(m, "default")]["status"] == "optimal"]
    at_root = sum(1 for r in rs if int(r["nodes"]) <= 1)
    best_root = sum(1 for r in rs if r["best_sol_s"] and r["root_s"] and float(r["best_sol_s"]) <= float(r["root_s"]) + 1e-6)
    rg = [(opt[r["inst"]] - float(r["root_bound"])) / opt[r["inst"]] * 100 for r in rs]
    lg = [(opt[r["inst"]] - float(r["lp"])) / opt[r["inst"]] * 100 for r in rs]
    print(f"{m:7} solved {len(rs)}: closed at root {at_root}, optimum found at root {best_root}, "
          f"mean gap LP {sum(lg)/len(lg):.4f}% -> root {sum(rg)/len(rg):.4f}%")
ws = [by[i][("AF", "tuned+ws")] for i in I]
wg = sorted((float(r["ws_obj"]) - opt[r["inst"]]) / opt[r["inst"]] * 100 for r in ws)
print(f"warm start quality: median {wg[len(wg)//2]:.3f}% max {wg[-1]:.3f}% above optimum; mean heuristic time "
      f"{sum(float(r['ws_s']) for r in ws)/len(ws):.1f}s")
