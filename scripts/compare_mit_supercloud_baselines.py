"""Run the exact f1 MIP and all offline baselines on a mit_supercloud instance,
and print every metric for each side by side.

Bypasses fedhpc's console `format_summary`/_stat_table_lines helper for the
"Bounded slowdown" row — that helper multiplies every stat by
`slot_size_seconds` uniformly, which is correct for the time-valued fields
(wait/run/turnaround) but wrong for bounded slowdown (a dimensionless ratio
that should never be scaled). This script reads `compute_stats()`'s raw dict
directly and scales only the fields that should be scaled.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fedhpc.data import Instance
from fedhpc.model import solve_f1
from fedhpc.viz import compute_stats
from fedhpc.baselines import onprem_only_spt, greedy_earliest_completion, threshold_bursting

_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _print_block(name: str, sol, inst: Instance, slot: float) -> dict:
    st = compute_stats(sol, inst)
    sys_ = st["system"]
    n_sched = st["n_scheduled"]
    n_total = st["n_total"]

    def scaled(block, key="avg"):
        return block[key] * slot

    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
    print(f"  status: {sol.status}    scheduled: {n_sched}/{n_total}")
    print(f"  {'Metric':<22}{'Avg':>12}{'Min':>12}{'Max':>12}{'Total':>14}")
    for label, key, unit in [
        ("Wait time", "wait_time", "s"),
        ("Run time", "run_time", "s"),
        ("Turnaround", "turnaround", "s"),
    ]:
        b = st[key]
        print(f"  {label + ' (' + unit + ')':<22}{scaled(b):>12.1f}{scaled(b,'min'):>12.1f}"
              f"{scaled(b,'max'):>12.1f}{scaled(b,'total'):>14.1f}")
    b = st["bounded_slowdown"]
    print(f"  {'Bounded slowdown':<22}{b['avg']:>12.3f}{b['min']:>12.3f}{b['max']:>12.3f}{'—':>14}")
    print()
    print(f"  f1 total turnaround (s): {sol.f1 * slot:,.1f}")
    print(f"  f2 total cost ($):       {sol.f2:,.2f}")
    print(f"  On-prem jobs/cost:  {sys_['onprem_jobs']:>6} / ${sys_['onprem_cost']:,.2f}")
    print(f"  Cloud   jobs/cost:  {sys_['cloud_jobs']:>6} / ${sys_['cloud_cost']:,.2f}")
    print(f"  On-prem utilization:     {sys_['onprem_util_pct']:.2f}%")

    return {
        "name": name, "status": sol.status, "n_sched": n_sched, "n_total": n_total,
        "wait_avg": scaled(st["wait_time"]), "wait_total": scaled(st["wait_time"], "total"),
        "run_avg": scaled(st["run_time"]),
        "ta_avg": scaled(st["turnaround"]), "ta_total": scaled(st["turnaround"], "total"),
        "ta_max": scaled(st["turnaround"], "max"),
        "bsd_avg": b["avg"], "bsd_max": b["max"],
        "cost_total": sol.f2, "util_pct": sys_["onprem_util_pct"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", default="mit_supercloud_20210108_10min.json",
                        help="Filename under fedhpc/data/ to load.")
    parser.add_argument("--mip-gap", type=float, default=0.0)
    parser.add_argument("--time-limit", type=float, default=None)
    args = parser.parse_args()

    instance_path = _DATA_DIR / args.instance
    inst = Instance.from_file(instance_path)
    slot = inst.slot_size_seconds
    print(f"Loaded {instance_path.name}: {len(inst.jobs)} jobs, "
          f"{len(inst.instance_types)} instance types, {len(inst.running_jobs)} running_jobs, "
          f"horizon={inst.horizon}, slot={slot}s")

    rows = []

    gurobi_params = {"MIPGap": args.mip_gap, "OutputFlag": 0}
    if args.time_limit is not None:
        gurobi_params["TimeLimit"] = args.time_limit
    sol_mip = solve_f1(inst, **gurobi_params)
    rows.append(_print_block("OFFLINE MIP (fedhpc f1, exact/Gurobi)", sol_mip, inst, slot))

    sol_spt = onprem_only_spt(inst)
    rows.append(_print_block("BASELINE: onprem_only_spt (SPT list-schedule)", sol_spt, inst, slot))

    sol_greedy = greedy_earliest_completion(inst)
    rows.append(_print_block("BASELINE: greedy_earliest_completion (SPT order)", sol_greedy, inst, slot))

    sol_thresh = threshold_bursting(inst, theta=inst.horizon)
    rows.append(_print_block("BASELINE: threshold_bursting (theta=horizon, arrival order)", sol_thresh, inst, slot))

    print(f"\n{'=' * 78}\nSUMMARY TABLE\n{'=' * 78}")
    hdr = f"  {'':<45}{'Sched':>7}{'AvgWait':>10}{'AvgTA':>10}{'MaxTA':>10}{'AvgBSD':>8}{'Util%':>7}"
    print(hdr)
    for r in rows:
        print(f"  {r['name']:<45}{r['n_sched']:>4}/{r['n_total']:<3}"
              f"{r['wait_avg']:>10.1f}{r['ta_avg']:>10.1f}{r['ta_max']:>10.1f}"
              f"{r['bsd_avg']:>8.3f}{r['util_pct']:>7.2f}")


if __name__ == "__main__":
    main()
