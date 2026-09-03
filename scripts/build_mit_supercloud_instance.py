"""Build a fedhpc Instance JSON for the mit_supercloud_20210108 case study.

Standalone converter — reads window_jobs / initial_state_jobs directly from
the overseer repo (python-dict-per-line files, produced by
experiments/mit_supercloud_20210108/prepare_data.py), WITHOUT running the
full overseer discrete-event simulator. Mirrors
build_pos_congestion_instance.py's structure; see that script for the
general approach this one reuses.

Differences from build_pos_congestion_instance.py, and why:
  - Multiple on-prem InstanceTypes ("capacity buckets"), not one flat 40c
    type: fedhpc's space-time formulation grants each job one whole
    InstanceType instance — no intra-node packing and no multi-node jobs.
    This trace is 98.3% single-core/5GB jobs with a thin heavy tail up to
    100 cores across 5 nodes, so a single 40c type either wastes ~39/40
    cores per job (measured: 20.1% on-prem utilization, avg wait ~1350s) or
    rejects the multi-node tail outright. `_choose_capacity_buckets()`
    (ported from wise_burst/scheduling_policies/fedhpc_policy.py, the
    live DES policy's already-tested mechanism) splits the 224-node pool
    into demand-weighted sub-node buckets (e.g. 1c, 8c, 10c, ...) AND
    multi-node buckets (e.g. 80c, 120c) from the SAME allocation — closing
    both the packing gap and the multi-node gap with one mechanism, rather
    than two separate patches. See the empirical validation this script's
    integration is based on (fixed 2-way 20c split alone cut average wait
    ~73% for this window; this generalizes that to a data-driven bucket set
    instead of a hardcoded size).
  - NO cloud types: the MIT Supercloud Dataset carries no cost/pricing
    signal, and this window has no congestion event motivating a burst-to-
    cloud story, so a pure on-prem instance is the honest first cut.
  - No jobs are dropped for exceeding the horizon: the longest window job is
    ~3.8h, well inside the default 48h horizon (unlike pos_congestion's
    multi-day reservoir-simulation tail).

Generation rules (mirrors build_pos_congestion_instance.py):
  t0                 = WINDOW_START (2021-01-08T07:00:00)
  time_scale_seconds = --slot-seconds (default 600 = 10 min)
  horizon             = ceil(48h / time_scale_seconds)
  job.arrival         = (@submit - t0) / time_scale
  job.exec_time       = elapsed / time_scale
  job.cpu             = total_cpus ; job.mem = mem_mb / 1024 (GB)
  job.stor            = 0 ; job.io_volume = 0
  instance_types      = one per capacity bucket chosen by
                        _choose_capacity_buckets() from window_jobs' demand
                        (cpu-hours-weighted), each bucket's mem scaled
                        proportionally to its share of one physical node.
  running_jobs        = initial_state_jobs rows with @start < t0, one entry
                        per real hostname (NOT one entry per (job, hostname)
                        pair — most hosts run several jobs concurrently at
                        t0), placed in the smallest bucket that fits that
                        host's aggregate concurrent cpu usage (same rule
                        window jobs use). end = ceil((@end - t0) /
                        time_scale).
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from pathlib import Path

import pandas as pd

_OVERSEER = Path("/home/bernardo/Documentos/programas/overseer")
_DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Per-experiment window start — the only thing that varies between
# mit_supercloud_* experiments; add new entries here rather than duplicating
# this script. horizon_hours defaults to 48 (matches build_pos_congestion_
# instance.py's convention) but can be overridden per experiment when jobs
# are short-lived enough that 48h of mostly-empty horizon would needlessly
# inflate the MIP. 20210317's 25h covers its full observed tail exactly
# (max arrival+exec_time = 147.5 slots = 24.58h, one 86,031s/23.9h job) with
# a small margin — a horizon that drops part of the workload changes what
# experiment is actually being run, so "some jobs don't fit" must be solved
# by sizing the horizon to the data, not by silently discarding the jobs
# that don't fit an arbitrary choice.
_EXPERIMENTS = {
    "mit_supercloud_20210108": dict(
        window_start="2021-01-08 07:00:00", horizon_hours=48.0,
    ),
    "mit_supercloud_20210317": dict(
        window_start="2021-03-17 03:30:00", horizon_hours=25.0,
    ),
    # Full-day window, chosen for real congestion (mean historical wait
    # 91,151s/25.3h) *and* job resource-request diversity (cpus_req spans
    # {1,4,5,6,8,10,12,20,21,30,40,80}, no dominant value — unlike 20210317's
    # 99.7%-single-value workload). 200h covers the observed tail (max
    # arrival+exec_time = 1179.5 slots = 196.6h, one 181.4h job) with margin.
    "mit_supercloud_20210604": dict(
        window_start="2021-06-04 00:00:00", horizon_hours=200.0,
    ),
    # 20210317 and 20210604 both turned out to have release times so tightly
    # clustered in one dominant burst that the classical "equal release
    # time" invariance result (SPT/FCFS already minimizes flow time when
    # jobs arrive together) applied almost exactly to their bulk job class —
    # real congestion + size diversity alone weren't enough. This window was
    # picked by explicitly scoring for LOW release-time clustering too (max
    # 9.1% of the day's jobs in any single minute, vs 20210317's 99.7% and
    # 20210604's ~54%), on top of wait (mean 9.77h) and diversity (24
    # distinct cpus_req values 1-1280, entropy 3.27 bits, highest of any
    # candidate day). 610h covers its extreme tail (max arrival+exec_time =
    # 607.5h, one 25-day job) with margin.
    "mit_supercloud_20210812": dict(
        window_start="2021-08-12 00:00:00", horizon_hours=610.0,
    ),
}

_NAN = float("nan")
_TIME_SCALE_SECONDS = 600.0  # overridden by --slot-seconds in main()
_HORIZON_HOURS = 48.0  # overridden per-experiment in main()

# 'normal' / xeon-g6 partition, per supercloud.mit.edu docs and
# node_pool.json's node_pool_full_partition_size (224, cross-checked against
# the publicly documented partition size).
_CORES_PER_NODE = 40
_N_NODES = 224


def _load_dict_lines(path: Path) -> list[dict]:
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(eval(line, {"nan": _NAN}))
    return rows


# ---------------------------------------------------------------------------
# Capacity-bucket selection — ported from
# wise_burst/scheduling_policies/fedhpc_policy.py's _choose_capacity_buckets /
# _select_bucket_type_id (the live DES policy's already-tested mechanism),
# adapted to work from a static window_jobs list instead of a live pending-job
# queue. See that function's docstring in fedhpc_policy.py for the full
# rationale; the algorithm here is unchanged.
# ---------------------------------------------------------------------------

def _choose_capacity_buckets(window: list[dict], node_cores: int, total_cores: int) -> dict[int, int]:
    num_nodes = total_cores // node_cores
    demand_weight: dict[int, float] = {}

    for r in window:
        cpu = int(math.ceil(r["total_cpus"]))
        if cpu <= 0 or cpu > total_cores:
            continue
        bucket_cpu = cpu if cpu <= node_cores else int(math.ceil(cpu / node_cores) * node_cores)
        bucket_cpu = min(bucket_cpu, total_cores)
        weight = cpu * max(1.0, float(r["elapsed"]))
        demand_weight[bucket_cpu] = demand_weight.get(bucket_cpu, 0.0) + weight

    if not demand_weight:
        return {node_cores: max(1, num_nodes)}

    # Preserve a full-node bucket so exclusive/full-node jobs stay feasible
    # even if the observed mix happens to skew away from that size.
    demand_weight.setdefault(node_cores, max(1.0, 0.01 * sum(demand_weight.values())))

    def _nodes_per_unit(bucket_cpu: int) -> int:
        return max(1, bucket_cpu // node_cores)

    def _cap_from_nodes(bucket_cpu: int, assigned_nodes: int) -> int:
        if bucket_cpu <= node_cores:
            return assigned_nodes * max(1, node_cores // bucket_cpu)
        return assigned_nodes // _nodes_per_unit(bucket_cpu)

    total_weight = sum(demand_weight.values())
    node_alloc: dict[int, int] = {}
    used_nodes = 0

    for bucket_cpu, weight in sorted(demand_weight.items()):
        share_nodes = num_nodes * (weight / total_weight)
        nodes = int(share_nodes)
        min_nodes = _nodes_per_unit(bucket_cpu)
        if nodes < min_nodes and min_nodes <= num_nodes:
            nodes = min_nodes
        node_alloc[bucket_cpu] = nodes
        used_nodes += nodes

    if used_nodes > num_nodes:
        for bucket_cpu, _ in sorted(demand_weight.items(), key=lambda item: item[1]):
            min_nodes = _nodes_per_unit(bucket_cpu)
            while used_nodes > num_nodes and node_alloc.get(bucket_cpu, 0) > min_nodes:
                node_alloc[bucket_cpu] -= 1
                used_nodes -= 1
        for bucket_cpu, _ in sorted(demand_weight.items(), key=lambda item: item[1]):
            if used_nodes <= num_nodes or len(node_alloc) <= 1:
                break
            if node_alloc.get(bucket_cpu, 0) > 0:
                used_nodes -= node_alloc.pop(bucket_cpu)

    for bucket_cpu, _ in sorted(demand_weight.items(), key=lambda item: item[1], reverse=True):
        if bucket_cpu not in node_alloc:
            continue
        while used_nodes < num_nodes:
            node_alloc[bucket_cpu] += 1
            used_nodes += 1

    return {
        cpu: cap
        for cpu, nodes in node_alloc.items()
        if (cap := _cap_from_nodes(cpu, nodes)) > 0
    }


def _select_bucket_type_id(type_ids_by_cpu: dict[int, int], cpu_demand: float) -> int | None:
    if not type_ids_by_cpu:
        return None
    demand = int(math.ceil(cpu_demand))
    for bucket_cpu in sorted(type_ids_by_cpu):
        if bucket_cpu >= demand:
            return type_ids_by_cpu[bucket_cpu]
    return type_ids_by_cpu[max(type_ids_by_cpu)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="mit_supercloud_20210108",
                        choices=sorted(_EXPERIMENTS), help="Which experiments/<name> window to build.")
    parser.add_argument("--slot-seconds", type=float, default=600.0,
                        help="Discretization slot size in seconds (default: 600 = 10 min).")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output path (default: data/<experiment>_10min.json).")
    args = parser.parse_args()

    global _TIME_SCALE_SECONDS, _HORIZON_HOURS, WINDOW_START, _EXP
    _TIME_SCALE_SECONDS = args.slot_seconds
    exp_cfg = _EXPERIMENTS[args.experiment]
    WINDOW_START = pd.Timestamp(exp_cfg["window_start"])
    _HORIZON_HOURS = exp_cfg["horizon_hours"]
    _EXP = _OVERSEER / "experiments" / args.experiment

    if args.out is not None:
        out_path = args.out
    elif args.slot_seconds == 600.0:
        out_path = _DATA_DIR / f"{args.experiment}_10min.json"
    else:
        out_path = _DATA_DIR / f"{args.experiment}_{int(args.slot_seconds)}s.json"

    print(f"Loading {args.experiment} window_jobs / initial_state_jobs …", file=sys.stderr)
    # prepare_data.py keeps both 'normal' and 'xeon-p8' partitions by design
    # (PARTITION_CORES has both) — this script only models the 'normal' pool,
    # so any stray other-partition job must be filtered here, not assumed
    # away. Verified necessary: mit_supercloud_20210317's initial_state_jobs
    # contains exactly one 'xeon-p8' entry (an 840-node/48c-per-node job,
    # matching xeon-p8's spec exactly) which, left unfiltered, corrupted the
    # running-jobs host count to 1,015 against a real 224-node 'normal' pool.
    window_raw = _load_dict_lines(_EXP / "window_jobs")
    initial_raw = _load_dict_lines(_EXP / "initial_state_jobs")
    window = [r for r in window_raw if r.get("partition") == "normal"]
    initial = [r for r in initial_raw if r.get("partition") == "normal"]
    if len(window) != len(window_raw) or len(initial) != len(initial_raw):
        print(f"  dropped {len(window_raw) - len(window)} non-'normal' window_jobs, "
              f"{len(initial_raw) - len(initial)} non-'normal' initial_state_jobs", file=sys.stderr)
    print(f"  window_jobs: {len(window)}  initial_state_jobs: {len(initial)}", file=sys.stderr)

    time_scale_seconds = float(_TIME_SCALE_SECONDS)
    horizon = math.ceil(_HORIZON_HOURS * 3600 / time_scale_seconds)
    print(f"  time_scale_seconds={time_scale_seconds}  horizon_hours={_HORIZON_HOURS}  horizon={horizon}",
          file=sys.stderr)

    # ── instance types: demand-weighted capacity buckets ────────────────────
    total_cores = _CORES_PER_NODE * _N_NODES
    bucket_caps = _choose_capacity_buckets(window, _CORES_PER_NODE, total_cores)
    bucket_cpus_sorted = sorted(bucket_caps)

    # Running jobs (pre-existing occupancy at t0) draw from the same physical
    # pool as window jobs but _choose_capacity_buckets() only saw window-job
    # demand — so bucket capacity can come back too small once running-job
    # occupancy is added, and Instance.build() would silently truncate the
    # excess (UserWarning: "Running jobs ... exceed capacity"), corrupting
    # the initial state exactly like the bug this script's history already
    # fixed once for the flat-40c version.
    #
    # Each host's *aggregate* concurrent cpu usage at t0 is consolidated into
    # ONE running-job entry, in the SMALLEST bucket that actually fits that
    # usage (the same rule _select_bucket_type_id applies to window jobs) —
    # NOT always the smallest bucket in the instance. Forcing every host into
    # 1c-equivalent units (an earlier version of this script did that)
    # wildly overcounts demand whenever running jobs are large: a single
    # 40-core running job would register as 40 separate 1c-slot requests
    # instead of one 40c-slot request, starving window-job buckets of
    # capacity that was never really needed there (verified: this exact
    # mistake dropped 186/820 window jobs as infeasible on
    # mit_supercloud_20210604, whose running jobs are large/organic, not the
    # small 1-core jobs 20210108/20210317's running jobs happened to be).
    def _nodes_per_unit(cpu: int) -> int:
        return max(1, cpu // _CORES_PER_NODE)

    def _cap_from_nodes(cpu: int, nodes: int) -> int:
        if cpu <= _CORES_PER_NODE:
            return nodes * max(1, _CORES_PER_NODE // cpu)
        return nodes // _nodes_per_unit(cpu)

    def _nodes_from_cap(cpu: int, cap: int) -> int:
        if cpu <= _CORES_PER_NODE:
            return -(-cap // max(1, _CORES_PER_NODE // cpu))  # ceil div
        return cap * _nodes_per_unit(cpu)

    def _smallest_fitting_bucket(demand: float) -> int:
        for cpu in bucket_cpus_sorted:
            if cpu >= demand:
                return cpu
        return bucket_cpus_sorted[-1]

    host_end_slot: dict[str, int] = {}
    host_peak_cpu: dict[str, float] = {}
    n_running_raw = 0
    for r in initial:
        start = pd.Timestamp(r["@start"])
        if start >= WINDOW_START:
            continue  # pending, not yet running at t0 — out of scope (schedule_scope=window_jobs)
        n_running_raw += 1
        end = pd.Timestamp(r["@end"])
        end_slot = max(1, math.ceil((end - WINDOW_START).total_seconds() / time_scale_seconds))
        nodes_str = str(r.get("nodes", ""))
        hostnames = [h.strip() for h in nodes_str.split(",") if h.strip() and h.strip() != "NONE"]
        per_node_cpu = r["total_cpus"] / max(1, len(hostnames))
        for h in hostnames:
            host_end_slot[h] = max(host_end_slot.get(h, 0), end_slot)
            host_peak_cpu[h] = host_peak_cpu.get(h, 0.0) + per_node_cpu

    host_bucket = {h: _smallest_fitting_bucket(peak) for h, peak in host_peak_cpu.items()}
    running_demand: dict[int, int] = {}
    for cpu in host_bucket.values():
        running_demand[cpu] = running_demand.get(cpu, 0) + 1

    # Reconcile in NODE units (the conserved quantity — total budget is a
    # fixed 224 nodes; slot-capacity varies by bucket size and isn't
    # conserved across buckets). Every bucket with running-job demand gets a
    # node *floor* it must never drop below; any bucket below its own floor
    # is raised to it; if that pushes total node usage over budget, the
    # excess is reclaimed only from each donor's genuine surplus (its
    # current nodes minus its own floor), largest surplus first.
    #
    # An earlier version of this reconciliation processed one shortfall
    # bucket at a time and let a later iteration steal from a bucket an
    # earlier iteration had just grown to meet ITS OWN running-job floor —
    # silently re-introducing the exact "RunningJob exceeds capacity"
    # truncation this mechanism exists to prevent (verified:
    # mit_supercloud_20210317's 40c bucket was grown to 64 slots to cover
    # its own running-job demand, then immediately raided back down to 44 —
    # 20 short — while fixing the 160c bucket's shortfall right after).
    # Computing every bucket's floor up front, before any stealing happens,
    # fixes that: a donor can only ever give up capacity above its own floor.
    node_alloc = {cpu: _nodes_from_cap(cpu, cap) for cpu, cap in bucket_caps.items()}
    floor_nodes = {cpu: _nodes_from_cap(cpu, needed) for cpu, needed in running_demand.items()}
    grown = []
    for cpu, floor in floor_nodes.items():
        if node_alloc.get(cpu, 0) < floor:
            node_alloc[cpu] = floor
            grown.append(cpu)

    over_budget = sum(node_alloc.values()) - _N_NODES
    if over_budget > 0:
        donors = sorted(
            node_alloc, key=lambda c: node_alloc[c] - floor_nodes.get(c, 0), reverse=True,
        )
        for donor in donors:
            if over_budget <= 0:
                break
            surplus = node_alloc[donor] - floor_nodes.get(donor, 0)
            take = min(surplus, over_budget)
            if take <= 0:
                continue
            node_alloc[donor] -= take
            over_budget -= take

    bucket_caps = {cpu: _cap_from_nodes(cpu, nodes) for cpu, nodes in node_alloc.items() if nodes > 0}
    bucket_cpus_sorted = sorted(bucket_caps)
    type_id_by_cpu = {cpu: idx for idx, cpu in enumerate(bucket_cpus_sorted)}
    for cpu in grown:
        print(f"  grew {cpu}c bucket capacity to {bucket_caps[cpu]} slots "
              f"({running_demand[cpu]} running-job slots needed) — took nodes from "
              f"other buckets' surplus", file=sys.stderr)

    # mem is deliberately unconstrained (_BIG, matching fedhpc_policy.py's
    # own convention exactly — that live-DES reference implementation sets
    # mem=_BIG=1e12 on every synthetic bucket too). An earlier version of
    # this script instead scaled each bucket's mem proportionally to its cpu
    # share of one node (e.g. a 1c bucket got 384/40=9.6GB) — a constraint
    # fedhpc_policy.py never applies, wise_burst's own DES doesn't track
    # memory at all (verified: no "mem" reference anywhere in
    # wise_burst/cluster_model/{node,partition,job}.py), and which turned
    # out to be actively wrong for this dataset: mit_supercloud_20210604's
    # 1-core jobs request up to 20GB (verified against mem_mb directly),
    # so proportional scaling spuriously declared 24/47 of them infeasible
    # even though they need only 1 core. job.mem is still populated from the
    # real mem_mb for anyone building a genuine memory-aware capacity model
    # later — it just isn't used as a bucket-selection constraint here.
    _BIG = 1e12
    instance_types = []
    for bucket_cpu in bucket_cpus_sorted:
        instance_types.append(dict(
            # kind must be exactly "on-prem" for fedhpc's own CLI/viz on-prem-vs-cloud
            # classification (viz.py: `m.kind == "on-prem" and m.capacity is not None`) —
            # a display-only field per data.py's docstring, but the report layer keys
            # off it, so a descriptive label here would silently mis-bucket every job as
            # "cloud" in the top-line summary even though these are all synthetic
            # capacity buckets of the same real 224-node on-prem pool.
            id=type_id_by_cpu[bucket_cpu], kind="on-prem", perf=1.0,
            io_time=0.0, deploy=0, cost_io=0.0, cost_vm=0.0, cost_stor=0.0,
            cpu=bucket_cpu, mem=_BIG, stor=_BIG, capacity=bucket_caps[bucket_cpu],
        ))
    print(f"  capacity buckets: {bucket_caps}", file=sys.stderr)

    # ── jobs (window_jobs only — mirrors fedhpc_schedule_scope=window_jobs) ─
    # Each job goes to the smallest bucket that covers its cpu demand
    # (_select_bucket_type_id, same as the live DES policy). A job can still
    # end up infeasible if its own demand exceeds every bucket fedhpc chose
    # to keep (e.g. the node budget was spent elsewhere) — drop and report,
    # mirroring build_pos_congestion_instance.py's horizon-drop precedent.
    dropped_infeasible = 0
    dropped_horizon = 0
    jobs: list[dict] = []
    for r in window:
        cpu_req = float(r["total_cpus"])
        type_id = _select_bucket_type_id(type_id_by_cpu, cpu_req)
        bucket_cpu = bucket_cpus_sorted[type_id]
        mem_mb = r.get("mem_mb")
        mem_gb = (mem_mb / 1024.0) if isinstance(mem_mb, (int, float)) and mem_mb == mem_mb else 1.0
        if cpu_req > bucket_cpu:
            dropped_infeasible += 1
            continue
        submit = pd.Timestamp(r["@submit"])
        arrival = (submit - WINDOW_START).total_seconds() / time_scale_seconds
        exec_time = r["elapsed"] / time_scale_seconds
        # Best-case (zero queueing) completion must fit the horizon, or
        # Instance.build() rejects the whole instance outright rather than
        # just this job — mirrors build_pos_congestion_instance.py's
        # horizon-drop precedent (there driven by multi-day reservoir-sim
        # outliers; here by a short but real long-running tail — 45/17,769
        # window jobs here run past 4h, up to 23.9h).
        if arrival + exec_time > horizon:
            dropped_horizon += 1
            continue
        jobs.append(dict(
            id=len(jobs), arrival=arrival, exec_time=exec_time, io_volume=0.0,
            cpu=cpu_req, mem=mem_gb, stor=0.0,
        ))
    if dropped_infeasible:
        print(f"  dropped {dropped_infeasible}/{len(window)} window jobs with no feasible bucket",
              file=sys.stderr)
    if dropped_horizon:
        print(f"  dropped {dropped_horizon}/{len(window)} window jobs whose best-case finish "
              f"exceeds horizon ({horizon} slots)", file=sys.stderr)

    # ── running jobs (initial_state jobs already running at t0) ─────────────
    # fedhpc's space-time network formulation grants each RunningJob one
    # whole INSTANCE of its type — no intra-slot packing — so a real
    # hostname's *aggregate* concurrent cpu usage at t0 must be expressed as
    # one entry in the bucket that fits it (host_bucket, computed above), not
    # one entry per (job, hostname) pair (verified: the 405 running-at-t0
    # jobs in the 20210108 window occupy only 63 distinct hostnames, up to 25
    # jobs sharing one host — one whole-node RunningJob per job would wildly
    # overcount occupied capacity).
    running_jobs: list[dict] = [
        dict(id=idx, type_id=type_id_by_cpu[host_bucket[h]], end=end_slot)
        for idx, (h, end_slot) in enumerate(host_end_slot.items())
    ]

    print(f"  jobs={len(jobs)}  running_jobs={len(running_jobs)} "
          f"(consolidated from {n_running_raw} running initial_state_jobs sharing "
          f"{len(host_end_slot)} distinct hosts, each in its own fitting bucket)", file=sys.stderr)

    out = dict(
        _info=dict(
            description=(
                f"FED-HPC offline scheduling instance — {args.experiment} case study "
                f"(window_start={WINDOW_START.isoformat()}, MIT Supercloud Dataset "
                "'normal'/xeon-g6 partition). "
                "exec_time = actual elapsed / slot_size_seconds (known runtimes). "
                "On-prem capacity is split into demand-weighted buckets "
                "(_choose_capacity_buckets, ported from fedhpc_policy.py) instead of one "
                "flat 40c type, so both sub-node packing and multi-node jobs are covered. "
                "Built by scripts/build_mit_supercloud_instance.py — a standalone converter "
                "reading window_jobs/initial_state_jobs directly (NOT the live overseer "
                "simulator); see that script's docstring for generation rules."
            ),
            capacity_buckets=bucket_caps,
            source_jobs=f"experiments/{args.experiment}/{{window_jobs,initial_state_jobs}}",
            fedhpc_settings=dict(
                objective="weighted_sum",
                alpha=0.5,
                time_scale_seconds=time_scale_seconds,
                horizon_slots=horizon,
                budget=1_000_000_000,
                onprem_cost=0.0,
            ),
        ),
        horizon=horizon,
        budget=1_000_000_000,
        instance_types=instance_types,
        jobs=jobs,
        running_jobs=running_jobs,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"Wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
