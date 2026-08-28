#!/usr/bin/env python3
"""Build a fedhpc Instance JSON for the pos_congestion_20240130 case study.

Standalone converter — reads window_jobs / initial_state_jobs directly from
the overseer repo (python-dict-per-line CSVs) and emits a JSON in the same
schema as data/fedhpc_known_runtime_offline.json, WITHOUT running the full
overseer discrete-event simulator.

Generation rules were reverse-engineered from data/fedhpc_known_runtime_offline.json
(the analogous tcc_2026 snapshot) by cross-checking its jobs/time_scale/horizon
against experiments/tcc_2026/offloading_strategy_comparison/window_jobs:

  t0                 = case-study window start (WINDOW_START from prepare_data.py)
  time_scale_seconds  = min positive `elapsed` among window_jobs
  horizon             = ceil(48h / time_scale_seconds)   (confirmed exactly:
                        5575 = ceil(172800 / 31) for the tcc_2026 snapshot)
  job.arrival         = (@submit - t0) / time_scale
  job.exec_time       = elapsed / time_scale
  job.cpu             = total_cpus ; job.mem = mem parsed from tres_req (GB)
  job.stor            = 0 ; job.io_volume = 0
  running_jobs        = initial_state_jobs rows with @start < t0 (i.e. truly
                        already running at t0), one entry per allocated
                        hostname, type_id resolved from the hostname's node
                        group prefix (clb1/csr1/csr2/csr3/ecn).
                        end = ceil((@end - t0) / time_scale).
  instance_types      = identical on-prem/cloud catalog to fedhpc_known_runtime_offline.json
                        (same physical cluster + AWS catalog is shared by both
                        experiments' .cfg files), with cost_vm recomputed for
                        this instance's own time_scale_seconds.

No per-job partition (Slurm) restriction is encoded — matching what is
observable in fedhpc_known_runtime_offline.json's jobs (F_j is resource-only,
i.e. cpu/mem/stor feasibility, no partition field exists in this instance
schema). This is an approximation of the live fedhpc_policy.py behaviour,
which additionally filters inst.F by allowed_partitions inside a running
simulation — that step cannot be reproduced from a static snapshot.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import pandas as pd

_OVERSEER = Path("/home/bernardo/Documentos/programas/overseer")
_EXP = _OVERSEER / "experiments" / "pos_congestion_20240130"
_DATA_DIR = Path(__file__).resolve().parents[1] / "data"

WINDOW_START = pd.Timestamp("2024-01-30 20:00:00")
WINDOW_END = pd.Timestamp("2024-01-31 00:00:00")

_NAN = float("nan")
_TIME_SCALE_SECONDS = 600.0  # overridden by --slot-seconds in main()


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
# Cluster / cloud catalog — identical node & AWS VM definitions to the
# fedhpc_integration / pos_congestion .cfg files (same physical cluster).
# ---------------------------------------------------------------------------

# group -> (cores_per_node, cost_per_hour_per_node, mem_gb_per_node, n_nodes)
_ONPREM_GROUPS = {
    "clb1": (40, 0.95, 300, 9 * 32),
    "csr1": (40, 0.95, 300, 12 * 32 + 22),
    "csr2": (40, 0.95, 300, 16 * 32),
    "csr3": (64, 0.95, 480, 10 * 41 + 38),
    "ecn":  (1000, 0.95, 8000, 2),
}
_ONPREM_ORDER = ["clb1", "csr1", "csr2", "csr3", "ecn"]

# prefix -> (cores, cost_per_hour, mem_gb) — mirrors fedhpc_known_runtime_offline.json
_CLOUD_TYPES = [
    ("aws_c5_24xlarge_spot",  48, 1.13, 512),
    ("aws_c5_24xlarge_od",    48, 4.08, 512),
    ("aws_c5a_24xlarge_spot", 48, 1.13, 512),
    ("aws_c5a_24xlarge_od",   48, 3.70, 512),
    ("aws_c6a_24xlarge_spot", 48, 1.48, 512),
    ("aws_c6a_24xlarge_od",   48, 3.67, 512),
    ("aws_m5_24xlarge_spot",  48, 1.07, 512),
    ("aws_m5_24xlarge_od",    48, 4.61, 512),
    ("aws_m5a_24xlarge_spot", 48, 1.24, 512),
    ("aws_m5a_24xlarge_od",   48, 4.13, 512),
    ("aws_c6i_24xlarge_spot", 48, 1.35, 512),
    ("aws_c6i_24xlarge_od",   48, 4.08, 512),
    ("aws_c6i_32xlarge_spot", 64, 1.53, 512),
    ("aws_c6i_32xlarge_od",   64, 5.44, 512),
    ("aws_c7a_32xlarge_spot", 64, 1.85, 512),
    ("aws_c7a_32xlarge_od",   64, 6.57, 512),
    ("aws_r6i_32xlarge_spot", 64, 2.09, 1024),
    ("aws_r6i_32xlarge_od",   64, 8.06, 1024),
    ("aws_m6i_32xlarge_spot", 64, 1.94, 512),
    ("aws_m6i_32xlarge_od",   64, 6.14, 512),
    ("aws_c7i_48xlarge_spot", 96, 2.32, 1024),
    ("aws_c7i_48xlarge_od",   96, 8.57, 1024),
    ("aws_c7a_48xlarge_spot", 96, 2.50, 1024),
    ("aws_c7a_48xlarge_od",   96, 9.85, 1024),
]

_STOR_GB = 10000


def _group_of_hostname(h: str) -> str | None:
    h = h.strip()
    if h.startswith("clb1"):
        return "clb1"
    if h.startswith("csr1"):
        return "csr1"
    if h.startswith("csr2"):
        return "csr2"
    if h.startswith("csr3"):
        return "csr3"
    if h in ("npaa1838", "npab1838"):
        return "ecn"
    return None


_MEM_RE = re.compile(r"mem=([\d.]+)([A-Za-z]?)")


def _parse_mem_gb(tres_req: str) -> float:
    m = _MEM_RE.search(tres_req or "")
    if not m:
        return 1.0
    val, unit = float(m.group(1)), m.group(2).upper()
    if unit == "M":
        return val / 1024.0
    if unit == "T":
        return val * 1024.0
    return val  # 'G' or unspecified


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slot-seconds", type=float, default=600.0,
                        help="Discretization slot size in seconds (default: 600 = 10 min, "
                             "matching fedhpc_known_runtime_offline_10min.json).")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output path (default: data/pos_congestion_known_runtime_offline_<slot>.json).")
    args = parser.parse_args()

    global _TIME_SCALE_SECONDS
    _TIME_SCALE_SECONDS = args.slot_seconds

    if args.out is not None:
        out_path = args.out
    elif args.slot_seconds == 600.0:
        out_path = _DATA_DIR / "pos_congestion_known_runtime_offline_10min.json"
    else:
        out_path = _DATA_DIR / f"pos_congestion_known_runtime_offline_{int(args.slot_seconds)}s.json"

    print("Loading pos_congestion_20240130 window_jobs / initial_state_jobs …", file=sys.stderr)
    window = _load_dict_lines(_EXP / "window_jobs")
    initial = _load_dict_lines(_EXP / "initial_state_jobs")
    print(f"  window_jobs: {len(window)}  initial_state_jobs: {len(initial)}", file=sys.stderr)

    # ── time scale + horizon ──────────────────────────────────────────────
    # tcc_2026's raw snapshot uses time_scale = min positive elapsed (31s,
    # giving horizon=ceil(48h/31s)=5575) — but that instance turned out to be
    # intractable to even *build* as a MIP (147M binary x-variables; the model
    # build alone exhausted 30+GB RAM and never reached optimize()). The repo
    # already ships a re-discretized fedhpc_known_runtime_offline_10min.json
    # (600s/10-min slots, horizon=289, 7.6M x-vars) for exactly this reason.
    # We mirror that here: fix time_scale_seconds directly instead of deriving
    # it from the smallest job, which also sidesteps pos_congestion's
    # pathological ~2s-job cluster.
    time_scale_seconds = float(_TIME_SCALE_SECONDS)
    horizon = math.ceil(48 * 3600 / time_scale_seconds)
    print(f"  time_scale_seconds={time_scale_seconds}  horizon={horizon}", file=sys.stderr)

    # ── instance types ────────────────────────────────────────────────────
    instance_types: list[dict] = []
    onprem_type_id: dict[str, int] = {}
    type_idx = 0
    for g in _ONPREM_ORDER:
        cores, cost_hr, mem_gb, n_nodes = _ONPREM_GROUPS[g]
        instance_types.append(dict(
            id=type_idx, kind="on-prem", perf=1.0, io_time=0.0, deploy=0,
            cost_io=0.0, cost_vm=0.0, cost_stor=0.0,
            cpu=cores, mem=mem_gb, stor=_STOR_GB, capacity=n_nodes,
        ))
        onprem_type_id[g] = type_idx
        type_idx += 1

    for prefix, cores, cost_hr, mem_gb in _CLOUD_TYPES:
        cost_vm = cost_hr / 3600.0 * time_scale_seconds
        instance_types.append(dict(
            id=type_idx, kind="cloud", perf=1.0, io_time=0.0, deploy=0,
            cost_io=0.0, cost_vm=cost_vm, cost_stor=0.0,
            cpu=cores, mem=mem_gb, stor=_STOR_GB, capacity=None,
        ))
        type_idx += 1

    # ── jobs (window_jobs only — mirrors fedhpc_schedule_scope=window_jobs) ─
    # A handful of window jobs (long reservoir-simulation runs, up to ~495h)
    # cannot finish within the horizon even in the best case (fastest type,
    # zero queueing) — Instance.build() would reject them outright ("no
    # feasible start time"). tcc_2026's own snapshot never hit this because
    # its longest job's best-case finish (1505 slots) was well under its
    # horizon (5575). Here the tail is far heavier (P99.5 = 2132 slots but
    # max = 29827 slots against horizon=2880), so we drop the jobs whose
    # best-case (perf=1, zero deploy) completion exceeds the horizon and
    # report how many/how much that discards.
    dropped = 0
    dropped_elapsed_h = 0.0
    jobs: list[dict] = []
    for r in window:
        submit = pd.Timestamp(r["@submit"])
        arrival = (submit - WINDOW_START).total_seconds() / time_scale_seconds
        exec_time = r["elapsed"] / time_scale_seconds
        if arrival + exec_time > horizon:
            dropped += 1
            dropped_elapsed_h += r["elapsed"] / 3600.0
            continue
        jobs.append(dict(
            id=len(jobs),
            arrival=arrival,
            exec_time=exec_time,
            io_volume=0.0,
            cpu=float(r["total_cpus"]),
            mem=_parse_mem_gb(r.get("tres_req", "")),
            stor=0.0,
        ))
    if dropped:
        print(
            f"  dropped {dropped}/{len(window)} window jobs whose best-case "
            f"finish exceeds horizon (total {dropped_elapsed_h:.1f}h of runtime)",
            file=sys.stderr,
        )

    # ── running jobs (initial_state jobs already running at t0) ────────────
    running_jobs: list[dict] = []
    n_running_raw = 0
    n_unmapped = 0
    for r in initial:
        start = pd.Timestamp(r["@start"])
        if start >= WINDOW_START:
            continue  # pending, not yet running at t0 — out of scope (schedule_scope=window_jobs)
        end = pd.Timestamp(r["@end"])
        end_slot = max(1, math.ceil((end - WINDOW_START).total_seconds() / time_scale_seconds))
        hostnames = [h.strip() for h in str(r.get("nodes", "")).split(",") if h.strip() and h.strip() != "NONE"]
        for h_idx, h in enumerate(hostnames):
            n_running_raw += 1
            g = _group_of_hostname(h)
            if g is None:
                n_unmapped += 1
                continue
            running_jobs.append(dict(
                id=int(r["jobid"]) * 100 + h_idx,
                type_id=onprem_type_id[g],
                end=end_slot,
            ))

    print(f"  jobs={len(jobs)}  running_jobs={len(running_jobs)}"
          f"  (raw hostname-occupancy entries={n_running_raw}, unmapped={n_unmapped})",
          file=sys.stderr)

    out = dict(
        _info=dict(
            description=(
                "FED-HPC offline scheduling instance — pos_congestion_20240130 case study "
                "(2024-01-30 20:00–24:00 UTC, pos/CSR1 congestion event). "
                "exec_time = actual elapsed / slot_size_seconds (known runtimes). "
                "Time slot = elapsed time of the smallest job in the dataset. "
                "Built by scripts/build_pos_congestion_instance.py — a standalone converter "
                "reading window_jobs/initial_state_jobs directly (NOT the live overseer "
                "simulator); see that script's docstring for the reverse-engineered "
                "generation rules and their limitations (no per-job partition restriction)."
            ),
            source_jobs="experiments/pos_congestion_20240130/{window_jobs,initial_state_jobs}",
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
