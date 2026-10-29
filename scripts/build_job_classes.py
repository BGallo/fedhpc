"""Job class map for the result-data model (fedhpc.io_model): which jobs of
each source workload are ensemble members and which are single runs.

Reservoir studies (history matching, optimisation, uncertainty analysis) run
the same deck many times with varied parameters; their job names share a
stem and differ by a trailing run number (``BL32_REVIT_OTIMIZA_V0_00384``).
A job is an *ensemble member* when at least ``--min-size`` jobs of the
workload share its (user, name stem); the stem is the job name with trailing
``[_-.]<digits>`` groups removed. Everything else is a *single run*.

Reads the Slurm records the workloads were built from (overseer
experiments/*/window_jobs) and matches them to the instance jobs:
``fedhpc`` through its ``job_trace_metadata`` job ids, ``pos`` by replaying
build_pos_congestion_instance.py's order (window order, horizon-infeasible
jobs removed — checked by elapsed time and core count). Only the 0/1 labels
are written (no names or users): data/job_classes_<workload>.json.
"""
from __future__ import annotations

import argparse
import ast
import collections
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OVERSEER = ROOT.parent / "overseer" / "experiments"
SOURCES = {
    "fedhpc": (ROOT / "data" / "fedhpc_known_runtime_offline_10min.json",
               OVERSEER / "tcc_2026" / "offloading_strategy_comparison" / "window_jobs"),
    "pos": (ROOT / "data" / "pos_congestion_known_runtime_offline_10min.json",
            OVERSEER / "pos_congestion_20240130" / "window_jobs"),
}


def load_records(path: Path) -> list[dict]:
    return [eval(line, {"nan": float("nan")}) for line in path.read_text().splitlines() if line.strip()]


def stem(name) -> str:
    name, prev = str(name), None
    while prev != name:
        prev, name = name, re.sub(r"[_\-.]?\d+$", "", name)
    return name


def match(workload: str, inst: dict, records: list[dict]) -> list[dict]:
    """The Slurm record of every instance job, in job-id order."""
    slot = inst["_info"]["fedhpc_settings"]["time_scale_seconds"]
    if workload == "fedhpc":
        meta = inst["_info"]["job_trace_metadata"]
        meta = ast.literal_eval(meta) if isinstance(meta, str) else meta
        by_id = {r["jobid"]: r for r in records}
        rows = [by_id[meta[str(j["id"])]["jobid"]] for j in inst["jobs"]]
    else:
        it, rows = iter(records), []
        for j in inst["jobs"]:
            rows.append(next(r for r in it if abs(r["elapsed"] / slot - j["exec_time"]) < 1e-6
                             and float(r["total_cpus"]) == j["cpu"]))
    for r, j in zip(rows, inst["jobs"]):
        if abs(r["elapsed"] / slot - j["exec_time"]) > 1e-6 or float(r["total_cpus"]) != j["cpu"]:
            raise RuntimeError(f"{workload}: job {j['id']} does not match Slurm job {r['jobid']}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-size", type=int, default=10)
    args = ap.parse_args()
    for workload, (inst_path, rec_path) in SOURCES.items():
        inst = json.loads(inst_path.read_text())
        rows = match(workload, inst, load_records(rec_path))
        key = [(r["username"], stem(r["job_name"])) for r in rows]
        size = collections.Counter(key)
        ensemble = [int(size[k] >= args.min_size) for k in key]
        out = ROOT / "data" / f"job_classes_{workload}.json"
        out.write_text(json.dumps(dict(
            _info=dict(source_workload=str(inst_path.relative_to(ROOT)),
                       rule=f"ensemble = at least {args.min_size} jobs share (user, job-name stem)",
                       built_by="scripts/build_job_classes.py"),
            ensemble=ensemble)) + "\n")
        print(f"{workload}: {sum(ensemble)}/{len(ensemble)} ensemble members "
              f"({len(size)} (user, stem) groups) -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
