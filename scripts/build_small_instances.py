"""Build small FED-HPC instances by subsetting the real
data/fedhpc_known_runtime_offline_10min.json trace, so `true_pareto_frontier`
(exact box-splitting) can enumerate the complete front quickly.

Only the first N jobs (by arrival) are kept, renumbered 0..N-1. Only a small
handful of instance types are kept (one on-prem type with *rescaled-down*
capacity so congestion actually bites at this scale, one on-prem type for the
bigger jobs, and cheap/expensive cloud alternatives) instead of all 29 -
otherwise the assignment search space stays large even with few jobs. Horizon
is picked per size to be tight enough to bound the box-splitting search while
still leaving a feasible schedule for every job.

Usage: uv run python scripts/build_small_instances.py
Writes data/tiny_smallest.json, tiny_small.json, tiny_medium.json,
tiny_large.json, tiny_xlarge.json.
"""
from __future__ import annotations

import math
import json

SRC = "data/fedhpc_known_runtime_offline_10min.json"
KEEP_TYPE_IDS = [0, 3, 5, 6, 17]  # on-prem(cpu40), on-prem(cpu64), cloud cheap, cloud expensive, cloud cpu64

# On-prem capacity is deliberately kept scarce relative to n_jobs (unlike the
# real trace, where it's plentiful) - otherwise every job fits on free
# capacity immediately and the turnaround/cost tradeoff that makes the front
# non-trivial disappears. (n_jobs, cap for type 0, cap for type 3, horizon slack)
SIZES = {
    "tiny_smallest": (5, 1, 1, 15),
    "tiny_small": (8, 2, 1, 20),
    "tiny_medium": (12, 2, 1, 25),
    "tiny_large": (18, 3, 2, 30),
    "tiny_xlarge": (25, 4, 2, 35),
}


def build(n_jobs: int, cap0: int, cap3: int, slack: int, src: dict) -> dict:
    jobs = sorted(src["jobs"], key=lambda j: j["arrival"])[:n_jobs]
    new_jobs = []
    for new_id, j in enumerate(jobs):
        nj = dict(j)
        nj["id"] = new_id
        new_jobs.append(nj)

    caps = {0: cap0, 3: cap3}
    types = []
    for t in src["instance_types"]:
        if t["id"] not in KEEP_TYPE_IDS:
            continue
        nt = dict(t)
        if t["id"] in caps:
            nt["capacity"] = caps[t["id"]]
        types.append(nt)

    # Minimum horizon needed for every job to have *some* feasible type/start:
    # the fastest feasible type's occupation time, from its own arrival.
    min_h = 0
    for j in new_jobs:
        best = min(
            math.ceil(j["exec_time"] / t["perf"] + t["io_time"] * j["io_volume"] + t["deploy"])
            for t in types
            if j["cpu"] <= t["cpu"] and j["mem"] <= t["mem"] and j["stor"] <= t["stor"]
        )
        min_h = max(min_h, math.ceil(j["arrival"]) + max(1, best))

    horizon = min_h + slack

    return {
        "horizon": horizon,
        "budget": src["budget"],
        "instance_types": types,
        "jobs": new_jobs,
        "running_jobs": [],
    }


def main():
    with open(SRC) as f:
        src = json.load(f)

    for name, (n_jobs, cap0, cap3, slack) in SIZES.items():
        inst = build(n_jobs, cap0, cap3, slack, src)
        out_path = f"data/{name}.json"
        with open(out_path, "w") as f:
            json.dump(inst, f, indent=2)
        print(f"wrote {out_path}  ({n_jobs} jobs, {len(inst['instance_types'])} types, H={inst['horizon']})")


if __name__ == "__main__":
    main()
