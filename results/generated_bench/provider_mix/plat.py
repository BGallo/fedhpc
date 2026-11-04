import json, numpy as np
from fedhpc import instance_gen as g
for w in ("fedhpc", "pos"):
    src = g._workload(w); J = src["jobs"]
    q = lambda k: np.percentile([j[k] for j in J], [0, 50, 90, 100])
    print(w, len(J), "H", src["horizon"], "slot", src["_info"]["fedhpc_settings"]["time_scale_seconds"])
    print("  keys", list(J[0].keys()))
    for k in ("cpu", "mem", "exec_time", "arrival"):
        print(f"  {k} p0/50/90/100", q(k).round(2))
    print("  core-slots total", sum(j["cpu"] * j["exec_time"] for j in J), " weights", set(j.get("weight", 1) for j in J))
    ends = [r["end"] for r in src["running_jobs"]]
    print("  running pool", len(ends), "end p10/50/90/max", np.percentile(ends, [10, 50, 90, 100]))
    for io in "LMH":
        d = json.load(open(f"data/generated/A_{w}_n300_lE_i{io}.json"))
        v = np.array([j["io_volume"] for j in d["jobs"]])
        print(f"  io {io}: total {v.sum():.0f} GB, median {np.median(v):.2f}, p90 {np.percentile(v,90):.1f}, max {v.max():.0f}")
    for c in "AB":
        print(" ", c, "onprem", [t.name for t in g.onprem_types(c, w)], "cloud", g.cloud_types(c, w))
