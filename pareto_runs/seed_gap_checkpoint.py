"""Carve a GAP checkpoint out of an in-progress map_pareto_frontier checkpoint.

The sparse tail of a Pareto front (typically the turnaround-minimal end) is
much cheaper to close than the dense end, but a single map_pareto_frontier
search always spends its budget on the *currently largest* gap, which for a
long time is in the dense region. Splitting the tail into its own independent
checkpoint lets a second process work it in parallel — own restart file, own
logs, no contention with the dense-region search.

This mirrors how pareto_runs/fedhpc_10min_gap_checkpoint.json was split off
pareto_runs/fedhpc_10min_map_checkpoint.json.

Usage:
    uv run python pareto_runs/seed_gap_checkpoint.py <main_ckpt> <gap_ckpt> <f1_split>

  * low anchor  = the proven point in <main_ckpt> with the smallest f1 >= f1_split
  * high anchor = the proven point with the largest f1 (anchor B)
  * one open box between them, direction "f1"

Refuses to overwrite an existing <gap_ckpt>. The two checkpoints stay fully
independent afterwards; merge them for analysis with _filter_dominated (see
scripts/compare_moea_vs_known_front.py).
"""
import json
import os
import sys

if len(sys.argv) != 4:
    sys.exit(__doc__)

main_ckpt, gap_ckpt, f1_split = sys.argv[1], sys.argv[2], float(sys.argv[3])

if os.path.exists(gap_ckpt):
    sys.exit(f"refusing to overwrite existing {gap_ckpt}")

with open(main_ckpt) as f:
    d = json.load(f)

sols = sorted(d["solutions"], key=lambda s: s["f1"])
lo = next((s for s in sols if s["f1"] >= f1_split), None)
if lo is None:
    sys.exit(f"no proven point with f1 >= {f1_split} in {main_ckpt} "
             f"(max f1 = {sols[-1]['f1']:.4f})")
hi = sols[-1]
if hi["f1"] - lo["f1"] <= 1:
    sys.exit("low and high anchor are adjacent — nothing to explore")

print(f"low  anchor  f1={lo['f1']:.4f}  f2={lo['f2']:.6g}")
print(f"high anchor  f1={hi['f1']:.4f}  f2={hi['f2']:.6g}")
print(f"gap width    {hi['f1'] - lo['f1']:.1f} (f1)")

gap = {
    "solutions": [lo, hi],
    "boxes": [[lo, hi]],
    "direction": "f1",
    "n_solves": 0,
    "build_time": 0.0,
    "n_boxes_unresolved": 0,
}

tmp = gap_ckpt + ".tmp"
with open(tmp, "w") as f:
    json.dump(gap, f)
os.replace(tmp, gap_ckpt)
print(f"wrote {gap_ckpt}")
