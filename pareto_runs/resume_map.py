"""Resume a checkpointed map_pareto_frontier() search.

Usage:
    uv run python pareto_runs/resume_map.py <instance.json> <checkpoint.json> \
        [time_budget_s] [per_solve_time_limit_s] [n_seed_retries]

Safe to re-run with the same checkpoint path as many times as needed — each
call picks up exactly where the last one left off (see map_pareto_frontier's
docstring for the checkpoint contract). If the checkpoint shows the search
already fully resolved (no boxes left), this returns immediately.
"""
import sys
import time

from fedhpc.data import Instance
from fedhpc.formulations import configure_env
from fedhpc.pareto import map_pareto_frontier

INSTANCE = sys.argv[1]
CHECKPOINT = sys.argv[2]
TIME_BUDGET = float(sys.argv[3]) if len(sys.argv) > 3 else 3600.0
PER_SOLVE_LIMIT = float(sys.argv[4]) if len(sys.argv) > 4 else 200.0
N_RETRIES = int(sys.argv[5]) if len(sys.argv) > 5 else 3

configure_env(verbose=True)

inst = Instance.from_file(INSTANCE)
print(f"jobs={len(inst.jobs)} types={len(inst.instance_types)}  "
      f"time_budget={TIME_BUDGET}s  per_solve_time_limit={PER_SOLVE_LIMIT}s  "
      f"n_seed_retries={N_RETRIES}  checkpoint={CHECKPOINT}", flush=True)

t0 = time.time()
sols = map_pareto_frontier(
    inst, verbose=True, time_budget=TIME_BUDGET,
    per_solve_time_limit=PER_SOLVE_LIMIT, n_seed_retries=N_RETRIES,
    checkpoint_path=CHECKPOINT, OutputFlag=1,
)
dt = time.time() - t0

print(f"\nmap_pareto_frontier session DONE in {dt:.1f}s -> {len(sols)} points "
      f"(check the log above for 'gap(s) still open' to see if this is complete)",
      flush=True)
for s in sorted(sols, key=lambda s: s.f1):
    print(f"  f1={s.f1:.4f} f2={s.f2:.6g}")
