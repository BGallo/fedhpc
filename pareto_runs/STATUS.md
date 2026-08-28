# Pareto frontier mapping — session state

## What this is

An in-progress exact mapping of the true Pareto frontier for
`data/fedhpc_known_runtime_offline_10min.json` (964 jobs, 29 types) using
`map_pareto_frontier()` (bisection-based exact exploration, see its
docstring in `src/fedhpc/pareto.py`). Every point below is individually
**proven optimal** (exact MIP solve, MIPGap forced to 1e-9) — this is not a
heuristic approximation, just an *incomplete* one: the true front turned out
to be far denser than expected, so full exhaustive coverage is a multi-hour
undertaking rather than the few-minutes job the anchor-only view suggested.

## How to resume

```bash
cd /home/bernardo/Documentos/programs/fedhpc
uv run python pareto_runs/resume_map.py \
    data/fedhpc_known_runtime_offline_10min.json \
    pareto_runs/fedhpc_10min_map_checkpoint.json \
    3600 200 3   # time_budget_s  per_solve_time_limit_s  n_seed_retries
```

Safe to run repeatedly — each call resumes exactly where the last one left
off and re-saves the checkpoint after every box resolution. If interrupted
(Ctrl-C, kill, crash), nothing is lost beyond the box currently mid-solve.

## Current state (as of 2026-08-27, after a further ~63 min resume session)

Previous checkpoint (2026-08-15): 9 points, 5 open boxes, 29 solves.
A 3600s resume run on 2026-08-27 added 17 interior points (63 solves total,
3771s wall). Backup of the pre-2026-08-27 checkpoint:
`fedhpc_10min_map_checkpoint.json.bak-20260827`.

- **26 exact points found** (2 anchors + 24 interior), all proven optimal:

  | f1 | f2 |
  |---|---|
  | 11151.04 | 361.96 |
  | 11160.04 | 357.09 |
  | 11168.04 | 352.96 |
  | 11189.04 | 344.32 |
  | 11213.04 | 335.49 |
  | 11238.04 | 327.14 |
  | 11253.04 | 322.58 |
  | 11267.04 | 318.29 |
  | 11298.04 | 309.63 |
  | 11332.04 | 300.89 |
  | 11367.04 | 292.47 |
  | 11843.04 | 223.43 |
  | 12853.04 | 148.51 |
  | 12979.04 | 140.21 |
  | 13105.04 | 131.98 |
  | 13231.04 | 123.88 |
  | 13294.04 | 119.85 |
  | 13358.04 | 115.81 |
  | 13421.04 | 111.87 |
  | 13484.04 | 107.95 |
  | 13547.04 | 104.06 |
  | 13611.04 | 100.13 |
  | 13737.04 | 92.49 |
  | 13800.04 | 88.70 |
  | 13864.04 | 84.94 |
  | 16577.04 | 0.0 |

- **22 boxes still open** (checkpoint `boxes`), all in f1 ∈ [11151, 13864],
  widths 8–126. The front is essentially continuous here — bisection keeps
  finding a new optimal point in nearly every box it splits, so each solve
  tends to spawn ~1 more box. Full closure of this region looks like many
  more hours of compute (rough order: 100–200+ total points).
- **The big f1 ∈ [13864, 16577] gap (width 2713)** — being attacked
  separately as of 2026-08-27 in its own checkpoint
  `fedhpc_10min_gap_checkpoint.json`, seeded with just the two bracketing
  anchors (13864.04/84.94 and 16577.04/0.0) as one box, run with a longer
  600s `per_solve_time_limit`. Resume it with:
  ```
  uv run python pareto_runs/resume_map.py \
      data/fedhpc_known_runtime_offline_10min.json \
      pareto_runs/fedhpc_10min_gap_checkpoint.json  3600 600 3
  ```
  Points found here must be merged back into the main solution set by hand
  (or just reported together) — the two checkpoints are independent.

  **First gap session (2026-08-27, 3795s, 15 solves, 0 inconclusive):**
  the region is NOT hard — the 600s limit was enough, every solve proved
  optimal. Found 7 new exact interior points:

  | f1 | f2 |
  |---|---|
  | 14154.04 | 68.39 |
  | 14466.04 | 51.84 |
  | 14650.04 | 42.54 |
  | 14843.04 | 33.34 |
  | 15220.04 | 18.81 |
  | 15898.04 | 7.13 |
  | 16237.04 | 3.01 |

  8 boxes still open in the gap (widths 184–678), much sparser than the
  [11151,13864] region — a few more hours should close it fully.
  **Combined exact-point count across both checkpoints: 33.**
- Nearly every solve now hits the 200s `per_solve_time_limit` (the region
  is genuinely hard), so throughput is ~1 point / 3–4 min.

## Known limitations / follow-ups

1. **Dropped boxes are gone, not retried.** When a box's query exhausts all
   `n_seed_retries` attempts without proving optimality, it's permanently
   removed from the search (by design — a deterministic re-solve with the
   same settings would just hang the same way again). Resuming this
   checkpoint will **not** revisit those 3 regions. To specifically retry
   them, they'd need to be re-derived manually (see the diagnostic approach
   used earlier this session: reproduce the exact sub-problem — bound
   values can be inferred from the gap between adjacent known points — and
   try a longer `per_solve_time_limit` or more seeds) — this isn't
   currently automated.
2. **`per_solve_time_limit` is a soft cap, not a hard one.** Gurobi's
   presolve phase doesn't always respect `TimeLimit` promptly — one solve
   in the last session ran presolve for 662s despite a 200s limit. The
   overall search still self-terminates correctly (checked between boxes),
   but a single box's true worst-case cost can exceed
   `n_seed_retries × per_solve_time_limit`.
3. **Root cause of the hard sub-problems**: confirmed (not assumed) to be
   Gurobi's parallel B&B non-determinism interacting with heavy solution
   symmetry (many structurally-interchangeable jobs/slots) — the *same*
   sub-problem solved fresh with `Seed=0` proved optimal in 144s, while
   `Seed=1` hit a 180s time limit stuck at `MIPGap=7e-6`. `Symmetry=2` and
   `MIPFocus=3` were tested and did not help (the latter was actively
   worse — timed out with zero feasible solutions found). The
   `n_seed_retries` mechanism is the mitigation currently in place.

## Related work this session

- `map_pareto_frontier()` and `true_pareto_frontier()` in
  `src/fedhpc/pareto.py` both gained: `max_solves`/`time_budget` (bounded
  partial runs), `per_solve_time_limit`/`n_seed_retries` (per-solve safety
  net with seed-retry), and `checkpoint_path` (this feature).
- `map_pareto_frontier` is the newer bisection-based sibling of
  `true_pareto_frontier` — same exactness guarantee per point, but explores
  the largest-known-gap first (alternating f1/f2 direction) instead of
  always resolving the immediate neighbour, so it gives broad coverage of
  the whole range quickly instead of exhaustively resolving one end first.
