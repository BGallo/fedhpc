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
cd /home/bernardo/Documentos/programas/fedhpc
# both regions, back-to-back (recommended):
bash scripts/resume_true_frontier.sh 3600

# or one checkpoint at a time:
uv run python pareto_runs/resume_map.py \
    data/fedhpc_known_runtime_offline_10min.json \
    pareto_runs/fedhpc_10min_map_checkpoint.json \
    3600 200 3   # time_budget_s  per_solve_time_limit_s  n_seed_retries
```

Safe to run repeatedly — each call resumes exactly where the last one left
off and re-saves the checkpoint after every box resolution. If interrupted
(Ctrl-C, kill, crash), nothing is lost beyond the box currently mid-solve.

## Current state (as of 2026-08-28, after a further ~1h + ~1h resume session)

History: 2026-08-15 checkpoint 9 pts / 5 boxes / 29 solves → 2026-08-27
+17 interior pts (63 solves, 3771s) → 2026-08-28 session below. Backup of the
pre-2026-08-27 MAIN checkpoint: `fedhpc_10min_map_checkpoint.json.bak-20260827`.

**Combined exact-point count across both checkpoints: 52** (up from 33).
Every point is individually proven optimal (MIPGap 1e-9); the front is still
incomplete (47 open boxes total + 5 boxes dropped as inconclusive over the
whole history).

### 2026-08-28 session — `scripts/resume_true_frontier.sh 3600`

Ran both checkpoints back-to-back, 3600s budget each (200s per-solve for MAIN,
600s for GAP), `n_seed_retries=3`. Added **19 new proven points** (+16 MAIN,
+3 GAP).

### MAIN checkpoint — `fedhpc_10min_map_checkpoint.json`

- **42 exact points** (2 anchors + 40 interior), all in f1 ∈ [11151, 16577].
  This session: 3682s wall, 32 solves, **0 inconclusive**, +16 pts.

  | f1 | f2 |   | f1 | f2 |
  |---|---|---|---|---|
  | 11151.04 | 361.96 | | 12979.04 | 140.21 |
  | 11156.04 | 359.21 | | 13042.04 | 136.08 |
  | 11160.04 | 357.09 | | 13105.04 | 131.98 |
  | 11168.04 | 352.96 | | 13168.04 | 127.91 |
  | 11179.04 | 348.40 | | 13231.04 | 123.88 |
  | 11189.04 | 344.32 | | 13294.04 | 119.85 |
  | 11201.04 | 339.71 | | 13326.04 | 117.84 |
  | 11213.04 | 335.49 | | 13358.04 | 115.81 |
  | 11226.04 | 331.00 | | 13421.04 | 111.87 |
  | 11238.04 | 327.14 | | 13484.04 | 107.95 |
  | 11253.04 | 322.58 | | 13515.04 | 106.03 |
  | 11267.04 | 318.29 | | 13547.04 | 104.06 |
  | 11275.04 | 315.92 | | 13579.04 | 102.08 |
  | 11283.04 | 313.68 | | 13611.04 | 100.13 |
  | 11298.04 | 309.63 | | 13674.04 |  96.29 |
  | 11315.04 | 305.20 | | 13737.04 |  92.49 |
  | 11332.04 | 300.89 | | 13800.04 |  88.70 |
  | 11349.04 | 296.67 | | 13832.04 |  86.80 |
  | 11367.04 | 292.47 | | 13864.04 |  84.94 |
  | 11843.04 | 223.43 | | 16577.04 |   0.00 |
  | 12853.04 | 148.51 | |          |        |
  | 12916.04 | 144.34 | |          |        |

- **38 boxes still open** (widths 4–63), all in f1 ∈ [11151, 13864]. The
  front is essentially continuous here — bisection finds a new optimal point
  in nearly every box it splits, so each solve tends to spawn ~1 more box.
  Open-box count keeps *rising* (22 → 38). Full closure of this region still
  looks like many more hours (rough order: 100–200+ total points).
- Nearly every MAIN solve hits the 200s `per_solve_time_limit`; throughput
  ~1 point / 3–4 min.
- 3 boxes dropped as inconclusive earlier in the history (see limitation #1);
  none new this session.

### GAP checkpoint — `fedhpc_10min_gap_checkpoint.json`

The big f1 ∈ [13864, 16577] gap, attacked separately (independent checkpoint,
600s `per_solve_time_limit`). Resume with:
```
uv run python pareto_runs/resume_map.py \
    data/fedhpc_known_runtime_offline_10min.json \
    pareto_runs/fedhpc_10min_gap_checkpoint.json  3600 600 3
```

- **12 exact points**, f1 ∈ [13864, 16577]:

  | f1 | f2 |
  |---|---|
  | 13864.04 | 84.94 |
  | 14154.04 | 68.39 |
  | 14308.04 | 60.09 |
  | 14466.04 | 51.84 |
  | 14650.04 | 42.54 |
  | 14843.04 | 33.34 |
  | 15031.04 | 25.27 |
  | 15220.04 | 18.81 |
  | 15559.04 | 12.44 |
  | 15898.04 |  7.13 |
  | 16237.04 |  3.01 |
  | 16577.04 |  0.00 |

- This session: 5145s wall (one solve overran the 600s cap in presolve,
  pushing past the 3600s budget), 12 solves, +3 pts, **2 inconclusive** —
  a change from the first gap session (0 inconclusive). The two dropped
  boxes are `[13864.04, 14154.04]` and `[14466.04, 14650.04]`; per
  limitation #1 they are gone from the search and will not be revisited on
  resume. To recover them: reproduce the sub-problem with a longer
  `per_solve_time_limit` / more seeds (manual).
- **9 boxes still open** (widths 154–340). Sparser than MAIN — a few more
  hours should close most of it.

### Merging the two checkpoints

They are independent. To get the combined front, load `solutions` from both
JSONs and run `_filter_dominated` (they share the two anchors 13864.04/84.94
and 16577.04/0.0). `scripts/compare_moea_vs_known_front.py` already does this.

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

## Related code

- `map_pareto_frontier()` and `true_pareto_frontier()` in
  `src/fedhpc/pareto.py` both gained: `max_solves`/`time_budget` (bounded
  partial runs), `per_solve_time_limit`/`n_seed_retries` (per-solve safety
  net with seed-retry), and `checkpoint_path` (this feature).
- `map_pareto_frontier` is the newer bisection-based sibling of
  `true_pareto_frontier` — same exactness guarantee per point, but explores
  the largest-known-gap first (alternating f1/f2 direction) instead of
  always resolving the immediate neighbour, so it gives broad coverage of
  the whole range quickly instead of exhaustively resolving one end first.
- `scripts/resume_true_frontier.sh <budget_s>` runs both checkpoints
  back-to-back with the right per-solve limits.
- `scripts/compare_moea_vs_known_front.py` scores the NSGA-II/III + MOEA/D
  heuristics against this combined proven front (loads both checkpoints,
  `_filter_dominated`). As of 2026-08-28 the EAs get IGD ≈ 0.05–0.10
  normalised, 0 exact hits, and never dominate a proven point — see the
  "FED-HPC Evolutionary Solver" artifact / `scripts/ea_improvement_prompt.md`.
