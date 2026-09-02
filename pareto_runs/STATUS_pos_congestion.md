# Pareto frontier mapping — pos_congestion 10min — session state

## What this is

An in-progress **exact** mapping of the true Pareto frontier for
`data/pos_congestion_known_runtime_offline_10min.json`
(3340 window jobs + 2617 running jobs, 29 types — the pos/CSR1 2024-01-30
congestion event) using `map_pareto_frontier()` (bisection-based exact
exploration — see its docstring in `src/fedhpc/pareto.py`).

Same method and same code as the fedhpc 964-job 10min front
(`pareto_runs/STATUS.md`), but a **completely separate** set of files so the
two fronts can be worked on simultaneously with no contention:

| purpose            | fedhpc 964-job front                        | this front (pos_congestion)                          |
|--------------------|---------------------------------------------|-----------------------------------------------------|
| MAIN restart file  | `fedhpc_10min_map_checkpoint.json`          | `pos_congestion_10min_map_checkpoint.json`          |
| GAP restart file   | `fedhpc_10min_gap_checkpoint.json`          | `pos_congestion_10min_gap_checkpoint.json`          |
| resume script      | `scripts/resume_true_frontier.sh`           | `scripts/resume_pos_congestion_frontier.sh`         |
| logs               | `resume_main_*.log` / `resume_gap_*.log`    | `resume_pc_main_*.log` / `resume_pc_gap_*.log`      |
| status doc         | `pareto_runs/STATUS.md`                      | this file                                            |

Every point recorded here is individually **proven optimal** (exact MIP solve,
MIPGap forced to 1e-9). The map is a partial-but-exact approximation: each
point is real, the front is just not yet exhaustively closed.

## How to resume

```bash
cd /home/bernardo/Documentos/programs/fedhpc

# MAIN region only (default per-solve limit 900s):
bash scripts/resume_pos_congestion_frontier.sh 3600 main

# GAP region only — ONLY after it has been seeded (see below):
bash scripts/resume_pos_congestion_frontier.sh 3600 gap

# both, back-to-back:
bash scripts/resume_pos_congestion_frontier.sh 3600 both
```

Safe to run repeatedly — `map_pareto_frontier` re-saves the checkpoint after
every box resolution, so each call picks up exactly where the last left off.
Interrupts lose at most the box currently mid-solve.

## Seeding the GAP checkpoint

Once the MAIN checkpoint has anchor B plus a sparse tail, split that tail into
its own independent checkpoint so it can be closed in parallel:

```bash
uv run python pareto_runs/seed_gap_checkpoint.py \
    pareto_runs/pos_congestion_10min_map_checkpoint.json \
    pareto_runs/pos_congestion_10min_gap_checkpoint.json \
    <f1_split>          # e.g. the f1 where the MAIN points start thinning out
```

## Current state

- **2026-09-01 (config fix)** — the oversubscribed on-prem "ecn" fat node
  (old type id 4: 1000 cpu / 8000 GB / capacity 2) was removed from
  `data/pos_congestion_known_runtime_offline_10min.json` and from the
  generator `scripts/build_pos_congestion_instance.py`. Instance now has
  **28 types** (4 on-prem ids 0–3, 24 cloud ids 4–27); the cloud ids were
  renumbered down by 1. No job or running job referenced the fat node, so
  nothing became infeasible (max job = 64 cpu / 480 GB, fits on-prem "csr3").
  Original 29-type JSON kept as `...json.bak-fatnode`. **All earlier anchor
  solves / checkpoints below were on the 29-type config and are archived in
  `pareto_runs/_stale_fatnode/` — the mapping was restarted from scratch.**

- **2026-09-01** — infrastructure created. A 90s probe solve of anchor A
  revealed how much bigger this MIP is than the fedhpc one:

  ```
  Optimize a model with 11722 rows, 25647491 columns and 98155041 nonzeros
  Variable types: 0 continuous, 25647491 integer (25639139 binary)
  Presolve time: 70.50s
  Explored 0 nodes ... Time limit reached ... Solution count 0
  ```

  ~25.6M binary variables, 98M nonzeros (`SpaceTimeFormulation`, ≈6000
  job-entities × 29 types × 288 slots). Presolve alone is ~70s and a 90s
  budget found **no feasible point at all**. This is a genuinely large MIP —
  a single exact anchor solve (MIPGap 1e-9) may take many hours or may not
  converge in a practical budget. Exact mapping of this front is therefore
  much more expensive than the fedhpc one and may turn out to be
  intractable at this tolerance.
  (On the 29-type config the anchors resolved fine — ~4 min per proven solve
  once B&B started; the probe was just too short. So 3600s per solve is
  generous.)
- Per-solve limits **3600s** (overridable via `PC_MAIN_PER_SOLVE` /
  `PC_GAP_PER_SOLVE`).
- **MAIN run: restarted 2026-09-01 on the 28-type config** via
  `pareto_runs/loop_pos_congestion_main.sh` (4h rounds, resumes the
  checkpoint until 0 open boxes). Logs `pareto_runs/loop_pc_main_*.log` /
  `resume_pc_main_*.log`.
- **PAUSED 2026-09-01 ~12:56** on user request, mid-round (round 0 of its
  14400s budget, killed via SIGTERM ~2h35m in — clean, checkpoint intact, no
  process left running). Current state: **5 proven points, 4 open boxes,
  10 solves, 0 dropped-inconclusive**:

  | f1 (turnaround) | f2 (cost $) |
  |---|---|
  | 44163.67 | 648.526 |
  | 45623.67 | 413.212 |
  | 47707.67 | 177.903 |
  | 49479.67 | 70.538 |
  | 51251.67 | 0.000 |

  Open boxes (all still wide — early days): [44163.67,45623.67] w=1460,
  [45623.67,47707.67] w=2084, [47707.67,49479.67] w=1772,
  [49479.67,51251.67] w=1772. **Resume with**
  `nohup bash pareto_runs/loop_pos_congestion_main.sh 14400 3600 40 > pareto_runs/loop_pc_main_$(date +%Y%m%d_%H%M%S).log 2>&1 &`
- **Objectives** (both formulations, `src/fedhpc/formulations.py:68-84`):
  `f1` = total turnaround `Σ (t + p_occ − a_j)·x_jmt`, in 600 s time-slot
  units; `f2` = total monetary cost `Σ c_proc·x_jmt`, in dollars. ("pos_congestion"
  is the scenario name — the pos/CSR1 congestion *event* — not an objective.)
- **Anchors — 28-type config (current, proven optimal MIPGap 1e-9):**

  | anchor              | f1 = turnaround | f2 = cost ($) | (29-type was) |
  |---------------------|-----------------|---------------|---------------|
  | A (min turnaround)  | 44163.67        | 648.526       | 643.892       |
  | B (min cost)        | 51251.67        | 0.0           | f1 51161.67   |

  Removing the fat node left the *turnaround minimum unchanged* (A.f1 identical
  to 4 dp) — it was never a real capacity bottleneck, just a $0 substitute for
  cloud burst. Cost at min-turnaround rose $4.63 (17 ex-fat-node jobs → cloud);
  turnaround at $0 rose 90 slot-units (22 ex-fat-node jobs → queue on the
  smaller on-prem nodes). Front now spans f1 ∈ [44163.67, 51251.67]
  (width ≈ 7088), f2 ∈ [0, 648.53].
- GAP checkpoint: **not seeded yet** — seed it once the MAIN checkpoint shows
  where the tail thins out:
  `uv run python pareto_runs/seed_gap_checkpoint.py pareto_runs/pos_congestion_10min_map_checkpoint.json pareto_runs/pos_congestion_10min_gap_checkpoint.json <f1_split>`

## Merging for analysis

The two checkpoints are independent. Combined front = load `solutions` from
both JSONs and run `_filter_dominated` (see
`scripts/compare_moea_vs_known_front.py`, which already does this for the
fedhpc front and can be pointed at these files).

## Known limitations

Same as the fedhpc front — see the "Known limitations / follow-ups" section of
`pareto_runs/STATUS.md`:
1. Dropped (inconclusive) boxes are removed permanently, not retried.
2. `per_solve_time_limit` is a soft cap (Gurobi presolve can overrun it).
3. Hard sub-problems trace to Gurobi B&B non-determinism × solution symmetry;
   `n_seed_retries` is the mitigation.
