# Why `weighted_solve_dispatch` is far from g* on the generated instances

Study of the heuristic used in `bench_io_noreduc`. That heuristic is
`weighted_solve_dispatch`, an ILS over a per-job auto/cloud/on-prem genome
with a list-scheduling decoder. The memetic GA (`weighted_solve_memetic`) is
one arm (Q6). Date: 2026-11-03.

## Sample and protocol

- **Instances:** 9, chosen from the cells with the largest bench gaps plus two
  controls.
  - A·pos: n100 E, n100 F, n300 H, n500 E
  - B·pos: n100 E, n100 F, n300 H
  - fedhpc: A n100 E, B n100 H
- **Run settings:** io M, λ ∈ {0.25, 0.5, 0.75}, 27 cells. Each instance uses
  the bench's reference points.
- **Measure:** gap = 100·(g − g*)/g*, with g* the bench Gurobi optimum.
- **Variants:** each changes one knob relative to the bench configuration.
  Seed 42, 4 threads, 3 instances in parallel.
- **Scripts:**
  - `scripts/dispatch_diagnose.py`: exact re-solve and oracle decodes, written
    to `diagnose.jsonl` and `exact/`.
  - `scripts/dispatch_ablation.py`: the ablation, written to `ablation.jsonl`.
  - Logs: `../dispatch_study_ablation*.log`.
- **Checks:**
  - The `default` arm reproduces the bench's dispatch g bit-exactly in all 27
    cells.
  - All 27 exact re-solves (reduced model) are optimal and equal to the bench g*.
  - `pytest -k "dispatch or moea"`: 499 passed after the C++ changes.
- **Caveat:** each cell is a single run (seed 42). Differences under about
  1 pp are noise-level.

## Hypotheses and verdicts

| | hypothesis | test | verdict |
|---|---|---|---|
| Q1 | The search budget is too small. | budget4 (4× K, 4× kicks), n_mu3, no_ls | **Rejected.** 12.9× the time buys 1.3 pp mean. |
| Q2 | The decoder cannot represent the optimum. | Decode the exact split (oracle_auto/forced/order). | **Mostly rejected.** The exact split in the exact order reproduces g* (≤ 1 %) in 15/27 cells, and within 2.5 % in 3 more. Of the other 9 cells, 6 are on F-load and 3 are on n300 H. On n300 H, EDF does better than the exact order (9.1 vs 26.1 % at A λ = 0.25), so the exact order is not the best one for this decoder. |
| Q3 | The deadline offload rule sends too many jobs to the cloud. | no_deadline, plus the `why` reasons from the decoder | **Confirmed as the mechanism.** The heuristic offloads 1,100–2,600 `auto` jobs while they wait. At A_pos_n300_lH λ = 0.25 it puts 1,234 jobs on the cloud against 178 in the optimum. Switching the rule off alone is unstable (wins 18, loses 6). |
| Q4 | The WSPT/μ priority order causes Q3: jobs starve, hit their indifference deadline and are offloaded. | Exact order oracle; serial, best_of; **edf**, **edf_add** | **Confirmed. This is the main cause.** EDF on the indifference deadline removes the offloads (off ≈ 0) and cuts the λ = 0.25 median gap from 30 % to 9 %. Changing the decoder (serial, best_of) does not help. |
| Q5 | The pool tie-break picks the wrong pool. | most_free | **Rejected.** Neutral to worse (6 wins, 14 losses). |
| Q6 | The ILS framework is the problem; the memetic GA would do better. | memetic | **Rejected.** Mean 17.6 %, max 75.8 %. It keeps the same decoder and the same failure. |
| Q7 | The io model creates the gap. | Old run without io vs bench by io level | **Rejected.** The gap is the same without io (see the bench summary). |
| P1 | Time goes into decoding. | decode_ms/total_ms | **Confirmed.** 99–100 % of the time; 40 µs–1.4 ms per decode. |
| P2 | Most decodes are wasted. | K64 | **Confirmed.** About 270 decodes per accepted move. K64 is 4.8× faster for +1.6 pp mean gap. |
| P3 | The ILS kicks are inefficient. | no_kick | **Partly.** Kicks take 98 % of the time for 4 pp mean. Without kicks, the result is fragile in one cell (B_pos_n300_lH λ = 0.25: 7.8 → 71.8 %). |

### Mechanism (Q3 + Q4)

1. The decoder serves jobs in WSPT/μ-regret order.
2. On congested pos, long low-priority jobs wait.
3. When a waiting `auto` job's projected turnaround cost exceeds its cloud
   cost (`w1(t+1+pmin−a) + w2·cmin ≥ cloud_score`), the decoder offloads it.
4. Each offload is locally rational, but in total the cost term is 20× the
   optimum's (0.277 vs 0.014 at A_pos_n300_lH λ = 0.25).
5. The optimum instead runs those jobs on-prem in a different order.
6. A gene flip cannot fix this, because forcing a job on-prem without changing
   the order makes it wait even longer (oracle_forced gap up to 267 %).
7. The search therefore moves in a space where the right answer is out of
   reach, and more search (Q1, Q6) does not help.

**The fix:** priority = earliest indifference deadline,
dl_j = min(latest start, a_j + regret_j/(w1·w_j)). It serves each job before
it becomes cheaper to offload, so the decoder rarely needs to offload.
- Spearman ρ between dl_j and the exact on-prem start: +0.6 to +0.9.
- EDF alone loses at λ = 0.75 on n100 pos, where the optimum offloads most
  jobs anyway.
- So the fix is to add EDF as an extra start beside the μ starts (`edf_add`),
  and to let the all-auto decode choose between them.

## Ablation results (27 cells; time = sum over the cells, 3 in parallel)

| variant | median gap λ .25 / .5 / .75 | mean | p90 | max | cells > 10 % | W/T/L vs default | time × |
|---|---|---|---|---|---|---|---|
| default | 30.2 / 19.1 / 7.0 | 21.9 | 76.0 | 94.7 | 12 | – | 1.00 |
| budget4 | 28.7 / 17.3 / 4.6 | 20.6 | 72.7 | 93.8 | 12 | 26/1/0 | 12.9 |
| n_mu3 | 30.2 / 19.1 / 7.0 | 21.7 | | | | 7/20/0 | 2.77 |
| no_ls | 34.5 / 20.6 / 9.1 | 24.8 | | | | 0/0/27 | ≈ 0 |
| no_kick | 62.7 / 19.7 / 8.8 | 26.1 | | | | 0/0/27 | 0.02 |
| K64 | 31.8 / 20.2 / 8.4 | 23.5 | | | | 0/0/27 | 0.21 |
| no_deadline | 29.5 / 11.9 / 3.2 | 16.6 | 38.4 | 96.4 | 12 | 18/3/6 | 1.03 |
| serial | 46.2 / 21.3 / 7.3 | 24.9 | | | | 9/0/18 | 1.05 |
| best_of | 30.2 / 19.1 / 7.1 | 22.0 | | | | 6/14/7 | 2.03 |
| most_free | 30.8 / 19.4 / 7.0 | 22.9 | | | | 6/7/14 | 0.98 |
| memetic | 32.4 / 19.3 / 6.9 | 17.6 | 32.9 | 75.8 | 14 | 15/2/10 | 0.69 |
| edf | 9.1 / 15.7 / 12.4 | 15.2 | | | | 15/0/12 | 1.16 |
| **edf_add** | 9.2 / 3.5 / 7.0 | 9.6 | 22.7 | 26.2 | 8 | 12/1/14 | 1.23 |
| edf_nodl | 9.1 / 3.4 / 2.6 | 12.1 | 28.8 | 78.9 | 9 | 22/0/5 | 1.04 |
| **edf_add_nodl** | 9.2 / 3.8 / 3.1 | **8.4** | 21.9 | 25.1 | 8 | 22/1/4 | 1.53 |
| edf_add_nodl_K64 | 11.1 / 4.8 / 5.8 | 10.7 | 27.3 | 34.2 | 11 | 19/0/8 | **0.34** |

Notes:
- **edf_add losses:** they are small (≤ 2 pp, e.g. A_pos_n300_lH λ = 0.75:
  8.7 → 9.0). Their cause is that the 60 kicks are now split between two starts.
- **edf_add_nodl's 4 losses:** they are on B_pos_n100_lE at λ = 0.5 and 0.75
  (14.7 → 20.0, 5.3 → 21.9). There, the deadline rule is right for the μ start,
  and `no_deadline` alone loses there too.
- **Per-start deadline flag (bound):** taking the better of default and
  edf_nodl in each cell gives a mean of 7.3 % and a max of 24.1 %. This is
  what a per-start flag would approach: the deadline rule on for the μ start,
  off for the EDF start. It is not implemented yet.

### What remains (gap > 10 % under edf_add_nodl)

The remaining gaps are all n100 pos with load F/E, at λ ≤ 0.5 (about 20 %).
- **F-load cells:** decoding the exact split in the exact start order already
  misses g* by 10–33 %, so this is a real decoder ceiling. It comes from the
  non-delay rule and the pool choice: the optimum delays jobs on purpose.
- **E-load n100 cells:** the exact order does reach g*, so better ordering
  (true deadlines from an LP or Lagrangian relaxation) could still close the
  gap.

## Recommendations

(Superseded by the follow-up below: the default became the per-start variant, `64|4096|8192`, with batched steps.)

### Quality

1. **Make EDF the extra start, with no deadline offload on that start.**
   - Gain: mean gap 21.9 → 8.4 %, max 94.7 → 25 %.
   - Implemented as flags `EDF_ADD | NO_DEADLINE` (= 64|4096|1, i.e.
     `flags=4161`).
   - Next step: a per-start deadline flag (bound: mean 7.3 %).
2. **Do not spend on more search budget, n_mu, the memetic GA, serial or
   best_of decoders, or the pool tie-break.** The data show no return from
   any of them.
3. **For the remaining n100 F/E cells:**
   - a decoder that may delay a job (non-delay → limited look-ahead / active
     schedule);
   - priorities seeded from an LP relaxation's start times.

### Performance

1. **Cut wasted decodes.**
   - K64 together with the EDF fix (`edf_add_nodl_K64`) is 3× faster than
     the default and still far better (mean 10.7 % vs 21.9 %).
   - Incremental decoding (restart from a snapshot at the flipped job's
     release) is the structural fix. Each decode is O(H × queue) and about
     270 are rejected per accepted move.
2. **Kicks:** they cost 98 % of the run time. Most of their gain arrives within
   40–60 % of the run, so an adaptive stop (no improvement in k kicks) is the
   cheap win. Removing kicks entirely is not safe (B_pos_n300_lH).
3. **No-deadline decodes are cheaper** (55 vs 83 µs median), because there are
   no offload checks while jobs wait.

## Code changes (uncommitted; default behaviour unchanged)

- `_ext/weighted_dispatch.hpp`:
  - new flags `DF_WHY` (1024), `DF_EDF` (2048), `DF_EDF_ADD` (4096);
  - a per-job offload reason;
  - `set_priority(..., edf)`.
- `_ext/moea.cpp`:
  - `stats["gen"]`;
  - a `dispatch_decode` binding (one decode with reasons, optional explicit
    order).
- `scripts/dispatch_diagnose.py` and `scripts/dispatch_ablation.py` (new).

---

# Follow-up: new defaults and a faster decoder (2026-11-03)

This section covers four requests:
1. Add EDF as an extra starting order.
2. Is EDF better only because it never offloads, so the cost is always lower?
3. K = 64 per step.
4. Stop the ILS after k kicks without improvement.

It also covers a measured plan for a faster decode.

## Is EDF better only because it never offloads?

**Partly. At λ ≤ 0.5, EDF does trade turnaround for cost**, which is the
optimum's trade-off.

| A_pos_n300_lH, λ = 0.25 | turnaround term | cost term |
|---|---|---|
| exact optimum | 0.232 | 0.014 |
| default (old) | 0.202 | 0.277 |
| EDF (`edf_add_nodl`) | 0.250 | 0.019 |

The old default over-offloads. It buys turnaround below the optimum's at 20×
the optimum's cost.

**But "never offloading" is not the mechanism.**
- `no_deadline` also makes 0 deadline offloads, yet its cost term stays high
  (A_pos_n100_lE, λ = 0.25: 0.250 vs EDF 0.064). Under WSPT the starved jobs
  still reach the cloud, through horizon overflow or through genes flipped by
  the search.
- EDF's order is what lets those jobs run on-prem in time.
- Of the 24 cells where EDF changed the split, 16 lowered only the cost term
  and 7 only the turnaround term (e.g. A_pos_n300_lH, λ = 0.75: 0.042 → 0.005);
  1 lowered both.
- EDF still uses the cloud: 209 jobs vs the optimum's 178. It does so through
  genes and horizon overflow, not deadline offloads.

## Changes and their measured effect (all 87 non-trivial instances × 3 λ = 261 cells)

| configuration | mean gap | median | p90 | max | cells > 10 % | time (sum) |
|---|---|---|---|---|---|---|
| old defaults (bench) | 15.1 | 7.3 | 40.7 | 156.5 | 97 | 3411 s |
| EDF start + K=64 + patience 5 (first try) | 8.9 | 5.1 | 21.0 | 62.1 | 82 | 570 s |
| … + K_max=256, kicks focused on the best start | 8.1 | 3.8 | 19.8 | 100.6 | 63 | 676 s |
| … focused, no patience | 7.4 | 3.1 | 18.9 | 98.9 | 56 | 1139 s |
| **… + K_max=256, kicks split over the starts, patience 5 (new default)** | **6.5** | **3.0** | **18.4** | **58.3** | **54** | **984 s** |
| … split, no patience | 6.3 | 2.9 | 18.2 | 59.0 | 51 | 1229 s |

Against the old defaults, the new default is:
- better in 156 of 261 cells and worse by more than 5 pp in 2;
- per λ, mean gap 27.2 / 11.0 / 6.9 → 9.3 / 5.4 / 4.7 at λ = 0.25 / 0.5 / 0.75;
- per class·workload, mean gap A·pos 27.8 → 9.4, B·pos 15.5 → 10.0,
  A·fedhpc 5.9 → 2.0, B·fedhpc 4.6 → 1.7.

Files: `validate_<tag>.jsonl`, written by `scripts/dispatch_validate.py`.

**Caveat:** the first try was chosen on the 9-instance sample and validated on
the 99 held-out instances. The final choice among the four rows below it was
made on all 108, so it has no held-out set. It is a choice among 4 coarse
options, and each cell is a single seed.

What each change did:
- **EDF as an extra start, with no deadline offload on that start**
  (`flags 4096|8192`). The μ start keeps the deadline rule, which helps it on
  B·pos. This is the main quality gain at λ ≤ 0.5.
- **K = 64 alone regressed B·pos n500** at λ ≥ 0.5, e.g. B_pos_n500_lH_iM at
  0.5: 1.3 → 24.2 %. With one batch of 64 (32 by estimate, 32 random out of
  about 3,000 candidates), the LS declared a local optimum too early.
  Restoring K = 256 fixed it, and switching off EDF did not (attribution in the
  session; cells B_pos_n500_lH_* and B_pos_n300_lF_iL).
- **Batched steps** (`K_max`). Each step decodes 64 candidates and decodes the
  next 64 only if none improves, up to 256. Improving steps cost 64 and the
  stopping depth equals the old K = 256 (that cell went back to 4.4 %).
- **Kick patience 5:** about 20 % less time for +0.2 pp mean gap (split rows).
- **Kicks split vs. focused on the best start after LS:** split is better. The
  LS-only winner is a poor predictor of the start that kicks best (focused
  max: 100 %).

## Decode speed: plan, ideas and measurements

Most of the run time is decoding: 98 % of C++ time on the reference decoder.
The single-thread profile showed where the decoder's work goes. On congested
A·pos it is the waiting-queue scans in slots without free room: about
105,000 job visits per decode, each recomputing `last`, `cmin` and the deadline
expression. Elsewhere it is the free-run scans and the capacity decrements.

The ideas below were measured two ways:
- **Micro:** single thread, 54 workloads (27 cells × incumbent at the final and
  the all-auto genome), 256 LS-shaped candidates each. Each candidate's schedule
  is checked against the reference decoder.
- **End to end:** `scripts/dispatch_speed.py`, the 27 sample cells, 4 threads,
  best of 2 runs. All arms take the same LS trajectory (asserted).

| idea | exact? | micro speedup vs reference (median / time-weighted) | end-to-end vs reference (total) | kept |
|---|---|---|---|---|
| bitset queue by priority rank + per-slot event bitsets (deadline/horizon slot precomputed; a slot without room only touches its event jobs) | yes, bit-identical schedules | 1.50 / 1.80 (max 4.6) | **1.76×** | yes |
| incremental free-run length (`next zero` pointer) | yes | ≈ 1.00 | – | no (run scans were not the cost) |
| restart from the recorded incumbent at the flipped gene's first decision slot, stop once the state reconverges (checkpoint every 4 slots) | schedules yes; g within 1e-13 relative | 1.79 / 2.27 (max 6.8) | **1.83×** (+4 % over the bitset decoder; 5 of 27 cells slower) | yes (default) |
| lower-bound pruning (decided + waiting min(cloud, next slot) + unreleased min(cloud, release)) | yes (never rejects an improving move) | on local optima: prunes 64–99 % of candidates; on rejected ones the bound crosses after 4–78 % of the horizon (saves 22–96 % of slots, least on A_pos_n100) | 1.78× with restart (up to +82 % on B_pos_n300; −7 % alone) | opt-in `DF_PRUNE` |
| tie-robust argmin (a later candidate must beat the best by 1e-12 relative) | – | – | – | yes. Without it, the engines' different summation orders broke exact ties (interchangeable jobs) differently and their trajectories diverged on the 10-min pos instance |

The end-to-end gains are below the micro gains for two reasons:
- **Recording is serial.** The incumbent is recorded once per accepted move, on
  one thread, so non-decode C++ time rises from about 50 ms to about 230 ms
  per run.
- **Thread scaling.** The candidate decodes run on 4 threads.

Ideas not yet built, in the order I would try them:
1. **Cheaper, parallel recording.** Snapshot during the accepted candidate's
   own restart decode, instead of re-decoding it in full. This removes the
   serial phase above.
2. **Prune only near local optima.** Switch pruning on when the last step had
   no improving move, where 64–99 % of candidates are pruned. Off elsewhere,
   its bookkeeping costs more than it saves.
3. **Fit-filtered room scans.** Per pool and per length bucket bitsets of
   waiting jobs, so a slot with room visits only the jobs that fit. Targets
   B·pos, where 31,000 visits per decode are of non-fitting jobs.
4. **Job data in rank order.** Removes the `order[r]` indirection and
   scattered loads per visited job (micro).
5. **Screen candidates before decoding.** Reject a flip whose estimate is far
   from improving. This changes the search, so it needs a quality ablation.

`-march=native` was not tried, because it changes the build for every user.

## Code (uncommitted)

**`_ext/weighted_dispatch.hpp`:**
- `DFast` / `DInc` / `DFlipWork` / `DEngine`: the bitset decoder, incumbent
  recording, restart and pruning.
- `best_improving`: the tie-robust argmin.
- Batched LS steps (`K_max`).
- New flags `DF_EDF_NODL` (8192), `DF_REF_DECODE` (16384),
  `DF_NO_RESTART` (32768), `DF_PRUNE` (65536), `DF_FOCUS_KICKS` (131072).
- `kick_patience`.

**`_ext/moea.cpp`:** the `K_max` and `kick_patience` arguments and the new stats
(`flip_slots`, `pruned`, `converged`, `records`).

**`moea.py`:** new defaults `K=64`, `K_max=256`, `flags=64|4096|8192`,
`kick_patience=5`.

**`tests/test_moea.py`:**
- the new flag sets;
- every engine arm decides identically (on the 10-min pos instance);
- batched steps are thread-independent;
- patience stops the kicks.

`pytest -k "dispatch or moea"`: 545 passed.

**Scripts:** `dispatch_validate.py` (with `--tag/--kw`) and `dispatch_speed.py`.
`dispatch_ablation.py` pins its variants to the old configuration (`BASE`), so
`ablation.jsonl` keeps its meaning.

`speed.jsonl` was measured before batching and kick patience became defaults:
K = 64 in one batch, kicks split. That does not affect the engine comparison,
since every arm decides identically.
