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

## The five next ideas, tested in order (2026-11-03)

Protocol: as above. Each exact change was checked twice:
- **Micro:** the 27 workloads, against the unchanged engine and the reference
  decoder; every candidate's pen_g had to match.
- **End to end:** `dispatch_speed.py`, sample cells, 4 threads, best of 2, with
  the two arms in the same process. Every arm reached the same g in every cell.

Cross-run totals drift by up to ±10 % on this machine (powersave governor,
desktop load), so only paired comparisons are reported. Idea 4 was paired by
alternating builds.

| # | idea | exact? | measured | verdict |
|---|---|---|---|---|
| 1 | Re-record the incumbent from the accepted move's restart (`record_flip`): re-snapshot only slots (t*, convergence]; patch the prefix (job j's queue bit, init score) and shift the tail's score/cv | yes. Checked field by field against a full record on 6,480 records (LS moves and random flips, both deadline modes) | record 1.4–4.3× cheaper. End to end 92.8 → 91.1 s (−1.9 %); non-decode time 9.8 → 8.1 s | **kept** (`DF_FULL_RECORD` restores the full record) |
| 2 | Prune only in a step's later batches (`DF_PRUNE_LATE`) | yes | 89.6 s unpruned, 90.5 s always pruned, 89.8 s late only | **rejected**, see below |
| 3 | Fit-filtered room scan: per pool, length-bucket rank bitsets (≤ 16 buckets). Used only in slots where some pool's free run is shorter than its longest job, and only on non-empty queue words | yes | micro: B 1.03–2.3×, A 0.98–0.99×. End to end 92.4 → 80.0 s (−13 %): B·pos 41.1 → 27.2 s (1.51×), A +3.6 % | **kept** (`DF_NO_FIT` turns it off) |
| 4 | Job data in rank order (`DFast::rjob/ropt`; the gene-0 test read from the `g0` rank bitset) | yes | micro 1.12× (1.02–1.23×). End to end, alternating builds on the 9 λ = 0.5 cells: 27.79 / 27.74 s → 24.86 / 24.66 s (1.12×; every cell 10–14 %) | **kept** |
| 5 | Screen candidates by estimate | – (changes the search) | premise false, see below. Variants tested on all 261 cells | **rejected** |

**Why pruning does not pay (idea 2).** Pruning everything skips 78 % of the
candidates and halves the decoded slots (567 M → 295 M), but the time does not
fall. A micro run with the bound tracked but never cutting splits the cost:
- tracking the bound costs 4–10 % per candidate;
- the slots it cuts are the cheap tail of each candidate. On B_pos_n100_lF
  λ 0.5, cutting 72 % of the slots saves about 1 % of the time.

The cost of a candidate sits in its first slots after t*, where the queue is
long. Only B_pos_n300 gains (−9 to −24 %). Pruning stays opt-in.

**Why the fit filter needed gating (idea 3).** Unconditional, it cost 3–9 % on
A·pos. On A_pos_n100 it was active in 40 % of slots with room but skipped only
0.4 % of job visits. Meanwhile it built a mask for 47 M queue words, against
13.5 M job visits. The two gates (a pool's run shorter than its longest job;
non-empty words only) bring the A class to within 1–2 % in the micro run.

**Idea 5: the estimate is not a screen.** Every decoded LS candidate of 18
default runs (λ 0.25 and 0.5, 2.1 M decodes) was logged with its estimate and
its decoded Δpen_g:
- **Hit rate:** 8.1 % of the estimate-ranked half improves; 16.3 % of the
  random half. In most cells 70–90 % of accepted moves come from the random
  half.
- **Positive estimates:** about 60 % of accepted moves had est > 0, a
  predicted worsening.
- **Best band:** near zero, est ∈ [−1e-5, 5e-5], with a 36 % hit rate. The
  most negative decile hits 14 %, the most positive 3 %.
- **Why:** the estimate scores the flipped job alone. Offloading a job frees
  capacity for the jobs behind it, a knock-on effect it does not see.

Rejecting candidates whose estimate is far from improving would therefore
reject most good moves. Two re-rankings were tested instead on all 87
non-trivial instances × 3 λ (`dispatch_validate.py`, the same seed):

| configuration | mean gap | max | λ 0.25 / 0.5 / 0.75 | worse / better by > 5 pp | time | decodes |
|---|---|---|---|---|---|---|
| default, after ideas 1, 3 and 4 (`validate_fast_default`) | **6.47 %** | 58.3 % | 9.3 / 5.4 / 4.7 | – | 862 s | 29.3 M |
| est half ranked by \|est\| (`DF_ABS_EST`, `validate_absest`) | 9.38 % | 70.0 % | 13.6 / 6.9 / 7.7 | 35 / 0 | 646 s | 23.6 M |
| all candidates random (flag 8, `validate_randrank`) | 7.78 % | 76.0 % | 11.3 / 6.2 / 5.8 | 18 / 0 | 630 s | 24.6 M |

The default reached the same g as `validate_split_pat5` in all 261 cells, so
ideas 1, 3 and 4 are exact across the whole bench. Its time fell from 984 s to
862 s (−12 %), a cross-run comparison.

Both re-rankings stop earlier, about 20 % fewer decodes and 35 % less time,
but reach worse optima. Neither is better in any cell. The estimate-ranked
half accepts fewer moves but steers the search, so the mixed ranking stays.

**Open:** a better estimate, one that counts the capacity a flip frees or
takes from the jobs behind it, would raise the hit rate of the ranked half.
That is a quality change and needs its own ablation.

## Code (uncommitted)

**`_ext/weighted_dispatch.hpp`:**
- `DFast` / `DInc` / `DFlipWork` / `DEngine`: the bitset decoder, incumbent
  recording, restart and pruning.
- `best_improving`: the tie-robust argmin.
- Batched LS steps (`K_max`).
- New flags `DF_EDF_NODL` (8192), `DF_REF_DECODE` (16384),
  `DF_NO_RESTART` (32768), `DF_PRUNE` (65536), `DF_FOCUS_KICKS` (131072).
- `kick_patience`.
- From the next-ideas round:
  - `DEngine::record_flip` / `commit` and `DInc::kind` (idea 1);
  - the fit filter (`DFast::fthr/fofs/fit`, idea 3);
  - rank-ordered scan data (`DFast::rjob/ropt`, idea 4);
  - flags `DF_FULL_RECORD` (262144), `DF_PRUNE_LATE` (524288),
    `DF_NO_FIT` (1048576) and `DF_ABS_EST` (2097152).

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

**Scripts:** `dispatch_validate.py` (with `--tag/--kw`) and `dispatch_speed.py`
(arms `restart_fullrec`, `restart_prune_late` and `restart_nofit` added; outputs
`speed_rec/_prune/_fit/_rank.jsonl`).
`dispatch_ablation.py` pins its variants to the old configuration (`BASE`), so
`ablation.jsonl` keeps its meaning.

`speed.jsonl` was measured before batching and kick patience became defaults:
K = 64 in one batch, kicks split. That does not affect the engine comparison,
since every arm decides identically.

---

# Memetic with EDF vs the ILS (2026-11-04)

**Question.** Q6 rejected the memetic GA only as the *cause* of the gap. On the
27 sample cells it had beaten the old ILS (mean 17.6 vs 21.9 %). Only the ILS
then got the EDF fix. Is the ILS better once both have EDF?

## Changes (`_ext/weighted_memetic.hpp`)

- **EDF priority** (`flags` 4096): the indifference-deadline EDF order joins
  the run's priority scalars after the μ grid, as one more value of each
  individual's evolved priority index (reported `mu` = −2). With 8192, EDF
  individuals decode without the deadline offload, as the ILS's EDF start
  does.
- **Fast LS decoder.** `dispatch_ls` now gets DEngine tables, one per
  (priority, deadline mode). Every individual's f1/f2 are re-summed in job
  order (`canon`). Without that, the engine's last-bit differences flip the
  GA's zero-tolerance comparisons: only 4/27 cells matched before `canon`.
  - **Checks:** with `canon`, the reference decoder (`flags` 16384) and the
    fast one give bit-identical g on 27/27 sample cells, 1.48× faster.
    `canon` itself changes only tie-breaks: mean gap on the sample 17.6 → 17.5 %.
  - **Tests:** `pytest -k "Memetic or dispatch"` 246 passed, including the new
    fast-vs-reference and EDF-population tests.

## Protocol

- All 87 non-trivial instances × λ 0.25/0.5/0.75 = 261 cells.
- Seed 42, 4 threads, 3 instances in parallel, the same reference points.
- Run in one session: `scripts/dispatch_validate.py --solver memetic`.
- Report: `scripts/memetic_vs_ils_report.py`.
- The ILS rerun reproduces `validate_fast_default` bit-exactly in 261/261 cells.

## Results

| arm | mean | median | p90 | max | >10 % | W/T/L vs ILS (±0.5 pp) | time (s) |
|---|---|---|---|---|---|---|---|
| ILS (current default) | **6.5** | **3.0** | **18.4** | **58.3** | **54** | – | **821** |
| memetic (adapt 7) | 11.4 | 6.2 | 26.9 | 75.9 | 96 | 39/76/146 | 2410 |
| memetic + EDF (12352) | 9.4 | 5.5 | 22.8 | 62.2 | 87 | 44/80/137 | 2370 |
| memetic + EDF, deadline evolved (4160) | 9.4 | 5.7 | 22.9 | 62.2 | 87 | 45/78/138 | 2428 |

| arm | λ .25 | λ .5 | λ .75 | A·fedhpc | A·pos | B·fedhpc | B·pos |
|---|---|---|---|---|---|---|---|
| ILS | 9.3 | 5.4 | 4.7 | 2.0 | 9.4 | 1.7 | 10.0 |
| memetic | 16.3 | 10.9 | 6.9 | 3.0 | 21.8 | 3.0 | 12.2 |
| memetic + EDF | 10.9 | 10.9 | 6.5 | 3.1 | 16.0 | 3.0 | 11.6 |

## Why the memetic stays behind

- **Where EDF wins in the GA, it nearly closes the gap.** The memetic's final
  incumbent uses EDF in 17 cells, all A·pos λ 0.25. There it scores 13.9 %
  (41.5 % without EDF), against the ILS's 12.6 %. On the sample cells EDF
  takes over the final population (28 of 32 individuals).
- **But the GA picks EDF far less often than the ILS.** The ILS's best comes
  from its EDF start in 178 of 261 cells, at every λ.
  - On those 178 cells: ILS 7.2 % vs memetic + EDF 11.6 %.
  - On the 83 cells the μ start wins: ILS 5.0 % vs memetic 4.7 %, on par.
- **Mechanism (hypothesis).** In the ILS each start gets its own full LS and
  kicks before being compared. In the GA, EDF is about 1/11 of the initial
  population, and it competes in tournaments after a capped LS of 4 moves.
  Where EDF needs search before it beats μ, which is the usual case outside
  A·pos λ 0.25, selection drops it early. That EDF and evolved-deadline arms
  are identical fits this: EDF individuals rarely survive long enough for the
  deadline bit to matter. Not yet tested directly, e.g. by tracking the EDF
  share per generation.
- **Cost.** The memetic is also 2.9× slower, so matching the ILS's time would
  only widen the gap.

## Verdict

Keep the ILS. With EDF the memetic improves (11.4 → 9.4 %), but it is behind
the ILS in quality (137 losses vs 44 wins) and takes 2.9× the time. Untested
way the memetic could catch up: protect the EDF niche (islands per priority,
or a per-priority quota in selection). That would make it an ILS-like
multi-start inside the GA.

**Caveat:** one seed per cell (±3 pp per-cell noise). The aggregate gap
(2.9 pp mean over 261 cells, 137/44 W/L) is well above that.

---

# Memetic mechanisms and fixes (2026-11-04)

**Question.** Which GA mechanisms hold the memetic back, and can fixing them
bring it level with the ILS or ahead of it?

## Diagnostics

`scripts/memetic_mechanisms.py`, with new memetic stats:
- `init_cloud`: gene-1 share of the initial population.
- `init_best`: best score per priority after the initial LS.
- `gens`: EDF count and best score per generation.
- `dec_x` / `dec_ls`: decodes in crossover vs the rest.
- `polish_gain`, `init_ms`, `seeds`.

Five cells: A_pos_n300_lH, A_pos_n100_lE, B_pos_n300_lH and A_fedhpc_n100_lE
at λ 0.5, and B_pos_n100_lE at λ 0.25.

| mechanism | evidence | verdict |
|---|---|---|
| **The capped LS kills EDF.** Init LS is 8 moves, a child's LS 4. | From all-auto, the EDF start (no deadline) needs 166–1,634 LS moves. After 8 moves it is at a 3,900–18,000 % gap. The μ start needs 3–46 moves, because the deadline rule offloads inside the decoder. After the init LS, EDF individuals sit at 60–128 % against 5–25 % for μ, and are gone by generation 0–1. | **Main cause** |
| **Random, cost-weighted initial genomes.** | About 10 % gene 1. Seeding all-auto genomes alone (bit 16 before the full-LS change) changed nothing: the cap still applies. | Secondary |
| **Plain elitism drops niches.** | Per-priority elitism alone (bit 64) helps only where EDF wins after few moves (B_pos_n100_lE λ 0.25: 32.6 → 25.1 %). | Helps once EDF is competitive |
| **The final polish stops early.** | Polish gain about 0 pp. | Not a cause |
| **The decoder picks μ by one all-auto decode** (ILS too). | A full LS on each μ of the grid reaches better μ starts (19.8 vs 23.9 %, 2.0 vs 8.3 %). The best μ after LS is usually 0.1–1, but those need 2,500–3,600 moves. | Headroom, costly |
| **Crossover is expensive** (MSXF, reference decoder). | 37–49 % of decodes. | Cost, not quality |

## Fixes (new `adapt` bits, experimental)

- **16 `MA_SEED_AUTO`:** all-auto seed individuals, each with an LS to a
  local optimum (K 64 / K_max 256), using the ILS start settings (deadline
  on, dispatch decoder).
- **128 `MA_SEED_TWO`:** with 16, seeds only the ILS's two starts: the best μ
  by all-auto decode, and EDF. Without it, every priority is seeded.
- **64 `MA_NICHE`:** each priority's best individual survives every generation.
- **32 `MA_POLISH_FULL`:** the final LS runs to a local optimum (not used
  below).

## Results (261 cells, seed 42, run in one session)

| arm | mean | median | p90 | max | >10 % | W/T/L vs ILS | time (s) |
|---|---|---|---|---|---|---|---|
| ILS (default) | 6.5 | 3.0 | 18.4 | 58.3 | 54 | – | 821 |
| ILS, 120 kicks, no patience | 5.8 | 2.6 | **16.9** | **56.6** | **47** | 80/178/3 | 1555 |
| memetic + EDF (adapt 7) | 9.4 | 5.5 | 22.8 | 62.2 | 87 | 44/80/137 | 2370 |
| **memetic + EDF, adapt 215** (7+16+64+128) | **5.5** | **1.7** | 18.3 | 62.1 | 49 | 122/108/31 | 1813 |
| … n_gen 30 | 6.1 | 2.1 | 18.9 | 62.3 | 54 | 109/101/51 | 1168 |

- **Against the long ILS** (1555 vs 1813 s): adapt 215 wins/ties/loses
  102/109/50. Mean 5.5 vs 5.8, median 1.7 vs 2.6. The ILS keeps the better
  tail (p90 16.9 vs 18.3, max 56.6 vs 62.1).
- **By class·workload:**

  | arm | A·fedhpc | B·fedhpc | A·pos | B·pos |
  |---|---|---|---|---|
  | memetic, adapt 215 | 0.7 | 0.7 | 8.7 | 9.1 |
  | ILS, long | 1.8 | 1.4 | 8.7 | 8.9 |

  The memetic's gain is on the fedhpc workloads; pos is level.
- **Seeding all 11 priorities** (adapt 87, 27 sample cells only): mean 6.3 vs
  the ILS's 7.9, W/L 16/2, but 7.9× the ILS's time. A full-LS μ start costs
  2,500–3,600 moves.

## Verdict

- **What hurt the memetic:** its LS budget per individual, not the GA
  framework. EDF only pays after hundreds to thousands of LS moves, and the
  memetic caps every individual at 4–8.
- **With the fixes:** the GA starts from the ILS's converged starts, each
  priority's best is kept, and the GA's recombination then adds to them. It
  then matches or slightly beats the ILS at about 1.2–1.4× the time.
- **Caveats:**
  - one seed (±3 pp per cell);
  - the configuration was picked on 5 cells, so it has no held-out set;
  - the bits are opt-in and the defaults are unchanged.
- **Next steps:**
  - a multi-seed run;
  - kicks on the niche bests (the ILS's strength on B·pos tails);
  - cheaper crossover through DEngine `full_pen`;
  - a full LS on 2–3 μ values ranked by a short LS rather than by decode.

## GA tuning, ablation and multi-seed ILS vs GA (2026-11-05)

Split (`split.json`): 22 tune instances (66 cells) for tuning and ablation, 65 held-out
test instances (195 cells) for the final comparison. Gap is measured against the bench Gurobi g*
(`bench_io_noreduc.jsonl`, 194 optimal + 1 feasible; Gurobi was not rerun). Reports:
`tuning/ablation_report.md`, `final/report.md`, `final/anytime.png`.

**Tuning (tune split, seed 42).** Shortlist 48 was the best value per unit time. Longer runs
(`n_gen`, `pop_size`, `ls_moves`) keep improving but cost proportionally more. Two-point
crossover and no crossover were both worse than MSXF, and `restart_patience` made no
difference. Final arms (all with `flags 12352, adapt 215, shortlist 48`):
GA-fast = pop 16 / 60 gen, GA-mid = pop 16 / 120 gen, GA-quality = pop 32 / 120 gen.

**Ablation of GA-mid (tune split, 66 cells, base 4.5 % mean / 1.3 % median).** Each row removes one mechanism; W/L counts cells that moved by more than 0.5 pp.

| ablation | mean | median | W/T/L vs base | verdict |
|---|---|---|---|---|
| no EDF priority (flags 64) | 9.5 | 4.8 | 2/25/39 | essential |
| no converged ILS-start seeds (adapt 71) | 7.1 | 3.4 | 3/21/42 | essential |
| no crossover | 6.2 | 2.7 | 0/28/38 | essential |
| EDF with deadline rule (4160) | 5.5 | 2.6 | 4/44/18 | the no-deadline EDF matters |
| seed all priorities (87) | 6.4 | 1.2 | 12/38/16 | worse and 2.6× slower |
| no mutation | 4.8 | 1.3 | 1/52/13 | small help |
| no niche elitism (151) | 4.3 | 1.0 | 17/46/3 | niche **hurts** quality slightly but saves time (610 s vs 408 s) |
| no μ-grid adapt bit (214) | 4.5 | 1.6 | 14/37/15 | neutral, 1.5× slower |
| no deadline / decoder bits, no ILS kick, no heuristic seeds, full polish | 4.2–4.5 | 1.3–1.4 | ≤ 8 losses | neutral (could be dropped) |

**Final comparison (195 test cells × seeds 42/1/7).**

| arm | mean gap ± sd (seeds) | median | time / seed | median t / cell |
|---|---|---|---|---|
| ILS | 6.87 ± 0.01 | 3.60 | 652 s | 2.2 s |
| ILS-long | 6.25 ± 0.02 | 2.99 | 1302 s | 5.6 s |
| GA-fast | 6.24 ± 0.14 | 2.44 | 845 s | 3.9 s |
| GA-mid | 5.66 ± 0.09 | 1.92 | 1307 s | 6.3 s |
| GA-quality | 5.09 ± 0.02 | 1.30 | 3330 s | 15.8 s |
| Gurobi (bench) | 0 | 0 | 22104 s | 70.6 s |

- Paired comparisons (seed-mean per cell, bootstrap CI, sign test):
  - GA-fast ties ILS-long (+0.01 pp, CI [−0.34, +0.37], p = 0.32) in 65 % of the time.
  - GA-mid beats ILS-long at equal time (−0.59 pp, CI [−0.92, −0.27], p = 4e-5).
  - GA-quality beats ILS-long by −1.16 pp (W/T/L 103/78/14) at 2.6× the time.
- **Anytime behaviour:**
  - The ILS has a solution after about 20 ms and stays ahead until about 2–3 s.
  - The GA has no incumbent until its initialization ends. That happens at a median of about 1–2 s, because the seeds run LS to convergence.
  - From about 3 s on, the GA's median gap is below the ILS.
  - Primal integral: the ILS is better at T = 10 s (0.074 vs 0.24) and roughly even at 60 s. At T = 600 s, GA-quality < GA-mid < ILS-long < ILS (0.047 / 0.051 / 0.054 / 0.059).
- **Against Gurobi:**
  - Every heuristic finishes in 3–19 % of Gurobi's time (median ratio: ILS 0.03, GA-mid 0.08, GA-quality 0.19).
  - At those times Gurobi has no reported solution, so the bench record is a single step at its finish. Pessimistic toward Gurobi, since intermediate incumbents weren't logged.
  - Gurobi reaches ≤ 1 % in 100 % of cells (median 71 s). The heuristics reach it in ILS 23 %, ILS-long 25 %, GA-fast 35 %, GA-mid 39 %, GA-quality 45 % of runs.
  - The remaining gap is concentrated in the pos workloads and at λ 0.25: 8–10 % mean for every method, versus < 1 % on fedhpc workloads for the GA.
- **Verdict:**
  - The ILS is the right choice for sub-second budgets.
  - At ≥ ~3 s per cell the tuned GA dominates: GA-fast ≈ ILS-long in less time; GA-mid is better at equal time.
  - The GA's seed variance is larger (sd 0.09–0.14 vs 0.01–0.02) but small relative to the differences.
