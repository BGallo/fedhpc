# Generated-instance benchmark — bench_io_noreduc

## Setup

Full description: `EXPERIMENT_SETUP.md` in this folder.

- Command: `nohup nice -n 10 uv run python scripts/bench_generated.py --manifest --no-reductions --workers 3 --threads 4 --out results/generated_bench/bench_io_noreduc.jsonl > results/generated_bench/bench_io_noreduc.log 2>&1 &`
- 108 instances, 0 failures; 2026-10-29T14:35-03:00 (approx., from the progress check at 15:17) → 2026-10-29T19:57:01-03:00 (mtime of the log) (5.4 h wall).
- 3 instances in parallel, 4 threads each; Gurobi {'OutputFlag': 0, 'Threads': 4, 'TimeLimit': 600, 'MIPGap': 0}; λ ∈ [0.25, 0.5, 0.75]; θ sweep [0, 1, 2, 3, 6, 12, 24, 48, 96] slots.
- Formulation: `SpaceTimeFormulation(collapse_unbounded=False, pareto_unbounded=False, rejection_deadlines=False, lex_deadlines=False)`.
- Dispatch: `fedhpc.moea.weighted_solve_dispatch` with K=256, max_iters=100000, n_kicks=60, kick_size=16, flags=64, ls=True, mu=-1.0, n_mu=1, seed=42, n_threads=4.
- Hardware: Intel Core i9-13900 (8 P-cores + 16 E-cores, 32 logical CPUs, max 5.6 GHz), 62 GB RAM, pinning: none (thread counts only); another user's LLM inference server ran on the same machine.
- Software: CachyOS Linux, kernel 7.3; CPython 3.14 (uv-managed .venv); gurobipy 13.0.1 (uv.lock), Gurobi academic WLS license; numpy 2.4.4.
- Code: fedhpc 97fb178 + `bench_io_noreduc.code.patch` (sha256 735bd3e349f5…).
- 2026-11-03: scripts/build_generated_instances.py --out <scratch> rebuilt all 108 instances; every sha256 equals data/generated/manifest.json.
- Re-run 2026-11-18: lex f1 re-solved on all 108 instances with Presolve=0 in its phase 2 only (scripts/bench_generated.py --lex-only --phase2-presolve 0 -> bench_io_noreduc_p2lex.jsonl; same workers/threads/TimeLimit). Phase-2 presolve on the dense f1-fix row had stalled past TimeLimit; lex f1 is now optimal on 108/108 (was 105) in 5873 s total (was 11610). Anchors changed on 3 instances (f2_T: B_pos_n300_lE_iH 11313.59->8640.79, A_pos_n300_lH_iH 13152.94->10809.01, B_fedhpc_n100_lE_iH 7958.91->7172.29); their weighted sums were re-solved (bench_io_noreduc_p2ws.jsonl, presolve default) and their heuristics re-run. lex f2 keeps the presolve-default results (Presolve=0 there lost 2 certificates). Merge: scripts/presolve0_merge.py; originals in archive_presolve_default/. Code: fedhpc bfb8651 + bench_io_noreduc_p2lex.code.patch.

## Dispatch results

The dispatch columns are **not** this run's own dispatch results. They are replaced by `results/generated_bench/dispatch_study/validate_ils_s42.jsonl`, `results/generated_bench/dispatch_study/validate_fast_default_trivial.jsonl` (`scripts/dispatch_validate.py`: the same instances, reference points and λ, 3 instances in parallel × 4 threads, one seed), run with the weighted_solve_dispatch defaults of 2026-11-04 (ILS: K=64 in batches up to K_max=256, flags=64|4096|8192, 60 kicks of 16 split over the two starts, kick_patience=5, plus the exact speed-ups: incremental record_flip, fit-filtered room scan, rank-ordered data; seed 42, 4 threads; fedhpc uncommitted). The trivial instances keep the 2026-11-03 run, whose g is identical. B_pos_n300_lE_iH, A_pos_n300_lH_iH and B_fedhpc_n100_lE_iH (new lex f1 anchors) were re-run on 2026-11-18 with the dispatch defaults of fedhpc bfb8651. The Dispatch line under Setup is the configuration of the original run.

## Gurobi statuses

{'optimal': 530, 'feasible': 10} over 540 solves. `feasible` = time limit reached with an incumbent (not proven optimal):

- B_pos_n100_lE_iL lex_f2
- B_pos_n100_lE_iH lex_f2
- B_pos_n100_lE_iM lex_f2
- B_pos_n100_lE_iM λ=0.25
- B_pos_n100_lH_iL lex_f2
- B_pos_n100_lH_iH lex_f2
- B_pos_n100_lH_iM lex_f2
- A_pos_n100_lE_iL λ=0.25
- A_pos_n100_lH_iH lex_f2
- A_pos_n100_lH_iL lex_f2

A non-optimal lex anchor shifts the normalisation (f1^T, f2^T or f1^0) for every method alike. A non-optimal weighted-sum incumbent is ≥ g*, so the heuristic gap reported for that (instance, λ) is a lower bound on the true one.

## Trivial instances (f2^T = 0: the fastest schedule is free, front is one point)

21 of 108; Gurobi and dispatch reach g = 0 on 21 of them. Excluded from every quality statistic below.

| class·workload | count | load | io levels |
|---|---|---|---|
| A·fedhpc | 300 | F | LMH |
| A·fedhpc | 500 | F | LMH |
| A·fedhpc | 500 | H | LMH |
| B·fedhpc | 300 | F | LMH |
| B·fedhpc | 500 | F | LMH |
| B·fedhpc | 500 | H | LMH |
| B·pos | 500 | F | LMH |

## Time (s): median [p90] per solve

| class·workload | n | trivial | lex f1 | lex f2 | WS Gurobi | WS dispatch | exact total | dispatch total |
|---|---|---|---|---|---|---|---|---|
| A·fedhpc | 27 | 9 | 12.5 [13.7] | 13.7 [16.2] | 15.4 [22.8] | 0.6 [1.0] | 74.2 [91.3] | 1.9 [3.0] |
| A·pos | 27 | 0 | 58.1 [62.8] | 87.0 [361.3] | 88.1 [235.1] | 5.5 [8.3] | 454.2 [1257.0] | 15.5 [23.3] |
| B·fedhpc | 27 | 9 | 21.6 [24.4] | 22.4 [24.4] | 25.9 [29.2] | 0.5 [0.9] | 123.5 [143.7] | 1.7 [2.3] |
| B·pos | 27 | 3 | 118.8 [143.4] | 125.8 [775.5] | 190.1 [314.6] | 3.6 [7.9] | 818.7 [1906.0] | 10.7 [22.6] |

## Weighted-sum quality: gap to exact g* (% of g*, non-trivial instances)

| class·workload | n | λ | dispatch median | p90 | max | dispatch = g* (≤1e-6) | threshold median | WS Gurobi faster than dispatch |
|---|---|---|---|---|---|---|---|---|
| A·fedhpc | 18 | 0.25 | 1.1 | 7.2 | 13.6 | 3/18 | 61.3 | 0/18 |
| A·fedhpc | 18 | 0.5 | 1.4 | 4.4 | 4.7 | 0/18 | 119.3 | 0/18 |
| A·fedhpc | 18 | 0.75 | 1.4 | 4.7 | 8.4 | 0/18 | 143.9 | 0/18 |
| A·pos | 27 | 0.25 | 12.0 | 48.4 | 58.3 | 3/27 | 184.5 | 0/27 |
| A·pos | 27 | 0.5 | 6.2 | 19.7 | 21.6 | 1/27 | 213.9 | 0/27 |
| A·pos | 27 | 0.75 | 4.3 | 14.8 | 16.2 | 0/27 | 209.3 | 0/27 |
| B·fedhpc | 18 | 0.25 | 1.1 | 3.6 | 4.3 | 3/18 | 48.0 | 0/18 |
| B·fedhpc | 18 | 0.5 | 1.5 | 5.6 | 10.1 | 2/18 | 78.0 | 0/18 |
| B·fedhpc | 18 | 0.75 | 1.5 | 3.3 | 5.1 | 0/18 | 74.0 | 0/18 |
| B·pos | 24 | 0.25 | 9.6 | 29.3 | 38.4 | 2/24 | 98.9 | 0/24 |
| B·pos | 24 | 0.5 | 4.8 | 18.4 | 22.8 | 0/24 | 147.6 | 0/24 |
| B·pos | 24 | 0.75 | 5.5 | 16.0 | 22.9 | 0/24 | 237.7 | 0/24 |

## Factor effects (median per cell; gap over non-trivial instances and all λ)

| class·workload | factor | level | n | trivial | f2^T (USD) | f1^0/f1^T − 1 (%) | dispatch gap (%) | exact total (s) |
|---|---|---|---|---|---|---|---|---|
| A·fedhpc | count | 100 | 9 | 0 | 3211 | 53.7 | 2.4 | 87 |
| A·fedhpc | count | 300 | 9 | 3 | 192 | 2.4 | 0.6 | 71 |
| A·fedhpc | count | 500 | 9 | 6 | 0 | 0.0 | 0.8 | 44 |
| A·fedhpc | load | F | 9 | 6 | 0 | 0.0 | 3.0 | 44 |
| A·fedhpc | load | H | 9 | 3 | 192 | 2.4 | 2.2 | 71 |
| A·fedhpc | load | E | 9 | 0 | 4819 | 78.3 | 0.8 | 83 |
| A·fedhpc | io | L | 9 | 3 | 747 | 15.6 | 0.7 | 74 |
| A·fedhpc | io | M | 9 | 3 | 829 | 15.6 | 1.3 | 73 |
| A·fedhpc | io | H | 9 | 3 | 1495 | 15.1 | 2.0 | 77 |
| A·pos | count | 100 | 9 | 0 | 23467 | 212.4 | 11.2 | 1037 |
| A·pos | count | 300 | 9 | 0 | 6712 | 36.5 | 4.0 | 432 |
| A·pos | count | 500 | 9 | 0 | 1873 | 7.7 | 3.7 | 334 |
| A·pos | load | F | 9 | 0 | 3178 | 13.4 | 7.3 | 332 |
| A·pos | load | H | 9 | 0 | 6712 | 36.5 | 8.5 | 432 |
| A·pos | load | E | 9 | 0 | 13050 | 90.8 | 2.4 | 506 |
| A·pos | io | L | 9 | 0 | 6929 | 48.4 | 3.7 | 506 |
| A·pos | io | M | 9 | 0 | 7519 | 48.4 | 5.0 | 454 |
| A·pos | io | H | 9 | 0 | 13050 | 46.9 | 9.3 | 426 |
| B·fedhpc | count | 100 | 9 | 0 | 1328 | 40.2 | 1.7 | 135 |
| B·fedhpc | count | 300 | 9 | 3 | 169 | 2.1 | 0.8 | 120 |
| B·fedhpc | count | 500 | 9 | 6 | 0 | 0.0 | 0.7 | 73 |
| B·fedhpc | load | F | 9 | 6 | 0 | 0.0 | 3.3 | 74 |
| B·fedhpc | load | H | 9 | 3 | 169 | 2.1 | 1.4 | 120 |
| B·fedhpc | load | E | 9 | 0 | 3949 | 68.9 | 1.0 | 126 |
| B·fedhpc | io | L | 9 | 3 | 466 | 9.6 | 1.1 | 123 |
| B·fedhpc | io | M | 9 | 3 | 504 | 9.6 | 1.3 | 124 |
| B·fedhpc | io | H | 9 | 3 | 942 | 9.6 | 1.5 | 124 |
| B·pos | count | 100 | 9 | 0 | 10754 | 85.4 | 18.0 | 1837 |
| B·pos | count | 300 | 9 | 0 | 1623 | 4.9 | 5.0 | 819 |
| B·pos | count | 500 | 9 | 3 | 122 | 0.2 | 4.0 | 533 |
| B·pos | load | F | 9 | 3 | 348 | 0.5 | 11.3 | 819 |
| B·pos | load | H | 9 | 0 | 1623 | 4.9 | 5.4 | 766 |
| B·pos | load | E | 9 | 0 | 4463 | 37.6 | 5.5 | 1012 |
| B·pos | io | L | 9 | 1 | 2255 | 11.7 | 5.1 | 704 |
| B·pos | io | M | 9 | 1 | 2427 | 11.7 | 5.8 | 819 |
| B·pos | io | H | 9 | 1 | 4168 | 11.4 | 8.8 | 841 |

## io level, paired over (class, workload, count, load) cells (non-trivial)

| class·workload | cells | total ι at M / L | at H / L | f2^T M / L | f2^T H / L | f1^T H / L | f1^0 H / L |
|---|---|---|---|---|---|---|---|
| A·fedhpc | 6 | 8.1 | 55.1 | 1.12 | 1.87 | 1.030 | 1.000 |
| A·pos | 9 | 9.1 | 39.3 | 1.08 | 1.88 | 1.010 | 1.000 |
| B·fedhpc | 6 | 8.1 | 55.1 | 1.11 | 1.96 | 1.026 | 1.000 |
| B·pos | 8 | 9.0 | 39.2 | 1.09 | 1.99 | 1.005 | 1.000 |

## What drives exact solve time (Spearman ρ with total Gurobi time)

- fedhpc: nodes per type: ρ = -0.77
- fedhpc: busy share at t=0: ρ = +0.13
- fedhpc: result volume ι (GB): ρ = -0.25
- fedhpc: f1^0 − f1^T: ρ = +0.61
- pos: nodes per type: ρ = -0.89
- pos: busy share at t=0: ρ = +0.04
- pos: result volume ι (GB): ρ = -0.25
- pos: f1^0 − f1^T: ρ = +0.65
