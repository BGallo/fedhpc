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

## Gurobi statuses

{'optimal': 527, 'feasible': 13} over 540 solves. `feasible` = time limit reached with an incumbent (not proven optimal):

- B_pos_n100_lE_iL lex_f2
- B_pos_n100_lE_iH lex_f2
- B_pos_n100_lE_iM lex_f2
- B_pos_n100_lE_iM λ=0.25
- B_pos_n100_lH_iL lex_f2
- B_pos_n100_lH_iH lex_f2
- B_pos_n100_lH_iM lex_f2
- B_pos_n300_lE_iH lex_f1
- A_pos_n100_lE_iL λ=0.25
- A_pos_n100_lH_iH lex_f2
- A_pos_n100_lH_iL lex_f2
- A_pos_n300_lH_iH lex_f1
- B_fedhpc_n100_lE_iH lex_f1

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
| A·fedhpc | 27 | 9 | 12.3 [17.7] | 13.7 [16.2] | 15.4 [22.8] | 2.3 [4.4] | 74.3 [91.0] | 7.0 [12.9] |
| A·pos | 27 | 0 | 53.7 [93.8] | 87.0 [361.3] | 88.1 [235.1] | 15.7 [41.3] | 488.5 [1311.3] | 57.5 [100.2] |
| B·fedhpc | 27 | 9 | 20.8 [29.3] | 22.4 [24.4] | 25.8 [28.5] | 2.3 [4.2] | 123.0 [142.1] | 7.1 [12.2] |
| B·pos | 27 | 3 | 130.2 [215.6] | 125.8 [775.5] | 190.1 [314.6] | 12.4 [41.5] | 858.9 [2004.4] | 38.0 [108.5] |

## Weighted-sum quality: gap to exact g* (% of g*, non-trivial instances)

| class·workload | n | λ | dispatch median | p90 | max | dispatch = g* (≤1e-6) | threshold median | WS Gurobi faster than dispatch |
|---|---|---|---|---|---|---|---|---|
| A·fedhpc | 18 | 0.25 | 1.8 | 27.6 | 58.4 | 3/18 | 61.3 | 0/18 |
| A·fedhpc | 18 | 0.5 | 4.0 | 8.8 | 9.0 | 2/18 | 119.3 | 0/18 |
| A·fedhpc | 18 | 0.75 | 3.0 | 9.4 | 10.8 | 0/18 | 143.9 | 0/18 |
| A·pos | 27 | 0.25 | 55.2 | 95.1 | 156.5 | 3/27 | 184.5 | 0/27 |
| A·pos | 27 | 0.5 | 21.2 | 25.8 | 27.7 | 3/27 | 213.9 | 0/27 |
| A·pos | 27 | 0.75 | 7.8 | 15.5 | 36.7 | 0/27 | 209.3 | 0/27 |
| B·fedhpc | 18 | 0.25 | 1.1 | 40.7 | 42.2 | 3/18 | 50.5 | 0/18 |
| B·fedhpc | 18 | 0.5 | 3.6 | 10.5 | 11.5 | 2/18 | 78.0 | 0/18 |
| B·fedhpc | 18 | 0.75 | 2.9 | 8.8 | 9.2 | 0/18 | 74.0 | 0/18 |
| B·pos | 24 | 0.25 | 11.8 | 86.8 | 125.2 | 5/24 | 98.9 | 0/24 |
| B·pos | 24 | 0.5 | 13.8 | 19.1 | 22.1 | 0/24 | 147.6 | 0/24 |
| B·pos | 24 | 0.75 | 6.9 | 15.1 | 20.9 | 0/24 | 237.7 | 0/24 |

## Factor effects (median per cell; gap over non-trivial instances and all λ)

| class·workload | factor | level | n | trivial | f2^T (USD) | f1^0/f1^T − 1 (%) | dispatch gap (%) | exact total (s) |
|---|---|---|---|---|---|---|---|---|
| A·fedhpc | count | 100 | 9 | 0 | 3211 | 53.7 | 6.8 | 88 |
| A·fedhpc | count | 300 | 9 | 3 | 192 | 2.4 | 2.1 | 70 |
| A·fedhpc | count | 500 | 9 | 6 | 0 | 0.0 | 2.3 | 44 |
| A·fedhpc | load | F | 9 | 6 | 0 | 0.0 | 9.3 | 44 |
| A·fedhpc | load | H | 9 | 3 | 192 | 2.4 | 5.1 | 70 |
| A·fedhpc | load | E | 9 | 0 | 4819 | 78.3 | 2.3 | 82 |
| A·fedhpc | io | L | 9 | 3 | 747 | 15.6 | 2.4 | 74 |
| A·fedhpc | io | M | 9 | 3 | 829 | 15.6 | 2.6 | 74 |
| A·fedhpc | io | H | 9 | 3 | 1495 | 15.1 | 3.3 | 80 |
| A·pos | count | 100 | 9 | 0 | 23467 | 212.4 | 23.0 | 1065 |
| A·pos | count | 300 | 9 | 0 | 6712 | 36.5 | 20.5 | 451 |
| A·pos | count | 500 | 9 | 0 | 1873 | 7.7 | 16.3 | 327 |
| A·pos | load | F | 9 | 0 | 3178 | 13.4 | 14.1 | 328 |
| A·pos | load | H | 9 | 0 | 6712 | 36.5 | 22.4 | 457 |
| A·pos | load | E | 9 | 0 | 13050 | 90.8 | 21.2 | 539 |
| A·pos | io | L | 9 | 0 | 6929 | 48.4 | 20.5 | 532 |
| A·pos | io | M | 9 | 0 | 7519 | 48.4 | 21.8 | 451 |
| A·pos | io | H | 9 | 0 | 13153 | 46.9 | 20.7 | 628 |
| B·fedhpc | count | 100 | 9 | 0 | 1328 | 40.2 | 5.6 | 133 |
| B·fedhpc | count | 300 | 9 | 3 | 169 | 2.1 | 1.3 | 119 |
| B·fedhpc | count | 500 | 9 | 6 | 0 | 0.0 | 0.8 | 72 |
| B·fedhpc | load | F | 9 | 6 | 0 | 0.0 | 8.8 | 73 |
| B·fedhpc | load | H | 9 | 3 | 169 | 2.1 | 1.1 | 119 |
| B·fedhpc | load | E | 9 | 0 | 3949 | 68.9 | 1.7 | 127 |
| B·fedhpc | io | L | 9 | 3 | 466 | 9.6 | 1.6 | 123 |
| B·fedhpc | io | M | 9 | 3 | 504 | 9.6 | 1.9 | 124 |
| B·fedhpc | io | H | 9 | 3 | 942 | 9.6 | 2.3 | 123 |
| B·pos | count | 100 | 9 | 0 | 10754 | 85.4 | 17.3 | 1818 |
| B·pos | count | 300 | 9 | 0 | 1623 | 4.9 | 7.6 | 859 |
| B·pos | count | 500 | 9 | 3 | 122 | 0.2 | 5.9 | 526 |
| B·pos | load | F | 9 | 3 | 348 | 0.5 | 11.7 | 839 |
| B·pos | load | H | 9 | 0 | 1623 | 4.9 | 7.6 | 763 |
| B·pos | load | E | 9 | 0 | 4463 | 37.6 | 11.5 | 1118 |
| B·pos | io | L | 9 | 1 | 2255 | 11.7 | 7.5 | 718 |
| B·pos | io | M | 9 | 1 | 2427 | 11.7 | 8.0 | 839 |
| B·pos | io | H | 9 | 1 | 4168 | 11.4 | 10.7 | 943 |

## io level, paired over (class, workload, count, load) cells (non-trivial)

| class·workload | cells | total ι at M / L | at H / L | f2^T M / L | f2^T H / L | f1^T H / L | f1^0 H / L |
|---|---|---|---|---|---|---|---|
| A·fedhpc | 6 | 8.1 | 55.1 | 1.12 | 1.87 | 1.030 | 1.000 |
| A·pos | 9 | 9.1 | 39.3 | 1.08 | 1.90 | 1.010 | 1.000 |
| B·fedhpc | 6 | 8.1 | 55.1 | 1.11 | 2.02 | 1.026 | 1.000 |
| B·pos | 8 | 9.0 | 39.2 | 1.09 | 1.99 | 1.005 | 1.000 |

## What drives exact solve time (Spearman ρ with total Gurobi time)

- fedhpc: nodes per type: ρ = -0.75
- fedhpc: busy share at t=0: ρ = +0.19
- fedhpc: result volume ι (GB): ρ = -0.18
- fedhpc: f1^0 − f1^T: ρ = +0.64
- pos: nodes per type: ρ = -0.83
- pos: busy share at t=0: ρ = +0.09
- pos: result volume ι (GB): ρ = -0.13
- pos: f1^0 − f1^T: ρ = +0.62
