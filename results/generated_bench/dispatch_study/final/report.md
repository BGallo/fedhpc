195 common cells, seeds [42, 1, 7]; Gurobi: bench solves (step at its time t)

## 1. Final gap and time

| arm | mean gap (per seed) | mean ± sd over seeds | median | p90 | max | cells > 10 % | time/seed (s) | median t/cell (s) |
|---|---|---|---|---|---|---|---|---|
| ILS | 6.87 / 6.91 / 6.88 | 6.89 ± 0.02 | 3.60 | 18.4 | 58.7 | 43.7 | 633 | 2.17 |
| ILS-long | 6.28 / 6.31 / 6.24 | 6.28 ± 0.03 | 2.99 | 18.1 | 56.6 | 39.0 | 1278 | 5.62 |
| GA-fast | 6.15 / 6.38 / 6.45 | 6.33 ± 0.15 | 2.44 | 19.1 | 62.3 | 40.0 | 835 | 3.89 |
| GA-mid | 5.62 / 5.80 / 5.77 | 5.73 ± 0.10 | 1.92 | 18.4 | 59.2 | 37.3 | 1295 | 6.27 |
| GA-quality | 5.14 / 5.19 / 5.12 | 5.15 ± 0.04 | 1.30 | 17.5 | 61.8 | 36.7 | 3306 | 15.33 |
| Gurobi (bench) | 0.000 | | 0.000 | | 0.00 | | 22052 | 71.0 |

Gurobi status: {'optimal': 194, 'feasible': 1}

## 2. Paired comparison (seed-mean gap per cell; tie ±0.5 pp)

| A vs B | mean diff A−B (pp) | 95 % CI | W/T/L (A better/tie/worse) | sign-test p |
|---|---|---|---|---|
| ILS vs ILS-long | +0.61 | [+0.47, +0.77] | 0/133/62 | 4.3e-19 |
| ILS vs GA-fast | +0.56 | [+0.18, +0.94] | 41/71/83 | 0.0002 |
| ILS vs GA-mid | +1.16 | [+0.78, +1.56] | 25/78/92 | 3.4e-10 |
| ILS vs GA-quality | +1.73 | [+1.32, +2.17] | 9/65/121 | 3.5e-26 |
| ILS-long vs GA-fast | -0.05 | [-0.42, +0.32] | 56/74/65 | 0.47 |
| ILS-long vs GA-mid | +0.55 | [+0.22, +0.88] | 39/73/83 | 8.4e-05 |
| ILS-long vs GA-quality | +1.12 | [+0.79, +1.47] | 15/77/103 | 2.5e-17 |
| GA-fast vs GA-mid | +0.60 | [+0.45, +0.76] | 0/139/56 | 2.8e-17 |
| GA-fast vs GA-quality | +1.17 | [+0.95, +1.42] | 1/90/104 | 5.2e-30 |
| GA-mid vs GA-quality | +0.58 | [+0.44, +0.72] | 2/110/83 | 1.9e-22 |

## 3. Gap of the best-so-far at a wall-clock budget (median over cells×seeds; % cells with a solution)

| method | 0.1 s | 0.3 s | 1 s | 3 s | 10 s | 30 s | 100 s | 600 s |
|---|---|---|---|---|---|---|---|---|
| ILS | 11.40 (100 %) | 9.71 (100 %) | 6.18 (100 %) | 4.39 (100 %) | 3.66 (100 %) | 3.60 (100 %) | 3.60 (100 %) | 3.60 (100 %) |
| ILS-long | 11.42 (100 %) | 10.00 (100 %) | 8.60 (100 %) | 5.30 (100 %) | 3.06 (100 %) | 2.99 (100 %) | 2.99 (100 %) | 2.99 (100 %) |
| GA-fast | – (0 %) | – (0 %) | – (40 %) | 3.82 (77 %) | 2.44 (100 %) | 2.44 (100 %) | 2.44 (100 %) | 2.44 (100 %) |
| GA-mid | – (0 %) | – (0 %) | – (40 %) | 3.88 (78 %) | 1.94 (100 %) | 1.92 (100 %) | 1.92 (100 %) | 1.92 (100 %) |
| GA-quality | – (0 %) | – (0 %) | – (35 %) | 5.64 (63 %) | 1.63 (100 %) | 1.30 (100 %) | 1.30 (100 %) | 1.30 (100 %) |
| Gurobi | – (0 %) | – (0 %) | – (0 %) | – (0 %) | – (0 %) | – (37 %) | 0.00 (60 %) | 0.00 (99 %) |

Mean gap at budget (cells without a solution excluded; n in brackets):

| method | 0.1 s | 0.3 s | 1 s | 3 s | 10 s | 30 s | 100 s | 600 s |
|---|---|---|---|---|---|---|---|---|
| ILS | 20.39 [585] | 17.66 [585] | 13.83 [585] | 9.08 [585] | 6.89 [585] | 6.89 [585] | 6.89 [585] | 6.89 [585] |
| ILS-long | 20.18 [585] | 18.22 [585] | 16.62 [585] | 13.94 [585] | 6.50 [585] | 6.28 [585] | 6.28 [585] | 6.28 [585] |
| GA-fast | – | – | 0.96 [234] | 6.52 [453] | 6.34 [585] | 6.33 [585] | 6.33 [585] | 6.33 [585] |
| GA-mid | – | – | 0.97 [234] | 6.61 [455] | 5.88 [585] | 5.73 [585] | 5.73 [585] | 5.73 [585] |
| GA-quality | – | – | 1.66 [207] | 5.58 [371] | 6.11 [585] | 5.23 [585] | 5.15 [585] | 5.15 [585] |
| Gurobi | – | – | – | – | – | 0.00 [73] | 0.00 [117] | 0.00 [194] |

## 4. Time to reach a gap target (median s over cells×seeds; % of runs reaching it)

| method | ≤ 10 % | ≤ 5 % | ≤ 2 % | ≤ 1 % | ≤ 0.01 % |
|---|---|---|---|---|---|
| ILS | 0.04 (78 %) | 0.26 (62 %) | 0.39 (38 %) | 0.32 (23 %) | 0.02 (5 %) |
| ILS-long | 0.04 (80 %) | 0.36 (64 %) | 0.78 (42 %) | 0.80 (25 %) | 0.04 (7 %) |
| GA-fast | 0.88 (79 %) | 0.78 (67 %) | 0.73 (46 %) | 0.75 (35 %) | 1.06 (14 %) |
| GA-mid | 1.12 (81 %) | 0.83 (70 %) | 0.79 (51 %) | 0.79 (39 %) | 1.12 (15 %) |
| GA-quality | 1.36 (81 %) | 1.10 (75 %) | 1.08 (60 %) | 1.09 (45 %) | 1.47 (17 %) |
| Gurobi | 71.02 (100 %) | 71.02 (100 %) | 71.02 (100 %) | 71.02 (100 %) | 71.02 (100 %) |

## 5. Primal integral / T (mean over cells×seeds; lower is better)

| method | T = 10 s | T = 60 s | T = 600 s |
|---|---|---|---|
| ILS | 0.0741 | 0.0611 | 0.0588 |
| ILS-long | 0.0842 | 0.0588 | 0.0541 |
| GA-fast | 0.2395 | 0.0840 | 0.0560 |
| GA-mid | 0.2410 | 0.0804 | 0.0514 |
| GA-quality | 0.2864 | 0.0846 | 0.0473 |
| Gurobi | 1.0000 | 0.7446 | 0.1883 |

## 6. Against Gurobi per cell (seed-mean)

| arm | heuristic t / Gurobi t (median) | Gurobi gap at the heuristic's finish (median) | heuristic better than Gurobi-at-same-time (% cells) |
|---|---|---|---|
| ILS | 0.031 | no solution (0 % have one) | 100 |
| ILS-long | 0.080 | no solution (0 % have one) | 100 |
| GA-fast | 0.051 | no solution (0 % have one) | 100 |
| GA-mid | 0.076 | no solution (0 % have one) | 100 |
| GA-quality | 0.187 | no solution (0 % have one) | 100 |

## 7. Seed-mean gap by λ and class·workload

| arm | λ 0.25 | λ 0.5 | λ 0.75 | A·fedhpc | A·pos | B·fedhpc | B·pos |
|---|---|---|---|---|---|---|---|
| ILS | 10.27 | 5.73 | 4.66 | 2.25 | 10.04 | 1.81 | 10.22 |
| ILS-long | 9.83 | 5.34 | 3.66 | 1.94 | 9.27 | 1.62 | 9.28 |
| GA-fast | 10.27 | 5.23 | 3.48 | 0.73 | 9.99 | 0.91 | 10.01 |
| GA-mid | 9.58 | 4.74 | 2.88 | 0.59 | 9.17 | 0.82 | 8.98 |
| GA-quality | 9.05 | 4.24 | 2.17 | 0.41 | 8.35 | 0.65 | 8.10 |

Figure: /home/bernardo/Documentos/programas/fedhpc/results/generated_bench/dispatch_study/final/anytime.png
