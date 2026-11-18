195 common cells, seeds [42, 1, 7]; Gurobi: bench solves (step at its time t)

## 1. Final gap and time

| arm | mean gap (per seed) | mean ± sd over seeds | median | p90 | max | cells > 10 % | time/seed (s) | median t/cell (s) |
|---|---|---|---|---|---|---|---|---|
| ILS | 6.86 / 6.88 / 6.86 | 6.87 ± 0.01 | 3.60 | 18.4 | 58.7 | 42.3 | 652 | 2.17 |
| ILS-long | 6.25 / 6.27 / 6.23 | 6.25 ± 0.02 | 2.99 | 18.1 | 56.6 | 37.0 | 1302 | 5.62 |
| GA-fast | 6.07 / 6.29 / 6.35 | 6.24 ± 0.14 | 2.44 | 18.9 | 62.3 | 39.3 | 845 | 3.89 |
| GA-mid | 5.56 / 5.72 / 5.69 | 5.66 ± 0.09 | 1.92 | 18.4 | 59.2 | 36.0 | 1307 | 6.27 |
| GA-quality | 5.09 / 5.12 / 5.07 | 5.09 ± 0.02 | 1.30 | 17.5 | 61.8 | 34.7 | 3330 | 15.79 |
| Gurobi (bench) | 0.000 | | 0.000 | | 0.00 | | 22104 | 70.6 |

Gurobi status: {'optimal': 194, 'feasible': 1}

## 2. Paired comparison (seed-mean gap per cell; tie ±0.5 pp)

| A vs B | mean diff A−B (pp) | 95 % CI | W/T/L (A better/tie/worse) | sign-test p |
|---|---|---|---|---|
| ILS vs ILS-long | +0.62 | [+0.47, +0.78] | 0/132/63 | 2.2e-19 |
| ILS vs GA-fast | +0.63 | [+0.27, +1.01] | 39/72/84 | 6.1e-05 |
| ILS vs GA-mid | +1.21 | [+0.83, +1.61] | 23/79/93 | 3.6e-11 |
| ILS vs GA-quality | +1.77 | [+1.36, +2.21] | 8/65/122 | 2.6e-27 |
| ILS-long vs GA-fast | +0.01 | [-0.34, +0.37] | 55/73/67 | 0.32 |
| ILS-long vs GA-mid | +0.59 | [+0.27, +0.92] | 38/73/84 | 3.8e-05 |
| ILS-long vs GA-quality | +1.16 | [+0.82, +1.51] | 14/78/103 | 6.4e-18 |
| GA-fast vs GA-mid | +0.58 | [+0.44, +0.74] | 0/138/57 | 1.4e-17 |
| GA-fast vs GA-quality | +1.14 | [+0.92, +1.38] | 1/90/104 | 5.2e-30 |
| GA-mid vs GA-quality | +0.56 | [+0.42, +0.70] | 2/111/82 | 3.7e-22 |

## 3. Gap of the best-so-far at a wall-clock budget (median over cells×seeds; % cells with a solution)

| method | 0.1 s | 0.3 s | 1 s | 3 s | 10 s | 30 s | 100 s | 600 s |
|---|---|---|---|---|---|---|---|---|
| ILS | 11.25 (100 %) | 9.67 (100 %) | 6.18 (100 %) | 4.39 (100 %) | 3.66 (100 %) | 3.60 (100 %) | 3.60 (100 %) | 3.60 (100 %) |
| ILS-long | 11.41 (100 %) | 9.77 (100 %) | 8.48 (100 %) | 5.30 (100 %) | 3.06 (100 %) | 2.99 (100 %) | 2.99 (100 %) | 2.99 (100 %) |
| GA-fast | – (0 %) | – (0 %) | – (40 %) | 3.82 (77 %) | 2.44 (100 %) | 2.44 (100 %) | 2.44 (100 %) | 2.44 (100 %) |
| GA-mid | – (0 %) | – (0 %) | – (40 %) | 3.88 (77 %) | 1.94 (100 %) | 1.92 (100 %) | 1.92 (100 %) | 1.92 (100 %) |
| GA-quality | – (0 %) | – (0 %) | – (35 %) | 5.64 (62 %) | 1.63 (100 %) | 1.30 (100 %) | 1.30 (100 %) | 1.30 (100 %) |
| Gurobi | – (0 %) | – (0 %) | – (0 %) | – (0 %) | – (0 %) | – (37 %) | 0.00 (60 %) | 0.00 (99 %) |

Mean gap at budget (cells without a solution excluded; n in brackets):

| method | 0.1 s | 0.3 s | 1 s | 3 s | 10 s | 30 s | 100 s | 600 s |
|---|---|---|---|---|---|---|---|---|
| ILS | 20.17 [585] | 17.43 [585] | 13.70 [585] | 9.17 [585] | 6.87 [585] | 6.87 [585] | 6.87 [585] | 6.87 [585] |
| ILS-long | 19.96 [585] | 17.98 [585] | 16.34 [585] | 13.70 [585] | 6.51 [585] | 6.25 [585] | 6.25 [585] | 6.25 [585] |
| GA-fast | – | – | 0.96 [234] | 6.24 [449] | 6.24 [585] | 6.24 [585] | 6.24 [585] | 6.24 [585] |
| GA-mid | – | – | 0.97 [234] | 6.34 [452] | 5.81 [585] | 5.66 [585] | 5.66 [585] | 5.66 [585] |
| GA-quality | – | – | 1.66 [207] | 5.06 [364] | 6.05 [585] | 5.17 [585] | 5.09 [585] | 5.09 [585] |
| Gurobi | – | – | – | – | – | 0.00 [73] | 0.00 [117] | 0.00 [194] |

## 4. Time to reach a gap target (median s over cells×seeds; % of runs reaching it)

| method | ≤ 10 % | ≤ 5 % | ≤ 2 % | ≤ 1 % | ≤ 0.01 % |
|---|---|---|---|---|---|
| ILS | 0.04 (78 %) | 0.26 (62 %) | 0.40 (38 %) | 0.32 (23 %) | 0.02 (5 %) |
| ILS-long | 0.04 (81 %) | 0.36 (64 %) | 0.78 (42 %) | 0.80 (25 %) | 0.04 (7 %) |
| GA-fast | 0.97 (80 %) | 0.78 (67 %) | 0.73 (46 %) | 0.75 (35 %) | 1.06 (14 %) |
| GA-mid | 1.14 (82 %) | 0.83 (70 %) | 0.79 (51 %) | 0.80 (39 %) | 1.12 (15 %) |
| GA-quality | 1.39 (82 %) | 1.12 (76 %) | 1.08 (60 %) | 1.09 (45 %) | 1.47 (17 %) |
| Gurobi | 70.61 (100 %) | 70.61 (100 %) | 70.61 (100 %) | 70.61 (100 %) | 70.61 (100 %) |

## 5. Primal integral / T (mean over cells×seeds; lower is better)

| method | T = 10 s | T = 60 s | T = 600 s |
|---|---|---|---|
| ILS | 0.0743 | 0.0610 | 0.0587 |
| ILS-long | 0.0840 | 0.0586 | 0.0539 |
| GA-fast | 0.2418 | 0.0837 | 0.0553 |
| GA-mid | 0.2432 | 0.0802 | 0.0508 |
| GA-quality | 0.2899 | 0.0847 | 0.0469 |
| Gurobi | 1.0000 | 0.7446 | 0.1888 |

## 6. Against Gurobi per cell (seed-mean)

| arm | heuristic t / Gurobi t (median) | Gurobi gap at the heuristic's finish (median) | heuristic better than Gurobi-at-same-time (% cells) |
|---|---|---|---|
| ILS | 0.032 | no solution (0 % have one) | 100 |
| ILS-long | 0.082 | no solution (0 % have one) | 100 |
| GA-fast | 0.051 | no solution (0 % have one) | 100 |
| GA-mid | 0.076 | no solution (0 % have one) | 100 |
| GA-quality | 0.187 | no solution (0 % have one) | 100 |

## 7. Seed-mean gap by λ and class·workload

| arm | λ 0.25 | λ 0.5 | λ 0.75 | A·fedhpc | A·pos | B·fedhpc | B·pos |
|---|---|---|---|---|---|---|---|
| ILS | 10.23 | 5.72 | 4.64 | 2.25 | 9.99 | 1.81 | 10.22 |
| ILS-long | 9.78 | 5.31 | 3.66 | 1.94 | 9.22 | 1.62 | 9.23 |
| GA-fast | 10.16 | 5.08 | 3.46 | 0.73 | 9.89 | 0.91 | 9.80 |
| GA-mid | 9.53 | 4.59 | 2.85 | 0.59 | 9.10 | 0.82 | 8.79 |
| GA-quality | 9.00 | 4.15 | 2.14 | 0.41 | 8.28 | 0.65 | 7.97 |

Figure: /home/bernardo/Documentos/programas/fedhpc/results/generated_bench/dispatch_study/final/anytime.png
