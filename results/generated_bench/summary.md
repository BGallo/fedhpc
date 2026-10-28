# Generated-instance benchmark

232 instances; Gurobi Threads=4, TimeLimit 600 s, MIPGap 0; dispatch heuristic n_threads=4; 4 instances in parallel.

Gurobi solve statuses: {'optimal': 1154, 'feasible': 6}

## Time (s): median [p90] per solve

| class·workload | n | trivial | lex f1 | lex f2 | WS Gurobi | WS dispatch | exact total | dispatch total |
|---|---|---|---|---|---|---|---|---|
| A·fedhpc | 81 | 24 | 0.0 [0.1] | 8.7 [11.8] | 1.2 [3.6] | 3.0 [4.6] | 14.0 [19.2] | 8.9 [13.1] |
| A·pos | 81 | 0 | 0.3 [0.4] | 61.2 [238.2] | 4.4 [31.3] | 18.7 [41.0] | 93.5 [350.5] | 58.2 [119.9] |
| B·fedhpc | 35 | 11 | 0.1 [0.1] | 10.9 [12.8] | 1.2 [4.4] | 3.6 [6.4] | 17.6 [24.0] | 10.9 [16.3] |
| B·pos | 35 | 2 | 0.4 [0.5] | 62.1 [134.8] | 3.0 [21.8] | 13.2 [41.1] | 74.4 [190.2] | 36.9 [127.2] |

## Weighted-sum quality: gap to exact g* (non-trivial instances)

| class·workload | λ | dispatch median gap % | dispatch p90 % | dispatch optimal (≤1e-6) | threshold median % | WS Gurobi faster than dispatch |
|---|---|---|---|---|---|---|
| A·fedhpc | 0.25 | 0.44 | 24.91 | 20/57 | 64.7 | 57/57 |
| A·fedhpc | 0.5 | 2.85 | 10.26 | 9/57 | 77.4 | 57/57 |
| A·fedhpc | 0.75 | 2.47 | 6.60 | 3/57 | 91.1 | 57/57 |
| A·pos | 0.25 | 64.26 | 144.67 | 3/81 | 175.6 | 69/81 |
| A·pos | 0.5 | 22.11 | 25.74 | 1/81 | 192.2 | 78/81 |
| A·pos | 0.75 | 4.60 | 8.94 | 0/81 | 124.2 | 81/81 |
| B·fedhpc | 0.25 | 0.44 | 17.06 | 6/24 | 58.0 | 24/24 |
| B·fedhpc | 0.5 | 1.79 | 8.57 | 6/24 | 43.8 | 24/24 |
| B·fedhpc | 0.75 | 1.99 | 6.48 | 1/24 | 29.8 | 24/24 |
| B·pos | 0.25 | 10.87 | 97.84 | 4/33 | 81.0 | 32/33 |
| B·pos | 0.5 | 9.52 | 31.68 | 2/33 | 113.5 | 33/33 |
| B·pos | 0.75 | 6.80 | 38.65 | 0/33 | 148.8 | 33/33 |

## Anchors vs initial load (median over instances)

| class·workload | load | n | f2^T (USD) | f1^0/f1^T − 1 (%) | lex f2 time (s) | exact total (s) |
|---|---|---|---|---|---|---|
| A·fedhpc | ≤25 % | 25 | 0 | 0.0 | 8.4 | 18.4 |
| A·fedhpc | 25–60 % | 25 | 169 | 2.4 | 8.5 | 12.0 |
| A·fedhpc | >60 % | 31 | 1408 | 41.5 | 9.3 | 12.7 |
| A·pos | ≤25 % | 25 | 3971 | 18.7 | 53.3 | 72.4 |
| A·pos | 25–60 % | 25 | 6214 | 38.1 | 59.7 | 78.3 |
| A·pos | >60 % | 31 | 9436 | 81.3 | 70.5 | 121.6 |
| B·fedhpc | ≤25 % | 5 | 0 | 0.0 | 9.1 | 19.1 |
| B·fedhpc | 25–60 % | 24 | 274 | 3.1 | 10.8 | 13.2 |
| B·fedhpc | >60 % | 6 | 1740 | 63.6 | 12.4 | 18.2 |
| B·pos | ≤25 % | 4 | 165 | 0.3 | 59.0 | 130.0 |
| B·pos | 25–60 % | 19 | 1593 | 5.4 | 62.1 | 71.7 |
| B·pos | >60 % | 12 | 2447 | 14.6 | 67.8 | 84.1 |

## What drives exact solve time (Spearman ρ with lex-f2 time)

- fedhpc: busy share at t=0: ρ = +0.32
- fedhpc: on-prem nodes: ρ = -0.12
- fedhpc: running jobs: ρ = +0.20
- fedhpc: f1^0 − f1^T: ρ = +0.51
- pos: busy share at t=0: ρ = +0.18
- pos: on-prem nodes: ρ = -0.56
- pos: running jobs: ρ = -0.16
- pos: f1^0 − f1^T: ρ = +0.67
