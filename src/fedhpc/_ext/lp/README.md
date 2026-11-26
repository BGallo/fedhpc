# fedlp — a simplex for column generation masters

`fedhpc._fedlp` solves the restricted master LPs of `fedhpc.bpc`
(`PricingOptions(master="fedlp" | "auto" | "gurobi")`). It is not a
general LP solver. It targets what branch and price needs and Gurobi is not
built for:

- many re-solves of one LP as columns arrive;
- few rows (tens to a few hundred) and many sparse columns;
- heavily degenerate set-partitioning structure.

## Layout

| File | Role |
|---|---|
| `model.hpp` | `Model`: the column store, plus an append-only row-wise copy for pivot rows. Row logicals are columns `0..m-1`, so slacks and structurals are handled alike. Columns can be appended, tombstoned and have their bounds changed. |
| `dense_inverse.hpp` | `DenseInverse`: an explicit B^-1 (Gauss-Jordan refactorisation, rank-1 updates). The sparse-row update and an AVX2 clone of the dense one are picked at load time. Its interface (factorize / ftran / row / update) is what a sparse LU would implement. |
| `primal.hpp` | `PrimalSimplex`: the algorithm. |
| `lp_module.cpp` | `Solver`: the Python façade. It does batch CSC column adds from numpy, and returns duals, primal values, reduced costs, basis status and stats. Solves release the GIL. |

## Algorithm

Bounded primal simplex, warm-started across column additions: appended
columns keep the basis primal feasible, so nothing is refactorised and
duals, reduced costs and weights carry over.

- **Pricing.** Devex (Harris 1973; Forrest & Goldfarb 1992). Reduced costs
  and weights are maintained through the pivot row.
- **Hybrid pivot rows.** The pivot row is computed row-wise from the
  nonzeros of `e_r' B^-1` when that is sparse (Hall & McKinnon
  hyper-sparsity, as in HiGHS). Otherwise it is computed column-wise, fused
  with the reduced-cost and weight updates and the choice of the next
  entering column.
- **Sifting.** Pivots and pricing see the basic columns plus a working set:
  the new columns and the recently attractive ones, capped at
  `max(work_min, work_per_row * m)`. Frozen columns are priced only when the
  working set is optimal. Exact: optimality is declared only after a fresh
  pricing pass over every column. In column generation almost all old
  columns never re-enter, so this cut LP time 2-4x at an unchanged pivot
  count.
- **Ratio test.** Harris two-pass test with bound flips. A leaving variable
  that overshoots its bound has the bound shifted, never a backward step
  (Koberstein 2005).
- **Degeneracy.** A streak of degenerate pivots, counted across the rounds
  of a solve, relaxes the bounds degenerate basics sit on by random amounts
  of about 5e-7. At optimality the true bounds are restored. Any residual
  infeasibility goes to a sum-of-infeasibilities phase 1, with the violated
  bound as the breakpoint; then phase 2 runs again.
- **Tolerances.** The dual tolerance of column j is
  `1e-9 (1 + |c_j|) + 1e-14 |y|_inf (1 + nnz_j)`. The second term is the
  rounding of `c_j - y'a_j` when big-M artificials make the duals large.
  Without it, cancellation noise of 1e-8 made a trap master ping-pong
  between rounds.
- **Stability.** The pivot element is compared as computed by row and by
  column, and a disagreement triggers refactorisation. There is also a
  fixed refactorisation interval.

## Validation

- **`tests/test_fedlp.py`, 64 tests against Gurobi.** Each checks the
  optimal objective and the optimality certificate (primal feasibility,
  dual signs, reduced-cost signs at bounds). The cases are random LPs,
  degenerate masters re-solved as columns arrive, removals and bound
  changes, and infeasible and unbounded LPs.
- **`tests/test_bpc_fedlp.py`, branch and price with fedlp masters.** It
  finds the same optima as with Gurobi masters under weights, bound rows,
  lex-min and fixed costs (the published Rm optimum). It finds the same
  Pareto front, and checks the `auto` selection.
- **Recorded RMP sequences** of four Rm solves (`scripts/bpc_bench.py`
  instances), solve by solve:
  - every objective agrees with Gurobi's;
  - the one 1.8e-6 difference was Gurobi's tolerance: a tight fresh Gurobi
    solve agrees with fedlp.

  | Master | Rows | Gurobi | fedlp | Speed-up |
  |---|---|---|---|---|
  | rm_M2N30_1 | 94 | 1.01 s | 0.17 s | 5.9x |
  | rm_M2N20_3 | 64 | 0.87 s | 0.14 s | 6.4x |
  | rm_M5N50_1 | 310 | 1.13 s | 0.39 s | 2.9x |
  | rm_M8N40_1 | 376 | 0.51 s | 0.34 s | 1.5x |

  About half as many pivots as Gurobi on each.
- **End to end** (`bpc_bench.py`, `master="auto"` against `"gurobi"`):

  | Suite | Gurobi | auto | Change |
  |---|---|---|---|
  | hard | 8.84 s | 7.24 s | -18 % |
  | gap | 6.71 s | 5.45 s | -19 % |
  | base | 3.68 s | 2.84 s | -23 % |

  trap_medium's Pareto front is identical, in 0.87 s instead of 1.40 s.

## Tried and rejected

- **Lazy reduced costs.** Skipping columns by an exact drift bound skipped
  only 10-20 % of updates, since the worst-case bound is loose, and was
  slower overall.
- **Positive Edge.** Not implemented: after perturbation only 14-24 % of
  pivots are degenerate, below the 25 % where Towhidi et al. found it pays.

## Next steps

- **A sparse LU with Forrest-Tomlin updates.** Above about 400 rows the
  O(m^2) explicit-inverse update loses to Gurobi: rm_M10N50_1 has 570 rows,
  mostly fixed-cost link rows. `auto` therefore keeps Gurobi there
  (`bpc._FEDLP_MAX_ROWS`).
- **Warm-starting tree nodes.** Keep one master across nodes and start each
  node from its parent's basis, with a dual simplex after the branching
  bound changes, instead of rebuilding every node.
- **Rows for cuts**, so limited-memory subset-row cuts can run on fedlp
  masters.
