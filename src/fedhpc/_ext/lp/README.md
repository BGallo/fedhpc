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
| `dense_inverse.hpp` | `DenseInverse`: an explicit B^-1 (Gauss-Jordan refactorisation, rank-1 updates). The sparse-row update and an AVX2 clone of the dense one are picked at load time. |
| `factor.hpp` | `BasisFactor`: the factor interface (factorize, which may reorder the basis positions, plus ftran, btran, row, update, which may report failure). `DenseFactor` wraps the explicit inverse; `SparsePFI` is a product-form inverse. |
| `lu_ft.hpp` | `LuFT`: sparse LU with Forrest-Tomlin updates (see below). `Options.factor` picks a factor: by default dense up to `dense_max_rows` = 128 rows, the LU above. |
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
  fixed refactorisation interval: 200 pivots dense, 64 etas sparse.
- **Dual steepest edge** (`Options.dual_pricing` = 1, the default;
  Forrest & Goldfarb 1992). The leaving row maximises violation^2 / beta_i,
  with beta_i ~ |e_i' B^-1|^2. After a pivot, one extra FTRAN of the pivot
  row gives tau = B^-1 rho_r, and the update runs over alpha's nonzeros.
  Weights start at 1 and survive refactorisation, except under the product
  form, which reorders positions. On 300-step branching dives over recorded
  masters, with the same bound changes for both rules and the same
  objective at every step, it beat dual Dantzig by 18-42 % in time:

  | Master | Dantzig | DSE |
  |---|---|---|
  | M5N50_1 | 3.63 s | 2.83 s |
  | M8N40_1 | 0.226 s | 0.155 s |
  | M2N30_1 | 0.176 s | 0.103 s |
  | M2N20_3 | 0.107 s | 0.088 s |

- **Sparse vectors.** The factor reports the nonzero pattern of
  alpha = B^-1 a_q (the LU collects it in its back-substitution), and the
  ratio test and x_B updates loop over it. e_r' B^-1's pattern drives the
  row-cost estimate, the row-wise pivot row and the dual update. The
  pivot's ratio test breaks |alpha| ties toward the lowest position,
  whatever order the pattern lists them in. Ties at 1 are everywhere in set
  partitioning, and leaving the order to the pattern sent rm_M8N40_1's
  column generation from 137 to 542 iterations.
- **Phase timers** compiled in with `-DFEDLP_PROFILE` (`stats()["t_*"]`).
- **Dual simplex** (`Options.dual`). When a solve starts primal infeasible
  but dual feasible, a bounded dual simplex runs first. That is the case
  when bounds changed and nothing was appended, as in a re-targeted
  branch-and-price node. It uses dual Dantzig rows, a Harris two-pass dual
  ratio test, and pivot rows over every column (no sifting). Violations no
  pivot can repair, below `feas_accept` = 1e-7, are rounding and get
  snapped. A dual ray hands over to phase 1, which proves infeasibility
  reliably: a tiny violation once looked like a ray.
- **Final refactorisation.** Optimality is declared from a fresh factor
  whenever updates have accumulated. With big-M costs in c_B, the updates'
  rounding could hide an improving column: about 2e-7 relative
  suboptimality with the LU.
- **Crash hint** (`Solver.set_crash_hint`). Columns of a known feasible
  solution take equality rows first in the crash. lpfirst passes its
  incumbent, a feasible start without artificials: pos_congestion's anchor
  went from 105.6 k to 49.8 k iterations.
- **LU with Forrest-Tomlin updates (`LuFT`).**
  - Factorisation is left-looking over the basis columns, sparsest first,
    with threshold pivoting. It is hyper-sparse: each column's nonzero
    pattern is tracked, and L etas are applied through a min-heap of the
    touched pivot rows.
  - U is stored by rows and by columns, so BTRAN is row-oriented and skips
    zeros.
  - An update moves the leaving position's spike to the end of the order
    and eliminates its pivot row into one row eta. It refactorises when the
    determinant check (new diagonal = alpha_r times the old one) fails or a
    multiplier exceeds 1e4.
- **Sparse factor (`SparsePFI`).** Left-looking Gauss-Jordan in product
  form over the basis columns, sparsest first: the master's slacks,
  artificials and null columns give trivial etas, so only a small bump
  fills in. Pivot rows use threshold partial pivoting (at least 0.1 of the
  column maximum), the sparsest row among those. FTRAN skips every eta whose
  pivot entry is zero, a hyper-sparse solve. A pivot appends one eta, which
  is O(nnz(alpha)) against the explicit inverse's O(m^2).

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

- **End to end with the sparse factor**, every master on fedlp
  (`lpf2_*`):

  | Suite | Gurobi | fedlp | Change |
  |---|---|---|---|
  | hard | 8.84 s | 6.90 s | -22 % |
  | gap | 6.71 s | 4.85 s | -28 % |

  The LP of rm_M10N50_1 (570 rows) went from 0.74 s with the dense
  inverse to 0.40 s, level with Gurobi; `auto` now covers masters up to
  2000 rows.

- **lpfirst masters** (`LPFirstOptions(lp_backend="fedlp")`, opt-in). The
  964-job trace's masters (~1.5 k rows) take 0.59 s of LP with the LU,
  against 1.02 s with Gurobi and 1.94 s with the product form: the five
  bench solves 9.07 -> 7.81 s. pos_congestion's (4.5 k rows, ~35 k
  columns) stay about 9x slower than Gurobi, from the per-pivot dense O(m)
  work. All results are exact.
- **bpc with the LU as the default sparse factor** (`lplu_*`):

  | Suite | Gurobi masters | fedlp |
  |---|---|---|
  | hard | 8.84 s | 6.33 s |
  | gap | 6.71 s | 4.95 s |
  | base | 3.68 s | 2.84 s |

## Tried and rejected

- **Lazy reduced costs.** Skipping columns by an exact drift bound skipped
  only 10-20 % of updates, since the worst-case bound is loose, and was
  slower overall.
- **Positive Edge.** Not implemented: after perturbation only 14-24 % of
  pivots are degenerate, below the 25 % where Towhidi et al. found it pays.

- **Warm-started tree nodes** (`PricingOptions.warm_nodes`, off). With
  the dual simplex they are level with rebuilding. With dual steepest edge
  as well:
  - gap tier (the most branching) 5.01 -> 4.65 s;
  - hard 6.24 vs 6.35 s;
  - base 2.84 vs 2.89 s.

  Even in total, so still off; it pays as trees deepen.
- **Partial pricing** (`Options.partial_min_cols`, off). Pricing sections
  of the working set from the duals, with no pivot row, halved the cost
  per pivot on pos_congestion's lpfirst masters but took 2.9x the pivots
  (Dantzig sections against Devex): 15.1 vs 11.5 s.
- **Working-set row lists for pivot rows.** On pos_congestion e_r' B^-1 is
  dense, so pivot rows touch most of the working set either way, and the
  lists made them slower (6.8 vs 5.2 s): reverted. One
  master for the whole search; each node re-targets its column and `z_m`
  bounds and continues from the last basis, with phase 1 repairing what
  branching made infeasible. Exact, but neutral to slightly negative:
  - hard tier 6.84 -> 7.12 s;
  - trap_medium's front 1.71 -> 1.83 s.

  A node master is cheap to rebuild, re-targeting costs about as much, and
  these trees are 1-15 nodes deep.

## Next steps

- **Pricing on the largest masters.** pos_congestion's lpfirst anchor
  (4.5 k rows, ~35 k columns) takes 11 s against Gurobi's 1.3 s: 49.8 k
  Devex pivots against 29 k, at about 230 us each against 46. The time is
  in the pivot rows and selection over a large working set. This is a
  pricing problem, not dense vectors: the per-pivot vector work is now
  sparse. Candidate list sizes, multiple pricing or a dual-simplex re-solve
  of the CG masters are the levers to try.
- **Rows for cuts.** Logicals are columns `0..m-1`, so added rows need a
  logical-index map. Subset-row cuts are off by default (a negative result
  on the Rm instances), so this waits for a use.
