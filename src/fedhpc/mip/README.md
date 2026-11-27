# fedhpc.mip: branch-and-cut on fedlp

A MIP solver built around fedlp, the simplex of `src/fedhpc/_ext/lp`. It
targets scheduling and arc-flow formulations first: Kramer et al.'s
P|r_j|Σw_jC_j models and MIPLIB 2017's scheduling and packing instances.

```python
from fedhpc.mip import read_mps, solve
r = solve(read_mps("model.mps.gz"), time_limit=60)
r.status, r.objective, r.bound, r.x
```

## Layout

| Module | Role |
|---|---|
| `problem.py` | `MipProblem`: min c'x + offset, row_lo ≤ Ax ≤ row_hi, lb ≤ x ≤ ub, integrality; a maximisation is stored negated |
| `mps.py` | MPS reader: fixed/free, gz, OBJSENSE, RANGES, every bound type except SC, integer markers. Checked against HiGHS's parser |
| `presolve.py` | reductions plus postsolve |
| `propagate.py` | activity-based domain propagation |
| `lp.py` | `LpRelaxation` over fedlp |
| `cuts.py` | Gomory mixed-integer and cover separators, cut pool |
| `heuristics.py` | rounding, fix-and-propagate, guided dive, RENS |
| `branching.py` | reliability pseudocost branching |
| `solver.py` | root cut loop and tree search |

## Mathematics

**Presolve.** The rounds apply:
- removal of fixed, empty and singleton columns and rows;
- propagation, with redundant-row removal;
- dual fixing: a column no row locks in its cost-improving direction goes
  to that bound (Achterberg 2007, §10.8);
- Savelsbergh coefficient tightening on one-sided rows over binaries: for
  a′x ≤ b with a_j > 0 and maxact − a_j < b, set a_j −= d and b −= d with
  d = b − (maxact − a_j).

Every reduction fixes, tightens or deletes, so postsolve only scatters the
kept and fixed values back. There are no aggregations.

**Propagation** (Brearley, Mitra & Williams 1975). Residual activities
without an entry's own contribution bound each column. Infinite
contributions are counted rather than summed. Integer bounds round inwards;
continuous bounds move only by at least 10⁻³ of their domain. The whole
pass is vectorised over CSR, with column reductions through a CSR → CSC
permutation.

**Gomory mixed-integer cuts** (Gomory 1960; Balas et al. 1996). A tableau
row x_B + Σ a_k v_k = β, with the nonbasics complemented to v_k ≥ 0 and
f₀ = frac(β), gives

  Σ_{int, f_k≤f₀} (f_k/f₀) v_k + Σ_{int, f_k>f₀} ((1−f_k)/(1−f₀)) v_k
  + Σ_{cont, a>0} (a_k/f₀) v_k + Σ_{cont, a<0} (−a_k/(1−f₀)) v_k ≥ 1.

Logicals are substituted through s_r = σ_r(rhs_r − a_r′x) and treated as
continuous (valid, but weaker than integral slacks). Safeguards:
- dynamism at most 10⁶;
- tiny coefficients relaxed out through the bounds;
- the right-hand side relaxed by 10⁻⁹;
- support at most half the columns.

**Cover cuts** (Crowder, Johnson & Padberg 1983). On one-sided binary rows
with negative coefficients complemented, a greedy cover by (1 − x*_j)/a_j
gives an extended cover inequality.

**Cut pool.** Cuts are selected by efficacy, with duplicates and
near-parallel ones (|cos| > 0.99) rejected. All root cuts are global. At
the end of the root, cuts with basic (slack) logicals are removed.

**Branching** (Achterberg, Koch & Martin 2005). Reliability pseudocosts
with the product score. Unreliable candidates are strong-branched with the
dual simplex under an iteration limit, within a budget: strong-branching
pivots stay within 50 % of the node LPs' plus 20 k. Only optimal or
infeasible strong-branch children may prune or fix: fedlp's objective at
an iteration limit is not a proven bound.

**Tree.** Nodes are taken by best bound, plunging into a child while it is
within 2 % of the best bound. A node is its path of branching bounds; its
domain is rebuilt and propagated when activated, and its parent's basis is
restored when it is not the child just solved. With an integral objective
(integer costs on integer columns only), bounds round up and the cutoff is
the incumbent minus 1.

**Heuristics:**
- simple rounding by locks;
- fix-and-propagate in doubling and halving batches;
- a guided dive (Danna et al. 2005);
- RENS (Berthold 2014) as a sub-MIP at the root.

## fedlp additions for MIP

These live in `lp_module.cpp` and `primal.hpp`, with tests in
`tests/test_fedlp_rows.py`:
- **Ranged rows:** `Solver.ranged(lo, hi)`. An L row's logical is bounded
  to [0, hi − lo].
- **`add_rows` / `remove_rows`:** the model is rebuilt with the same column
  ids, options and basis, and new logicals basic, so the dual simplex
  re-solves.
- **`get_basis` / `set_basis`:** through `PrimalSimplex::warm_start`, which
  falls back to the crash when the basis is singular or the wrong size.
- **`basic_heads`, `tableau_row`, `logicals`:** for cut separation.
- **Options outside the simplex:** the façade owns the `Options`, so they
  survive rebuilds.
- **A dual-simplex robustness fix.** An overflowed DSE weight scored its
  violated row 0, and the dual declared optimality over rows violated by up
  to 32 (cold AF_N100_M10 LP). Weights now stay finite, a violated row that
  no score selects triggers exact weights, and the dual confirms optimality
  on a fresh factor.

- **Infeasibility proofs and a time limit.** A dual row with no entering
  column, rechecked on a fresh factor, proves the LP infeasible. Before, the
  proof went to primal phase 1, which cycled for 8 M pivots on a csched008
  node LP. `Options.time_limit` caps a solve's wall clock, and the MIP passes
  its deadline to every LP. A singular refactorisation repairs the basis
  with logicals instead of failing.

## Benchmarks

`scripts/mip_bench.py`: one thread per solver (fedlp's included), relative
gap 1e-4, one pinned subprocess per run. HiGHS is 1.16 and Gurobi 12.
Times are shifted geometric means (shift 1 s; unsolved counted at the
limit).

| Set | Limit | fedhpc.mip | HiGHS | Gurobi |
|---|---|---|---|---|
| Kramer TI + AF, N = 20 (18) | 60 s | **18/18, 0.94 s** | 18/18, 1.40 s | 18/18, 0.48 s |
| Kramer TI + AF, N = 50 (18) | 120 s | 13/18 | 14/18 | 17/18 |
| MIPLIB scheduling + packing (10) | 120 s | 0/10 | 1/10 | 5/10 |
| N = 50 and MIPLIB together (28) | 120 s | 13/28, 31.6 s | 15/28, 30.3 s | 22/28, 12.0 s |

Every objective fedhpc.mip reports matches the reference solvers' and
MIPLIB's `.solu`.

**Where fedhpc.mip is ahead.** On the time-indexed models it beats HiGHS
throughout; on two of them it beats Gurobi too:

| Instance | fedhpc.mip | HiGHS | Gurobi |
|---|---|---|---|
| TI_N50_M2_R9_1 | 18 s | 84 s | 24 s |
| TI_N50_M5_R9_1 | 6 s | 34 s | 7 s |
| TI_N20_M2_R9_1 | 2 s | 9 s | — |

fedlp's warm dual re-solves on TI's dense columns are fast.

**Where it is behind, and why:**
- **Kramer M = 2, R0/R4, both models** (four instances): no incumbent
  within 120 s. The AF models have 160 k columns, and the root LP is fedlp's
  cold arc-flow weakness (about 10× Gurobi at N = 100: dense pivot rows of
  30 k breakpoints). The TI ones have the same LP-speed problem.
- **Root cuts on AF:** Gomory rows have about 27 k nonzeros out of 40 k
  columns and are rejected (support limit), so the root gets no cuts
  (bound 82,044 against Gurobi's 82,060 on AF_N50_M5_R4_1).
- **MIPLIB:**
  - csched007/008, lectsched, gfd and physiciansched find no incumbent.
  - physiciansched's LP (100 k rows after presolve) does not finish in the
    limit.
  - csched's big-M orderings defeat fix-and-propagate and the pump.

## Next steps

1. **The cold LP on arc-flow and time-indexed models.** A crash basis from
   a list-scheduling solution, and a ratio test for dense pivot rows
   (neither the heap nor grouped passes handle 30 k breakpoints well).
2. **Arc-flow cuts.** Sparse cuts suited to flow conservation (MIR /
   flow cover on aggregated rows, as the Kramer et al. arc-flow papers
   report) in place of dense Gomory rows.
3. **Primal side.** Conflict-driven backtracking in fix-and-propagate,
   local branching / RINS once an incumbent exists, and a job-sequence
   heuristic for Kramer's models.
4. **Presolve.** Aggregation of doubleton equations, parallel rows and
   columns, with a postsolve stack.
5. **Phase-1 cycling in fedlp's primal** (seen before the infeasibility fix
   routed around it) needs perturbation in phase 1.
