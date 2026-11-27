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
| `propagate.py` | activity-based domain propagation: the C++ kernel (`_ext/mip/propagate.hpp`, module `fedhpc._fedmip`) with a bound trail, and the NumPy reference |
| `reformulate.py` | interval row blocks differenced into network rows, for the LP only |
| `lp.py` | `LpRelaxation` over fedlp: bounds, cuts, bases with DSE weights, tableau rows, crash from a point |
| `cuts.py` | Gomory mixed-integer, cover, zero-half and c-MIR separators, implied integers, cut pool |
| `heuristics.py` | rounding, fix-and-propagate with backjumping, dives, feasibility pump, RENS, RINS |
| `scheduling.py` | list scheduling read off interval (TI) and arc-flow (AF) structure |
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
- fix-and-propagate (now a depth-first search with backjumping, below);
- a guided dive (Danna et al. 2005);
- RENS (Berthold 2014) as a sub-MIP at the root.

**Interval rows as network rows** (`reformulate.py`; Veinott & Wagner
1962). A block of consecutive rows in which every column's entries are one
run of adjacent rows with a constant coefficient (time-indexed capacity
rows: x_js covers periods s..s+p_j-1) is multiplied by the lower
bidiagonal matrix that differences adjacent rows, with explicit slacks
a_t'x + sigma_t s_t = rhs_t. A run [a, e] with coefficient v leaves +v at
row a and -v at row e+1; a slack leaves +sigma at its row and -sigma at the
next. The feasible set in x and every tableau B^-1 A are unchanged. Only the
LP sees it: slack s_t is reported as row t's logical, the differenced rows'
own logicals are fixed at 0 and dropped from tableau rows, and duals map
back by the transpose (y_t = y'_t - y'_{t+1}). Detection chains adjacent
rows sharing continuing runs, splits a chain where a column re-enters it
(the fewest cuts, by greedy interval stabbing) and keeps a block when it at
least halves the block's nonzeros. TI_N50_M2: 3.6 M to 0.2 M nonzeros.

**Propagation in C++** (`_ext/mip/propagate.hpp`). Event driven: a row is
examined only when one of its columns changed, its activity recomputed
exactly from the domain as it is scanned (nothing stored, nothing drifts),
and skipped after that first pass when its slack is at least its largest
|a_j| (u_j - l_j). The domain has a bound trail (level / change / undo), so
node domains are the root's fixed point plus the path's changes,
propagated from their rows only, and dives and fix-and-propagate push and
pop without copying arrays.

**Zero-half cuts** (Caprara & Fischetti 1996; Koster, Zymolka & Kutschka
2009). Rows with integral data over (implied) integer columns, columns
complemented to their nearest bound (v >= 0, value y*): half the sum of a
row set with odd right-hand side, rounded down, is valid and violated by
(1 - sum slack - sum_{odd u_j} y*_j) / 2. Sets are found by Gaussian
elimination over GF(2) on bitsets, the fractional columns eliminated by
decreasing y*. The cut family Gurobi reports most on Kramer's models.

**Implied integers.** A continuous column with integral bounds, unit
coefficient in an equality row with integral right-hand side over
otherwise integral terms, is integral in every feasible point (arc-flow
loss arcs). Used by the cut separators, never by branching.

**c-MIR** (Marchand & Wolsey 2001). From tight rows with fractional
integers: aggregation along the continuous column farthest inside its
bounds (up to 5 rows), bound substitution, divisors from the coefficients
of integers strictly inside their bounds, halved up to three times.

**Root LP choice.** On wide LPs (columns >= 20 x rows) the primal simplex
with partial pricing from the slack basis (on a network-form LP: the
slacks of the differenced rows, the original slack basis), capped at
3 x rows phase-1 pivots (10 x on network form), then the dual. A
heuristic schedule's basis (``LpRelaxation.basis_from_point``: inside
columns basic, matched to rows of their own support, unmatched rows'
logicals basic) starts the primal in phase 2.

**Heuristics added:**
- fix-and-propagate as a depth-first search with conflict-directed
  backjumping (Prosser 1993): on a dead end, back to the latest decision
  on a column of the failing rows;
- interval list scheduling (`scheduling.py`): jobs = assignment rows,
  runs = interval-block entries; first-fit cheapest column per job in LP
  mean-start and alpha-point orders (Hall, Schulz, Shmoys & Wein 1997),
  insertion local search;
- RINS (Danna, Rothberg & Le Pape 2005) at the root and every 50 nodes.

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

- **Round 2:** a dual ratio test over compact candidate arrays with
  breakpoints taken in sample-quantile chunks (the heap over ~10 k
  breakpoints cost ~70 us a pivot; AF_N50_M5_R0's ratio test 4.9 -> 1.4 s,
  pos_congestion's replay 3.38 -> 2.86 s); partial pricing in phase 1;
  `Options.phase1_limit`; DSE weights saved with a basis
  (`get_dual_weights` / `set_dual_weights`), where every restore had cost
  the next dual solve an O(m^2) exact initialisation - most of strong
  branching's time on AF_N50_M2.
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
limit). Labels `k20_v4`, `k50_mip_v4` (v2: before this round).

| Set | Limit | fedhpc.mip v2 | fedhpc.mip v4 | HiGHS | Gurobi |
|---|---|---|---|---|---|
| Kramer TI + AF, N = 20 (18) | 60 s | 18/18, 0.94 s | **18/18, 0.74 s** | 18/18, 1.40 s | 18/18, 0.48 s |
| Kramer TI + AF, N = 50 (18) | 120 s | 13/18 | 14/18 | 14/18 | 17/18 |
| MIPLIB scheduling + packing (10) | 120 s | 0/10 | 0/10 | 1/10 | 5/10 |
| N = 50 and MIPLIB together (28) | 120 s | 13/28, 31.6 s | 14/28, **27.9 s** | 15/28, 30.3 s | 22/28, 12.0 s |

Every Kramer N = 50 instance now has an incumbent; the four unsolved ones
end at gaps of 0.22-0.28 % (v2: no incumbent at all). On TI_N50_M2_R4
fedhpc.mip (156,091) and Gurobi (156,051) both hit the limit; HiGHS is at
313,608. Every objective matches the reference solvers' and MIPLIB's.

| Instance | v2 | v4 | HiGHS | Gurobi |
|---|---|---|---|---|
| TI_N50_M2_R9_1 | 18 s | 12 s | 84 s | 24 s |
| TI_N50_M5_R9_1 | 6 s | 4 s | 34 s | 7 s |
| TI_N50_M5_R4_1 | limit | 33 s | 59 s | 7 s |
| TI_N50_M2_R0_1 | no incumbent | 0.22 % | 43 % | 30 s |
| AF_N50_M5_R0_1 | 53 s | 11 s | 24 s | 6 s |
| AF_N50_M2_R0_1 | no incumbent | 0.25 % | 1.8 % | 52 s |

**Where the time went, and what changed it** (per-phase `stats["time"]`):
- cold root LPs: TI_N50_M2_R0 from over 116 s to 8 s (network form,
  primal with partial pricing), AF_N50_M2_R0 from 66 s to 8 s;
- root cuts: 18-25 s of Python per round to under 1 s (vectorised GMI and
  pool);
- strong branching: the O(m^2) DSE re-initialisation per restore removed;
- incumbents: list scheduling within 0.1-0.4 % of the optimum at the root
  (TI and AF), where fix-and-propagate and the dives found none.

**Where it is behind.** Node LPs and strong branching are 70-90 % of the
time on the unsolved Kramer instances: AF child LPs take 100-11 k dual
pivots at ~0.45 ms each (about 3x Gurobi's). Path variance is large:
TI_N50_M5_R0 took 15 s in one version and 118 s in the next from a
different root vertex (6 against 206 nodes). On MIPLIB: physiciansched's
root LP does not finish, and csched, gfd, lectsched and the fhnw bin
packing find no incumbent.

## Analysis: what the engine has, what it lacks

### What it has

| Area | In fedhpc.mip | Notes |
|---|---|---|
| Presolve | fixed, empty and singleton rows and columns; propagation with redundant rows; dual fixing; Savelsbergh coefficient tightening | scatter-only postsolve |
| LP | fedlp: bounded primal (Devex, sifting, partial pricing) and dual (DSE, BFRT, cost perturbation, flip start), FT-LU with hyper-sparse solves | cold root: primal on wide LPs, falling back to the dual; warm nodes: dual with saved DSE weights |
| LP reformulation | interval rows to network rows | TI's LP at ~5 % of its nonzeros |
| Propagation | C++, event driven, bound trail | node domains propagate from the path's rows only |
| Cuts | GMI (with implied integers), extended covers, zero-half, c-MIR with aggregation; efficacy and parallelism pool; slack cuts removed after the root | root only |
| Branching | reliability pseudocosts, strong branching under a pivot budget | |
| Search | best bound with plunging, deepest-first ties, integral-objective rounding | |
| Heuristics | rounding, fix-and-propagate with backjumping, two dives, objective feasibility pump, RENS, RINS, interval and arc-flow list scheduling | |

### What it lacks, by expected gain on the benchmark families

1. **Per-pivot speed of the dual on node LPs.** After this round, node LPs
   and strong branching are 70-90 % of the time on the unsolved Kramer
   instances. Child LPs need 100-11 k dual pivots on the degenerate flow
   LPs, at ~0.45 ms each on AF_N50_M2 (about 3x Gurobi's). The pivot row's
   PRICE and candidate pass cost the most (20-56 k entries a row).
2. **Local cuts and a cut pool across the tree.** All cuts are root cuts;
   Gurobi also separates in the tree (zero-half and MIR again at nodes).
3. **Conflict analysis.** Infeasible nodes teach nothing; SCIP's conflict
   graphs prune csched-like trees much faster.
4. **Symmetry handling.** Kramer's identical machines are symmetric in the
   TI and AF models only through the aggregated capacity (no machine
   index), but MIPLIB's bin packing (bppc, fhnw-binpack) needs orbital
   fixing or orbitopes.
5. **Presolve aggregation** (doubleton equations, parallel rows and
   columns) with a postsolve stack; probing and clique detection.
6. **MIPLIB-scale robustness.** physiciansched's root LP (100 k rows) does
   not finish in 120 s: scaling, a crash for big equality-heavy LPs, and
   a barrier for cold solves would be needed. gfd/lectsched/csched find no
   incumbent: they need problem-independent heuristics stronger than
   diving (local branching, ...).
7. **Restarts and root node reoptimisation**, which Gurobi uses when the
   root fixes many columns.

### Options implemented but off by default

| Option | Where | Why off | Where it could pay |
|---|---|---|---|
| `mp_min_cols` (multiple pricing) | fedlp | 3x the pivots of full Devex on pos_congestion | LPs with cheap dot products and expensive pivot rows |
| `partial_min_cols` (partial pricing) | fedlp | 2.9x the pivots on pos_congestion's masters | on now for the cold root primal of wide LPs (Kramer: 2-5x faster); candidates: other LPs with n >> m |
| `dual_sift` (dual over the working set) | fedlp | lost on pos_congestion (3.1 -> 6.3 s) | masters with many irrelevant columns |
| `dual_flip_min_rows` = 3000 | fedlp | the primal won below 3 k rows on the lpfirst trace | boxed LPs of any size where the crash is far from dual feasible |
| `phase1_limit` = 0 | fedlp | a caller's budget | set by the MIP root policy |
| `schedule_crash` | mip | the crash from a Smith schedule did not beat the slack basis on TI (9.9 vs 8.0 s); from a near-optimal point AF_N50_M5_R0 went 1.15 -> 0.53 s | AF models with a good schedule before the LP |
| `PricingOptions.warm_nodes` | bpc | even in total | deeper B&P trees |
| SR cuts (bpc) | bpc | negative on the Rm instances | other set-partitioning masters |
| `root_lp` = "dual" / "primal" | mip | "auto" chooses | LPs where the size rule misjudges (AF_N50_M5_R0's dive succeeded from the dual's vertex: 7.6 s against a timeout) |
| `network_lp` = False | mip | on by default | (debugging: same LP, original rows) |

## Next steps

1. **The dual's cost per pivot on node LPs**: fuse the column-wise PRICE
   with the ratio test's candidate pass (both stream the row), and
   density-predicted U solves (see `_ext/lp/README.md`).
2. **Cuts in the tree**: zero-half and c-MIR at nodes, a global pool with
   ageing.
3. **Conflict analysis** from infeasible nodes and propagation failures.
4. **Symmetry** (orbital fixing) for the bin-packing instances.
5. **Presolve aggregation** with a postsolve stack, probing, cliques.
6. **Big LPs** (physiciansched): scaling, a triangular crash, barrier.
7. **Reproducible budgets**: the remaining wall-clock budgets (dives,
   pump, cut loop) as work limits, as zero-half's and the schedule search's
   already are.
