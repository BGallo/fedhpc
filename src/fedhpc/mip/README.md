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

**LP-first in the tree: reduced-cost fixing and restarts.** lpfirst proves
optimality by solving the LP, then keeping only the placements whose
reduced cost is below the incumbent-bound gap and solving the MIP over
those. Inside a general MIP that argument is reduced-cost fixing (Dantzig;
Nemhauser & Wolsey): with the root LP's value z, its reduced costs d and the
cutoff U (the incumbent, less one step when the objective is integral), a
column nonbasic at its lower bound l satisfies x_j <= l + (U - z) / d_j in
every improving solution (symmetrically at an upper bound). The root's
duals are kept; at every new incumbent the global domain is tightened and
propagated, and every node rebuilds its domain from it. When the fixings
reach `restart_min_fixed` (30 %) of the integer columns, the problem
restricted to that domain is presolved again (the fixed columns leave the
LP) and solved with the incumbent's value as cutoff: lpfirst's MIP over the
eligible placements. Nothing better there proves the incumbent optimal. On
Kramer's models the root fixes 72-95 % of the columns (TI_N50_M5_R0: 28,219
of 31,412).

**Presolve.** The rounds apply:
- removal of fixed, empty and singleton columns and rows;
- propagation, with redundant-row removal;
- dual fixing: a column no row locks in its cost-improving direction goes
  to that bound (Achterberg 2007, §10.8);
- Savelsbergh coefficient tightening on one-sided rows over binaries: for
  a′x ≤ b with a_j > 0 and maxact − a_j < b, set a_j −= d and b −= d with
  d = b − (maxact − a_j).

- parallel rows (hashed after normalisation, compared exactly);
- doubleton equations a_i x_i + a_j x_j = b, substituted in batches as one
  sparse product A T (the eliminated column continuous, or integer with
  integral ratios);
- implied-free continuous column singletons in equality rows, with their
  row;
- probing on binaries on the C++ propagator's trail, within a work budget
  (a value that empties the domain fixes the other; bounds both values
  imply hold);
- coefficient tightening also on complemented binaries (x - M y <= 0 with
  x <= U becomes x - U y <= 0).

Substitutions push affine records x_j = const + sum coef x onto a
postsolve stack, replayed in reverse; every other reduction fixes, tightens
or deletes. Each is a `presolve()` switch (`MipOptions.presolve_options`).
On Kramer's TI/AF models presolve finds nothing to do: they are evaluated
with the general MIPLIB set (`miplib_general`).

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

**c-MIR with variable bounds.** A continuous column with a variable bound
(x_j <= d y_k + e, from a two-entry row over a continuous and an integer
column: a fixed-charge link) is substituted through it when that bound is
the closest to the LP point; y_k then joins the integer part of the MIR,
which is what flow-cover cuts are (Marchand & Wolsey 2001). Base rows
include flow rows whose columns have variable bounds on fractional
integers; aggregation measures "inside" against variable bounds too.
Root bounds: exp-1-500-5-5 52,698 -> 64,694 (optimum 65,887), beasleyC3 346
-> 476 (754), tr12-30 -> 122,125 (130,596).

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

**Clique table** (`cliques.py`): cliques from rows (the conflicting prefix
of a sorted knapsack side; Atamtürk, Nemhauser & Savelsbergh 2000) and from
probing implications, separated greedily as clique cuts and handed to the
propagator as extra rows. **Conflict analysis** on propagation-infeasible
nodes: the deletion filter shrinks the path's binary decisions to a nogood,
batched into the propagator. **Feasibility Jump** (Luteberget & Sartor
2023, C++ `_ext/mip/fj.hpp`): LP-free weighted local search before and
after the root LP. A node LP that fails numerically is split on its widest
integer column with its parent's bound, instead of ending the solve.

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

v5 (`k50_mip_v5`, the faster dual pivot below): 14/28 at 28.1 s, N = 20
18/18 at 0.75 s; the unsolved Kramer instances end closer (AF_N50_M2_R0
0.13 %, TI_N50_M2_R0 0.095 %), mcsched at 212,920 (optimum 211,913).

**v6** (`k20_v6`, `k50_mip_v6`, `k100_v1`, `general_v2`: reduced-cost
fixing and restarts, deeper presolve, c-MIR with variable bounds, batched
crossings, the feasibility-check fix):

| Set | Limit | fedhpc.mip v6 | HiGHS | Gurobi |
|---|---|---|---|---|
| Kramer N = 20 (18) | 60 s | **18/18, 0.65 s** | 18/18, 1.40 s | 18/18, 0.48 s |
| Kramer N = 50 + MIPLIB sched/pack (28) | 120 s | 14/28, **23.8 s** | 15/28, 30.3 s | 22/28, 12.0 s |
| Kramer N = 100 (18, new) | 120 s | **10/18**, 47.8 s | 6/18, 65.8 s | 11/18, 26.1 s |
| MIPLIB general (24, new) | 60 s | 1/24 | 12/24 | 14/24 |

- Kramer N = 50: AF_N50_M5_R4 6 s (v5 23 s), TI_N50_M5_R0 14 s, TI_N50_M5_R4
  16 s; TI_N50_M2_R4 ends at 156,051, Gurobi's value at its limit.
- Kramer N = 100: **TI_N100_M2_R0 470,664 at 0.42 % against Gurobi's
  604,220 at 22 %** (HiGHS: no incumbent); TI_N100_M2_R9 and TI_N100_M5_R9
  level with Gurobi (58 s, 21 s); HiGHS solves none of the TI models.
- MIPLIB general: the engine's weakest family - fixed charge and network
  design need flow-path and network cuts, the knapsack families stronger
  MIR; presolve depth changes little there yet (A/B in `general_v1`).

**v9** (`k20_v9`, `k50_mip_v9`, `k100_v4`, `general_v5`: clique table,
Feasibility Jump, propagation conflicts, the Gomory support fix, the
adaptive cut loop, node-LP failure recovery):

| Set | Limit | fedhpc.mip v9 | v6 | HiGHS | Gurobi |
|---|---|---|---|---|---|
| Kramer N = 20 (18) | 60 s | 18/18, 0.71 s | 0.65 s | 18/18, 1.40 s | 18/18, 0.48 s |
| Kramer N = 50 + MIPLIB sched/pack (28) | 120 s | 14/28, 24.0 s | 23.8 s | 15/28, 30.3 s | 22/28, 12.0 s |
| Kramer N = 100 (18) | 120 s | 10/18, 53.5 s | 47.8 s | 6/18, 65.8 s | 11/18, 26.1 s |
| MIPLIB general (24) | 60 s | 2/24, mean gap 0.28 | 1/24, 0.29 | 12/24, 0.11 | 14/24, 0.09 |

On the general set the incumbents moved most: mas74 11,891 (HiGHS
12,015), mas76 at the optimum, gen-ip054 6,854 (HiGHS 6,856), neos-911970
57.2 (was 74 at the root), exp-1-500-5-5 and pg solved. The scheduling sets
keep v6's level: Feasibility Jump and the long cut loop are off when a
scheduling structure is detected (they cost AF_N20 1.3 s for nothing), and
the primal root no longer crashes from a Feasibility Jump point (65 s
against 0.6 s on TI_N50_M2_R9).

**A correctness fix.** `MipProblem.violations` computed a one-sided row's
infinite side as -inf / inf = NaN, and np.maximum then hid the other side's
violation: a violated <= row read as satisfied. On neos859080 (infeasible)
the feasibility pump's point was accepted and the solve reported "optimal
0". Earlier benchmark objectives all matched the reference solvers, so no
reported optimum was wrong; the check is now NaN-free and tested.

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

## Profiling

`scripts/profile.py` (perf from the system; py-spy and pybind11 are dev
dependencies):

- `build timers|perf|native`: standalone `_fedlp`/`_fedmip` builds
  (fedlp's phase timers; frame pointers and symbols; `-march=native`),
  selected per run with `--build` (via `FEDLP_DIR`).
- `mip FILE -t 60 --build timers`: per-phase wall time and pivots of the
  MIP (`stats["time"]`, `stats["pivots"]`; nested phases such as `cmir`
  inside `root_cuts` overlap) and fedlp's timers and per-pivot sizes
  summed over every LP solve (`stats["fedlp"]`).
- `lp FILE "" "dual=0,partial_min_cols=2000" --build timers`: the root LP
  under option sets.
- `perf FILE --build perf`: perf record with Python's perf trampolines; time
  by component (interpreter, fedlp, fedmip, NumPy/SciPy, libc, kernel) and
  the top symbols; `perf report -i results/profile/<label>/perf.data` for
  more (`perf annotate` for instruction-level hot spots).
- `pyspy FILE --build perf`: py-spy --native stacks, the native time
  attributed to its innermost Python frame.

## Python and C++: where the time goes, and what the split costs

The engine is Python orchestration over two C++ kernels: fedlp (the LP,
`_ext/lp`) and fedmip (propagation, `_ext/mip`). Presolve, cuts,
heuristics, branching and the tree are Python (NumPy/SciPy for the vector
work). `scripts/profile.py perf` measures the split with hardware sampling
(Python 3.12+ `-X perf` frames); self time by component:

| Instance (60 s, one thread) | LP size | fedlp | fedmip | Python interpreter | NumPy/SciPy | node cost |
|---|---|---|---|---|---|---|
| csched008 | 351 x 1.3 k | 91.0 % | - | 2.2 % | 0.2 % | 6.2 ms (LP 5.2) |
| AF_N50_M2_R4_1 | 2.5 k x 100 k | 87.1 % | - | 3.3 % | 2.8 % | ~0.9 s |
| TI_N50_M2_R0_1 (network form) | 1.6 k x 70 k | 73.6 % | 1.5 % | 9.4 % | 6.4 % | ~1.5 s |
| bppc4-08 | 111 x 1.5 k | 49.0 % | 6.1 % | 24.4 % | 5.7 % | 2.9 ms (LP 1.0) |

**What this says.**
- On the benchmark's big LPs (Kramer N = 50, the MIPLIB scheduling set) the
  split is not the bottleneck: 85-95 % of the time is in C++, most of it
  inside dual pivots. Moving Python code to C++ would buy at most the
  remaining 5-15 %.
- On small LPs it is: bppc4-08's nodes cost 2.9 ms, of which the LP is
  1.0 ms. The interpreter's quarter is per-node bookkeeping - the node's
  domain arrays, the heap of nodes, pseudocost NumPy ops on tiny arrays,
  rounding's Python loop over fractional columns, `lp.x()` copies and the
  per-column `set_bounds` calls. SCIP and HiGHS process such nodes in tens
  of microseconds of overhead; here each crossing costs a few microseconds
  and each NumPy call on a tiny array ~1-5 us, dozens of them per node.
- Python hurt most where a loop ran per element: GMI's per-entry
  substitution and the cut pool's dict dot products took 18-25 s per
  root cut round on TI/AF until vectorised (now < 1 s); the vectorised
  NumPy propagator re-derived every row each round (now C++, event
  driven). Both cases are the same lesson: Python is fine per *phase*,
  per *node* it is marginal, per *entry* it is fatal.
- py-spy (`profile.py pyspy`) attributes native time to the Python call
  sites (which heuristic, which separator); its totals include idle pool
  threads, so the split above is taken from perf.

**Costs of the boundary itself**, small but structural:
- every LP call converts arguments and returns copies (`x()`, bases,
  DSE weights: O(n) per call);
- the GIL is released in `solve()` only;
- state lives on both sides (the LP's bounds in fedlp and in
  `LpRelaxation`, the domain in the propagator's trail and in NumPy
  arrays), copied at every node.

**Options.**
1. *Keep the split, batch the crossings* (cheapest): vector calls
   (`set_bounds_many` is the first), the domain kept in the propagator and
   handed to fedlp in C++ without a Python round trip, rounding and
   fractionality in C++. Recovers most of the small-LP overhead.
2. *A C++ node engine, Python as plug-ins* (what SCIP's plug-in design and
   HiGHS's MIP solver look like from PySCIPOpt / highspy): node selection,
   domain via the trail, LP warm start, pseudocost branching and the
   node-level heuristics in C++; Python callbacks for root heuristics,
   structure detection (list scheduling), separators under development, at
   a controlled frequency. Node overhead drops to microseconds; the
   research loop (new heuristics, new separators in NumPy) stays in Python.
3. *All C++*: what the commercial and the open-source codes do; the
   fastest, and the costliest to change. Not warranted while the LP's
   pivots are 70-90 % of the time on the target models.

Recommendation: 1 now (it is mechanical), 2 when the small-LP families
(MIPLIB packing, scheduling with small LPs) become the target. The direct
Python integration is worth keeping as the plug-in surface either way.

## Against full-scale MILP solvers: the biggest missing features

HiGHS (open source; Huangfu & Hall's simplex, Achterberg-style branch and
cut), SCIP (open source; the reference academic framework, Achterberg 2007
and the SCIP Optimization Suite reports) and Gurobi (commercial; features as
its documentation and logs report). Ordered by how much each is worth on
general MILP, with what it would take here.

| # | Feature | HiGHS | SCIP | Gurobi | fedhpc.mip | Expected effect |
|---|---|---|---|---|---|---|
| 1 | **Presolve depth**: aggregation (doubleton/tripleton equations, free column substitution), dominated columns, parallel rows/columns, implied free columns, probing, clique extraction and merging, coefficient strengthening on all rows, sparsification; restarts | yes (most) | yes (all) | yes (all, plus more) | removals, FBBT, dual fixing, Savelsbergh on binary rows; no aggregation, no probing, no cliques, no restarts | the largest single factor on MIPLIB: presolve often removes 30-70 % of rows and columns and tightens the LP bound before the first pivot |
| 2 | **Cuts at scale and in the tree**: c-MIR over all aggregations, flow cover, lifted knapsack cover, clique, implied bound, multi-commodity flow, zero-half, GMI with a stable selection, mod-k, lift-and-project; separation at nodes; a cut pool with ageing; cut selection by efficacy, parallelism, integral support and objective parallelism | MIR, cover, clique, implied bound, GMI, path aggregation; local cuts | all of these and more | all, plus network/MCF, RLT, BQP | GMI, cover, zero-half, c-MIR (root only), efficacy and parallelism | Gurobi reports cuts as the second largest factor after presolve (often 2-10x on MIPLIB) |
| 3 | **Conflict analysis** (infeasible and bound-exceeding nodes, propagation conflicts, CDCL-style learned constraints) | yes (since 1.3) | yes (graph and dual-ray conflicts) | yes | none | prunes trees by 10-30 % on average, much more on scheduling/combinatorial models (csched) |
| 4 | **Symmetry handling**: orbital fixing, orbitopes, symretopes, symmetry-breaking inequalities, orbital branching; automorphism detection (bliss/nauty) | orbitopes and orbital fixing | full (Pfetsch & Rehn; Hojny) | yes | none | decisive on bin packing and identical-machine models (fhnw-binpack, bppc): orders of magnitude |
| 5 | **LP engine for MIP**: dual simplex with bound flipping tuned for re-solves, presolve and scaling inside the LP, crash bases, interior point with crossover for cold large roots, concurrent root solves, stabilised pricing, parallel PRICE | dual + IPX + PDLP | SoPlex / any LP | dual, primal, barrier, concurrent, PDLP | fedlp: primal/dual simplex, no scaling, no LP presolve, no barrier | physiciansched's root LP does not finish; per pivot fedlp's dual is ~3x Gurobi's on the Kramer masters |
| 6 | **Primal heuristics portfolio**: RINS, RENS, local branching, crossover, DINS, proximity search, feasibility pump with stage 3, ZI round, shifting, octane, alternating (MIP-start repair), CP-style probing dives, adaptive large neighbourhood search; scheduling by frequency and success | RINS, RENS, feasibility pump, ZI round, shifting, randomized rounding | ~50 heuristics with ALNS | many, plus solution improvement in parallel | rounding, fix-and-propagate with backjumping, two dives, pump, RENS, RINS, list scheduling | gfd, lectsched, csched, fhnw: no incumbent here |
| 7 | **Branching**: reliability branching with hybrid inference/conflict scores, pseudocost initialisation by strong branching on the LP's best candidates with domain propagation, lookahead, cloud branching, multi-aggregated branching; node selection with estimates (best estimate, BFS/DFS interleaving), restarts, root reoptimisation | reliability + more | reliability with conflict/inference scores, best estimate with plunging | proprietary, strong | reliability pseudocosts, best bound with plunging | 10-30 % of tree size typically |
| 8 | **Parallelism**: deterministic parallel tree search, concurrent heuristics, racing/concurrent root | parallel LP and some tree parallelism | UG (ParaSCIP/FiberSCIP) | deterministic parallel B&B | none (one thread by design here) | 2-8x on 8-16 cores for Gurobi |
| 9 | **Numerical safety**: scaling, exact/safe dual bounds (SCIP exact), tolerance management, feasibility repair of solutions (polishing), cut numerics (dynamism, scaling, relaxation) | yes | yes, plus exact mode | yes, NumericFocus | cut safeguards only; no scaling | robustness on MIPLIB's numerically hard instances |
| 10 | **Model features**: general constraints (SOS1/2, indicators, semicontinuous, quadratic/MIQP/MIQCP, nonlinear in SCIP), objective-cutoff, MIP starts, callbacks, warm starts, solution pools, multi-objective | most | most (MINLP) | all | linear MILP only, MPS without SC/SOS/QP; no callbacks | coverage, not speed |

**What closes the most gap per effort, for this code base.**
1. *Presolve aggregation and probing* (item 1): self-contained, in Python
   on sparse matrices with a postsolve stack; probing on binaries reuses
   the C++ propagator's trail directly (fix, propagate, undo).
2. *Conflict analysis* (item 3): the propagator already knows the failing
   row; with reasons recorded on its trail (which row implied each bound)
   it can derive a conflict clause as SCIP does, and the solver can add it
   as a cut or a propagation constraint.
3. *Cuts in the tree and a pool with ageing* (item 2): the separators exist;
   the solver needs local cuts (valid in a subtree), a pool, and row
   management in fedlp (already incremental).
4. *Symmetry* (item 4): orbitopes for identical machines/bins need the
   permutation structure; detection via bliss/nauty would be a new
   dependency (could be installed as `pynauty` or by building bliss).
5. *LP robustness* (item 5): scaling and an LP presolve inside fedlp;
   barrier is a large project (HiGHS's IPX could be called through highspy
   for cold roots instead).
6. *Heuristic portfolio scheduled by success* (item 6), *branching scores
   with conflicts and inference* (item 7).

## Architecture against HiGHS, SCIP and Gurobi, and the plan

**Measured at the root** (`miplib_general`, one node each; % of the gap
between the LP bound and the optimum closed; HiGHS 1.16 `mip_max_nodes=1`,
Gurobi 13 `NodeLimit=1`, which includes its root heuristics and restarts):

| Instance | fedhpc.mip before | after this round | HiGHS | Gurobi |
|---|---|---|---|---|
| mas76 / mas74 | 0 / 0 | 8.3 / 7.2 | 15.2 / 10.1 | 15.5 / 9.0 |
| assign1-5-8 | 0 | 9.4 | 37.1 | 30.2 |
| neos5 | 4.2 | **21.5** | 15.9 | 17.3 |
| neos-911970 | 0 | 79 (23.3 -> 48.2) | 94.9 | 0 |
| beasleyC3 | 45 | 57 | 95.2 | 98.1 |
| p200x1188c / sp150x300d | 18 / 56 | 18 / 60 | 100 / 100 | 100 / 100 |
| exp-1-500-5-5 / tr12-30 | 96.8 / 92.5 | 96.8 / 92.5 | 100 / 99.7 | 80.2 / 99.1 |

The two biggest root fixes were mechanical: Gomory cuts were rejected on
every small problem by a support cap meant for the 40 k-column arc-flow
LPs (now: half the columns, but never below 1,000), and the cut loop
stopped after 3 rounds without bound progress, where on degenerate LPs
violated cuts move the vertex for several rounds before the bound moves
(now 12 rounds when the root LP is cheap; neos-911970's root went from
23.3 to 48.2).

**Components, side by side** (HiGHS from its MIP solver sources and
Bernhard et al.'s design; SCIP from Achterberg 2007 and the SCIP
Optimization Suite reports; Gurobi from its documentation and logs):

| | fedhpc.mip | HiGHS | SCIP | Gurobi |
|---|---|---|---|---|
| Language / structure | Python orchestration, C++ LP and propagation | C++ | C, plug-ins | C, closed |
| Presolve | removals, FBBT, dual fixing, coefficient tightening, doubletons, free singletons, parallel rows, probing | + dominated columns, sparsification, clique merging, implied integers | + symmetry, many more | the most extensive |
| Clique table | rows + probing, cuts, propagation | yes (central: propagation, cuts, conflicts) | yes | yes |
| Cuts | GMI, cover, zero-half, c-MIR (VUB, aggregation), clique; root only | tableau (GMI), path MIR, mod-k, clique, implied bound; root + cut pool in tree | ~20 separators, in tree | ~20, network/flow path, in tree |
| Conflicts | propagation nogoods (deletion filter) | conflict pool from propagation and LP | graph + dual-proof conflicts | yes |
| Heuristics | rounding, FJ, fix-and-propagate (backjumping), dives, pump, RENS, RINS, list scheduling | FJ, randomized/central/ZI rounding, shifting, pump, RENS, RINS | ~50, ALNS | many, parallel |
| Branching / search | reliability pseudocosts, best bound with plunging, RC fixing, restart | reliability + inference, hybrid estimate, restarts | hybrid scores, best estimate | proprietary |
| Symmetry | none | orbitopes, orbital fixing | full | yes |
| LP | own dual/primal simplex, warm DSE, no scaling | own dual simplex, IPX for cold LPs | SoPlex | primal/dual/barrier/concurrent |

**Biggest levers for general MIP (to match HiGHS), in order:**
1. *Flow and path cuts on fixed-charge/network models* (p200x1188c,
   sp150x300d, mc11, beasleyC3: HiGHS's path separator and Gurobi's flow
   cover/flow path/network cuts close 95-100 % at the root, we 18-60 %).
   Our c-MIR substitutes variable bounds but the links' big-M (x <= 3950 y
   in p200x1188c) is uniform; path aggregation across several nodes with
   the bound substitution chosen per path is what is missing.
2. *Search speed on small LPs*: HiGHS processes nodes in tens of
   microseconds of overhead, we in ~0.5 ms of Python bookkeeping plus the
   LP (the Python/C++ section above); the C++ node engine is the fix.
3. *LP-infeasibility conflicts* (dual proofs from fedlp's Farkas row),
   which is where csched-like trees fail; propagation nogoods alone found
   none on csched008.
4. *Cuts in the tree* with a pool and ageing.
5. *Symmetry* for the bin-packing instances.
6. *Huge LPs* (gfd, physiciansched): scaling, crash, IPX for cold roots.

**Arc-flow and time-indexed scheduling against Gurobi.** Where we stand:
level or ahead on TI with many machines or late release dates, ahead on
TI_N100_M2_R0, behind on M = 2 with early releases (Gurobi proves
AF_N50_M2_R0 at its root node in 52 s; we end at 0.13-0.14 %). What is
missing there:
- *Node LP speed*: AF child LPs need 100-11 k dual pivots at ~0.45 ms (3x
  Gurobi's per pivot); restarts after reduced-cost fixing shrink them by
  72-95 %, which closed most of the N = 50 M = 5 gap.
- *Root bound*: Gurobi's root adds +16 on AF_N50_M5_R4 against our +4
  (zero-half, Gomory, MIR, clique); the gap left is mostly the zero-half
  separator's reach (it caps rows and columns) and cuts in the tree.
- *Branching on time*: SOS/GUB branching on a job's start time (split its
  placements at the LP's mean start: Sousa & Wolsey 1992; van den Akker,
  Hurkens & Savelsbergh 2000) balances TI/AF trees far better than
  variable branching; not implemented.
- *Column generation for N >= 100*: lpfirst's pricing avoids building all
  placements; the MIP builds them all (TI_N100_M2: 32 MB of MPS).

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

1. **Network and path cuts** for fixed-charge/network design (p200x1188c,
   sp150x300d, mc11, beasleyC3: Gurobi's root closes them with flow cover,
   flow path and network cuts; our c-MIR with variable bounds gets part).
2. **The dual on huge LPs** (gfd, physiciansched: 100-250 k rows, the root
   LP does not finish): scaling, a crash, an LP presolve, or IPX through
   highspy for cold roots.
3. **Node-local reduced-cost fixing and in-tree restarts**: the root's
   fixing is global only.
4. **Conflict analysis** from the propagator's failing rows.
5. **The C++ node engine** when small-LP families become the target (the
   Python/C++ section above).
6. **Kramer N = 100 M = 2 AF**: no incumbent on AF_N100_M2_R0 (the root LP
   and the restart's sub-solve take the time); the flow list scheduler runs
   only after the root LP.
