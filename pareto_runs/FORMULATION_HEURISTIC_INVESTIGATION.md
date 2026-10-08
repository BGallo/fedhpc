# AF/B&P competitiveness investigation (2026-09-29)

## Root cause of the AF timing difference

`normal_patterns=True` filtered job-start variables but retained the full
time grid: one idle variable and one flow row per time unit. Kramer et al.'s
AF graph also compresses idle flow, using continuous loss arcs that jump from
a normal-pattern node to the next release/event time. Thus the earlier option
did not actually reproduce the paper's graph.

`SpaceTimeFormulation(compressed_flow=True)` now implements the compressed
graph, including running-job completion events and per-type releases. It is
off by default pending broader replicated benchmarks.

### Paired same-hardware pilot

Twelve `(n,m,alpha)` cases (`n=20,50`, `m=2,5`, `alpha=0.2,1,3`, pmax=100),
one thread, identical Gurobi settings:

| quantity (geometric ratio to paper AF) | full-grid `fed_st` | compressed `fed_af` |
|---|---:|---:|
| variables | 1.012 | 1.000 |
| constraints | 1.071 | 1.002 |
| solve time | 1.274 | 0.943 |
| Python build time | 1.617 | 2.195 |

The compressed model has the same LP bound and essentially exactly the same
size as AF. Its construction is slower because the generic FED-HPC builder
supports multiple types, running jobs, budgets and fixed costs and builds the
graph in Python. Solve time is indistinguishable from AF at this sample size.

### Compiled graph builder

Normal-pattern closure, job-specific start filtering, node construction and
loss-arc generation now run in the dedicated pybind11 module `_graph`, with an
exact pure-Python fallback. Median graph-only times (9 repetitions) were:

| case | horizon | Python | C++ | speedup |
|---|---:|---:|---:|---:|
| p100, n100, m3 | 1,997 | 28.1 ms | 6.1 ms | 4.6x |
| p1000, n50, m2 | 15,077 | 118.4 ms | 26.1 ms | 4.5x |
| p1000, n100, m2 | 30,432 | 512.7 ms | 136.0 ms | 3.8x |

On the first case, complete Gurobi model construction fell only from a 510 ms
median to 497 ms (2.7%). Graph generation is no longer the bottleneck; creating
roughly 180,000 named `gurobipy` variables and their expressions dominates.
The next construction optimization must therefore use bulk Gurobi APIs and/or
omit variable/constraint names in benchmark mode, rather than further tuning
the graph algorithm.

On the harder `p100_n100_m3_a0.2` case, the same run produced 596 s (AF),
147 s (full grid), and 243 s (compressed), all with the same LP bound and
optimum. Earlier stored runs were about 170 s for both AF and full grid. This
large reversal shows that one-shot wall times are dominated by B&B variability;
claims about formulation speed need several Gurobi seeds per instance.

## Where each flow formulation wins

The original 120-instance comparison contains 112 cases solved by both AF and
full-grid FED-HPC. Its strongest predictor is release-date spread `alpha`, not
horizon, processing-time scale, machine count, or the percentage of normal
patterns removed (Spearman correlation between `alpha` and log runtime ratio:
0.35; every other measured feature was below 0.24 in absolute value).

| regime | full-grid time / AF (geomean) | FED-HPC wins |
|---|---:|---:|
| alpha=0.2 | **0.54x** | 15/18 |
| alpha=0.6 | 1.17x | 6/15 |
| alpha=1.0 | 1.34x | 3/19 |
| alpha=1.5 | 1.33x | 4/20 |
| alpha=2.0 | 1.20x | 3/20 |
| alpha=3.0 | 1.30x | 1/20 |

Thus there is no globally best graph. Dense-release instances favor an
extended unit-idle network; spread-out releases favor compressed loss arcs.

### Factorial multi-seed diagnostics

Six variants isolate job-arc filtering, idle compression, assignment sense,
and declared idle integrality. Each result below is the median of Gurobi seeds
0/1/2, one thread.

**Dense releases, root-hard (`p100_n50_m2_a0.2`)**

| variant | time | nodes | simplex iterations |
|---|---:|---:|---:|
| paper AF | 63.3 s | 1 | 542,687 |
| compressed, covering | 54.7 s | 1 | 543,075 |
| compressed, equality | 50.1 s | 1 | 509,809 |
| full grid, covering/continuous | 47.6 s | 1 | 407,196 |
| full grid, equality/integer | **23.6 s** | 1 | 238,448 |
| normal job arcs + unit idle arcs | 25.8 s | 1 | **237,243** |

All solve at the root. Compression loses here because the jump-arc network is
a poorer simplex/cut representation of this dense root relaxation: it needs
over twice as many iterations. Normal-pattern job arcs plus unit idle arcs are
the best hybrid, showing that the useful ingredient is the unit-idle extended
network, not the extra unreachable job starts.

**Spread releases, root-easy (`p100_n50_m2_a1.0`)**

| variant | time | nodes | simplex iterations |
|---|---:|---:|---:|
| paper AF | **0.52 s** | 1 | **1,437** |
| compressed, equality | 0.59 s | 1 | 1,448 |
| compressed, covering | 0.63 s | 1 | 1,543 |
| normal job arcs + unit idle arcs | 0.85 s | 1 | 1,441 |
| full grid, equality/integer | 1.13 s | 1 | 3,172 |
| full grid, covering/continuous | 1.33 s | 1 | 4,355 |

When the root is easy, the smaller compressed graph wins; the extended unit
network adds overhead without reducing meaningful search work.

**Branch-heavy (`p100_n50_m5_a0.6`)**

| variant | time | nodes | simplex iterations |
|---|---:|---:|---:|
| compressed, covering | **3.91 s** | **1,228** | 77,089 |
| compressed, equality | 4.84 s | 1,430 | **72,341** |
| paper AF | 6.83 s | 1,736 | 96,896 |
| normal job arcs + unit idle arcs | 14.57 s | 5,357 | 279,313 |
| full grid, equality/integer | 15.18 s | 6,019 | 268,508 |
| full grid, covering/continuous | 19.58 s | 6,563 | 336,400 |

When branch-and-bound dominates, compressed loss arcs reduce both nodes and
LP work per tree, and covering rows are slightly preferable.

### Structural conclusions

* Paper AF and `compressed_flow + cover_ge` presolve to identical dimensions
  on the diagnostic cases. FED-HPC's one extra source loss arc/two rows vanish
  in presolve. Their residual timing differences are solver noise.
* Gurobi infers idle-flow integrality: after presolve, even models declaring
  loss/idle variables continuous report every remaining variable as integer.
  Declared integrality itself does not explain performance.
* Equality versus covering is secondary and regime-dependent. It helps the
  dense root-hard case; covering helps the branch-heavy case.
* Raw variable count is a poor predictor. The relevant quantities are root
  simplex/cut work for root-only cases and node count times per-node LP work
  for branch-heavy cases.

An adaptive policy is therefore defensible: use normal-pattern job arcs with
unit idle arcs for very concentrated releases (`alpha`-like spread near 0.2),
and compressed covering AF otherwise. A production selector should use
observable instance statistics (release span divided by average machine load)
rather than generator `alpha`, then be validated out of sample.

As a retrospective check, define
`rho = (max release - min release) / (sum processing / machine count)`.
Choosing full-grid when `rho < 0.4` and paper AF otherwise selected the faster
of the two on 93/112 solved pairs. Its geometric runtime was 9.5% below always
using AF and only 5.5% above an oracle that knew the faster formulation after
the run. This threshold is exploratory—it was evaluated on the same data used
to discover it—but it confirms release concentration is usable as an adaptive
feature rather than merely an artifact of the generator's `alpha` label.

## Metaheuristic root cause and WSPT result

The previous decoder sequenced mainly by SPT/SRPT even though the paper
benchmark minimizes *weighted* completion time. Job weights are now passed to
the C++ layer and `decode_order=2` tries a WSPT (Smith-rule) SGS schedule while
retaining the old schedule whenever WSPT is worse. Existing behavior remains
the default.

On the ten selected paper cases with a proven optimum:

| configuration | mean gap | median gap | max gap |
|---|---:|---:|---:|
| default, 32 x 60 | 9.56% | 10.43% | 21.40% |
| WSPT, 32 x 60 | 5.09% | 5.55% | 10.20% |
| long, 64 x 300 | 6.25% | 7.40% | 12.31% |
| WSPT long, 64 x 300 | 3.97% | 4.29% | 7.54% |

WSPT improved 7/10 cases and regressed 2/10. It roughly halves the default
gap, but is still not competitive with the paper's B&P quality (about 0.4%
on its reported hard set). The next heuristic step should be a truly dynamic
release-aware WSPT dispatch and machine/type assignment neighborhood, not a
larger GA budget.

## Recommended next experiments

1. Run AF/full-grid/compressed with 5--10 `Seed` values on the published
   benchmark and report shifted geometric means plus solved counts/gaps.
2. Optimize model construction with bulk `addMVar`/matrix constraints or a
   C++ graph builder; compression already removes the solver-side difference,
   leaving Python build overhead.
3. Add dynamic WSPT dispatch: whenever a machine becomes free, select the
   minimum `p_j/w_j` among released jobs; advance to the next release only if
   none are available. Use it both as a seed and a finishing decoder.
4. Add assignment moves that account for the displaced jobs' completion-time
   change. The present local search estimates only the moved job's `delta p`,
   which is weak for weighted completion objectives.
5. For B&P, prioritize the paper's full heuristic-pricing cascade and a
   set-covering/non-elementary primary pricer. The already-tested relaxed
   prescreen and SR-1 cuts do not improve this implementation because exact
   elementary pricing still dominates runtime.
