# Research plan: advisor review (2026-11-24)

A review of the repository written from a doctoral advisor's point of view
(mathematical optimization): what is publishable, the most promising avenues,
the literature to compare against, and what is unlikely to pay off.
Literature references are from memory. Check them, especially for work from
2025-2026.

## 1. Where the project stands

| Asset | Assessment |
|---|---|
| Exact solving at trace scale: dominance reductions (40-60x), LP-first on HiGHS proving 3340-job instances in 6-58 s | Strong, and unusual |
| Root LP gap ~0 on the real traces, and it can be measured | The most interesting scientific fact in the repo |
| Rm\|\|TWCT+TMUC (Chen et al., EJOR 2024): all 1540 known optima matched, 42 new optima, all 120 of the largest instances proven, an error found in the published sheet | Strong, self-contained |
| z-as-flow-source link: root gap 5-18% -> <=0.22% with activation costs | Promising, not in the code yet |
| SpaceTime == Kramer arc-flow (same LP, about the same speed) | Validation only, not novelty |
| EAs: 86-100% of points dominated, 0 exact hits, a "wall" that more generations don't break | A negative result |
| Lean proof of the compressed-flow removal lemma (the naive version was false) | A nice differentiator, but niche |

**Main risk to the CCGrid draft:** the abstract sells the GA ("4.36% gap in
under 6 s" against an MILP at 91 s). Our own later measurements show the
reduced MILP / LP-first is faster *and* exact. Recast the GA as an
online-latency component or move it to a baseline. "A strong arc-flow
formulation" is not novel by itself (Valerio de Carvalho; Kramer et al.). The
novelty is the federated model, the exact reductions, and exactness at scale.

## 2. Avenues, ranked

### Tier 1: a thesis core

**A. Why the LP is near-integral: turn the observation into a theorem.**

In the time-indexed form each placement column covers consecutive slots, so
the capacity rows of one type form an interval matrix, which is totally
unimodular. Only the assignment rows sum x_j = 1 couple types and times.
Crucially, the cloud is an unbounded overflow resource.

That suggests a rounding argument in the style of Lenstra-Shmoys-Tardos and
Shmoys-Tardos:
- take a basic optimal LP solution;
- the number of fractional jobs is at most the number of tight capacity rows,
  by the usual rank argument;
- send every fractional job to its best cloud option (removing load never
  breaks time-indexed capacity);
- the integrality gap is then at most the sum, over fractional jobs, of each
  job's cloud regret.

That gives an a-posteriori bound, and likely an asymptotic one: gap/OPT -> 0
as N_m grows relative to the number of fractional jobs. It also explains the
contrasts: Rm (capacity 1, no overflow) has real gaps, and the trace
instances (N_m in the hundreds, cloud everywhere) don't.

- Cheap first test: count the fractional jobs in the LP optimum on fedhpc,
  pos and the A/B classes as N_m goes 100 -> 500.
- Condition to check: every job needs a cloud-feasible type (GPU and
  multi-node jobs may break this).
- Natural stress test: multi-node jobs (q_j > 1) turn the rows into
  knapsack-like rows. Does near-integrality survive?
- Literature: Lenstra, Shmoys & Tardos (1990) and Shmoys & Tardos (1993) for
  the generalized assignment problem; unsplittable flow on paths under the
  no-bottleneck / large-capacity assumption (Chekuri, Mydlarz & Shepherd;
  Raghavan & Thompson rounding); time-indexed LP strength (Sousa & Wolsey
  1992; van den Akker, Hurkens & Savelsbergh 2000); Pessoa et al. (2010)
  arc-time-indexed; the interval scheduling survey (Kolen et al. 2007).

**B. An exact bi-objective algorithm that exploits near-integrality.**

Today the search needs about 2 MIP solves per Pareto point. Instead:
- use the parametric bi-objective LP as a lower bound set (it gives the
  supported points almost for free);
- close each box with LP-first's reduced-cost certificate instead of a full
  MIP;
- warm-start duals and columns across boxes.

This is a methods paper (INFORMS JoC / C&OR), with A as its theory section.
- Compare against: the balanced box method (Boland, Charkhgard & Savelsbergh
  2015), Kirlik & Sayin (2014), bound sets (Ehrgott & Gandibleux 2007),
  bi-objective branch-and-bound (Parragh & Tricoire 2019), bi-objective
  branch-and-cut (Gadegaard, Nielsen & Ehrgott 2019), and the surveys by
  Przybylski & Gandibleux (2017) and Halffmann et al. (2022).
- Report: IP solves per point, time per point, and front size at 3340 jobs.
- Also address front density. With f1 on an integer grid the front can have
  thousands of points. Discuss representation quality (Sayin 2000) and what a
  decision-maker actually uses.

**C. Close the Rm||TWCT+TMUC benchmark: a quick win, low risk.**

The pipeline is ILS-SP, then a Lagrangian bound, then column enumeration, then
a set-partitioning MIP. Combined with the z-flow-source formulation, that is a
short journal paper (C&OR, J. Scheduling, or an EJOR short communication).
- To do: search the 2024-2026 papers that cite Chen et al. in case someone
  closed it first; normalise hardware (Chen et al. used a Celeron with a 1 h
  limit); run multiple seeds; put the z link into `formulations.py`.
- Related work: Kowalczyk & Leus (2018, ZDD branch-and-price);
  machine-activation approximation (Khuller, Li & Saha 2010).

### Tier 2: what reviewers of the applied paper will ask for

**D. Online / rolling horizon with uncertain runtimes.** Every instance is
"known_runtime_offline", and real walltime estimates are poor (Tsafrir, Etsion
& Feitelson 2007). Our exact offline optima are a rare asset: almost nobody
can compute the "value of clairvoyance" at trace scale. Plan: rolling-horizon
re-optimisation with a warm-started LP-first, against offline OPT and against
the cloud-bursting baselines already in the bibliography ("Waiting Game",
Starburst, Marshall et al.). For theory hooks, see stochastic online
scheduling (Megow, Uetz & Vredeveld 2006). This is what makes a systems venue
(TPDS, FGCS, CCGrid) take the work seriously.

**E. A public benchmark library.** The A/B generator plus certified fronts
plus a Gurobi-free path (HiGHS) is a cheap but citable contribution. It also
makes B reproducible.

### Tier 3: lower priority or less likely to yield results

- More EA work (NSGA-II/III, MOEA/D variants, decoders, tie-breaks). The wall
  is established on every instance class, and the weighted heuristics are
  slower and worse than the reduced MILP. Write it up once as a cautionary
  benchmark and stop.
- Cuts or more LP strengthening on the large-capacity instances. There is no
  gap to close. Cuts only matter for capacity-1 (Rm).
- BPC engineering for the trace instances. LP-first dominates there. Keep BPC
  for capacity-1 and fixed-cost cases.
- P|r_j|sum wC arc-flow comparisons. They're equivalent to Kramer's model.
  Keep them as one validation table.
- More Lean formalisation. The current lemma is a good story. Formalising
  R1-R3 costs months for little return. For certified optimality, VIPR
  certificates (Cheung, Gleixner & Steffy 2017; exact SCIP) are the more
  mainstream route.
- Geo-distributed transfer. Only worth it if shared links couple the types
  through bandwidth rows, which tests whether A survives. Otherwise it is
  modelling without new mathematics.
- Spot instances, lex-(f2,f1) reductions, weighted-sum vs epsilon-constraint.
  Already removed, ruled out, or textbook.

## 3. Thesis shape

1. Applied paper (CCGrid -> journal extension): the model, exact reductions,
   exactness at scale, and simulation against cloud bursting, with D added
   for the journal version. Reposition the GA.
2. Methods paper: A + B, near-integrality theory plus the LP-first
   bi-objective algorithm, with E as its benchmark.
3. Short paper: C.

## 4. Next 4-6 weeks

1. Commit the open work (lpfirst, lp_backend, pareto, the reference-points
   code).
2. Run the fractional-job count experiment for A. If the counts are small,
   write the rounding lemma.
3. Literature check for C, then a multi-seed rerun with hardware
   normalisation, then a draft.
4. Fix the GA framing in the CCGrid draft.
5. Prototype the LP bound set for B on fedhpc_known and compare solves per
   point against the current box search.

## 5. Progress on B (2026-11-24)

All runs on HiGHS (`LPFirstOptions(backend="highs", lns_sweeps=0)`): the
Gurobi licence has expired.

### Where the time of an exact front goes

Serial box search (`true_pareto_frontier(engine="lpfirst")`) on the 964-job
trace, first 40 phase solves (20 Pareto points, 265 s, ~13 s per point):

| Component | Time |
|---|---|
| Column generation (LP + pricing) | 11 s |
| Primal heuristics | 31 s |
| Restricted-master MIP | 16 s |
| **Eligible-placement MIP (the proof)** | **199 s (75%)** |

The high-cost side of the front is cheap (1-2 s per phase, 2-6k eligible
placements). The cost-minimal side (f2 -> 0) is expensive: `min f2 s.t.
f1 <= k` has ~70k eligible placements and takes 5-60 s per proof.

### Why the cost-minimal side is hard

There the LP is *not* near-integral in cost steps:

| f1 bound | LP | optimum | gap (delta steps) |
|---|---|---|---|
| 16575 | 0.0104 | 0.0468 | 364 |
| 16558 | 0.1040 | 0.1450 | 410 |
| 16548 | 0.2392 | 0.4559 | 2167 |
| 16539 | 0.5218 | 0.7262 | 2044 |

The alpha-point heuristic already finds the optimum. The time is all proof.

Evidence for avenue A, from the same LPs: only **4-12 of 964 jobs are
fractional** (57-65 tight capacity rows), almost all split between two
*on-prem* start times. At most one job is partly in the cloud. A weighted sum
with no bound row has 0-6 fractional jobs.

Ruled out:
- **Integer-knapsack Lagrangian** (dualize capacity, keep assignment plus the
  f1 row integral, solve the multiple-choice knapsack by DP over the f1
  budget). It gives *exactly* the LP bound on all five boxes tested. With ~9
  Pareto options per job and unit regret steps, the knapsack has no
  integrality gap. The gap comes from the capacity rows interacting with a
  tight f1 budget, not from the knapsack row.
- **Placement branch-and-price** (`_Tree`, forced): 1577 nodes in 300 s, with
  the bound stuck at the LP value.
- **Two-phase / bound-set shortcuts.** The LP is about the convex hull of the
  front. A triangle's LP bound is the supporting line, so every nonsupported
  point still needs its own reduced-cost enumeration. A bound set cannot skip
  those proofs. It only confirms the supported points, which are already
  cheap.

So the paper-B story is: near-integrality (A) makes supported points almost
free and pushes the whole cost of exactness onto nonsupported points, which
sit on the cost-minimal side. The open research question is a bound that sees
the capacity x f1-budget interaction, for example lifted cover inequalities on
the joint capacity/f1 rows, or a Lagrangian keeping one type's capacity
integral.

### Parallel exact search (implemented)

`pareto.parallel_pareto_frontier`: workers in a process pool, each with its
own engine, built once. Boxes are split at the f1 midpoint by an
epsilon-solve while fewer boxes are open than workers (as in the balanced box
method), and grown from alternating ends otherwise. Exact: same sub-solves
and certificates as the serial search. Test: `tests/test_parallel_pareto.py`
(identical fronts on 4 instances). Benchmark script:
`scripts/parallel_front_bench.py`.

| Instance | Parallel (14 workers, 1 thread each) | Serial (lpfirst-HiGHS) |
|---|---|---|
| 100-job trace slice | complete front, 548 points, 163 s | 505 points in 600 s, all in the parallel front |
| 964-job trace (`fedhpc_known`), 7200 s budget, 300 s per-solve cap | **partial**: 3686 proven points in 7224 s (~2 s/point); holes at f1 15235-16521 and 14900-15162 (cost-minimal side); 3 workers overran the 300 s cap by >1h50 and blocked the run | ~13 s/point (20 points in 265 s) |

Literature for the parallel part: Pettersson & Ozlen (2019, INFORMS JoC,
parallel criterion-space search); Boland, Charkhgard & Savelsbergh (2015,
balanced box).

The 964-job run above is the box search's last word: three of its workers
overran the 300 s cap by more than 1h50 and blocked it (section 6 has the fix).
Points: `results/parallel_front/fedhpc_known_runtime_offline_10min_box_w14.points.jsonl`.

## 6. Making the exact search faster (2026-11-24)

### What the hard proofs are

The proofs that stall sit at the cost-minimal end. They are `M(b) = min f2
s.t. f1 <= b` with b close to the f1 of the f2 = 0 schedule. I captured the
eligible MIP for b = 16558 (scratch `bb/capture.py`). It has 64,344 columns,
and only **3** of them carry cost: three cloud placements costing
468 + 468 + 514 (scaled), which add up to the incumbent's 1450. LP bound:
1040.5. So the "min-cost" MIP is a combinatorial feasibility question: can
the f1 budget be met with any two of those three jobs? HiGHS answers it by
blind branching (259 nodes, 90 s). In the middle of the front the same
direction is easy (b = 16000 and b = 15500: 3-4 s, root gap ~0).

The asymmetry runs both ways. Times for the two ε-constraint directions at
adjacent front points, one process (`bb/dirs.py`):

| f1 | Q = min f1 s.t. f2 <= c | M = min f2 s.t. f1 <= b |
|---|---|---|
| 11300-12800 | 5.6-7.9 s | 2.2-4.3 s |
| 13300 | 64 s | 9 s |
| 13800-14700 | 11-26 s | 11-32 s |
| 16540 | 22 s | 165 s |
| 16558, 16548, 16539 | 8-9 s | 50-93 s |

Caveat: in that script each M ran right after the Q at the same point, so
it had a near-optimal incumbent. Part of M's edge on the high-cost side is
a warm start.

### Single-phase lazy ε-constraint search (implemented)

The box search pays one solve of each kind per point (lexmin = two phases),
so every point pays for the expensive kind. `pareto.parallel_epsilon_frontier`
uses one *single-phase* solve per step instead:
- `Q(c) = a` proves only that some Pareto point has f1 = a and f2 <= the
  returned schedule's cost (a "Q-candidate"); `M` is the mirror image;
- a candidate is confirmed lazily, by the next growth from it
  (`Q(f2 - δ) > f1`), or replaced by that growth's cheaper schedule;
- boxes carry two clearing levels (no schedule with f2 <= lo has f1 < N.f1;
  none with f1 <= hi has f2 < P.f2), so splits and growths of either kind
  can be mixed in one box;
- `direction="adaptive"` picks Q or M per box from the median time of the
  nearest solves of each kind. For each candidate it also decides whether
  to grow lazily (cost t/(1-p), with p the observed replacement rate) or to
  settle it first with one solve of the other kind (t_settle + t), seeded
  with the candidate's own schedule;
- every solve is seeded with both box ends: one is feasible, the other is
  next to the answer.

Exactness tests (`tests/test_parallel_pareto.py`, 24 tests): adaptive, Q-only,
M-only and random-direction searches all equal the serial front on trap_small
and on 9 random instances with nonsupported points. Mixing kinds at random
stresses the candidate bookkeeping.

100-job trace slice, 548 points, 14 workers:

| Method | Wall | Solves |
|---|---|---|
| box search (`parallel_pareto_frontier`) | 163 s | - |
| ε, Q only (lazy) | 83 s | 1399 (774 replacements) |
| ε, Q + capped 5 s M phase | 117 s | 639 |
| ε, M only | 140 s | 672 |
| ε, adaptive (no settle policy) | 74 s | 1060 |
| ε, adaptive + settle policy + two seeds | 92 s | 1080 |

All produce the identical front.

964-job trace (`fedhpc_known`), 14 workers, 300 s per-solve cap. Proven
points by time (in brackets: those with f1 >= 14000, the cost-minimal side
where the box search stalls):

| Method | 10 min | 20 min | 30 min | 60 min |
|---|---|---|---|---|
| box search | 496 (12) | 947 (24) | 1319 (30) | 2342 (54) |
| ε, Q only | 279 at 10.4 min (aborted) | | | |
| ε, Q + capped 5 s M phase | 188 at 9.5 min (aborted) | | | |
| ε adaptive v1 (no settle policy, one seed) | 330 (61) | 684 (150) | 995 (226) | 2244 (490) |
| ε adaptive v2 (settle policy, two seeds) | 631 (18) | 1122 (36) | 1605 (55) | (30 min run) |

- v2 is the fastest in total: +22% over the box search at 30 min,
  2.09 solves per point, 7.4 s per solve (v1: 2.2 solves per point, 9.9 s
  each, 54% of solves wasted on replacements; v2: 6%).
- v1 is the one that gets into the hard side: 9x the box search's points
  at f1 >= 14000 after an hour. v2 spends its time on the dense, cheap
  high-cost side. Box selection (widest f1 range first) and the settle
  policy interact; neither run completed the front.
- Cross-check: the three runs together give 4050 distinct points, and no
  point of one run dominates a point of another, although they use
  different proof paths.

### Hard time limit (implemented)

The HiGHS-backed solves do not always honour `time_limit`. One M solve in
each ε run went past twice the cap. `pareto._SolvePool` kills the worker
processes of a solve that overruns `hard_time_limit` (default 2x the cap +
60 s), rebuilds the pool, resubmits the other in-flight solves and reports
the box unresolved. In both runs it fired once and the search carried on.

### Next steps for branch and bound, ranked

1. **Finish one complete exact front at 964 jobs.** Run ε-adaptive v2 to
   completion and rerun the dropped boxes with a larger cap. This is the
   paper-B headline number.
2. **Few-cost-column structure in M.** When the eligible set has k
   cost-carrying columns, M is a search over at most 2^k cost levels, each
   one a feasibility check of the easy kind (an f1 budget). Branching on the
   cost level (sorted subset sums), or the Q direction, which the adaptive
   search already uses, avoids HiGHS's blind branching. Worth a lemma: why
   the reduced-cost filter leaves so few cloud columns at the cost-minimal
   end.
3. **Worker affinity.** Send a box's next solve to the worker that did its
   previous one (its column pool and schedules are the relevant ones). That
   needs a custom process pool instead of `ProcessPoolExecutor`.
4. **Tie-break Q's schedule on f2.** Every replacement is a Q schedule with
   spare cost at its optimal f1. Preferring the cheapest incumbent among
   equal-f1 ones costs nothing and lowers p.
5. **Other MIP solvers on the captured models.** The MPS files of the hard
   eligible MIPs are in the scratchpad (`bb/mip_b*.mps`). SCIP or CP-SAT would
   show whether the difficulty is HiGHS-specific. Not run: installing
   packages was not permitted in this session. Gurobi's licence has expired.
6. Box ordering policy (which open box gets a worker) decides whether the
   hard side gets attention early. v1 vs v2 shows a 9x swing.
