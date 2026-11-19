"""Branch-price-and-cut (BPC) exact solver for FED-HPC — Phases 1-5 complete.

Dantzig-Wolfe reformulation of the space-time network formulation
(:class:`fedhpc.formulations.SpaceTimeFormulation`, matching
``MOFE__Multi_Objective_Scheduling_on_Federated_Enviroments.pdf`` §3): one
set-partitioning master over "instance-lifetime" columns (source-to-sink paths
in each type's space-time network ``G_m``) plus one shortest-path pricing
subproblem per instance type.

Running jobs, unbounded types and reduced start sets (``_Prep``)
-----------------------------------------------------------------
Every entry point preprocesses the instance once (:func:`_prepare`; a Pareto
search shares one result across all its box solves):

- **Running jobs as release groups.** A capacitated type's instances are
  grouped by the slot they become free: 0 for the ``N_m - n_running[m]``
  idle ones, ``min(e, H)`` for one held by a running job ending at ``e``.
  Each group ``(m, r)`` has its own fleet row, and its columns are paths
  from node ``r`` — exactly the compact model's ``y_{m,-1}`` boundary plus
  ghost completions. (This replaced one pinned pseudo-job per running job,
  which cost the pricing DP one mask bit each — 1525 on pos_congestion.)
- **Unbounded types get no pricing.** On a type whose capacity never binds
  (``formulations.unbounded_types``, e.g. cloud), any column splits into
  single-job columns of the same total cost and the fleet is never short,
  so such a type has no fleet row and all its single-job columns are
  enumerated upfront (``static_columns``).
- **Exact start-set reductions** (nonnegative data only): normal-pattern
  starts (Kramer et al. 2020), R1 (a job on an unbounded type starts at its
  release) and R2 (dominated unbounded options dropped), from
  ``formulations``. None of them raises f1 or f2 for any schedule, so they
  stay valid under both bound rows and under every branching decision. The
  objective-dependent rules (R3, lex deadlines) are not valid under an f2
  cap and are not used. ``reduce=False`` turns the reductions off.
- **Seeds and incumbents.** :func:`heuristic_schedules` (list-scheduling
  baselines, optionally an NSGA-II front) seed the root's columns, and the
  best one inside the bound rows is the starting incumbent. A lex-min's
  phase 2 starts from phase 1's schedule and columns.

Search tree (``_branch_and_price``)
-----------------------------------
- **Branching on placements** (``branching="time"``, :func:`_time_branch`):
  a job split across types goes on/off its most fractional type; otherwise
  its start window is split at the point holding half its LP mass
  (Kramer et al. 2020). Each decision is a set of forbidden ``(job, type,
  start)`` triples, so pricing just loses arcs — Ryan-Foster same-pairs
  (``branching="rf"``) instead cost the DP a parity bit each, which on
  rm_M8N40_1 kept the old tree from finishing in 120 s (now 8 s).
- **Reduced-cost fixing** (:func:`_rc_fixing`, ``PricingOptions.fixing``):
  after a node's column generation, every placement that no solution below
  the cutoff can use is forbidden in its subtree.
- **Best-first** node selection (``node_selection``), with the open bound
  reported on timeout (``BPCTimeLimit.bound``); nodes an incumbent found
  later dominates are dropped without a solve.
- The RMP integer heuristic runs at the root and every 10th node
  (``heuristic_every``), not at every node.

Together: trap_medium's 27-point Pareto front in 1.4 s (76 s before), all
base-tier solves in 9 s (100 s), and rm_M8N40_1 proven optimal at the
paper's best known 155509. Branching and fixing compound — each alone gave
about 10 s on the Pareto front.

Compiled pricing, ng-routes and the Lagrangian bound
----------------------------------------------------
Column generation prices with the C++ label-setting DP ``_bpcprice``
(:class:`PricingOptions`, default ``engine="cpp"``; the Python
:func:`_price_type` remains the reference and runs whenever subset-row cuts
are active):

- **One active set, no wait copies.** A label created at slot ``t`` is
  available from ``t`` on, so dominance (cost ``<=``, job memory a subset,
  same Ryan-Foster parity) is checked against everything available, and
  there is no bucket cap: a call is exact unless it exhausts
  ``max_labels``, and says so.
- **Completion bound.** ``g(t)``, the cheapest completion from slot ``t``
  with repeats allowed (one backward pass), retires every label that can no
  longer reach a negative reduced cost. The call returns the type's
  minimum reduced cost as a valid lower bound either way.
- **Many columns per call** (``cols_per_type``, the most negative distinct
  labels), instead of one per type and iteration.
- **ng-routes** (Baldacci, Mingozzi & Roberti 2011). A label remembers
  only the jobs in its last job's neighbourhood (``ng_size`` nearest by
  start window and length), so a column may repeat a job outside it. Such a
  column gets the visit count as its coefficient in the coverage and
  fixed-cost rows, so it can never be part of an integer solution, and the
  LP lies between the compact LP and the elementary master's. When the
  converged LP still uses one, the neighbourhoods on its cycle grow (up to
  ``ng_max``), every repeating column leaves the RMP and generation resumes
  (dynamic ng, Roberti & Mingozzi 2014) — usually ending at the
  elementary bound: on rm_M5N25_1 the root takes 0.5 s instead of the
  elementary DP's 116 s, at the same bound. ``ng_size=None`` prices
  elementary paths.
- **Lagrangian bound** (:func:`_lagrangian_bound`): every exact pricing
  round gives a bound valid for any sign-feasible duals. A node stops as
  soon as it reaches the cutoff (``"pruned"``) or the RMP value (the RMP
  is then optimal, though its duals may not price everything out —
  degenerate masters have many optimal duals). ``smoothing > 0`` prices at
  Wentges-smoothed duals around the best-bound duals first, falling back
  to the RMP duals on a mispricing.
- **Activation rows.** For each fixed-cost path type the master adds
  ``sum_{p non-null on m} lambda_p <= N_m z_m`` (``_build_rmp``'s
  ``activation``), priced as a once-per-column constant. With ``N_m = 1``
  (Rm machines) this is Chen et al.'s "``U_k`` in the column cost": on the
  Rm instances it cuts the f1 + f2 root gap from about 13% to 0-2.7%. The
  per-job rows alone let a column pay ``U_m`` only in proportion to its
  largest single job share.
- **Exactness.** The Python DP's per-node ``bucket_cap`` (200) drops labels
  and can miss the best column (found by ``tests/test_bpc_pricing.py``:
  the compiled pricer's columns are valid and up to 9% cheaper there), so
  the Python engine may stop column generation early. It remains the
  engine under subset-row cuts only.
- **Objective grid.** Pruning uses the step of ``w1·f1 + w2·f2``
  (:func:`_objective_step`) for any weights on a common grid, not only
  ``w2 = 0``.

Limited-memory subset-row cuts (``PricingOptions.sr_cuts``, off by default)
--------------------------------------------------------------------------
The compiled engine can separate subset-row cuts on job triples with
limited memory (:class:`_LmCut`, :func:`_separate_lm_src`; Jepsen et al.
2008, Pecin et al. 2017). This revisits the old Python cuts below, which
backfired because the DP needed one exact state bit per cut:

- The pricer carries one state bit per cut. A visit outside the cut's
  memory resets the bit, so the bit is forgotten off the cut's cycle.
- Dominance is soft. A label may dominate one whose bits are lower if it
  still wins after paying those cuts' duals.
- Separation adds the least memory that keeps each cut's violation. Cuts
  are found at the root (``sr_depth``) and stay in every later node's master.
- The Lagrangian bound, reduced-cost fixing and ng-routes all remain valid.

Benchmarked 2026-11 on the Rm instances with root gaps (M2N10_1/3/9,
M2N20_3/4/9, M5N25_1/6, M8N40_1; 0.3-3 % gap):

- **Bound and tree.** Cuts lift every root bound, by up to 0.8 % of the
  objective with 16 cuts (rm_M2N10_3). They cut nodes on some instances
  (rm_M5N25_1 11 -> 3, rm_M8N40_1 9 -> 5).
- **Time.** The whole set takes 36.5 s without cuts, 45.7 s with 8 and
  78.8 s with 16. On the 2-machine instances a path visits half the jobs,
  so memories cover nearly everything. Label counts then grow by about
  3x per round, and with 31 cuts rm_M2N10_9's root did not finish in
  6 min.

Phase 3's fixing and time-window branching already close these gaps in
5-13 nodes, so cuts stay off. The larger Rm families have essentially no
root gap at all.

Large near-integral instances: :mod:`fedhpc.lpfirst`
----------------------------------------------------
On the real trace (964 jobs, hundreds of identical instances per on-prem
type) the compact LP is essentially the integer optimum. For those
instances, :mod:`fedhpc.lpfirst` (``pareto`` engine ``"lpfirst"``)
replaces path pricing:

- column generation over single placements of the time-indexed LP;
- a Lagrangian bound;
- a MIP over the placements whose reduced cost fits in the gap, which
  proves optimality without branching.

The development notes below, and some function docstrings, cite timings on
``tiny_*`` instances (5-25-job subsets of the fedhpc trace). Those instances
have been deleted; the tests
now use the trap instances in ``tests/conftest.py`` (7 and 12 jobs, with
running jobs).

Phase 1: root column generation
-------------------------------
Root-node column generation only — no branching, no cuts. This gives a standalone,
independently checkable LP bound: by Geoffrion's integrality property, the
root LP relaxation of this master must equal the LP relaxation of the compact
``SpaceTimeFormulation`` MIP exactly, since each type's pricing subproblem
polytope (a pure DAG flow) is integral. That equality is Phase 1's load-
bearing correctness test (``tests/test_bpc.py``).

Caveat: the pricing below enforces *elementary* paths (no job twice on one
column), which the DAG-flow polytope does not, so in general the master's
bound can exceed the compact LP's; equality on the test instances means
elementarity does not bind there, not that it never can. The reduced start
sets of ``_Prep`` likewise can only raise the bound
(``tests/test_bpc_prep.py``).

Master (RMP)
------------
A column ``p`` for type ``m`` is an ordered sequence of non-overlapping
``(job_id, start)`` pairs — one candidate full-horizon timeline for a single
instance of type ``m``. Variables ``lambda_{m,p} >= 0`` count how many
instances of type ``m`` follow pattern ``p``. Rows:

- job coverage (equality, dual ``sigma_j``): every job covered exactly once.
- fleet-size (equality, dual ``rho_{m,r}``, one per capacitated type and
  release group ``r`` — see "Running jobs" below; unbounded types such as
  cloud have none): ``sum_p lambda_{m,r,p} = N_{m,r}``. (Originally one
  row per type, cloud included, with ``N_m = n_jobs`` for unlimited types.) MOFE fixes ``y_{m,-1} := N_m`` and
  ``y_{m,H} := N_m`` as *parameters* (not a variable to optimize), so flow
  conservation forces exactly ``N_m`` instance-lifetime paths through ``G_m``
  at all times, including idle ones. A canonical **null column** (no jobs,
  cost 0) is seeded per type so idle capacity is never something pricing has
  to discover.
- up to two optional bound rows (inequality, duals ``nu1``/``nu2``), capping
  the aggregate f1 and/or f2 contribution — both may be active
  *simultaneously* (see ``make_bpc_lexmin_bounded``: ``pareto.py``'s box
  search always needs an external box-boundary cap on one objective plus a
  lex-min tie-break cap on the other, at the same time, which is why this
  is two independent rows rather than one). This is what plugs the engine
  into the existing box-search Pareto algorithms (``pareto.true_pareto_frontier``/
  ``map_pareto_frontier``) unchanged, via ``engine="bpc"``.

Per-job coverage rows are bootstrapped with a big-M artificial variable each,
so the RMP is always LP-feasible even before any real column touches a given
job (standard column-generation phase-1 bootstrapping) — ``artificial_total``
in the result must be ~0 at convergence for the bound to be trusted.

Phase 2 + 3: branch-and-price, Ryan-Foster with a variable-branching fallback
------------------------------------------------------------------------------
A custom branch-and-bound tree on top of Phase 1's root relaxation — no cuts
yet. Each node gets its **own fresh Gurobi RMP** (seeded with the parent's
already-discovered, still-valid columns) rather than mutating one shared
model across the tree — matching this project's own established caution
about reusing solved Gurobi model state across structurally different
constraint sets (see ``pareto._make_lexmin_bounded``'s docstring for the
precedent and the concrete bug it was tracking).

A node's solution is accepted as integral once every job's aggregated
``x_jmt = sum_{p containing (j,t) on type m} lambda_p`` reaches ~1 at some
single ``(m,t)`` — individual ``lambda_p`` values may still be fractional,
but that residual is provably resolvable without changing the objective
(Geoffrion's integrality property again, applied to each type's now-pinned
sub-selection), so job-level integrality is both necessary and sufficient to
accept a node's objective as a valid incumbent.

**Ryan-Foster on job pairs** (``branching="rf"``; no longer the default — see "Search tree" above). For every pair of jobs
``(j, k)`` that co-occur on some positive-``lambda`` column, ``joint_jk =
sum_{p containing both j and k} lambda_p`` measures how "tied together" they
currently are. The most fractional such pair (closest to 0.5) is branched:

- *differ*: forbid any column containing both ``j`` and ``k``.
- *same*: forbid any column containing exactly one of ``j``, ``k`` (both or
  neither).

This is the branching that actually attacks the identical-instance symmetry
the compact model suffers from (see ``pareto.true_pareto_frontier``'s
docstring for the concrete pathology — heavy solution symmetry on
identically-capacitied instances making B&B search order, and hence runtime,
wildly non-deterministic): unlike branching on a single ``(job, type,
start)`` triple, fixing *which two jobs share an instance* directly collapses
the many structurally-interchangeable ways of assigning them across
identical instances, rather than picking on one arbitrary instance/time slot
at a time. Both directions are enforced purely by filtering, symmetric to
Phase 2's mechanism: ``differ_pairs``/``same_pairs`` frozensets threaded into
``_price_type`` — *differ* as a per-arc mask check (no DP state growth,
same shape as the elementary-path check), *same* as (1) dropping a job from
a type's arcs entirely if its partner isn't even feasible there, and (2) a
mask-consistency filter over the labels reaching node H (see
``_price_type``'s docstring) — and into the inherited column pool's filter
(``_column_allowed``).

**Fallback: Phase 2's single-variable branching.** Ryan-Foster needs some
job to co-occur with another in a fractional column; a lone job whose only
ambiguity is *which start time* (no other job ever sharing its column) has
no useful pair, so ``_solve_node`` falls back to branching directly on the
most-fractional ``(job, type, start)`` triple in that case — *down* forbids
the exact triple, *up* pins the job to it (dropping every other placement,
any type, any other start). Still enforced the same way as before (a
``forbidden`` set / ``pinned`` dict, arc-level skip, no DP state growth).

**Known limitation: Ryan-Foster does not fix every slow case.** It targets
identical-instance symmetry specifically — collapsing the many
interchangeable ways to assign two jobs across identical-capacity instances.
It does *not* by itself strengthen a weak LP bound. Concretely,
``solve_f2_bpc``'s second phase (minimise f1 subject to f2 <= its own
minimum, which on ``tiny_large``/``tiny_xlarge`` forces every job onto the
5 free on-prem instance-slots total) stayed slow (30+ min, not reduced from
Phase 2's variable-branching baseline) even with Ryan-Foster as the primary
branching rule. That sub-problem's hardness is minimising weighted
completion time across a handful of machines for 18-25 jobs — a genuinely
hard scheduling problem in its own right (closer to classical
``Pm||sum(w_j C_j)``), not a symmetry artifact — so no branching *rule*
alone should be expected to fix it; the LP relaxation's bound is simply not
tight enough to prune effectively, which is what valid inequalities (Phase
4/5's cuts, e.g. subset-row cuts) exist to address. Every other tested case
(all weighted-sum vectors, all lex-min directions except this one on the two
largest instances) is both correct and fast (well under a second to
~90s worst case) with Ryan-Foster active.

Job weights and fixed activation costs
---------------------------------------
The current formulation (``data.py``/``formulations.py``, past what either PDF
documents) adds two extras on top of MOFE §3, both handled here:

- ``Job.weight`` (``w_j``, default 1): f1 becomes ``sum w_j (C_j - a_j)``.
  Purely a per-arc cost1 multiplier — no structural change.
- ``InstanceType.fixed_cost`` (``U_m``, default 0): f2 gains ``sum U_m z_m``
  where the compact model's ``z_m in {0,1}`` is linked per (job, type) pair
  via ``sum_t x_jmt <= z_m`` (``formulations.add_fixed_cost_vars``) — i.e. a
  one-off charge if *any* job runs on type ``m``, independent of how many
  columns/instances of that type end up doing work. This is reproduced at
  master level, matching the compact model's linking exactly (both are
  disaggregated per job, not one aggregate row per type, to keep the LP bound
  tight): one continuous var ``z_m in [0,1]`` per fixed-cost type with
  objective coefficient ``weight2 * U_m``, and one row per (job, type) pair
  eligible for that type, ``sum_{p containing j} lambda_{m,p} <= z_m``, dual
  ``theta_{j,m}``. Because this row's membership is per (job, type) — not per
  column-content — it is still exactly arc-additive: pricing just subtracts
  ``theta_{j,m}`` from that job's arc on that type, same shape as ``sigma_j``.
  No extra DP state is needed (unlike a naive "pay once per column" framing
  would require).

Phase 4: pareto.py integration, and two more real bugs it exposed
---------------------------------------------------------------------
``make_bpc_lexmin_bounded`` wires this engine into
``pareto.true_pareto_frontier``/``map_pareto_frontier`` (``engine="bpc"``),
which needs *two* bound rows active simultaneously — the box search's own
external boundary on one objective, plus the lex-min tie-break fix on the
other — hence ``bound1_value``/``bound2_value`` as two independent optional
caps rather than the earlier single ``bound_coef``/``bound_value``. This
combination (a genuinely binding bound row, not just a loose one as in every
Phase 1-3 test) surfaced two real, previously-latent bugs together, both
around ``InstanceType.fixed_cost``:

1. **``z_m`` was missing from the f2 bound row.** Its fixed-cost
   contribution to f2 only ever appeared in the RMP's *objective*
   (``obj=weight2*U_m``) — its ``gp.Column`` never gave it a coefficient in
   either bound row, unlike the compact model's f2 expression, which is built
   from ``f2_expr`` and so already includes it (see
   ``formulations.add_fixed_cost_vars``). Every prior test used either no
   bound row or one loose enough to never actually bind on this, so it went
   uncaught through the whole of Phases 1-3. Fixed by giving ``z_m`` a coefficient of ``U_m``
   in ``bound2`` (never ``bound1`` — a fixed cost never contributes to f1)
   whenever active.

2. **``z_m`` is not structurally guaranteed integral, and the acceptance
   tolerance was too loose to catch it.** Geoffrion's integrality property
   covers the coverage/fleet-size rows (arising from each type's own
   integral network-flow subproblem) — it says nothing about a variable that
   exists purely for the bound rows and the objective. A node could have every job's
   placement fully resolved while ``z_m`` sat at, say, ``0.9999971`` — "close
   enough" under the 1e-5 tolerance used elsewhere for column-generation tie-
   breaking, but multiplied by a ``fixed_cost`` of 15 that residual is a real
   ~1e-4 error in f2, which is exactly the size of gap a bound row set right
   at a lex-min tie-break boundary can let slip through silently: the
   fractional LP solution barely satisfied the row, but the *clean* assignment
   extracted from it (ignoring the residual) had a higher true cost that
   violated it. The same failure mode, same magnitude, showed up in ordinary
   job-placement fractionality too — a job's coverage split ~99.9997% on its
   obvious ``(m,t)`` with a sliver actually sitting on a *different*, cheaper
   placement for the same job. Both are fixed the same way: ``_solve_node``'s
   acceptance check (as opposed to ``_branch_and_price``'s own bound-pruning
   comparisons, which stay at the looser tolerance — only *this* determination
   needed tightening) now uses a dedicated, much tighter tolerance
   (``_INTEGRALITY_TOL = 1e-7``) for both per-job settlement and ``z_m``
   fractionality, and branches on a fractional ``z_m`` exactly like Ryan-Foster
   branches on a fractional job pair — one child forces ``z_m=0`` (type
   entirely forbidden: setting its RMP upper bound to 0 also forces every
   job-carrying column's lambda on that type to 0 via the existing linking
   rows, no separate arc filtering needed), the other forces ``z_m=1`` (lower
   bound 1, not tied to any specific job).

Phase 5: rank-1 subset-row (SR-1) cuts, off by default
--------------------------------------------------------
``_separate_sr_cuts`` (Jepsen et al. 2008) finds job triples ``{i,j,k}``
whose active-column coverage sums to > 1 — impossible for any integral
solution, since no elementary path can cover 2+ of 3 given jobs more than
once — and adds ``sum(lambda_p : |jobs(p) ∩ {i,j,k}| >= 2) <= 1`` as a new
RMP row. ``_root_cuts_and_price`` separates these once at the root, before
branching, alternating with column generation until no more violated triples
are found or a small round/count budget is hit. In pricing, each active cut
contributes a dual credit to a column's reduced cost the *first* time a
label picks up its 2nd job of the triple (``_pending_sig`` extends its
existing same-pair pending-bit mechanism with one more bit per active cut,
so the credit is applied exactly once, not once per qualifying job).

This is implemented and unit-tested (``tests/test_bpc_cuts.py`` — separation,
the pricing dual-trigger arithmetic, and end-to-end agreement with
``use_cuts=False`` on ``known_pareto_inst``/the job-weight+fixed-cost
fixture), and cuts by construction can only tighten the LP bound, never
change the true optimum. Running that test suite for the first time found
two more real bugs, both in the test's own first draft as much as in
``bpc.py``:

* **Candidate-triple generation never proposed the classic 3-columns shape.**
  The first version built each pair's "third job" candidates only from
  *within the same column* that contained the pair — but the textbook SR-1
  violation is three *separate* columns, each covering a different pair of
  the triple (``{i,j}``, ``{i,k}``, ``{j,k}``), where ``i`` and ``k`` may
  never co-occur in any single column at all. A 2-job-per-column instance
  (exactly ``TestSeparateSrCuts.test_finds_violated_triple``'s synthetic
  case) could then never produce *any* candidate triple, silently
  neutering separation. Fixed by building a global job co-occurrence graph
  across all active columns first, then chaining through shared jobs (edge
  ``i-k``, extended by anything adjacent to *either* endpoint) to propose
  the triple — this is what actually lets the cut fire on the shape it's
  meant for.
* **The end-to-end test asserted an invariant cuts never promised.**
  ``TestCutsEndToEnd``'s first draft compared f1 *and* f2 individually
  between ``use_cuts=True/False`` and failed on the job-weight+fixed-cost
  fixture at ``weight1=1, weight2=0`` (47.0 vs 39.0) — but with
  ``weight2=0`` that solve never constrains f2 at all, so a tie in the
  *actual* objective (f1, which matched exactly: 32.0 both ways) is free to
  resolve to a different assignment — and hence a different f2 — depending
  on B&B exploration order, cuts or not, the same as any two runs that
  explore the tree differently. (This is exactly why the public
  ``solve_f1_bpc``/``solve_f2_bpc`` wrappers add their own explicit
  tie-break bound row on the *other* objective — raw ``_branch_and_price``
  alone never promised single-objective determinism.) Fixed by comparing
  only the scalar weighted objective ``_branch_and_price`` itself returns,
  which is the actual invariant cuts preserve.

Validating the (now-fixed) mechanism against the plan's actual target (the
``solve_f2_bpc`` tie-break case documented as slow under Phase 3) turned up
a further two findings that argue for **``use_cuts=False`` by default**,
which is now the default (with much smaller separation limits than first
written — ``max_cut_rounds``/``max_cuts_per_round``/``max_total_cuts`` of
3/2/5 rather than 10/5/30 — for callers who opt in):

1. **Every active cut costs the pricing DP a dominance bit.** A label that
   has triggered a cut and one that hasn't are no longer comparable for
   dominance (same as an open same-pair decision), so more active cuts
   shrink how much the label-setting DP can prune — the same mechanism that
   already makes ``weight1=0``/deep-in-the-tree pricing slow (see
   "Elementary-path enforcement" below) gets *worse* with cuts on, not
   better. Concretely: with cuts on, ``tiny_medium.json``'s
   ``solve_f2_bpc`` tie-break phase (minimise f1 s.t. f2 <= its own minimum)
   went from finishing in seconds to not finishing in 10 minutes, on an
   instance that Phases 2-4 always handled without trouble.
2. **That same phase is *also* slow with ``use_cuts=False``** — confirmed by
   direct reproduction (still not finished after 10 minutes). So the
   regression observed while validating Phase 5 was not actually caused by
   cuts at all: it was unmasked by the Phase 4/5 bugfix described below
   (``_solve_node``'s objective for an "integral" node), which had been
   silently over-pruning the tree since Phase 2. Before the fix, a
   falsely-optimistic ``mdl.ObjVal`` could make ``best_obj`` look better than
   it truly was, pruning away branches that a *correct* bound would have
   kept open — for ``tiny_smallest`` this pruned away the actual optimum
   (a wrong answer, the bug that motivated the fix), but for ``tiny_medium``
   it happened to only prune away branches that weren't optimal anyway (a
   right answer, reached for the wrong reason, fast only because the search
   was — incorrectly — cut short). With the bug fixed, the search is
   correct but now has to actually explore the tree, exposing that this
   phase's underlying hardness (tightly-pinned, on-prem-packing weighted
   completion time — the same ``Pm||ΣwⱼCⱼ``-flavoured shape Phase 3 already
   diagnosed for ``tiny_large``/``tiny_xlarge``) was there all along.
   ``tiny_medium`` simply joins that known-slow set now that the bug that
   was accidentally hiding it is gone.

Net: cuts are a real, correctly-implemented, bound-only-tightening tool that
remains available (``use_cuts=True``), but they are not what's needed for
this project's actual hard case, and at real cost to pricing-DP dominance
they can make a previously-fine case slow too. The pathological
``Pm||ΣwⱼCⱼ``-style hardness itself (weak LP bound, not symmetry) is still
open — per the plan, if it needs solving, it likely wants a different lever
(e.g. a stronger scheduling-specific bound, or accepting this as a scale
limit) rather than more cuts along this same axis.

Comparison with Kramer et al. 2020, and a second negative result
--------------------------------------------------------------------
Kramer, Dell'Amico, Feillet & Iori, "Scheduling jobs with release dates on
identical parallel machines by minimizing the total weighted completion
time" (2020, arXiv:2005.10097) targets ``P|rⱼ|ΣwⱼCⱼ`` — the exact
``Pm||ΣwⱼCⱼ``-flavoured hardness this project has documented for
``tiny_large``/``tiny_xlarge``/``tiny_medium`` above, via a published,
dedicated branch-and-price. Worth comparing directly: their master is
set-*covering* (``≥1``/``≤m`` rows, not our set-*partitioning* ``=1``/
``=N_m``) over a single homogeneous machine pool, with completion-time-
window branching (not Ryan-Foster) — their own DW decomposition already
absorbs identical-machine symmetry for free, so unlike this project they
never need symmetry-breaking branching at all. Their headline result: even
a dedicated, peer-reviewed B&P for exactly this problem leaves real gaps
and hits their 1800s time limit at n=150-200 jobs (their Table 4) —
independent confirmation that ``Pm||ΣwⱼCⱼ`` is genuinely hard, not an
artifact of this project's own implementation.

Their biggest lever for scaling pricing is giving up strict elementary-path
enforcement: instead of full bitmask dominance (what ``_price_type`` does,
worst-case exponential), their ``LC_2l`` pricer solves a *1-cycle-free*
relaxation — a job may be reused non-adjacently, only an immediate repeat is
forbidden — via just two labels per node (best, and best with a different
last job), giving strict ``O(n*H)``. ``_price_type_relaxed`` adapts this
directly (same two-label update rule). But their master's ``≥1`` coverage
makes a non-elementary column merely wasteful (paying to process a job
twice is never beneficial, so the LP relaxation naturally avoids it even
though it's technically feasible), never *infeasible* — so they can safely
run their **entire** B&P on ``S_1-cycle`` columns. Our master's ``=1``
*equality* coverage has no such escape hatch: a column that revisits a job
would need ``lambda_p`` values that don't exist for the row to still sum to
exactly 1, so accepting one into the RMP is a correctness risk, not just an
efficiency question — see ``_price_type_relaxed``'s own docstring for the
full argument. This project's adaptation therefore uses
``_price_type_relaxed`` only as a **provably safe pre-screening bound**,
never a column source: since ``S ⊆ S_1-cycle``, its minimum reduced cost is
always <= ``_price_type``'s true elementary one, so whenever it already
finds nothing improving, ``_price_type`` is guaranteed to find nothing
either and can be skipped that iteration — see
``_run_column_generation``'s ``use_relaxed_prescreen`` parameter. Validated
both as a bound (``tests/test_bpc_relaxed.py``'s ``TestRelaxedLowerBound``,
including 8 random-dual draws per type on ``known_pareto_inst``, not just
one hand-picked example) and end-to-end (screened vs. unscreened
``_branch_and_price`` results must match exactly, which they do).

Despite being correct, it does not meaningfully speed up this project's
documented hard case — a second negative result, alongside cuts, but for a
different, specific, measured reason. Direct instrumentation on both
``tiny_medium`` (30 nodes, 8-node-budget comparison) and ``tiny_large`` (the
plan's original motivating case) showed real skip rates (19% and 55% of
exact calls respectively) but negligible wall-clock change (roughly 0%/4%,
both within Gurobi's own run-to-run LP-solve noise). Root cause, from timing
``_price_type`` by outcome on ``tiny_large``: calls that *confirm nothing
improving* cost about the same per call (~0.31s avg) as calls that *find*
something (~0.38s avg) — the elementary DP's cost comes from exploring the
label space, not from what the final answer turns out to be, so knowing in
advance "nothing improving exists" (what the relaxed screen buys) doesn't
actually make the exact call itself any cheaper when it still has to run.
In other words: this project's per-node bottleneck isn't "confirming
convergence" (Kramer et al.'s target, where two-label relaxation genuinely
is much cheaper than full elementary search) but "computing the actual best
column when one exists", for which the full elementary DP is unavoidable
regardless of how cheaply a screen can rule out the *other* case. Kept
available (``use_relaxed_prescreen=True``, provably never changes results)
but off by default, same rollout posture as cuts, for the same reason: a
correctly-implemented tool this project's actual hard case doesn't need.

Pricing
-------
Reduced cost decomposes additively over a path's processing arcs (wait arcs
cost 0, so they never affect it): arc reduced cost = weighted (cost1, cost2)
coefficient minus the job's coverage dual minus the bound duals' share
of that arc's cost2/cost1 contribution, minus the fixed-cost linking dual
``theta_{j,m}`` when the type has a fixed cost. The fleet-size dual ``rho_m``
applies once per whole column (its row has coefficient 1 for every column of
type ``m``), not per arc. Because ``G_m`` is a DAG ordered by time, minimizing
the arc-additive part is a single forward DP pass over nodes ``0..H`` —
``O(n_m * H)`` per type, no full resource-constrained-shortest-path label
machinery needed at the root.

Elementary-path enforcement (job reuse within one column)
-----------------------------------------------------------
A plain DAG shortest path has no notion of "don't revisit the same job" — and
this is not just a theoretical gap: it fired on essentially *every* first
column-generation iteration during development, because the big-M coverage
duals (see above) make every uncovered job's arc attractive at *every* one of
its reachable start times simultaneously, so the naive DP chained the same
job onto one path repeatedly. That's not a harmless artifact: it double-
claims the job's coverage-row credit and would corrupt the LP bound in a way
the compact model's per-``(j,m,t)`` binary variables structurally cannot
(``x_jmt in {0,1}`` can't represent "job j twice"; a column-level aggregate
can, unless prevented).

The first fix tried — rejecting a transition if the single best-known path to
that node already used the job — was not sufficient: once "no job reused" is
a constraint, the shortest path to an intermediate node is no longer
guaranteed to be part of the shortest *elementary* path end-to-end (the DP's
optimal-substructure assumption breaks), so a naive single-best-path-per-node
DP can genuinely miss the true optimum. This showed up concretely as a root
LP bound *weaker* (too high) than the compact model's, e.g. 29.5 vs. the true
25.0 on a 3-job/2-type instance where the compact optimum runs all 3 jobs
sequentially on one capacity-1 on-prem instance — a 3-job elementary path the
walk-back check couldn't reach.

The real fix is standard resource-constrained-shortest-path label-setting:
``_price_type`` keeps, per node, a set of Pareto-efficient ``_Label``s, each
carrying a bitmask of jobs used so far (``bit_of``, one bit per job feasible
on this type) alongside its cost. A label dominates another at the same node
iff its cost is <= and its job-set is a *subset* (strictly more remaining
flexibility, never worse) — dominated labels are dropped, new labels that are
already dominated are never added. The final answer is the min-cost label
reaching node ``H``, reconstructed by walking ``.pred`` links back to the
source. This is exact (true minimum-reduced-cost elementary path), matching
Phase 1's root-LP-bound-equality guarantee exactly, not just "usually" —
tractable at this project's current (tiny-instance) scale since label counts
per node stay small in practice; would need dominance-rule tightening (e.g.
resource-window relaxation / ng-routes) before scaling to much larger
instances, same as any RCSP-based column generation.
"""
from __future__ import annotations

import heapq
import math
import time
from collections import Counter
from dataclasses import dataclass, field, replace

import gurobipy as gp
from gurobipy import GRB

from . import formulations
from .data import Instance, Job
from .model import Solution, grid_floor

try:
    from . import _bpcprice  # type: ignore[attr-defined]
except ImportError:  # extension not built: Python pricing only
    _bpcprice = None

__all__ = [
    "BPCTimeLimit", "Column", "PricingOptions", "RootResult", "solve_root_relaxation",
    "solve_f1_bpc", "solve_f2_bpc", "solve_weighted_sum_bpc",
    "make_bpc_lexmin_bounded",
]


class BPCTimeLimit(RuntimeError):
    """``_branch_and_price`` ran past its ``time_limit``.

    ``objective``/``assignment`` hold the best schedule found by then (the
    instance's own jobs only), or ``inf``/``None`` when there is none. It is
    a feasible schedule, not a proven optimum.
    """

    def __init__(self, objective: float = float("inf"),
                 assignment: dict[int, tuple[int, int]] | None = None,
                 bound: float = -float("inf")):
        super().__init__("BPC branch-and-price exceeded its time limit")
        self.objective = objective
        self.assignment = assignment
        self.bound = bound   # proven lower bound on the optimum (open nodes)


def _check_deadline(deadline: float | None) -> None:
    """Raise :class:`BPCTimeLimit` (no incumbent yet) once ``deadline``
    (a ``time.perf_counter()`` value) has passed."""
    if deadline is not None and time.perf_counter() > deadline:
        raise BPCTimeLimit


# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Column:
    """One candidate full-horizon timeline for a single instance of ``type_id``.

    ``jobs`` is the ordered sequence of ``(job_id, start)`` pairs the instance
    processes (empty for the type's null/all-idle column). ``cost1``/``cost2``
    are this column's total contribution to f1 (turnaround) and f2 (cost).
    """
    type_id: int
    jobs: tuple[tuple[int, int], ...]
    cost1: float
    cost2: float
    release: int = 0   # slot the instance becomes free (a running job's end, or 0)


@dataclass(frozen=True)
class _Arc:
    job_id: int
    p_occ: int
    starts: range | frozenset   # allowed start slots (T_jm, or its reduced subset)
    raw_cost2: float    # inst.c[job_id, type_id]
    weight: float       # j.weight — multiplies this arc's cost1 contribution


def _build_arcs_by_type(
    inst: Instance,
    starts: dict[tuple[int, int], range | frozenset] | None = None,
) -> dict[int, list[_Arc]]:
    """Processing arcs per type. ``starts`` (default ``inst.T``) maps each
    kept ``(job, type)`` pair to its allowed starts; pairs absent from it are
    dropped."""
    result: dict[int, list[_Arc]] = {itype.id: [] for itype in inst.instance_types}
    for j in inst.jobs:
        for m_id in inst.F[j.id]:
            rng = inst.T[j.id, m_id] if starts is None else starts.get((j.id, m_id))
            if not rng:
                continue
            result[m_id].append(_Arc(
                job_id=j.id,
                p_occ=inst.p_occ[j.id, m_id],
                starts=rng,
                raw_cost2=inst.c[j.id, m_id],
                weight=j.weight,
            ))
    return result


def _release_classes(inst: Instance, type_id: int) -> dict[int, int]:
    """``{release slot: instances}`` of a capacitated type: the instances free
    at 0, plus one per running job, available from its (horizon-clipped)
    end. Mirrors :meth:`Instance.build`'s ``n_running``/``rj_arrivals``; an
    instance released at H can never run anything and is left out."""
    itype = next(m for m in inst.instance_types if m.id == type_id)
    out = {0: itype.capacity - inst.n_running.get(type_id, 0)}
    for (mid, r), n in inst.rj_arrivals.items():
        if mid == type_id and r < inst.horizon:
            r = max(r, 0)
            out[r] = out.get(r, 0) + n
    return {r: n for r, n in out.items() if n > 0}


def _fixed_cost_types(inst: Instance) -> list:
    """Instance types with a per-type fixed activation cost (U_m > 0)."""
    return [itype for itype in inst.instance_types if itype.fixed_cost]


def _single_job_column(inst: Instance, j: Job, mid: int, t: int) -> Column:
    jid = j.id
    return Column(type_id=mid, jobs=((jid, t),),
                  cost1=j.weight * (t + inst.p_occ[jid, mid] - j.arrival),
                  cost2=inst.c[jid, mid])


@dataclass
class _Prep:
    """Everything about an instance the master and pricing need, computed once
    (and shared across every box solve of a Pareto search).

    - ``starts``: allowed starts per ``(job, type)``. With nonnegative data
      these are the objective-independent exact reductions of
      ``formulations``: normal-pattern starts (Kramer et al. 2020) plus R1
      (a job on an unbounded type starts at its release) and R2 (dominated
      unbounded options dropped). None of them raises f1 or f2 for any
      schedule, so they stay valid under both bound rows. The
      objective-dependent rules (R3, lex deadlines) are invalid under an f2
      cap and are not used.
    - ``unbounded``: types whose capacity never binds
      (:func:`formulations.unbounded_types`, e.g. cloud). Any column on such
      a type splits into single-job columns of equal total cost, and the
      fleet is never short, so those types get no fleet row and no pricing:
      their single-job columns (``static_columns``) are all enumerated
      upfront.
    - ``releases``: per remaining (*path*) type, its instances grouped by the
      slot they become free — 0, or a running job's end. Each group has its
      own fleet row, and a column of group ``r`` is a path from node ``r``;
      this replaces the earlier pinned pseudo-job per running job.
    """
    inst: Instance
    starts: dict[tuple[int, int], range | frozenset]
    unbounded: frozenset[int]
    path_types: tuple[int, ...]
    releases: dict[int, dict[int, int]]
    arcs_by_type: dict[int, list[_Arc]]
    static_columns: list[Column]
    arrival_of: dict[int, float]
    weight_of: dict[int, float]
    # ng-route neighbourhoods per path type, {job: other jobs remembered after
    # it}; built on first use and grown by _augment_ng (see _NodePricing).
    ng: dict[int, dict[int, set[int]]] = field(default_factory=dict)


def _prepare(inst: Instance, reduce: bool = True) -> _Prep:
    if reduce and formulations.nonnegative_data(inst):
        S = formulations._normal_pattern_graph_data(inst)[0]
        S, _ = formulations.SpaceTimeFormulation()._reduce(inst, S, None)
    else:
        S = {(j.id, m): inst.T[j.id, m] for j in inst.jobs for m in inst.F[j.id]}
    unbounded = frozenset(formulations.unbounded_types(inst))
    path_types = tuple(m.id for m in inst.instance_types if m.id not in unbounded)
    releases = {m: _release_classes(inst, m) for m in path_types}
    arcs = _build_arcs_by_type(inst, S)
    job_by_id = {j.id: j for j in inst.jobs}
    static = [
        _single_job_column(inst, job_by_id[jid], mid, t)
        for (jid, mid), ts in sorted(S.items(), key=lambda kv: kv[0]) if mid in unbounded
        for t in sorted(ts)
    ]
    return _Prep(
        inst=inst, starts=S, unbounded=unbounded, path_types=path_types,
        releases=releases,
        arcs_by_type={m: arcs[m] for m in path_types},
        static_columns=static,
        arrival_of={j.id: j.arrival for j in inst.jobs},
        weight_of={j.id: j.weight for j in inst.jobs},
    )


def _column_key(col: Column) -> tuple:
    return (col.type_id, col.release, col.jobs)


def _schedule_columns(prep: _Prep, assignment: dict[int, tuple[int, int]]) -> list[Column]:
    """Split a (possibly partial) schedule into master columns: one single-job
    column per job on an unbounded type, and per path type one chain per
    instance, built by interval partitioning over the instances' release
    slots. Processing jobs by start, any instance free at a job's start stays
    free for every later job, so a capacity-feasible schedule always fits;
    a job that does not fit (an infeasible input) is left out.

    Starts outside the reduced start sets are fine: such a column is still a
    valid column of the unreduced problem."""
    inst = prep.inst
    job_by_id = {j.id: j for j in inst.jobs}
    cols: list[Column] = []
    by_type: dict[int, list[tuple[int, int, int]]] = {}
    for jid, (mid, t) in assignment.items():
        if jid not in job_by_id or mid not in inst.F[jid]:
            continue
        if mid in prep.unbounded:
            cols.append(_single_job_column(inst, job_by_id[jid], mid, t))
        elif mid in prep.releases:
            by_type.setdefault(mid, []).append((t, t + inst.p_occ[jid, mid], jid))
    for mid, items in by_type.items():
        items.sort()
        chains = [[r, r, []] for r, n in sorted(prep.releases[mid].items()) for _ in range(n)]
        for start, end, jid in items:
            pick = None
            for ch in chains:
                if ch[1] <= start and (pick is None or ch[1] > pick[1]):
                    pick = ch
            if pick is None:
                continue
            pick[1] = end
            pick[2].append((jid, start))
        for release, _end, jobs in chains:
            if not jobs:
                continue
            cost1 = sum(job_by_id[jid].weight * (t + inst.p_occ[jid, mid] - job_by_id[jid].arrival)
                        for jid, t in jobs)
            cost2 = sum(inst.c[jid, mid] for jid, t in jobs)
            cols.append(Column(type_id=mid, jobs=tuple(jobs), cost1=cost1, cost2=cost2,
                               release=release))
    return cols


def heuristic_schedules(inst: Instance, *, ea: bool = False) -> list[dict[int, tuple[int, int]]]:
    """Feasible schedules from cheap list-scheduling heuristics, used to seed
    the master's columns and the search's incumbent.

    The baselines span the trade-off: earliest completion (min-turnaround
    side), on-prem only (min-cost side) and a threshold-bursting sweep in
    between. With ``ea`` (and the compiled extension, no fixed costs), an
    NSGA-II front is added — a few seconds that the box searches of a
    Pareto mapping amortise. Partial schedules are dropped.
    """
    inst.require_unit_demand("branch-and-price (bpc)")
    from . import baselines

    sols = [baselines.greedy_earliest_completion(inst),
            baselines.greedy_earliest_completion(inst, order="arrival"),
            baselines.onprem_only_spt(inst)]
    sols += [s for _th, s in baselines.threshold_bursting_sweep(inst)]
    if ea:
        try:
            from . import moea
            sols += moea.nsga2_frontier(inst, pop_size=100, n_gen=60, n_threads=1)
        except Exception:  # extension missing or fixed costs: baselines only
            pass
    out, seen = [], set()
    for s in sols:
        a = dict(s.assignment)
        key = tuple(sorted(a.items()))
        if len(a) == len(inst.jobs) and key not in seen and not inst.schedule_violations(a):
            seen.add(key)
            out.append(a)
    return out


def _heuristic_seed_columns(inst: Instance, prep: _Prep | None = None) -> list[Column]:
    """Columns of :func:`heuristic_schedules`' schedules (see there), to seed
    the very root of a search.

    Without real starting columns the root RMP holds only null columns plus
    big-M coverage artificials, which makes *every* job's dual look uniformly
    huge on iteration 1 — the degenerate regime that defeats the
    elementary-path label-setting DP's dominance pruning (see the module
    docstring's "Elementary-path enforcement" section).
    """
    prep = prep or _prepare(inst)
    cols: dict[tuple, Column] = {}
    for a in heuristic_schedules(inst):
        for c in _schedule_columns(prep, a):
            cols.setdefault(_column_key(c), c)
    return list(cols.values())


def _schedule_objective(
    inst: Instance, assignment: dict[int, tuple[int, int]],
    weight1: float, weight2: float,
    bound1_value: float | None, bound2_value: float | None,
) -> float | None:
    """``weight1·f1 + weight2·f2`` of a complete feasible schedule within both
    bound rows, else None."""
    if len(assignment) != len(inst.jobs) or inst.schedule_violations(assignment):
        return None
    completion = {jid: t + inst.p_occ[jid, mid] for jid, (mid, t) in assignment.items()}
    f1, f2 = inst.f1_of(completion), inst.f2_of(assignment)
    if bound1_value is not None and f1 > bound1_value + 1e-6:
        return None
    if bound2_value is not None and f2 > bound2_value + 1e-6:
        return None
    return weight1 * f1 + weight2 * f2


# ---------------------------------------------------------------------------
# Pricing: one DAG shortest-path DP per instance type
# ---------------------------------------------------------------------------

@dataclass
class _Label:
    """One Pareto-efficient partial path in the elementary shortest-path DP.

    ``mask`` is a bitmask over this type's feasible job ids (see
    ``_price_type``'s ``bit_of``) recording which jobs this path has already
    used — the state dimension a plain DAG shortest path is missing (see the
    module docstring's "Elementary-path enforcement" section).
    """
    cost: float
    mask: int
    pred: "_Label | None"
    job_id: int | None   # job used on the transition into this label, or None
    t_prev: int           # node the transition came from
    pending: int = 0       # Ryan-Foster same-pair pending signature — see _pending_sig


def _price_type(
    inst: Instance,
    type_id: int,
    arcs: list[_Arc],
    weight1: float,
    weight2: float,
    sigma: dict[int, float],
    nu1: float | None,
    nu2: float | None,
    rho_m: float,
    theta: dict[tuple[int, int], float],
    arrival_of: dict[int, float],
    weight_of: dict[int, float],
    tol: float,
    forbidden: frozenset[tuple[int, int, int]] = frozenset(),
    pinned: dict[int, tuple[int, int]] | None = None,
    differ_pairs: frozenset[tuple[int, int]] = frozenset(),
    same_pairs: frozenset[tuple[int, int]] = frozenset(),
    cut_duals: dict[frozenset[int], float] | None = None,
    sources: dict[int, float] | None = None,
    bucket_cap: int | None = 200,
) -> tuple[float, Column] | None:
    """Return ``(reduced_cost, column)`` for the min-reduced-cost *elementary*
    (no job reused) path of ``type_id``, or ``None`` if none prices out below
    ``-tol``. Label-setting DP with dominance — see the module docstring's
    "Elementary-path enforcement" section for why a plain shortest path isn't
    sufficient here.

    ``cut_duals`` — ``{triple: dual}`` for active rank-1 subset-row cuts (see
    the module docstring's "Phase 5: cuts" section and
    ``_separate_sr_cuts``): each triple's dual is subtracted from reduced
    cost exactly once per column, at the transition where the *second* of
    the triple's three jobs joins the path (the point ``count(mask, triple)``
    crosses from 1 to 2, which is where the cut's own row — ``sum_{p: |p.jobs
    & triple| >= 2} lambda_p <= 1`` — starts counting a column). Purely
    arc-additive, same shape as ``theta``, once one extra DP-state bit per
    active cut is added to the dominance signature (see ``_pending_sig``).

    ``forbidden`` — set of exact ``(job_id, type_id, start)`` triples this
    node's branching has ruled out (a Phase-2-style *down* decision).
    ``pinned`` — ``{job_id: (type_id, start)}`` for jobs a Phase-2-style *up*
    decision has fixed exactly; any other placement of that job (any type,
    any other start) is skipped, on every type including this one.
    ``differ_pairs`` — Ryan-Foster *differ* decisions: ``(j, k)`` pairs that
    must never both appear on the same column. Enforced per transition (an
    arc for job whose partner is already in the path's mask is skipped) —
    purely arc-additive, no extra DP state.
    ``same_pairs`` — Ryan-Foster *same* decisions: ``(j, k)`` pairs that must
    appear together or not at all on a column. Enforced two ways: (1) if a
    job in the pair isn't even feasible on this type, its partner is dropped
    from this type's arcs entirely (they could never be paired here); (2)
    among the labels reaching node H, only mask-consistent ones (both or
    neither job present) are eligible — see the loop after the DP below.
    ``sources`` — ``{release slot r: rho_{m,r}}``, the fleet-row dual of each
    group of instances that becomes free at ``r`` (see :class:`_Prep`). The
    DP starts one label per group at node ``r`` with cost ``-rho_{m,r}``, so
    one pass prices every group; the column carries the group it started
    from. ``None`` means one group free at 0 with dual ``rho_m``.
    ``bucket_cap`` — labels kept per node (``None``: no cap). The default
    cap makes the DP *inexact*: once a node holds that many labels, further
    ones are dropped, which can lose the best column (and so end column
    generation early). The compiled pricer has no such cap.
    ``nu1``/``nu2`` — dual of the RMP's optional f1-cap/f2-cap bound rows
    (``None`` when that row doesn't exist at all, vs. a float — possibly
    0.0 — when it exists but isn't currently binding; see ``_build_rmp``).
    Both can be active simultaneously (``pareto.py``'s box search needs an
    external box-boundary cap on one objective *and* a lex-min tie-break cap
    on the other, at the same time — see ``make_bpc_lexmin_bounded``).
    """
    H = inst.horizon
    pinned = pinned or {}

    # When cost1 never enters the objective or the f1-cap bound row
    # (weight1 == 0 and nu1 is None), reduced_arc no longer depends on t
    # (raw_cost1's coefficient is 0 throughout), so a job's cost is fixed
    # regardless of when/how it's scheduled. Two consequences: (1) any job
    # whose own reduced cost is non-negative can never help *when jobs are
    # otherwise independent* — dropping it only frees capacity for other
    # jobs, never hurts — so it's safe to prune it before the DP even
    # starts; (2) this is also exactly the scenario where the label-setting
    # DP's dominance is at its weakest (many same-cost, mutually
    # non-dominated job subsets survive, since nothing about *order* can
    # ever break a tie) — this
    # prefilter is what keeps it tractable at this project's current instance
    # sizes for that corner of the objective space (see the module
    # docstring's "Elementary-path enforcement" section).
    #
    # Jobs coupled by an active same-pair (or an active cut, same reasoning —
    # a job that looks unattractive alone can still be worth including once
    # it's the *second* of a triple, triggering that cut's dual credit) are
    # NOT independent, so this reasoning does not apply to them: a job's own
    # reduced cost being non-negative does not mean it "can never help" once
    # same_pairs/cuts can force/reward it coming along with others. Pruning
    # it anyway was a real bug found during development (for same_pairs) — it
    # silently made the partner-required job permanently unavailable, so
    # *neither* job in the pair could ever be used, well before same_pairs'
    # own consistency logic got a chance to weigh them together.
    same_pair_jobs = {jid for pair in same_pairs for jid in pair}
    cut_jobs = {jid for triple in (cut_duals or {}) for jid in triple}
    if weight1 == 0.0 and nu1 is None:
        pruned = []
        for arc in arcs:
            if arc.job_id in same_pair_jobs or arc.job_id in cut_jobs:
                pruned.append(arc)
                continue
            reduced_job = weight2 * arc.raw_cost2 - sigma.get(arc.job_id, 0.0)
            reduced_job -= theta.get((arc.job_id, type_id), 0.0)
            if nu2 is not None:
                reduced_job -= nu2 * arc.raw_cost2
            if reduced_job < -1e-9:
                pruned.append(arc)
        arcs = pruned

    if same_pairs:
        feasible_here = {arc.job_id for arc in arcs}
        exclude = set()
        for j, k in same_pairs:
            j_ok, k_ok = j in feasible_here, k in feasible_here
            if j_ok and not k_ok:
                exclude.add(j)
            elif k_ok and not j_ok:
                exclude.add(k)
        if exclude:
            arcs = [a for a in arcs if a.job_id not in exclude]

    bit_of = {jid: 1 << i for i, jid in enumerate(sorted({arc.job_id for arc in arcs}))}

    differ_of: dict[int, int] = {}  # job_id -> bitmask of jobs it must not co-occur with
    for j, k in differ_pairs:
        jb, kb = bit_of.get(j), bit_of.get(k)
        if jb is None or kb is None:
            continue  # one of them isn't even feasible on this type — nothing to enforce
        differ_of[j] = differ_of.get(j, 0) | kb
        differ_of[k] = differ_of.get(k, 0) | jb

    # (job_bit, partner_bit) for every same-pair with both feasible here —
    # used both by the pending-signature dominance guard below and by the
    # final mask-consistency filter at node H.
    same_bit_pairs = [
        (bit_of[j], bit_of[k]) for j, k in same_pairs if j in bit_of and k in bit_of
    ]

    # (bits of the triple's members feasible here, dual) for every active cut
    # with >=2 members feasible on this type — a cut can never trigger here
    # otherwise, so it's dropped entirely for this type's pricing (its row
    # still exists in the RMP; other types may still trigger it).
    # cut_idx_by_job: job_id -> indices into cut_specs it participates in, for
    # a fast per-arc lookup in the main DP loop below.
    cut_specs: list[tuple[list[int], float]] = []
    cut_idx_by_job: dict[int, list[int]] = {}
    for triple, dual in (cut_duals or {}).items():
        bits = [bit_of[jid] for jid in triple if jid in bit_of]
        if len(bits) < 2:
            continue
        idx = len(cut_specs)
        cut_specs.append((bits, dual))
        for jid in triple:
            if jid in bit_of:
                cut_idx_by_job.setdefault(jid, []).append(idx)

    def _cut_count(mask: int, bits: list[int]) -> int:
        return sum(1 for b in bits if mask & b)

    def _pending_sig(mask: int) -> int:
        """Bitmask of (a) which same-pairs have *exactly one* partner in
        ``mask`` so far — an unresolved future obligation — and (b) which
        cuts have already triggered (>=2 of their triple present). Two
        labels with different signatures are not comparable for dominance
        purposes: a cheaper-and-subset label that still owes a same-pair
        completion, or hasn't yet triggered a cut a costlier label already
        has, is not actually "at least as good", because catching up costs
        it strictly more time/cost going forward than a label that already
        resolved/triggered it ever needed to spend. Discovered as a real bug
        during development (for same_pairs — see the module docstring):
        without this guard, a same-pair-*inconsistent* partial path (e.g.
        job 0 alone, pending its same-pair partner job 1) could get purged by
        a cheaper, same-pair-*compliant*-looking-but-actually-irrelevant
        subset label before it ever had the chance to add job 1 and complete
        legally — silently returning a worse-than-optimal (but still
        same-pair-valid) column instead of the true optimum. Cuts reuse the
        identical mechanism, same reasoning, one bit per active cut.
        """
        sig = 0
        for idx, (jb, kb) in enumerate(same_bit_pairs):
            if bool(mask & jb) != bool(mask & kb):
                sig |= 1 << idx
        base = len(same_bit_pairs)
        for cidx, (bits, _dual) in enumerate(cut_specs):
            if _cut_count(mask, bits) >= 2:
                sig |= 1 << (base + cidx)
        return sig

    arcs_at: list[list[_Arc]] = [[] for _ in range(H)]
    for arc in arcs:
        for t in arc.starts:
            if 0 <= t < H:
                arcs_at[t].append(arc)

    labels: list[list[_Label]] = [[] for _ in range(H + 1)]
    srcs = {0: rho_m} if sources is None else sources
    for r, rho_r in sorted(srcs.items()):
        # one source label per release group; t_prev records the group
        labels[min(r, H)].append(
            _Label(cost=-rho_r, mask=0, pred=None, job_id=None, t_prev=r))

    # Cross-node dominance for the *same* mask, applied only to genuine
    # job-arc transitions (job_id is not None) — i.e. only when a transition
    # actually creates this mask via some ordering, never to a wait-arc
    # transition carrying an *already-counted* label forward. Reaching a
    # given mask earlier is never worse (wait arcs are free), so a prior
    # arrival (t1, mask, cost1) with t1 <= t2 and cost1 <= cost2 makes a new
    # *different-arrival* (t2, mask, cost2) useless. Restricting this to
    # job-arc transitions matters: applying it to wait transitions too was a
    # real bug during development — a label's own routine wait-propagation
    # to t+1 was being compared against the very entry it had just added for
    # itself at t, always "dominated" by its immediately-preceding self, so
    # nothing ever survived to reach later nodes at all (labels[H] came back
    # empty). Without any cross-node check at all, a label sitting idle
    # instead gets re-copied forward node-by-node via the wait arc, each
    # copy surviving the (same-node-only) bucket dominance check below since
    # no *other* label at that later node happens to share its mask yet —
    # pure waste that, combined with the per-node check alone, is what made
    # this DP effectively hang past ~12 jobs (see the module docstring's
    # "Elementary-path enforcement" section).
    best_for_mask: dict[int, list[tuple[int, float]]] = {}

    def _mask_frontier_dominates(mask: int, t2: int, cost: float) -> bool:
        return any(
            t_e <= t2 and c_e <= cost + 1e-9
            for t_e, c_e in best_for_mask.get(mask, ())
        )

    def _update_mask_frontier(mask: int, t2: int, cost: float) -> None:
        entries = best_for_mask.get(mask, [])
        best_for_mask[mask] = [
            (t_e, c_e) for t_e, c_e in entries if not (t2 <= t_e and cost <= c_e + 1e-9)
        ] + [(t2, cost)]

    # Safety cap: bounds *every* node's bucket size directly, not just the
    # total label count — a global count alone doesn't bound the cost, since
    # the expensive part (the O(bucket size) dominance scan in try_add) runs
    # on every *attempt*, including rejected ones, and a global cap checked
    # only between outer-loop nodes lets any single node's bucket (and hence
    # every scan against it, and every downstream wait-propagation copy of
    # it) grow unboundedly before the next check catches it — that's what a
    # global-count-only cap actually measured as an 18s single pricing call
    # on 17 arcs during development, capped or not. Bounding the bucket
    # itself bounds every cost that depends on it, deterministically.
    # New labels beyond the cap are silently dropped (not an error): the
    # trivial all-wait path is always added first via labels[0] and never
    # competes for room, so node H is always reached, and the DP returns
    # the best column found among what was kept — a valid, safe (never
    # wrong-direction) column if one was found, or a clean "no improving
    # column this round" if not. Same graceful-degradation precedent as
    # pareto.py's per_solve_time_limit/max_solves. See the module
    # docstring's "Elementary-path enforcement" section for why this
    # instance/dual combination (typically weight1 == 0, many similar-cost
    # incomparable job subsets) needs it at all.
    _BUCKET_CAP = bucket_cap if bucket_cap is not None else float("inf")

    def try_add(t2: int, cost: float, mask: int, pred: _Label, job_id: int | None, t_prev: int) -> None:
        if job_id is not None and _mask_frontier_dominates(mask, t2, cost):
            return
        pending = _pending_sig(mask) if (same_bit_pairs or cut_specs) else 0
        bucket = labels[t2]
        for lbl in bucket:
            # existing dominates new: same-or-lower cost, subset of new's
            # jobs, AND matching same-pair pending signature (see
            # _pending_sig's docstring for why that last condition is
            # required, not optional, whenever same_pairs is active).
            if (lbl.cost <= cost + 1e-9 and (lbl.mask & mask) == lbl.mask
                    and lbl.pending == pending):
                return
        kept = [
            lbl for lbl in bucket
            if not (cost <= lbl.cost + 1e-9 and (mask & lbl.mask) == mask
                    and pending == lbl.pending)
        ]
        if len(kept) >= _BUCKET_CAP:
            # The mask==0 (all-wait, "do nothing") label must never be
            # evicted here: it is the only entry every node is guaranteed to
            # have, and losing it at some intermediate node broke the "node
            # H is always reachable" invariant during development — labels
            # ran out entirely partway through, tripping the "unreachable
            # node H" assertion below on a perfectly feasible instance,
            # simply because 200+ cheaper (but mask != 0) labels had
            # accumulated at that node first. Sorting excludes it from the
            # cost-based truncation and re-appends it unconditionally.
            trivial = [lbl for lbl in kept if lbl.mask == 0]
            rest = [lbl for lbl in kept if lbl.mask != 0]
            rest.sort(key=lambda lbl: lbl.cost)
            kept = trivial + rest[: _BUCKET_CAP - 1 - len(trivial)]
        kept.append(_Label(cost=cost, mask=mask, pred=pred, job_id=job_id, t_prev=t_prev,
                            pending=pending))
        labels[t2] = kept
        if job_id is not None:
            _update_mask_frontier(mask, t2, cost)

    for t in range(H):
        for lbl in labels[t]:
            try_add(t + 1, lbl.cost, lbl.mask, lbl, None, t)  # wait arc, cost 0, always allowed

        for lbl in labels[t]:
            for arc in arcs_at[t]:
                bit = bit_of[arc.job_id]
                if lbl.mask & bit:
                    continue  # elementary-path: job already used on this label
                if lbl.mask & differ_of.get(arc.job_id, 0):
                    continue  # Ryan-Foster differ: partner already on this path
                if (arc.job_id, type_id, t) in forbidden:
                    continue
                pin = pinned.get(arc.job_id)
                if pin is not None and pin != (type_id, t):
                    continue
                t2 = t + arc.p_occ
                raw_cost1 = arc.weight * (t2 - arrival_of[arc.job_id])
                reduced_arc = (
                    weight1 * raw_cost1 + weight2 * arc.raw_cost2
                    - sigma.get(arc.job_id, 0.0)
                    - theta.get((arc.job_id, type_id), 0.0)
                )
                if nu1 is not None:
                    reduced_arc -= nu1 * raw_cost1
                if nu2 is not None:
                    reduced_arc -= nu2 * arc.raw_cost2
                for cidx in cut_idx_by_job.get(arc.job_id, ()):
                    bits, dual = cut_specs[cidx]
                    if _cut_count(lbl.mask, bits) == 1:
                        # this arc's job is the *second* of the triple —
                        # the unique transition where the cut's row starts
                        # counting this column; credit it exactly once.
                        reduced_arc -= dual
                try_add(t2, lbl.cost + reduced_arc, lbl.mask | bit, lbl, arc.job_id, t)

    final_labels = labels[H]
    if same_pairs:
        same_bits = [
            (bit_of[j], bit_of[k]) for j, k in same_pairs if j in bit_of and k in bit_of
        ]
        # both-feasible-here same-pairs need the runtime mask check (the
        # exclude-if-only-one-feasible case above already handled the rest);
        # mask=0 (all-wait) always passes trivially, so this never empties
        # final_labels outright.
        final_labels = [
            lbl for lbl in labels[H]
            if all(bool(lbl.mask & jb) == bool(lbl.mask & kb) for jb, kb in same_bits)
        ]

    assert final_labels, "unreachable node H — instance should be feasible"
    best_label = min(final_labels, key=lambda lbl: lbl.cost)
    best = best_label.cost
    if best >= -tol:
        return None

    jobs_seq: list[tuple[int, int]] = []
    cost1 = 0.0
    cost2 = 0.0
    lbl = best_label
    while lbl.pred is not None:
        if lbl.job_id is not None:
            completion = lbl.t_prev + inst.p_occ[lbl.job_id, type_id]
            cost1 += weight_of[lbl.job_id] * (completion - arrival_of[lbl.job_id])
            cost2 += inst.c[lbl.job_id, type_id]
            jobs_seq.append((lbl.job_id, lbl.t_prev))
        lbl = lbl.pred
    jobs_seq.reverse()
    release = lbl.t_prev   # the source label: its release group

    seen = {jid for jid, _t in jobs_seq}
    if len(seen) != len(jobs_seq):
        # Should be structurally unreachable: the mask check above rejects
        # any transition reusing a job. Kept as a cheap defensive check
        # rather than trusting that invariant silently — see the module
        # docstring's "Elementary-path enforcement" section for the incident
        # that motivated this whole label-setting rewrite in the first place.
        counts = Counter(jid for jid, _t in jobs_seq)
        dupes = sorted(jid for jid, n in counts.items() if n > 1)
        raise RuntimeError(
            f"_price_type produced a column reusing job(s) {dupes} on type {type_id} "
            f"— the label mask should have prevented this; see the comment above."
        )

    return best, Column(type_id=type_id, jobs=tuple(jobs_seq), cost1=cost1, cost2=cost2,
                        release=release)


@dataclass
class _RelaxedLabel:
    cost: float
    last_job: int | None


def _price_type_relaxed(
    inst: Instance,
    type_id: int,
    arcs: list[_Arc],
    weight1: float,
    weight2: float,
    sigma: dict[int, float],
    nu1: float | None,
    nu2: float | None,
    rho_m: float,
    theta: dict[tuple[int, int], float],
    arrival_of: dict[int, float],
    weight_of: dict[int, float],
    forbidden: frozenset[tuple[int, int, int]] = frozenset(),
    pinned: dict[int, tuple[int, int]] | None = None,
    sources: dict[int, float] | None = None,
) -> float:
    """1-cycle-free relaxation of ``_price_type``, adapted from Kramer,
    Dell'Amico, Feillet & Iori (2020)'s ``LC_2l`` (see the module docstring's
    "Comparison with Kramer et al. 2020" section). Allows a job to be reused
    non-adjacently within a path — only an *immediate* repeat (using the
    same job twice back-to-back, a "1-cycle") is forbidden — by keeping just
    two labels per node instead of one per distinct job subset: ``f1`` (best
    overall) and ``f2`` (best whose last job differs from ``f1``'s), so an
    arc for job ``j`` only ever has to fall back from ``f1`` to ``f2`` when
    ``f1``'s own last job is ``j``. O(n_m * H), no dominance scan, no
    exponential worst case — the entire point of the relaxation.

    Returns *only* the best reduced cost found, never a ``Column``: since
    the 1-cycle-free path space is a strict superset of the elementary path
    space ``_price_type`` searches, this relaxation's minimum reduced cost
    is always <= ``_price_type``'s true elementary minimum — a valid *lower
    bound* on it, never a valid *source of columns* for the RMP (a
    non-elementary path would double-count a job's coverage-row credit,
    corrupting the master exactly as documented in the module docstring's
    "Elementary-path enforcement" section for our set-*partitioning* master
    specifically — unlike Kramer et al.'s set-*covering* one, where
    over-covering is merely wasteful, not infeasible; see the "Comparison"
    section for the full argument on why this project cannot reuse
    relaxed columns directly).

    Used purely as a cheap pre-check: if this returns >= ``-tol`` (no
    improving column even in the enlarged relaxed space), ``_price_type`` is
    *guaranteed* to find nothing either for this type this iteration, and
    can be skipped entirely — same direction, and same superset argument, as
    why ``same_pairs``/``cut_duals``/``differ_pairs`` are intentionally not
    enforced here either: ignoring more constraints only enlarges the
    relaxed feasible region further, which can only make this bound looser
    (more negative, more likely to trigger a real ``_price_type`` call), but
    never *unsound* — it can never cause a real improving elementary column
    to be missed.
    """
    H = inst.horizon
    pinned = pinned or {}

    f1: list[_RelaxedLabel | None] = [None] * (H + 1)
    f2: list[_RelaxedLabel | None] = [None] * (H + 1)

    def update(t: int, cand: _RelaxedLabel) -> None:
        cur1 = f1[t]
        if cur1 is None:
            f1[t] = cand
            return
        if cand.cost < cur1.cost - 1e-9:
            if cand.last_job != cur1.last_job:
                f2[t] = cur1
            f1[t] = cand
        elif cand.last_job != cur1.last_job:
            cur2 = f2[t]
            if cur2 is None or cand.cost < cur2.cost - 1e-9:
                f2[t] = cand

    # release groups, as in _price_type's ``sources``
    for r, rho_r in sorted(({0: rho_m} if sources is None else sources).items()):
        update(min(r, H), _RelaxedLabel(cost=-rho_r, last_job=None))

    arcs_at: list[list[_Arc]] = [[] for _ in range(H)]
    for arc in arcs:
        for t in arc.starts:
            if 0 <= t < H:
                arcs_at[t].append(arc)

    for t in range(H):
        l1 = f1[t]
        if l1 is None:
            continue
        update(t + 1, _RelaxedLabel(cost=l1.cost, last_job=l1.last_job))
        l2 = f2[t]
        if l2 is not None:
            update(t + 1, _RelaxedLabel(cost=l2.cost, last_job=l2.last_job))

        for arc in arcs_at[t]:
            if (arc.job_id, type_id, t) in forbidden:
                continue
            pin = pinned.get(arc.job_id)
            if pin is not None and pin != (type_id, t):
                continue
            src = l1 if l1.last_job != arc.job_id else l2
            if src is None:
                continue  # only known predecessor here just used this job — 1-cycle
            t2 = t + arc.p_occ
            raw_cost1 = arc.weight * (t2 - arrival_of[arc.job_id])
            reduced_arc = (
                weight1 * raw_cost1 + weight2 * arc.raw_cost2
                - sigma.get(arc.job_id, 0.0)
                - theta.get((arc.job_id, type_id), 0.0)
            )
            if nu1 is not None:
                reduced_arc -= nu1 * raw_cost1
            if nu2 is not None:
                reduced_arc -= nu2 * arc.raw_cost2
            update(t2, _RelaxedLabel(cost=src.cost + reduced_arc, last_job=arc.job_id))

    candidates = [lbl.cost for lbl in (f1[H], f2[H]) if lbl is not None]
    assert candidates, "unreachable node H in relaxed pricer — instance should be feasible"
    return min(candidates)


# ---------------------------------------------------------------------------
# Compiled pricing (``_bpcprice``): ng-routes, many columns per call, and the
# Lagrangian bound
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PricingOptions:
    """How column generation prices (see "Compiled pricing" in the module
    docstring).

    ``engine`` — ``"cpp"`` (the ``_bpcprice`` label-setting DP; falls back to
    ``"python"`` when the extension is missing or subset-row cuts are active)
    or ``"python"`` (:func:`_price_type`, one column per type per iteration).
    ``ng_size`` — ng-route neighbourhood size, the job itself included;
    ``None`` prices elementary paths. ``ng_max`` — the size a neighbourhood
    may grow to when the LP keeps using a column that repeats a job.
    ``smoothing`` — Wentges dual-smoothing weight on the best-bound duals
    (0 disables). ``cols_per_type`` — columns added per type per iteration.
    ``max_labels`` — label budget per pricing call; past it the call is not
    exact and gives no bound. ``fixing`` — reduced-cost fixing at the end of
    a node's column generation (:func:`_rc_fixing`).

    Limited-memory subset-row cuts (:func:`_separate_lm_src`, compiled
    engine only): ``sr_cuts`` — most cuts in the whole search (at most 64,
    0 disables them); ``sr_rounds`` — separation rounds per node;
    ``sr_per_round`` — cuts added per round; ``sr_depth`` — separate at
    nodes up to this depth (0: root only; the cuts stay in every later
    node); ``sr_min_viol`` — least violation of a cut worth adding. Off
    by default: on the gapped Rm instances cuts raised the bound but
    slowed pricing more than they saved in nodes (see "Limited-memory
    subset-row cuts" in the module docstring).

    Defaults from ``scripts/bpc_bench.py`` (2026-11): ng 4 beat 8 and 12
    (Rm f1 + f2: 7 s vs 37 s and 50 s on rm_M2N10_3), 30 columns per type
    edged 10. Smoothing 0.5 is neutral on the base tier but cuts the
    root-bound hard Rm solves by 40-48% (rm_M2N30_1 48 -> 25 s, rm_M5N50_1
    72 -> 42 s; ``--suite hard``).
    """
    engine: str = "cpp"
    ng_size: int | None = 4
    ng_max: int = 24
    smoothing: float = 0.5
    cols_per_type: int = 30
    max_labels: int = 2_000_000
    fixing: bool = True
    sr_cuts: int = 0
    sr_rounds: int = 5
    sr_per_round: int = 16
    sr_depth: int = 0
    sr_min_viol: float = 0.05


# Used wherever no ``pricing`` argument is given (e.g. under pareto.py's
# box search); replace it to change every solve's pricing at once.
DEFAULT_PRICING = PricingOptions()


def _compiled(opts: PricingOptions | None, cuts) -> bool:
    """Whether column generation under ``opts`` runs the compiled loop (it
    needs the extension and no active subset-row cut)."""
    opts = opts or DEFAULT_PRICING
    return opts.engine == "cpp" and _bpcprice is not None and not cuts


def _is_elementary(col: Column) -> bool:
    return len({jid for jid, _t in col.jobs}) == len(col.jobs)


def _ensure_ng(prep: _Prep, mid: int, size: int) -> dict[int, set[int]]:
    """Initial neighbourhoods of the type's jobs: the ``size - 1`` jobs whose
    feasible windows (start-window centre and processing time) on ``mid``
    are closest — the jobs a short repeat cycle would be built from."""
    ng = prep.ng.get(mid)
    if ng is None:
        arcs = prep.arcs_by_type[mid]
        key = {a.job_id: ((min(a.starts) + max(a.starts) + a.p_occ) / 2.0, a.p_occ) for a in arcs}
        ng = {}
        for a in arcs:
            c, p = key[a.job_id]
            others = sorted((abs(c - c2) + abs(p - p2), j2)
                            for j2, (c2, p2) in key.items() if j2 != a.job_id)
            ng[a.job_id] = {j2 for _d, j2 in others[:max(0, size - 1)]}
        prep.ng[mid] = ng
    return ng


def _augment_ng(prep: _Prep, cols: list[Column], ng_max: int) -> bool:
    """Grow neighbourhoods so the repeat cycles of ``cols`` become infeasible
    (dynamic ng-routes, Roberti & Mingozzi 2014): for a job ``j`` visited
    at positions ``a < b``, ``j`` joins the neighbourhood of every job in
    between. Returns whether any neighbourhood grew."""
    grew = False
    for col in cols:
        ng = prep.ng.get(col.type_id)
        if ng is None:
            continue
        seq = [jid for jid, _t in col.jobs]
        last: dict[int, int] = {}
        for b, j in enumerate(seq):
            a = last.get(j)
            if a is not None:
                for i in seq[a + 1:b]:
                    nb = ng.setdefault(i, set())
                    if j != i and j not in nb and len(nb) + 1 < ng_max:
                        nb.add(j)
                        grew = True
            last[j] = b
    return grew


@dataclass(frozen=True)
class _LmCut:
    """A limited-memory subset-row cut on three jobs (Pecin et al. 2017):
    ``sum_p a_p lambda_p <= 1`` with ``a_p`` from :func:`_lm_coef`.

    For ``|S| = 3`` and memory ``M = all jobs`` this is Jepsen et al.'s
    SR cut, ``a_p = floor(|S ∩ p| / 2)``: an integer solution covers each
    job once, so at most one of its columns holds two of ``S``. A smaller
    ``M`` only lowers coefficients, so the cut stays valid, and its pricing
    state is forgotten on any job outside ``M`` — the DP keeps more
    dominance. Columns that repeat a job (ng-routes) just get some
    nonnegative coefficient; they are never part of an integer solution.
    """
    S: frozenset[int]
    M: frozenset[int]


def _lm_coef(cut: _LmCut, jobs) -> int:
    """Coefficient of a column visiting ``jobs`` (in order) in ``cut``,
    by the rule the compiled pricer charges: a job outside ``M`` resets the
    state; a job of ``S`` pays one if the state is set (and clears it),
    else sets it."""
    state, n = 0, 0
    for jid, *_rest in jobs:
        if jid in cut.S:
            if state:
                n += 1
            state ^= 1
        elif jid not in cut.M:
            state = 0
    return n


@dataclass
class _Duals:
    """One dual vector of the RMP's rows. ``beta`` are the duals (<= 0) of
    the limited-memory subset-row cuts ``cuts``, in that order."""
    sigma: dict[int, float]
    rho: dict[tuple[int, int], float]
    nu1: float | None
    nu2: float | None
    theta: dict[tuple[int, int], float]
    eta: dict[int, float] = field(default_factory=dict)
    cuts: tuple[_LmCut, ...] = ()
    beta: tuple[float, ...] = ()

    def mix(self, other: "_Duals", a: float) -> "_Duals":
        """``a * self + (1 - a) * other`` (both over the same cuts)."""
        def lin(x, y):
            return None if x is None else a * x + (1 - a) * y
        return _Duals(
            sigma={k: a * v + (1 - a) * other.sigma[k] for k, v in self.sigma.items()},
            rho={k: a * v + (1 - a) * other.rho[k] for k, v in self.rho.items()},
            nu1=lin(self.nu1, other.nu1), nu2=lin(self.nu2, other.nu2),
            theta={k: a * v + (1 - a) * other.theta[k] for k, v in self.theta.items()},
            eta={k: a * v + (1 - a) * other.eta[k] for k, v in self.eta.items()},
            cuts=self.cuts,
            beta=tuple(a * x + (1 - a) * y for x, y in zip(self.beta, other.beta)),
        )


def _read_duals(state: "_RmpState") -> _Duals:
    return _Duals(
        sigma={jid: c.Pi for jid, c in state.coverage.items()},
        rho={key: c.Pi for key, c in state.fleet.items()},
        nu1=state.bound1.Pi if state.bound1 is not None else None,
        nu2=state.bound2.Pi if state.bound2 is not None else None,
        theta={key: c.Pi for key, c in state.fixed_link.items()},
        eta={m: c.Pi for m, c in state.activation.items()},
        cuts=tuple(cut for cut, _row in state.lm_cuts),
        beta=tuple(min(0.0, row.Pi) for _cut, row in state.lm_cuts),
    )


def _column_rc(col: Column, d: _Duals, w1: float, w2: float) -> float:
    """Reduced cost of ``col`` at duals ``d`` (visit counts as coefficients)."""
    rc = (w1 - (d.nu1 or 0.0)) * col.cost1 + (w2 - (d.nu2 or 0.0)) * col.cost2
    for jid, _t in col.jobs:
        rc -= d.sigma[jid] + d.theta.get((jid, col.type_id), 0.0)
    if col.jobs:
        rc -= d.eta.get(col.type_id, 0.0)
        for cut, b in zip(d.cuts, d.beta):
            if b:
                rc -= b * _lm_coef(cut, col.jobs)
    return rc - d.rho.get((col.type_id, col.release), 0.0)


class _NodePricing:
    """The compiled pricers of one B&B node, one per path type. The node's
    ``forbidden``/``pinned`` decisions are applied by filtering start sets,
    Ryan-Foster pairs inside the DP."""

    def __init__(self, prep: _Prep, opts: PricingOptions,
                 forbidden: frozenset, pinned: dict,
                 differ_pairs: frozenset, same_pairs: frozenset):
        self.prep, self.opts = prep, opts
        self.types: dict[int, tuple] = {}
        H = prep.inst.horizon
        for mid in prep.path_types:
            kept = []
            for arc in prep.arcs_by_type[mid]:
                pin = pinned.get(arc.job_id)
                if pin is not None:
                    ts = [pin[1]] if pin[0] == mid and pin[1] in arc.starts else []
                else:
                    ts = sorted(arc.starts)
                if forbidden:
                    ts = [t for t in ts if (arc.job_id, mid, t) not in forbidden]
                if ts:
                    kept.append((arc, ts))
            here = {arc.job_id for arc, _ts in kept}
            # a same pair with one job impossible here: the other can't come either
            excl = {k for j, k in same_pairs if j not in here} | {j for j, k in same_pairs if k not in here}
            kept = [(arc, ts) for arc, ts in kept if arc.job_id not in excl]
            local = {arc.job_id: i for i, (arc, _ts) in enumerate(kept)}
            differ = [(local[j], local[k]) for j, k in differ_pairs if j in local and k in local]
            same = [(local[j], local[k]) for j, k in same_pairs if j in local and k in local]
            pricer = _bpcprice.Pricer(H, [arc.p_occ for arc, _ts in kept],
                                      [ts for _arc, ts in kept], differ, same)
            self.types[mid] = (pricer, [arc for arc, _ts in kept], local)
        self.cut_ids: dict[int, list[int]] = {mid: [] for mid in self.types}
        self.refresh_memory()

    def set_cuts(self, cuts) -> None:
        """Load the limited-memory subset-row cuts into every type's pricer;
        a type only sees the cuts with two or more of their jobs on it."""
        for mid, (pricer, _arcs, local) in self.types.items():
            ids, spec = [], []
            for c, cut in enumerate(cuts):
                S = [local[j] for j in cut.S if j in local]
                if len(S) >= 2:
                    ids.append(c)
                    spec.append((S, [local[j] for j in cut.M if j in local]))
            pricer.set_cuts(spec)
            self.cut_ids[mid] = ids

    def refresh_memory(self) -> None:
        for mid, (pricer, arcs, local) in self.types.items():
            if self.opts.ng_size is None:
                pricer.set_memory(None)
                continue
            ng = _ensure_ng(self.prep, mid, self.opts.ng_size)
            pricer.set_memory([[local[x] for x in ng.get(arc.job_id, ()) if x in local]
                               for arc in arcs])

    def _inputs(self, mid: int, d: _Duals, w1: float, w2: float):
        """Arc coefficients: an arc of job j started at t has reduced cost
        ``A_j + B_j t``; sources are the release groups' ``-rho``."""
        _pricer, arcs, _local = self.types[mid]
        prep = self.prep
        c1 = w1 - (d.nu1 or 0.0)
        c2 = w2 - (d.nu2 or 0.0)
        A, B = [], []
        for arc in arcs:
            jid = arc.job_id
            b = c1 * arc.weight
            A.append(b * (arc.p_occ - prep.arrival_of[jid]) + c2 * arc.raw_cost2
                     - d.sigma[jid] - d.theta.get((jid, mid), 0.0))
            B.append(b)
        sources = [(r, -d.rho[mid, r]) for r in sorted(prep.releases[mid])]
        return A, B, sources

    def fixable(self, mid: int, d: _Duals, w1: float, w2: float, threshold: float
                ) -> list[tuple[int, int]]:
        """``(job, start)`` arcs of ``mid`` every column through which has
        reduced cost >= ``threshold`` at ``d``."""
        pricer, arcs, _local = self.types[mid]
        A, B, sources = self._inputs(mid, d, w1, w2)
        return [(arcs[i].job_id, t) for i, t in pricer.fixable(A, B, sources, threshold)]

    def price(self, mid: int, d: _Duals, w1: float, w2: float, tol: float
              ) -> tuple[float, bool, int, list[Column]]:
        """``(min_lb, exact, n_labels, columns)``: ``min_lb`` lower-bounds the
        type's minimum reduced cost over ng-paths (``-inf`` when not exact)."""
        pricer, arcs, _local = self.types[mid]
        prep = self.prep
        A, B, sources = self._inputs(mid, d, w1, w2)
        min_lb, exact, n_labels, raw = pricer.price(
            A, B, sources, self.opts.cols_per_type, tol, self.opts.max_labels,
            max(0.0, -d.eta.get(mid, 0.0)),
            [max(0.0, -d.beta[c]) for c in self.cut_ids[mid]])
        cols = []
        for _rc, release, path in raw:
            jobs = tuple((arcs[i].job_id, t) for i, t in path)
            cost1 = sum(arcs[i].weight * (t + arcs[i].p_occ - prep.arrival_of[arcs[i].job_id])
                        for i, t in path)
            cost2 = sum(arcs[i].raw_cost2 for i, _t in path)
            cols.append(Column(type_id=mid, jobs=jobs, cost1=cost1, cost2=cost2,
                               release=release))
        return min_lb, exact, n_labels, cols


def _lagrangian_bound(
    prep: _Prep, state: "_RmpState", d: _Duals, mins: dict[int, float],
    w1: float, w2: float, big_m: float, statics_by_job: dict[int, list[Column]],
    fixed_cost: dict[int, float],
) -> float:
    """Lagrangian bound ``L(d)`` of the node's master LP, valid for *any*
    sign-feasible duals ``d`` (``nu``, ``theta`` <= 0), hence for smoothed
    ones: the dual objective plus, for every variable, its least possible
    reduced-cost contribution (cut duals included). Each release group ``(m, r)`` picks
    ``N_{m,r}`` paths, each no cheaper than ``mins[m]`` (the type's minimum
    reduced cost over every group); job ``j``'s artificial and static
    columns together have value at most 1; ``z_m`` lies in its bounds."""
    L = sum(d.sigma.values())
    for (mid, r), rho in d.rho.items():
        L += prep.releases[mid][r] * rho
    if d.nu1 is not None:
        L += d.nu1 * state.bound1.RHS
    if d.nu2 is not None:
        L += d.nu2 * state.bound2.RHS
    L += sum(d.beta)   # subset-row cuts, right-hand side 1
    for mid in prep.path_types:
        L += sum(prep.releases[mid].values()) * mins[mid]
    for jid, sig in d.sigma.items():
        best = big_m - sig
        for col in statics_by_job.get(jid, ()):
            best = min(best, _column_rc(col, d, w1, w2))
        L += min(0.0, best)
    for mid, zvar in state.z.items():
        U = fixed_cost[mid]
        rc = w2 * U - (d.nu2 or 0.0) * U + sum(
            th for (_jid, m2), th in d.theta.items() if m2 == mid)
        if mid in d.eta:
            rc += sum(prep.releases[mid].values()) * d.eta[mid]
        L += rc * (zvar.LB if rc >= 0 else zvar.UB)
    return L


def _objective_step(inst: Instance, w1: float, w2: float) -> float:
    """Smallest positive difference between two schedules' ``w1·f1 + w2·f2``
    when the objective lies on a grid, else 0. f1 moves in multiples of
    ``gcd(w_j)`` (integral weights, integer completions), f2 in multiples of
    ``δ·gcd(cost units)``."""
    def gcd_all(vals) -> int:
        g = 0
        for v in vals:
            g = math.gcd(g, int(v))
        return g
    s1 = gcd_all(j.weight for j in inst.jobs)
    delta = inst.cost_resolution
    s2 = gcd_all([*inst.c_units.values(),
                  *(round(m.fixed_cost / delta) for m in inst.instance_types)]) * delta
    a, b = w1 * s1, w2 * s2
    if w2 == 0 or b == 0:
        return a
    if w1 == 0 or a == 0:
        return b
    lo, hi = min(a, b), max(a, b)
    ratio = hi / lo
    return lo if abs(ratio - round(ratio)) < 1e-9 else 0.0


# ---------------------------------------------------------------------------
# Restricted master problem
# ---------------------------------------------------------------------------

@dataclass
class _RmpState:
    coverage: dict[int, gp.Constr]
    fleet: dict[tuple[int, int], gp.Constr]   # (path type, release slot) -> row
    bound1: gp.Constr | None   # optional f1-cap row (pareto.py's box boundary / lex-min tie-break)
    bound2: gp.Constr | None   # optional f2-cap row — both may be active simultaneously
    art: dict[int, gp.Var]
    fixed_link: dict[tuple[int, int], gp.Constr] = field(default_factory=dict)
    # Activation rows of fixed-cost path types (compiled engine only):
    # sum of non-null columns on m <= N_m z_m — see _build_rmp.
    activation: dict[int, gp.Constr] = field(default_factory=dict)
    z: dict[int, gp.Var] = field(default_factory=dict)
    cuts: dict[frozenset[int], gp.Constr] = field(default_factory=dict)
    # limited-memory subset-row cuts (compiled engine), see _LmCut
    lm_cuts: list[tuple[_LmCut, gp.Constr]] = field(default_factory=list)
    columns: list[Column] = field(default_factory=list)
    vars: list[gp.Var] = field(default_factory=list)
    unbounded: frozenset[int] = frozenset()   # types with no fleet row


def _add_column_var(
    mdl: gp.Model,
    col: Column,
    weight1: float,
    weight2: float,
    state: _RmpState,
    name: str,
) -> gp.Var:
    fleet = state.fleet.get((col.type_id, col.release))
    if fleet is None and col.type_id not in state.unbounded:
        raise ValueError(f"column on type {col.type_id} from release {col.release}: "
                         "no such fleet row")
    coeffs = [1.0] if fleet is not None else []
    constrs: list[gp.Constr] = [fleet] if fleet is not None else []
    # An ng-route column may visit a job twice: its coefficient is the visit
    # count (it can then never be part of an integer solution).
    for job_id, n in Counter(jid for jid, _start in col.jobs).items():
        coeffs.append(float(n))
        constrs.append(state.coverage[job_id])
        link = state.fixed_link.get((job_id, col.type_id))
        if link is not None:
            coeffs.append(float(n))
            constrs.append(link)
    act = state.activation.get(col.type_id)
    if act is not None and col.jobs:
        coeffs.append(1.0)
        constrs.append(act)
    if state.bound1 is not None:
        coeffs.append(col.cost1)
        constrs.append(state.bound1)
    if state.bound2 is not None:
        coeffs.append(col.cost2)
        constrs.append(state.bound2)
    if state.cuts:
        job_set = {jid for jid, _t in col.jobs}
        for triple, constr in state.cuts.items():
            if len(job_set & triple) >= 2:
                coeffs.append(1.0)
                constrs.append(constr)
    if state.lm_cuts and len(col.jobs) >= 2:
        for cut, constr in state.lm_cuts:
            a = _lm_coef(cut, col.jobs)
            if a:
                coeffs.append(float(a))
                constrs.append(constr)

    obj = weight1 * col.cost1 + weight2 * col.cost2
    return mdl.addVar(
        obj=obj, lb=0.0, column=gp.Column(coeffs, constrs), name=name,
    )


def _column_allowed(
    col: Column,
    forbidden: frozenset[tuple[int, int, int]],
    pinned: dict[int, tuple[int, int]],
    differ_pairs: frozenset[tuple[int, int]] = frozenset(),
    same_pairs: frozenset[tuple[int, int]] = frozenset(),
) -> bool:
    """Whether ``col`` survives a node's branching decisions (see _price_type)."""
    job_set = {jid for jid, _t in col.jobs}
    for job_id, start in col.jobs:
        if (job_id, col.type_id, start) in forbidden:
            return False
        pin = pinned.get(job_id)
        if pin is not None and pin != (col.type_id, start):
            return False
    for j, k in differ_pairs:
        if j in job_set and k in job_set:
            return False
    for j, k in same_pairs:
        if (j in job_set) != (k in job_set):
            return False
    return True


def _build_rmp(
    inst: Instance,
    weight1: float,
    weight2: float,
    bound1_value: float | None,
    bound2_value: float | None,
    big_m: float,
    extra_columns: list[Column] = (),
    forbidden_types: frozenset[int] = frozenset(),
    forced_types: frozenset[int] = frozenset(),
    cuts: frozenset[frozenset[int]] = frozenset(),
    prep: _Prep | None = None,
    activation: bool = False,
    lm_cuts: tuple[_LmCut, ...] = (),
) -> tuple[gp.Model, _RmpState]:
    """The restricted master. ``activation`` adds, per fixed-cost path type
    ``m`` with ``N_m`` instances, ``sum_{p non-null on m} lambda_p <= N_m z_m``
    (dual ``eta_m``): valid, and with ``N_m = 1`` (e.g. Rm machines) it puts
    ``U_m`` into the column cost as Chen et al.'s master does, which the
    per-job rows alone do not. Only the compiled pricing prices its dual."""
    prep = prep or _prepare(inst)
    formulations.configure_env()
    mdl = gp.Model("bpc-rmp", env=formulations._shared_env)
    mdl.setParam("OutputFlag", 0)

    coverage = {
        j.id: mdl.addConstr(gp.LinExpr() == 1.0, name=f"cover[{j.id}]")
        for j in inst.jobs
    }
    # One fleet row per (path type, release group): exactly n instance
    # timelines start at r. Unbounded types have none (see _Prep).
    fleet = {
        (mid, r): mdl.addConstr(gp.LinExpr() == float(n), name=f"fleet[{mid},{r}]")
        for mid in prep.path_types for r, n in sorted(prep.releases[mid].items())
    }
    bound1 = mdl.addConstr(gp.LinExpr() <= bound1_value, name="bound1") if bound1_value is not None else None
    bound2 = mdl.addConstr(gp.LinExpr() <= bound2_value, name="bound2") if bound2_value is not None else None

    # Rank-1 subset-row cuts (Phase 5 — see the module docstring and
    # _separate_sr_cuts): one row per active job-triple, "sum_{p: |p.jobs &
    # triple| >= 2} lambda_p <= 1" — valid because any integer solution
    # covers each job exactly once, so at most one selected column can cover
    # 2+ of any 3 jobs. Decided once at the root and held fixed for the rest
    # of the tree (see _branch_and_price) — not branching state, so (unlike
    # forbidden_types/forced_types) there is no "cuts_value" per node, just
    # this one fixed set threaded straight through.
    cut_rows: dict[frozenset[int], gp.Constr] = {
        triple: mdl.addConstr(gp.LinExpr() <= 1.0, name=f"sr[{sorted(triple)}]")
        for triple in cuts
    }

    mdl.update()

    art = {
        j.id: mdl.addVar(obj=big_m, lb=0.0, column=gp.Column([1.0], [coverage[j.id]]),
                          name=f"art[{j.id}]")
        for j in inst.jobs
    }

    # Fixed-cost linking rows, per (job, type) pair — mirrors
    # add_fixed_cost_vars's "sum_t x[j,m,t] <= z_m" exactly (disaggregated per
    # job, not one aggregate row, to keep the LP bound as tight as the compact
    # model's). Rows first (empty), then z_m with coefficient -1 on each of
    # its type's rows, so real columns only ever need to add +1 per job later.
    fixed_link: dict[tuple[int, int], gp.Constr] = {}
    for itype in _fixed_cost_types(inst):
        for j in inst.jobs:
            if itype.id in inst.F[j.id]:
                fixed_link[j.id, itype.id] = mdl.addConstr(
                    gp.LinExpr() <= 0.0, name=f"use[{j.id},{itype.id}]"
                )
    activation_rows: dict[int, gp.Constr] = {}
    if activation:
        for itype in _fixed_cost_types(inst):
            if itype.id in prep.releases:
                activation_rows[itype.id] = mdl.addConstr(
                    gp.LinExpr() <= 0.0, name=f"act[{itype.id}]")
    mdl.update()

    z: dict[int, gp.Var] = {}
    for itype in _fixed_cost_types(inst):
        rows = [c for (jid, mid), c in fixed_link.items() if mid == itype.id]
        coeffs = [-1.0] * len(rows)
        constrs = list(rows)
        if itype.id in activation_rows:
            coeffs.append(-float(sum(prep.releases[itype.id].values())))
            constrs.append(activation_rows[itype.id])
        # z_m's own contribution to the f2-cap bound row when one is active
        # (matches the compact model's f2_expr — z_m included, see
        # formulations.add_fixed_cost_vars). Missing this was a real bug
        # found during Phase 4 validation: with no coefficient here, z_m's
        # fixed cost only ever showed up in the *objective*, so the bound row
        # silently undercounted f2 by exactly U_m whenever a fixed-cost type
        # got activated. z_m contributes nothing to f1, so bound1 never
        # needs it.
        if bound2 is not None:
            coeffs.append(itype.fixed_cost)
            constrs.append(bound2)
        # forbidden_types/forced_types: Ryan-Foster-style branching on z_m
        # itself (see _solve_node) — z_m is not guaranteed integral by the
        # rest of the master's structure the way job placements are
        # (Geoffrion's property covers the coverage/fleet-size rows, not
        # this bound-row/objective-only variable), so it needs its own
        # fix/forbid mechanism once found fractional. ub=0 forces z_m=0,
        # which (via the linking rows above) also forces every job-carrying
        # column's lambda on this type to 0 — no separate arc filtering
        # needed. lb=1 forces z_m=1 (its ub is already 1) without
        # constraining which job(s) trigger it.
        lb = 1.0 if itype.id in forced_types else 0.0
        ub = 0.0 if itype.id in forbidden_types else 1.0
        z[itype.id] = mdl.addVar(
            lb=lb, ub=ub, obj=weight2 * itype.fixed_cost,
            column=gp.Column(coeffs, constrs),
            name=f"z[{itype.id}]",
        )

    state = _RmpState(coverage=coverage, fleet=fleet, bound1=bound1, bound2=bound2,
                       art=art, fixed_link=fixed_link, z=z, cuts=cut_rows,
                       activation=activation_rows,
                       lm_cuts=[(cut, mdl.addConstr(gp.LinExpr() <= 1.0, name=f"lmsr[{c}]"))
                                for c, cut in enumerate(lm_cuts)],
                       unbounded=prep.unbounded)
    mdl.update()

    for mid, r in fleet:
        null_col = Column(type_id=mid, jobs=(), cost1=0.0, cost2=0.0, release=r)
        var = _add_column_var(mdl, null_col, weight1, weight2, state,
                               name=f"lam[{mid},{r}]_null")
        state.columns.append(null_col)
        state.vars.append(var)

    for i, col in enumerate(extra_columns):
        if not col.jobs:
            continue   # null columns are added above
        var = _add_column_var(mdl, col, weight1, weight2, state,
                               name=f"lam[{col.type_id}]_seed{i}")
        state.columns.append(col)
        state.vars.append(var)

    mdl.update()
    return mdl, state


class _RmpInfeasible(Exception):
    """Raised when an RMP LP is infeasible outright (e.g. a bound row
    too tight for even the all-idle solution) — distinct from the ordinary
    "some job not yet covered" case, which the big-M artificials absorb
    instead of ever making the LP infeasible.
    """


@dataclass
class _CGResult:
    """Outcome of one node's column generation.

    ``status``: ``"converged"`` (no improving column; ``objective`` is the
    RMP value), ``"bound"`` (the Lagrangian bound met the RMP value first;
    ``objective`` is that bound) or ``"pruned"`` (the bound reached the
    cutoff; the RMP solution is stale and must not be read).
    """
    iterations: int
    status: str
    objective: float
    lagrangian: float = -math.inf
    fixed: frozenset = frozenset()   # (job, type, start) triples, see _rc_fixing


def _run_column_generation(
    mdl: gp.Model,
    state: _RmpState,
    inst: Instance,
    weight1: float,
    weight2: float,
    prep: _Prep,
    max_iters: int,
    tol: float,
    forbidden: frozenset[tuple[int, int, int]] = frozenset(),
    pinned: dict[int, tuple[int, int]] | None = None,
    differ_pairs: frozenset[tuple[int, int]] = frozenset(),
    same_pairs: frozenset[tuple[int, int]] = frozenset(),
    use_relaxed_prescreen: bool = False,
    deadline: float | None = None,
    timers: dict | None = None,
    *,
    opts: PricingOptions | None = None,
    cutoff: float = math.inf,
    big_m: float | None = None,
) -> _CGResult:
    """Column generation at one node (or the root): the compiled loop
    (:func:`_run_cg_compiled`) when ``opts.engine == "cpp"``, the extension
    is built and no subset-row cut is active, else :func:`_run_cg_python`."""
    opts = opts or DEFAULT_PRICING
    if (state.activation or state.lm_cuts) and not _compiled(opts, state.cuts):
        raise ValueError("activation rows and limited-memory cuts need the compiled pricing")
    if _compiled(opts, state.cuts):
        return _run_cg_compiled(
            mdl, state, inst, weight1, weight2, prep, max_iters, tol,
            forbidden, pinned or {}, differ_pairs, same_pairs, deadline, timers,
            opts, cutoff, big_m if big_m is not None else _big_m_for(inst))
    it = _run_cg_python(mdl, state, inst, weight1, weight2, prep, max_iters, tol,
                        forbidden, pinned, differ_pairs, same_pairs,
                        use_relaxed_prescreen, deadline, timers)
    return _CGResult(it, "converged", mdl.ObjVal)


def _run_cg_compiled(
    mdl: gp.Model, state: _RmpState, inst: Instance, weight1: float, weight2: float,
    prep: _Prep, max_iters: int, tol: float,
    forbidden: frozenset, pinned: dict, differ_pairs: frozenset, same_pairs: frozenset,
    deadline: float | None, timers: dict | None, opts: PricingOptions,
    cutoff: float, big_m: float,
) -> _CGResult:
    """Column generation with the compiled pricer.

    Each iteration solves the RMP and prices every path type, adding up to
    ``opts.cols_per_type`` columns per type. With ``opts.smoothing = a > 0``
    pricing first runs at the Wentges duals ``a·center + (1−a)·duals``
    (``center``: the duals of the best Lagrangian bound so far); if none of
    those columns prices out at the RMP duals (a mispricing), it reprices at
    the RMP duals, so convergence is unchanged.

    Every exact pricing round gives a Lagrangian bound
    (:func:`_lagrangian_bound`). The node stops early when the best bound
    reaches ``cutoff`` (``"pruned"``) or the RMP value (``"bound"``: the RMP
    solution is optimal up to the gap).

    ng-routes: when the converged LP uses a column that repeats a job, the
    neighbourhoods grow to forbid that cycle (:func:`_augment_ng`), every
    repeating column leaves the RMP, and generation resumes; it stops when
    the LP is elementary or no neighbourhood can grow.
    """
    if timers is None:
        timers = {}
    pricing = _NodePricing(prep, opts, forbidden, pinned, differ_pairs, same_pairs)
    if state.lm_cuts:
        pricing.set_cuts([cut for cut, _row in state.lm_cuts])
    statics: dict[int, list[Column]] = {}
    for col in prep.static_columns:
        if _column_allowed(col, forbidden, pinned, differ_pairs, same_pairs):
            statics.setdefault(col.jobs[0][0], []).append(col)
    fixed_cost = {m.id: m.fixed_cost for m in inst.instance_types}
    keys = {_column_key(c) for c in state.columns}
    best_lb, center, best_mins = -math.inf, None, None

    def finish(it: int, status: str, objective: float) -> _CGResult:
        fixed = frozenset()
        if (opts.fixing and center is not None and math.isfinite(cutoff)
                and math.isfinite(best_lb)):
            t0 = time.perf_counter()
            fixed = _rc_fixing(pricing, prep, center, best_mins, best_lb, cutoff,
                               weight1, weight2, big_m, statics)
            timers["fixing_s"] = timers.get("fixing_s", 0.0) + time.perf_counter() - t0
            timers["fixed"] = timers.get("fixed", 0) + len(fixed)
        return _CGResult(it, status, objective, best_lb, fixed)

    def price_all(d: _Duals) -> tuple[float | None, list[Column], dict]:
        t0 = time.perf_counter()
        mins, cols, exact = {}, [], True
        for mid in prep.path_types:
            mn, ex, n_labels, cs = pricing.price(mid, d, weight1, weight2, tol)
            mins[mid] = mn
            exact &= ex
            cols += cs
            timers["labels"] = max(timers.get("labels", 0), n_labels)
        lag = (_lagrangian_bound(prep, state, d, mins, weight1, weight2, big_m,
                                 statics, fixed_cost) if exact else None)
        timers["pricing_s"] = timers.get("pricing_s", 0.0) + time.perf_counter() - t0
        if not exact:
            timers["inexact_pricing"] = timers.get("inexact_pricing", 0) + 1
        return lag, cols, mins

    for it in range(max_iters):
        _check_deadline(deadline)
        t_lp = time.perf_counter()
        mdl.optimize()
        timers["lp_s"] = timers.get("lp_s", 0.0) + time.perf_counter() - t_lp
        timers["cg_iters"] = timers.get("cg_iters", 0) + 1
        if mdl.Status == GRB.INFEASIBLE:
            raise _RmpInfeasible
        if mdl.Status != GRB.OPTIMAL:
            raise RuntimeError(f"BPC RMP LP not optimal: status={mdl.Status}")
        rmp = mdl.ObjVal
        d = _read_duals(state)

        new: list[Column] = []
        if center is not None and opts.smoothing > 0:
            sd = center.mix(d, opts.smoothing)
            lag, cols, mins = price_all(sd)
            if lag is not None and lag > best_lb:
                best_lb, center, best_mins = lag, sd, mins
            new = [c for c in cols if _column_key(c) not in keys
                   and _column_rc(c, d, weight1, weight2) < -tol]
        if not new:
            lag, cols, mins = price_all(d)
            if lag is not None and lag > best_lb:
                best_lb, center, best_mins = lag, d, mins
            new = [c for c in cols if _column_key(c) not in keys]
            if not new and lag is None:
                raise RuntimeError("BPC pricing hit its label budget (max_labels) "
                                   "without an improving column")
        if best_lb >= cutoff:
            timers["lb_pruned"] = timers.get("lb_pruned", 0) + 1
            return _CGResult(it + 1, "pruned", best_lb, best_lb)
        done = not new
        if not done and best_lb >= rmp - max(tol, 1e-7 * abs(rmp)):
            timers["lb_stops"] = timers.get("lb_stops", 0) + 1
            done = True
        if done:
            if opts.ng_size is not None and _grow_ng_and_drop(mdl, state, prep, opts):
                pricing.refresh_memory()
                keys = {_column_key(c) for c in state.columns}
                timers["ng_rounds"] = timers.get("ng_rounds", 0) + 1
                continue
            if new:   # stopped by the bound: report the bound, not the RMP value
                return finish(it + 1, "bound", min(best_lb, rmp))
            return finish(it + 1, "converged", rmp)
        for col in new:
            keys.add(_column_key(col))
            state.vars.append(_add_column_var(mdl, col, weight1, weight2, state,
                                              name=f"lam[{col.type_id}]_{len(state.columns)}"))
            state.columns.append(col)
    raise RuntimeError("BPC column generation did not converge within max_iters")


def _rc_fixing(
    pricing: _NodePricing, prep: _Prep, d: _Duals, mins: dict[int, float], L: float,
    cutoff: float, w1: float, w2: float, big_m: float,
    statics: dict[int, list[Column]],
) -> frozenset:
    """Reduced-cost fixing (e.g. Pessoa et al. 2010 for arc-time-indexed
    scheduling): ``(job, type, start)`` placements no solution cheaper than
    ``cutoff`` can use, given a Lagrangian bound ``L`` at duals ``d``.

    ``L`` charges each instance of a path type ``m`` its minimum reduced cost
    ``mins[m]``, so a solution with a column ``p`` on ``m`` costs at least
    ``L − mins[m] + rc_p``; an arc every column through which has
    ``rc_p >= cutoff − L + mins[m]`` is unusable (``_NodePricing.fixable``
    gives a lower bound on such ``rc_p``). Likewise a static column whose
    reduced cost exceeds its job's term in ``L`` by ``cutoff − L``. Valid in
    the node's whole subtree, since ``cutoff`` only decreases."""
    gap = cutoff - L
    safety = 1e-6 * max(1.0, abs(cutoff))
    out: set[tuple[int, int, int]] = set()
    for mid in prep.path_types:
        for jid, t in pricing.fixable(mid, d, w1, w2, gap + mins[mid] + safety):
            out.add((jid, mid, t))
    for jid, cols in statics.items():
        rcs = [_column_rc(c, d, w1, w2) for c in cols]
        term = min(0.0, big_m - d.sigma[jid], *rcs)
        for c, rc in zip(cols, rcs):
            if rc - term >= gap + safety:
                out.add((jid, c.type_id, c.jobs[0][1]))
    return frozenset(out)


def _grow_ng_and_drop(mdl: gp.Model, state: _RmpState, prep: _Prep,
                      opts: PricingOptions) -> bool:
    """After convergence: if the LP uses repeating columns, grow the
    neighbourhoods against their cycles and remove every repeating column
    from the RMP. Returns whether generation must resume."""
    used = [c for c, v in zip(state.columns, state.vars) if v.X > 1e-9 and not _is_elementary(c)]
    if not used or not _augment_ng(prep, used, opts.ng_max):
        return False
    keep_cols, keep_vars = [], []
    for c, v in zip(state.columns, state.vars):
        if _is_elementary(c):
            keep_cols.append(c)
            keep_vars.append(v)
        else:
            mdl.remove(v)
    state.columns[:], state.vars[:] = keep_cols, keep_vars
    mdl.update()
    return True


def _run_cg_python(
    mdl: gp.Model,
    state: _RmpState,
    inst: Instance,
    weight1: float,
    weight2: float,
    prep: _Prep,
    max_iters: int,
    tol: float,
    forbidden: frozenset[tuple[int, int, int]] = frozenset(),
    pinned: dict[int, tuple[int, int]] | None = None,
    differ_pairs: frozenset[tuple[int, int]] = frozenset(),
    same_pairs: frozenset[tuple[int, int]] = frozenset(),
    use_relaxed_prescreen: bool = False,
    deadline: float | None = None,
    timers: dict | None = None,
) -> int:
    """Solve -> price every type -> add improving columns -> repeat, until no
    type prices out a negative-reduced-cost column. Returns the iteration
    count. The pricing is :func:`_price_type`, one column per type.

    ``deadline`` — a ``time.perf_counter()`` value; checked once per
    iteration, raising :class:`BPCTimeLimit` once passed.

    ``use_relaxed_prescreen`` — before calling the exact elementary
    ``_price_type`` for a type, first call the much cheaper 1-cycle-free
    ``_price_type_relaxed`` (see its docstring and the module docstring's
    "Comparison with Kramer et al. 2020" section); if the relaxed check
    already finds nothing (>= ``-tol``), the exact call is skipped entirely
    for this type this iteration — provably safe (the relaxed minimum is
    always <= the exact one), never changes what columns get added, only
    how much exact-DP work is spent finding out none exist.
    """
    for it in range(max_iters):
        _check_deadline(deadline)
        t_lp = time.perf_counter()
        mdl.optimize()
        if timers is not None:
            timers["lp_s"] = timers.get("lp_s", 0.0) + time.perf_counter() - t_lp
        if mdl.Status == GRB.INFEASIBLE:
            raise _RmpInfeasible
        if mdl.Status != GRB.OPTIMAL:
            raise RuntimeError(f"BPC RMP LP not optimal: status={mdl.Status}")

        sigma = {jid: c.Pi for jid, c in state.coverage.items()}
        rho = {key: c.Pi for key, c in state.fleet.items()}
        nu1 = state.bound1.Pi if state.bound1 is not None else None
        nu2 = state.bound2.Pi if state.bound2 is not None else None
        theta = {key: c.Pi for key, c in state.fixed_link.items()}
        cut_duals = {triple: c.Pi for triple, c in state.cuts.items()}

        improved = False
        t_price = time.perf_counter()
        for mid in prep.path_types:   # unbounded types: all columns are static
            sources = {r: rho[mid, r] for r in prep.releases[mid]}
            if use_relaxed_prescreen:
                relaxed_best = _price_type_relaxed(
                    inst, mid, prep.arcs_by_type[mid], weight1, weight2,
                    sigma, nu1, nu2, 0.0, theta,
                    prep.arrival_of, prep.weight_of, forbidden, pinned, sources,
                )
                if relaxed_best >= -tol:
                    continue
            result = _price_type(
                inst, mid, prep.arcs_by_type[mid], weight1, weight2,
                sigma, nu1, nu2, 0.0, theta,
                prep.arrival_of, prep.weight_of, tol, forbidden, pinned,
                differ_pairs, same_pairs, cut_duals, sources,
            )
            if result is None:
                continue
            _reduced_cost, col = result
            var = _add_column_var(
                mdl, col, weight1, weight2, state,
                name=f"lam[{mid}]_{len(state.columns)}",
            )
            state.columns.append(col)
            state.vars.append(var)
            improved = True
        if timers is not None:
            timers["pricing_s"] = timers.get("pricing_s", 0.0) + time.perf_counter() - t_price
            timers["cg_iters"] = timers.get("cg_iters", 0) + 1

        if not improved:
            return it + 1
    raise RuntimeError("BPC column generation did not converge within max_iters")


def _big_m_for(inst: Instance) -> float:
    cost_scale = max((abs(c) for c in inst.c.values()), default=1.0)
    fixed_scale = max((itype.fixed_cost for itype in inst.instance_types), default=0.0)
    time_scale = max(inst.horizon, 1)
    weight_scale = max((j.weight for j in inst.jobs), default=1.0)
    return 1e6 * max(cost_scale, fixed_scale, time_scale * weight_scale, 1.0)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

@dataclass
class RootResult:
    objective: float
    iterations: int
    n_columns: int
    artificial_total: float
    model: gp.Model
    state: _RmpState = field(repr=False)


def solve_root_relaxation(
    inst: Instance,
    weight1: float,
    weight2: float,
    bound1_value: float | None = None,
    bound2_value: float | None = None,
    max_iters: int = 10_000,
    tol: float = 1e-7,
    big_m: float | None = None,
    use_relaxed_prescreen: bool = False,
    reduce: bool = True,
    pricing: PricingOptions | None = None,
) -> RootResult:
    """Column-generate the root LP relaxation of the BPC master.

    Minimises ``weight1 * f1 + weight2 * f2``, optionally with an extra
    ``f1 <= bound1_value`` and/or ``f2 <= bound2_value`` row (both may be
    active at once) — mirrors ``pareto._make_lexmin_bounded``'s
    ``bound_expr_fn``/``bound_value``, generalised to two independent caps
    since ``pareto.py``'s box search needs both simultaneously (see
    ``make_bpc_lexmin_bounded``).

    No branching: this is the LP bound only, Phase 1 of the BPC plan.
    ``reduce`` — use the exact start-set reductions of :class:`_Prep`
    (``False``: every start in ``T_jm``). ``pricing`` — see
    :class:`PricingOptions`; with ng-routes the bound lies between the
    compact LP and the elementary master's.
    """
    inst.require_unit_demand("branch-and-price (bpc)")
    if big_m is None:
        big_m = _big_m_for(inst)
    prep = _prepare(inst, reduce)

    seed = {_column_key(c): c for c in prep.static_columns}
    for c in _heuristic_seed_columns(inst, prep):
        seed.setdefault(_column_key(c), c)
    mdl, state = _build_rmp(inst, weight1, weight2, bound1_value, bound2_value, big_m,
                             extra_columns=list(seed.values()), prep=prep,
                             activation=_compiled(pricing, frozenset()))

    iterations = _run_column_generation(
        mdl, state, inst, weight1, weight2, prep, max_iters, tol,
        use_relaxed_prescreen=use_relaxed_prescreen, opts=pricing, big_m=big_m,
    ).iterations

    artificial_total = sum(v.X for v in state.art.values())
    return RootResult(
        objective=mdl.ObjVal,
        iterations=iterations,
        n_columns=len(state.columns),
        artificial_total=artificial_total,
        model=mdl,
        state=state,
    )


# ---------------------------------------------------------------------------
# Phase 3: branch-and-price with Ryan-Foster job-pair branching
# ---------------------------------------------------------------------------

@dataclass
class _NodeResult:
    status: str  # "integral" | "fractional" | "infeasible" | "pruned" (bound >= cutoff)
    objective: float
    columns: list[Column]
    assignment: dict[int, tuple[int, int]] | None = None
    branch_pair: tuple[int, int] | None = None          # Ryan-Foster (job_j, job_k)
    branch_var: tuple[int, int, int] | None = None       # Phase-2 fallback (job_id, type_id, start)
    branch_type: int | None = None                       # fractional z_m — see _solve_node
    incumbent: tuple[float, dict[int, tuple[int, int]]] | None = None  # see _rmp_integer_heuristic
    # type/time-window branching: the (job, type, start) triples each child
    # forbids, the child to explore first first — see _time_branch
    branch_forbid: tuple[frozenset, frozenset] | None = None
    fixed: frozenset = frozenset()   # reduced-cost fixing, inherited by children


def _rmp_integer_heuristic(
    mdl: gp.Model,
    state: _RmpState,
    inst: Instance,
    weight1: float,
    weight2: float,
    bound1_value: float | None,
    bound2_value: float | None,
    time_limit: float,
) -> tuple[float, dict[int, tuple[int, int]]] | None:
    """Restricted-master IP heuristic ("price-and-branch"): solve the node's
    RMP with integer column variables over the columns generated so far.

    Any integer RMP solution with no artificial mass is a feasible schedule
    (each job covered once, at most N_m timelines per type, bound rows
    respected), so its objective is a valid incumbent for pruning. Without
    one, depth-first search prunes nothing until it reaches an integral leaf
    — on tiny_large's (a deleted development instance, see the module
    docstring) lex-min (f2, f1) phase 2 that took > 15 min although
    the root bound already equals the optimum. Changes the model in place:
    call after every LP value has been read.

    Returns ``(objective, assignment)`` recomputed from the assignment, as an
    integral node does, or None when no feasible integer solution was found.
    """
    for var in state.vars:
        var.VType = GRB.INTEGER
    for var in state.z.values():
        var.VType = GRB.BINARY
    mdl.setParam("TimeLimit", time_limit)
    mdl.optimize()
    if mdl.SolCount == 0 or sum(v.X for v in state.art.values()) > 1e-6:
        return None
    assignment: dict[int, tuple[int, int]] = {}
    for col, var in zip(state.columns, state.vars):
        if var.X > 0.5:
            for job_id, start in col.jobs:
                if job_id in assignment:
                    return None
                assignment[job_id] = (col.type_id, start)
    if len(assignment) != len(inst.jobs):
        return None
    completion = {jid: t + inst.p_occ[jid, mid] for jid, (mid, t) in assignment.items()}
    f1, f2 = inst.f1_of(completion), inst.f2_of(assignment)
    if bound1_value is not None and f1 > bound1_value + 1e-6:
        return None
    if bound2_value is not None and f2 > bound2_value + 1e-6:
        return None
    return weight1 * f1 + weight2 * f2, assignment


def _solve_node(
    inst: Instance,
    weight1: float,
    weight2: float,
    bound1_value: float | None,
    bound2_value: float | None,
    forbidden: frozenset[tuple[int, int, int]],
    pinned: dict[int, tuple[int, int]],
    differ_pairs: frozenset[tuple[int, int]],
    same_pairs: frozenset[tuple[int, int]],
    forbidden_types: frozenset[int],
    forced_types: frozenset[int],
    cuts: frozenset[frozenset[int]],
    seed_pool: list[Column],
    prep: _Prep,
    max_iters: int,
    tol: float,
    big_m: float,
    use_relaxed_prescreen: bool = False,
    heuristic_time_limit: float | None = None,
    deadline: float | None = None,
    timers: dict | None = None,
    opts: PricingOptions | None = None,
    cutoff: float = math.inf,
    branching: str = "time",
    depth: int = 0,
    lm_cuts: list[_LmCut] | None = None,
) -> _NodeResult:
    """Column generation at one node, then its status and branching choice.

    ``heuristic_time_limit`` — when set and the node is fractional, also run
    :func:`_rmp_integer_heuristic` with that limit (capped by ``deadline``);
    its result is the returned ``incumbent``. ``cutoff`` — the node is
    ``"pruned"`` as soon as its Lagrangian bound reaches it. ``branching`` —
    ``"time"`` (:func:`_time_branch`: a job's type, then its start window)
    or ``"rf"`` (Ryan-Foster job pairs, falling back to single placements).

    Columns that repeat a job (ng-routes) are not inherited: the node's own
    pricing regenerates the ones its neighbourhoods still allow.

    ``lm_cuts`` — the search's limited-memory subset-row cuts (shared, in
    order). Every node's master holds all of them; at depth up to
    ``opts.sr_depth`` the node separates more (:func:`_separate_lm_src`) and
    appends them, re-running column generation after each round. Rounds
    stop when no cut is violated, the bound gains less than 1 % of the
    gap to the cutoff, or ``opts.sr_cuts`` is reached.
    """
    opts = opts or DEFAULT_PRICING
    if lm_cuts is None:
        lm_cuts = []
    valid_seed = [
        c for c in seed_pool
        if _is_elementary(c) and _column_allowed(c, forbidden, pinned, differ_pairs, same_pairs)
    ]
    mdl, state = _build_rmp(inst, weight1, weight2, bound1_value, bound2_value, big_m,
                             extra_columns=valid_seed,
                             forbidden_types=forbidden_types, forced_types=forced_types,
                             cuts=cuts, prep=prep, activation=_compiled(opts, cuts),
                             lm_cuts=tuple(lm_cuts) if _compiled(opts, cuts) else ())

    def generate() -> _CGResult:
        return _run_column_generation(
            mdl, state, inst, weight1, weight2, prep, max_iters, tol,
            forbidden, pinned, differ_pairs, same_pairs,
            use_relaxed_prescreen, deadline, timers,
            opts=opts, cutoff=cutoff, big_m=big_m,
        )

    try:
        cg = generate()
        separate = (_compiled(opts, cuts) and depth <= opts.sr_depth
                    and min(opts.sr_cuts, 64) > len(lm_cuts))
        for _round in range(opts.sr_rounds if separate else 0):
            if cg.status == "pruned" or sum(v.X for v in state.art.values()) > 1e-6:
                break
            room = min(opts.sr_cuts, 64) - len(lm_cuts)
            t0 = time.perf_counter()
            new = _separate_lm_src(state, min(opts.sr_per_round, room), opts.sr_min_viol,
                                   {c.S for c in lm_cuts})
            if timers is not None:
                timers["sep_s"] = timers.get("sep_s", 0.0) + time.perf_counter() - t0
            if not new:
                break
            _add_lm_cuts(mdl, state, new)
            lm_cuts.extend(new)
            if timers is not None:
                timers["sr_cuts"] = len(lm_cuts)
            before = cg.objective
            cg = generate()
            if cg.status != "pruned" and depth == 0 and timers is not None:
                timers["cut_bound"] = cg.objective
            gap = cutoff - before if math.isfinite(cutoff) else abs(before)
            if cg.status != "pruned" and cg.objective - before < 0.01 * max(gap, 1e-9):
                break   # tailing off
    except _RmpInfeasible:
        return _NodeResult(status="infeasible", objective=float("inf"), columns=seed_pool)
    if cg.status == "pruned":
        return _NodeResult(status="pruned", objective=cg.objective, columns=state.columns)

    artificial_total = sum(v.X for v in state.art.values())
    if artificial_total > 1e-6:
        return _NodeResult(status="infeasible", objective=float("inf"), columns=state.columns)

    # Aggregate x_jmt = sum of lambda_p over columns placing job at (m,t), and
    # joint_jk = sum of lambda_p over columns placing *both* j and k (any
    # times/types) — the Ryan-Foster branching signal: how "tied together"
    # two jobs currently are in the fractional solution.
    agg: dict[tuple[int, int, int], float] = {}
    joint: dict[tuple[int, int], float] = {}
    for col, var in zip(state.columns, state.vars):
        if not col.jobs or var.X <= 1e-9:
            continue
        for job_id, start in col.jobs:
            key = (job_id, col.type_id, start)
            agg[key] = agg.get(key, 0.0) + var.X
        if len(col.jobs) >= 2:
            job_ids = sorted({jid for jid, _t in col.jobs})
            for i in range(len(job_ids)):
                for jx in range(i + 1, len(job_ids)):
                    pkey = (job_ids[i], job_ids[jx])
                    joint[pkey] = joint.get(pkey, 0.0) + var.X

    # A tight, fixed tolerance for *acceptance* (not the caller's, often-
    # loosened `tol`, which stays fine for RF branching-candidate selection
    # below — picking which fractional pair to branch on next isn't
    # correctness-critical the way "is this node truly done" is). A real bug
    # found during Phase 4 validation: a job's coverage can be ~99.9997% on
    # its "obvious" (m,t) with a tiny residual actually sitting on a
    # *different*, cheaper (m,t) for the same job (e.g. mostly on paid cloud,
    # a sliver on free on-prem) — `per_job_max >= 1 - tol` at the loose
    # tol=1e-5 called that job "settled" and the extracted clean assignment
    # silently dropped the residual, but the RMP's own bound rows had
    # already banked that residual's (lower) cost — so a bound row that was
    # only just barely satisfied by the *fractional* solution came back
    # violated once cleanly assigned. Same root cause, same fix, as z_m's
    # own fractionality below: use an acceptance tolerance tight enough that
    # its cost impact (residual * a cost/fixed_cost magnitude) stays well
    # under any real comparison tolerance (1e-4-ish), not the loose one used
    # for the different purpose of breaking ties in column generation.
    _INTEGRALITY_TOL = 1e-7

    per_job_max: dict[int, float] = {}
    for (job_id, _mid, _t), v in agg.items():
        if v > per_job_max.get(job_id, 0.0):
            per_job_max[job_id] = v

    unsettled = {j.id for j in inst.jobs if per_job_max.get(j.id, 0.0) < 1 - _INTEGRALITY_TOL}

    if not unsettled:
        # z_m (the fixed-activation-cost variable) is not structurally
        # guaranteed integral just because every job is placed — Geoffrion's
        # integrality property covers the coverage/fleet-size rows, not a
        # variable that exists purely for the bound rows and the objective. See the
        # tolerance comment above; the same tight bound applies here for the
        # same reason (fixed_cost can be large, amplifying any residual).
        frac_z = [
            mid for mid, zvar in state.z.items()
            if _INTEGRALITY_TOL < zvar.X < 1 - _INTEGRALITY_TOL
        ]
        if frac_z:
            return _NodeResult(status="fractional", objective=cg.objective,
                                columns=state.columns, branch_type=frac_z[0], fixed=cg.fixed)

        assignment = {
            job_id: (mid, t) for (job_id, mid, t), v in agg.items()
            if v >= 1 - _INTEGRALITY_TOL
        }
        # mdl.ObjVal is NOT used here, deliberately: it's the RMP's raw LP
        # objective, which can still diverge from what the *clean* extracted
        # assignment actually costs — the residual-mass phenomenon documented
        # above (and in the module docstring's Phase 4 section) shows up here
        # too, not just in bound-row accounting. A real bug found during
        # Phase 5 validation: this node's objective feeds `best_obj`, which
        # every other node gets pruned against — a falsely-optimistic
        # mdl.ObjVal for one integral node silently pruned away the *actual*
        # optimum elsewhere in the tree (best_obj looked already unbeatable).
        # Recomputing cleanly from the extracted assignment — exactly like
        # every public solve_*_bpc wrapper already does for its *returned*
        # f1/f2 — makes this node's objective trustworthy for pruning too.
        # (Fractional nodes are unaffected: their mdl.ObjVal is a valid LP
        # relaxation bound by construction, not meant to match any one
        # concrete assignment, so there's nothing to "clean" there.)
        completion = {jid: t + inst.p_occ[jid, mid] for jid, (mid, t) in assignment.items()}
        true_objective = weight1 * inst.f1_of(completion) + weight2 * inst.f2_of(assignment)
        return _NodeResult(status="integral", objective=true_objective,
                            columns=state.columns, assignment=assignment)

    # Prefer Ryan-Foster: the job pair whose joint coverage is most fractional
    # (closest to 0.5) among those genuinely undetermined right now — this is
    # the branching that actually breaks the compact model's identical-
    # instance symmetry (see the module docstring). Falls back to Phase 2's
    # single-variable branching when no pair is a useful candidate, e.g. one
    # unsettled job split across start times with no job ever co-occurring
    # with it in any positive-weight column.
    pair_candidates = [
        (abs(v - 0.5), jid, kid)
        for (jid, kid), v in joint.items()
        if tol < v < 1 - tol
    ] if branching == "rf" else []
    if branching == "time":
        result = _NodeResult(status="fractional", objective=cg.objective, columns=state.columns,
                             branch_forbid=_time_branch(inst, prep, agg, unsettled))
    elif pair_candidates:
        _dist, branch_j, branch_k = min(pair_candidates)
        result = _NodeResult(status="fractional", objective=cg.objective, columns=state.columns,
                             branch_pair=(branch_j, branch_k))
    else:
        _dist, branch_j, branch_m, branch_t = min(
            (abs(v - 0.5), jid, mid, t)
            for (jid, mid, t), v in agg.items()
            if jid in unsettled
        )
        result = _NodeResult(status="fractional", objective=cg.objective, columns=state.columns,
                             branch_var=(branch_j, branch_m, branch_t))
    if heuristic_time_limit is not None:
        if deadline is not None:
            heuristic_time_limit = min(heuristic_time_limit,
                                       max(0.0, deadline - time.perf_counter()))
        result.incumbent = _rmp_integer_heuristic(
            mdl, state, inst, weight1, weight2, bound1_value, bound2_value,
            heuristic_time_limit,
        )
    result.fixed = cg.fixed
    return result


def _time_branch(inst: Instance, prep: _Prep, agg: dict[tuple[int, int, int], float],
                 unsettled: set[int]) -> tuple[frozenset, frozenset]:
    """Branch on one unsettled job's placement, as the forbidden triples of
    each child (the side holding more LP mass first).

    1. Its type, if the job is split across types: the job whose mass on a
       type ``m`` is closest to 1/2 goes on ``m`` in one child, off it in the
       other.
    2. Else its start window on its one type: the split point ``t*`` whose
       cumulative mass is closest to 1/2; children ``S_j <= t*`` and
       ``S_j > t*`` (completion-time-window branching, Kramer et al. 2020).

    Every decision shrinks a start set, so the tree is finite, and a node
    with each job's mass at one placement is integral (see _solve_node).
    Unlike Ryan-Foster pairs it needs no pricing state, and both children
    move the LP: the mass on the forbidden side must go elsewhere.
    """
    by_type: dict[tuple[int, int], float] = {}
    by_start: dict[tuple[int, int], list[tuple[int, float]]] = {}
    for (jid, mid, t), v in agg.items():
        if jid in unsettled and v > 1e-9:
            by_type[jid, mid] = by_type.get((jid, mid), 0.0) + v
            by_start.setdefault((jid, mid), []).append((t, v))
    types = [(abs(x - 0.5), jid, mid, x) for (jid, mid), x in by_type.items() if 1e-6 < x < 1 - 1e-6]
    if types:
        _d, jid, mid, x = min(types)
        on = frozenset((jid, m2, t) for m2 in inst.F[jid] if m2 != mid
                       for t in prep.starts.get((jid, m2), ()))
        off = frozenset((jid, mid, t) for t in prep.starts.get((jid, mid), ()))
        return (on, off) if x >= 0.5 else (off, on)
    best = None
    for (jid, mid), lst in by_start.items():
        lst.sort()
        cum = 0.0
        for t, v in lst[:-1]:
            cum += v
            cand = (abs(cum - 0.5), jid, mid, t, cum)
            if best is None or cand < best:
                best = cand
    if best is None:
        raise RuntimeError("no branching candidate on a fractional node")
    _d, jid, mid, t_star, cum = best
    starts = prep.starts.get((jid, mid), ())
    early = frozenset((jid, mid, t) for t in starts if t > t_star)    # S_j <= t*
    late = frozenset((jid, mid, t) for t in starts if t <= t_star)    # S_j > t*
    return (early, late) if cum >= 0.5 else (late, early)


# ---------------------------------------------------------------------------
# Phase 5: rank-1 subset-row cuts, separated once at the root
# ---------------------------------------------------------------------------

def _separate_sr_cuts(
    state: _RmpState,
    tol: float,
    max_new: int,
) -> list[frozenset[int]]:
    """Find up to ``max_new`` violated rank-1 subset-row (SR-1) cuts — job
    triples ``{i,j,k}`` where the current fractional solution has
    ``sum_{p: |p.jobs & {i,j,k}| >= 2} lambda_p > 1`` — from the just-solved
    RMP. Valid because any *integer* solution covers each job exactly once,
    so at most one selected column can cover 2+ of any 3 jobs (Jepsen et al.
    2008's classic strengthening for set-partitioning column generation —
    the standard next step once columns + Ryan-Foster branching alone don't
    close a stubborn gap; see the module docstring's "Phase 5" section for
    why this project specifically needed it and what it targets).

    Candidate triples are built from a job co-occurrence graph (edge i-k iff
    some positive-lambda column covers both), extending each edge by a third
    job adjacent to *either* endpoint — cheap, and exactly the triples where
    a violation is actually possible (a triple with no two jobs ever sharing
    a column trivially has LHS 0). This has to look at co-occurrence across
    *all* columns, not just within one: the classic fractional-vertex shape
    this cut targets is three separate columns each covering a different pair
    of the triple (e.g. {i,j},{i,k},{j,k}), where i and k in particular may
    never co-occur in any single column at all — the candidate generation
    must still propose {i,j,k} by chaining through the shared jobs i-j and
    j-k. Returns the most-violated ones first.
    """
    active = [
        ({jid for jid, _t in col.jobs}, var.X)
        for col, var in zip(state.columns, state.vars)
        if col.jobs and var.X > 1e-9
    ]

    adjacent: dict[int, set[int]] = {}
    for job_set, _lam in active:
        for j in job_set:
            adjacent.setdefault(j, set()).update(job_set - {j})

    candidates: set[frozenset[int]] = set()
    for i, neighbors in adjacent.items():
        for k in neighbors:
            if k <= i:
                continue
            for j in adjacent[i] | adjacent[k]:
                if j != i and j != k:
                    candidates.add(frozenset((i, j, k)))

    violations: list[tuple[float, frozenset[int]]] = []
    for triple in candidates:
        lhs = sum(lam for job_set, lam in active if len(job_set & triple) >= 2)
        if lhs > 1.0 + tol:
            violations.append((lhs, triple))

    violations.sort(key=lambda x: -x[0])
    return [triple for _lhs, triple in violations[:max_new]]


def _root_cuts_and_price(
    inst: Instance,
    weight1: float,
    weight2: float,
    bound1_value: float | None,
    bound2_value: float | None,
    big_m: float,
    prep: _Prep,
    root_pool: list[Column],
    max_iters: int,
    tol: float,
    max_cut_rounds: int,
    max_cuts_per_round: int,
    max_total_cuts: int,
    use_relaxed_prescreen: bool = False,
    deadline: float | None = None,
    opts: PricingOptions | None = None,
) -> tuple[bool, frozenset[frozenset[int]], list[Column]]:
    """Separate + add SR-1 cuts at the root only, before the B&B tree starts
    — the simpler, standard variant of branch-and-cut-and-price that avoids
    having to track which cuts are still meaningful under which branching
    decisions at every node. Cuts found here are valid everywhere in the
    tree regardless (a triple's constraint doesn't depend on forbidden/
    pinned/differ/same decisions), so holding them fixed for the rest of the
    search only costs some tightness at deeper nodes, never correctness.

    Returns ``(feasible, cuts, pool)`` — ``pool`` is the column pool
    accumulated across cut rounds, handed to the tree as its root's seed
    pool so that work isn't wasted; ``feasible=False`` short-circuits
    ``_branch_and_price`` straight to its own infeasible-result path.
    """
    cuts: frozenset[frozenset[int]] = frozenset()
    pool = root_pool
    for _round in range(max_cut_rounds):
        mdl, state = _build_rmp(inst, weight1, weight2, bound1_value, bound2_value, big_m,
                                 extra_columns=pool, cuts=cuts, prep=prep,
                                 activation=_compiled(opts, cuts))
        try:
            _run_column_generation(mdl, state, inst, weight1, weight2, prep, max_iters, tol,
                                    use_relaxed_prescreen=use_relaxed_prescreen,
                                    deadline=deadline, opts=opts, big_m=big_m)
        except _RmpInfeasible:
            return False, frozenset(), []
        pool = state.columns
        if len(cuts) >= max_total_cuts:
            break
        new_cuts = _separate_sr_cuts(state, tol, max_cuts_per_round)
        if not new_cuts:
            break
        cuts = cuts | frozenset(new_cuts)
    return True, cuts, pool


def _separate_lm_src(state: _RmpState, max_new: int, min_viol: float,
                     existing: set[frozenset[int]]) -> list[_LmCut]:
    """Violated subset-row cuts on job triples at the current RMP solution,
    most violated first, each with the least memory that keeps its full
    violation (Pecin et al. 2017): for every positive column holding two
    of ``S``, the jobs it visits between its first and last job of ``S``.

    A triple's left-hand side is ``sum_p lambda_p floor(|S ∩ p| / 2)``
    (visit counts); at least two of its pairs then share a column, so the
    candidates are each co-occurring pair plus any third job. No job is in
    more than three of the returned cuts, so one fractional cycle does not
    take the whole round."""
    import numpy as np

    act = [(col, var.X) for col, var in zip(state.columns, state.vars)
           if len(col.jobs) >= 2 and var.X > 1e-6]
    if not act or max_new <= 0:
        return []
    jobs = sorted({jid for col, _x in act for jid, _t in col.jobs})
    idx = {j: i for i, j in enumerate(jobs)}
    X = np.zeros((len(act), len(jobs)))
    for r, (col, _x) in enumerate(act):
        for jid, _t in col.jobs:
            X[r, idx[jid]] += 1.0
    lam = np.array([x for _col, x in act])
    co = (X.T * lam) @ (X > 0)          # co[a, b] > 0: a and b share a column
    found: dict[frozenset[int], float] = {}
    for a, b in zip(*np.nonzero(np.triu(co, 1) > 1e-9)):
        cnt = X[:, a:a + 1] + X[:, b:b + 1] + X
        lhs = lam @ np.floor(cnt / 2.0)
        for k in np.nonzero(lhs > 1.0 + min_viol)[0]:
            if k != a and k != b:
                S = frozenset((jobs[a], jobs[b], jobs[k]))
                if S not in existing:
                    found[S] = float(lhs[k])
    uses: Counter = Counter()
    out: list[_LmCut] = []
    for S, _lhs in sorted(found.items(), key=lambda kv: (-kv[1], sorted(kv[0]))):
        if len(out) >= max_new:
            break
        if any(uses[j] >= 3 for j in S):
            continue
        M = set(S)
        for col, _x in act:
            pos = [i for i, (jid, _t) in enumerate(col.jobs) if jid in S]
            if len(pos) >= 2:
                M.update(jid for jid, _t in col.jobs[pos[0]:pos[-1] + 1])
        out.append(_LmCut(S=S, M=frozenset(M)))
        uses.update(S)
    return out


def _add_lm_cuts(mdl: gp.Model, state: _RmpState, cuts: list[_LmCut]) -> None:
    """Add cut rows to a live RMP, with every current column's coefficient."""
    for cut in cuts:
        expr = gp.LinExpr()
        for col, var in zip(state.columns, state.vars):
            if len(col.jobs) >= 2:
                a = _lm_coef(cut, col.jobs)
                if a:
                    expr.add(var, float(a))
        state.lm_cuts.append(
            (cut, mdl.addConstr(expr <= 1.0, name=f"lmsr[{len(state.lm_cuts)}]")))
    mdl.update()


_POOL_CAP = 50_000   # columns kept across solves (pool_out)

# _branch_and_price's tree options when not given (as DEFAULT_PRICING).
TREE_DEFAULTS = {"node_selection": "best", "branching": "time", "heuristic_every": 10}


def _branch_and_price(
    inst: Instance,
    weight1: float,
    weight2: float,
    bound1_value: float | None = None,
    bound2_value: float | None = None,
    max_nodes: int = 5_000,
    max_iters_per_node: int = 10_000,
    tol: float = 1e-5,
    big_m: float | None = None,
    use_cuts: bool = False,
    max_cut_rounds: int = 3,
    max_cuts_per_round: int = 2,
    max_total_cuts: int = 5,
    use_relaxed_prescreen: bool = False,
    heuristic_time_limit: float = 5.0,
    time_limit: float | None = None,
    stats: dict | None = None,
    *,
    prep: _Prep | None = None,
    reduce: bool = True,
    seed_schedules: list[dict[int, tuple[int, int]]] | None = None,
    column_pool: list[Column] | None = None,
    pool_out: dict | None = None,
    pricing: PricingOptions | None = None,
    node_selection: str | None = None,
    branching: str | None = None,
    heuristic_every: int | None = None,
) -> tuple[float, dict[int, tuple[int, int]] | None]:
    """Exact branch-and-price. Returns ``(objective, assignment)``, or
    ``(inf, None)`` if the instance (restricted to any bound row) is
    infeasible. See the module docstring for the branching scheme.

    ``time_limit`` — wall-clock seconds for this call (None: no limit),
    checked between nodes and between column-generation iterations, so it
    is overshot by at most one LP solve and one pricing round. Past it,
    raises :class:`BPCTimeLimit` carrying the best schedule found so far.

    ``tol`` defaults looser than the root relaxation's (1e-5 vs. Phase 1's
    1e-7): with ``weight1=0`` (or any node deep in an "up"-heavy branch,
    where most jobs are pinned), f2 alone often doesn't depend on a job's
    exact start time, so many columns tie exactly — at 1e-7 that tie noise
    made column generation cycle (repeatedly re-adding near-zero-reduced-cost
    columns) instead of converging on the (deleted) tiny_large/tiny_xlarge. 1e-5 is still
    two orders of magnitude tighter than the ``abs=1e-4`` comparison
    tolerance the tests check f1/f2 against.

    ``use_cuts`` — separate rank-1 subset-row cuts at the root before
    branching (see ``_root_cuts_and_price``/``_separate_sr_cuts`` and the
    module docstring's "Phase 5" section). Cuts never change the optimum —
    only the bound/search order — but each *active* cut costs the pricing
    DP an extra dominance-signature bit (a label triggered on a cut and one
    that isn't become mutually non-dominating), which shrinks how much the
    elementary-path dominance can prune. On the (deleted) tiny_medium.json this measurably
    backfires: with cuts on, the ``solve_f2_bpc`` tie-break phase (minimise
    f1 s.t. f2 <= its own minimum — the same tightly-pinned, on-prem-packing
    shape Phase 3 already found hard) went from ~seconds to not finishing in
    90s, on an instance that Phase 2-4 handled fine *without* cuts. So the
    default is now **off**, with much smaller separation limits than the
    original (max_cut_rounds 10->3, max_cuts_per_round 5->2, max_total_cuts
    30->5) for when a caller opts in — keeping the active-cut count low is
    what keeps this dominance cost bounded. Set True to try cuts; verify
    against ``use_cuts=False`` on the specific case you're targeting before
    trusting it's a net win, the same way ``tests/test_bpc_cuts.py``'s
    ``TestCutsEndToEnd`` does.

    ``use_relaxed_prescreen`` — before every exact ``_price_type`` call,
    first try the much cheaper 1-cycle-free ``_price_type_relaxed`` (see its
    docstring and the module docstring's "Comparison with Kramer et al.
    2020" section) and skip the exact call whenever the relaxed one already
    finds nothing; provably safe (never changes which columns are found),
    intended for the same tightly-pinned scheduling cases cuts didn't help.

    ``heuristic_time_limit`` — seconds for :func:`_rmp_integer_heuristic`,
    run at each fractional node (the node's grown column pool often holds
    a better schedule than the last incumbent); ``0`` disables it. It only supplies incumbents, so the optimum
    is unchanged.

    Pruning uses objective integrality: f1 = Σ w_j C_j − Σ w_j a_j over
    integer completions C_j and integral weights (an assumption
    :meth:`Instance.build` enforces) moves in steps of ``gcd(w_j)``, f2 in
    steps of ``δ·gcd(cost units)``, so when ``weight1·f1 + weight2·f2`` lies
    on a grid of step ``s >= 1e-3`` (:func:`_objective_step`) a node holds a
    strictly better schedule only if its bound is at most ``best − s``; the
    1e-3 margin absorbs the column generation's ``tol``-level bound error.
    Without it, bounds within that error of the optimum (e.g. 187.51663 vs
    187.516667 on the deleted tiny_large) never prune.
    ``pricing`` — see :class:`PricingOptions`. Its Lagrangian bound prunes a
    node as soon as it reaches ``best − s``, before column generation ends.
    ``prep`` — a :func:`_prepare` result to reuse (built here otherwise, with
    ``reduce``). ``seed_schedules`` — complete schedules whose columns seed
    the root and whose best one within the bound rows is the starting
    incumbent (default: :func:`heuristic_schedules`). ``column_pool`` —
    further columns for the root, e.g. from earlier solves on the same
    ``prep``: the bound rows are aggregate, so a column is valid under any
    bounds. ``pool_out`` — when a dict, receives every column the search
    generated (keyed by :func:`_column_key`), for the next solve.

    ``node_selection`` — ``"best"`` (lowest parent bound first) or
    ``"dfs"``. ``branching`` — ``"time"`` (job type, then start window,
    :func:`_time_branch`) or ``"rf"`` (Ryan-Foster pairs). Children inherit
    their parent's reduced-cost fixing (``PricingOptions.fixing``).
    ``heuristic_every`` — run the RMP integer heuristic at the root and every
    that-many nodes.

    ``stats`` — when a dict, filled with search counters (``nodes``,
    ``root_bound``, ``root_s``, ``cg_iters``, ``lp_s``, ``pricing_s``,
    ``columns``, ``open_bound``: the proven lower bound at the end) for
    benchmarking; it never changes the search.
    """
    t_start = time.perf_counter()
    node_selection = node_selection or TREE_DEFAULTS["node_selection"]
    branching = branching or TREE_DEFAULTS["branching"]
    heuristic_every = heuristic_every or TREE_DEFAULTS["heuristic_every"]
    timers: dict = stats if stats is not None else {}
    timers.setdefault("nodes", 0)
    deadline = None if time_limit is None else time.perf_counter() + time_limit
    if big_m is None:
        big_m = _big_m_for(inst)
    prune_gap = tol
    step = _objective_step(inst, weight1, weight2)
    if step >= 1e-3:
        prune_gap = max(tol, step - 1e-3 * min(step, 1.0))
    if prep is None:
        prep = _prepare(inst, reduce)
    if seed_schedules is None:
        seed_schedules = heuristic_schedules(inst)

    best_obj = float("inf")
    best_assignment: dict[int, tuple[int, int]] | None = None
    pool: dict[tuple, Column] = {_column_key(c): c for c in prep.static_columns}
    for c in column_pool or ():
        pool.setdefault(_column_key(c), c)
    for a in seed_schedules:
        for c in _schedule_columns(prep, a):
            pool.setdefault(_column_key(c), c)
        obj = _schedule_objective(inst, a, weight1, weight2, bound1_value, bound2_value)
        if obj is not None and obj < best_obj:
            best_obj, best_assignment = obj, dict(a)
    timers["seed_obj"] = best_obj
    root_pool = list(pool.values())

    cuts: frozenset[frozenset[int]] = frozenset()
    if use_cuts:
        feasible, cuts, root_pool = _root_cuts_and_price(
            inst, weight1, weight2, bound1_value, bound2_value, big_m,
            prep, root_pool, max_iters_per_node, tol,
            max_cut_rounds, max_cuts_per_round, max_total_cuts,
            use_relaxed_prescreen, deadline, pricing,
        )
        if not feasible:
            return float("inf"), None

    # Open nodes: (bound, -depth, seq, decisions); a child's bound is its
    # parent's. Best-first pops the lowest bound (deeper, then older, first);
    # DFS pops the newest, exploring each node's first child first.
    best_first = node_selection == "best"
    frontier: list = []
    seq = 0

    def push(bound: float, depth: int, children: list[tuple]) -> None:
        nonlocal seq
        entries = []
        for payload in children:
            seq += 1
            entries.append((bound, -depth, seq, payload))
        if best_first:
            for e in entries:
                heapq.heappush(frontier, e)
        else:
            frontier.extend(reversed(entries))

    def open_bound(current: float) -> float:
        return min([current, best_obj] + [e[0] for e in frontier])

    push(-math.inf, 0, [(frozenset(), {}, frozenset(), frozenset(), frozenset(), frozenset(),
                         root_pool)])
    n_nodes = 0
    lm_cuts: list[_LmCut] = []   # limited-memory SR cuts, valid in the whole tree

    while frontier:
        bound, neg_depth, _seq, payload = heapq.heappop(frontier) if best_first else frontier.pop()
        if bound >= best_obj - prune_gap:
            continue   # a later incumbent already prunes it
        if n_nodes >= max_nodes:
            raise RuntimeError("BPC branch-and-price exceeded max_nodes")
        forbidden, pinned, differ_pairs, same_pairs, forbidden_types, forced_types, node_pool = payload
        depth = -neg_depth
        n_nodes += 1
        run_heuristic = heuristic_time_limit > 0 and (n_nodes == 1 or n_nodes % heuristic_every == 0)

        try:
            _check_deadline(deadline)
            node = _solve_node(
                inst, weight1, weight2, bound1_value, bound2_value,
                forbidden, pinned, differ_pairs, same_pairs, forbidden_types, forced_types,
                cuts, node_pool,
                prep, max_iters_per_node, tol, big_m,
                use_relaxed_prescreen,
                heuristic_time_limit if run_heuristic else None,
                deadline, timers, pricing, best_obj - prune_gap, branching,
                depth=depth, lm_cuts=lm_cuts,
            )
        except BPCTimeLimit:
            timers["open_bound"] = open_bound(bound)
            raise BPCTimeLimit(best_obj, best_assignment, timers["open_bound"]) from None
        timers["nodes"] = n_nodes
        timers["max_depth"] = max(timers.get("max_depth", 0), depth)
        timers["columns"] = max(timers.get("columns", 0), len(node.columns))
        if n_nodes == 1:
            timers["root_bound"] = node.objective
            timers["root_s"] = time.perf_counter() - t_start
            timers["root_fixed"] = len(node.fixed)
        if pool_out is not None and len(pool_out) < _POOL_CAP:
            for c in node.columns:
                if c.jobs and _is_elementary(c):
                    pool_out.setdefault(_column_key(c), c)
        if node.incumbent is not None and node.incumbent[0] < best_obj:
            best_obj, best_assignment = node.incumbent
        if node.status in ("infeasible", "pruned") or node.objective >= best_obj - prune_gap:
            continue

        if node.status == "integral":
            best_obj = node.objective
            best_assignment = node.assignment
            continue

        fb = forbidden | node.fixed if node.fixed else forbidden
        cols = node.columns
        if node.branch_type is not None:
            m = node.branch_type
            children = [(fb, pinned, differ_pairs, same_pairs, forbidden_types, forced_types | {m}, cols),   # z_m = 1
                        (fb, pinned, differ_pairs, same_pairs, forbidden_types | {m}, forced_types, cols)]   # z_m = 0
        elif node.branch_forbid is not None:
            first, second = node.branch_forbid
            children = [(fb | first, pinned, differ_pairs, same_pairs, forbidden_types, forced_types, cols),
                        (fb | second, pinned, differ_pairs, same_pairs, forbidden_types, forced_types, cols)]
        elif node.branch_pair is not None:
            j, k = node.branch_pair
            children = [(fb, pinned, differ_pairs, same_pairs | {(j, k)}, forbidden_types, forced_types, cols),   # same
                        (fb, pinned, differ_pairs | {(j, k)}, same_pairs, forbidden_types, forced_types, cols)]   # differ
        else:
            jid, mid, t = node.branch_var
            children = [(fb, {**pinned, jid: (mid, t)}, differ_pairs, same_pairs, forbidden_types, forced_types, cols),  # up
                        (fb | {(jid, mid, t)}, pinned, differ_pairs, same_pairs, forbidden_types, forced_types, cols)]  # down
        push(node.objective, depth + 1, children)

    timers["open_bound"] = best_obj
    return best_obj, best_assignment


def _assignment_to_solution(inst: Instance, objective: float,
                             assignment: dict[int, tuple[int, int]] | None) -> Solution:
    if assignment is None:
        return Solution(status="infeasible", objective=None, f1=None, f2=None,
                         assignment={}, completion={})
    inst.check_schedule(assignment)
    completion = {jid: t + inst.p_occ[jid, mid] for jid, (mid, t) in assignment.items()}
    return Solution(
        status="optimal", objective=objective,
        f1=inst.f1_of(completion), f2=inst.f2_of(assignment),
        assignment=assignment, completion=completion,
    )


def solve_weighted_sum_bpc(inst: Instance, weight1: float, weight2: float, **kwargs) -> Solution:
    """Exact branch-and-price solve of ``min weight1*f1 + weight2*f2``."""
    inst.require_unit_demand("branch-and-price (bpc)")
    obj, assignment = _branch_and_price(inst, weight1, weight2, **kwargs)
    return _assignment_to_solution(inst, obj, assignment)


def _lex_tiebreak(inst: Instance, assignment1: dict[int, tuple[int, int]],
                  weight1: float, weight2: float, **kwargs) -> Solution:
    """Phase 2 of a lex-min: ``_branch_and_price(inst, weight1, weight2,
    **kwargs)`` with the bound row fixing phase 1's optimum.

    The result is ``"optimal"`` only when phase 2 itself finished. If it
    raised (``max_nodes``/``max_iters_per_node``/``time_limit``) or came back
    infeasible — impossible in exact arithmetic, since phase 1's schedule
    satisfies the bound — the result is ``"feasible"``: phase 2's incumbent
    when it timed out with one, else phase 1's schedule. Either is
    primary-optimal, but its secondary objective is unproven, so it may be
    weakly dominated and must not be trusted as a Pareto point.
    """
    try:
        _obj2, assignment2 = _branch_and_price(inst, weight1, weight2, **kwargs)
    except BPCTimeLimit as e:
        fallback = e.assignment or assignment1
        return replace(_assignment_to_solution(inst, None, fallback), status="feasible")
    except RuntimeError:
        assignment2 = None
    if assignment2 is None:
        return replace(_assignment_to_solution(inst, None, assignment1), status="feasible")
    return _assignment_to_solution(inst, None, assignment2)


def _reuse_kwargs(inst: Instance, kwargs: dict) -> dict:
    """``kwargs`` with a shared ``prep`` and ``seed_schedules`` filled in, so
    several ``_branch_and_price`` calls on one instance build them once."""
    kw = dict(kwargs)
    reduce = kw.pop("reduce", True)
    if kw.get("prep") is None:
        kw["prep"] = _prepare(inst, reduce)
    if kw.get("seed_schedules") is None:
        kw["seed_schedules"] = heuristic_schedules(inst)
    return kw


def _lexmin(inst: Instance, primary: str, **kwargs) -> Solution:
    """Both phases of a lex-min. Phase 2 starts from phase 1's columns and
    from its schedule, which satisfies phase 2's bound row and so is an
    incumbent from the first node on."""
    kw = _reuse_kwargs(inst, kwargs)
    pool: dict = {}
    w = (1.0, 0.0) if primary == "f1" else (0.0, 1.0)
    obj1, assignment1 = _branch_and_price(inst, *w, pool_out=pool, **kw)
    if assignment1 is None:
        return _assignment_to_solution(inst, obj1, None)
    kw2 = dict(kw, seed_schedules=[assignment1, *kw["seed_schedules"]],
               column_pool=[*(kw.get("column_pool") or ()), *pool.values()])
    if primary == "f1":
        return _lex_tiebreak(inst, assignment1, 0.0, 1.0, bound1_value=obj1, **kw2)
    return _lex_tiebreak(inst, assignment1, 1.0, 0.0, bound2_value=obj1, **kw2)


def solve_f1_bpc(inst: Instance, **kwargs) -> Solution:
    """Lex-min (f1, f2) via branch-and-price — mirrors ``model.solve_f1``.

    Status ``"feasible"`` when the tie-break phase failed (see
    :func:`_lex_tiebreak`). A ``time_limit`` applies to each phase; past it
    in phase 1, :class:`BPCTimeLimit` propagates.
    """
    inst.require_unit_demand("branch-and-price (bpc)")
    return _lexmin(inst, "f1", **kwargs)


def solve_f2_bpc(inst: Instance, **kwargs) -> Solution:
    """Lex-min (f2, f1) via branch-and-price — mirrors ``model.solve_f2``.

    Status ``"feasible"`` when the tie-break phase failed (see
    :func:`_lex_tiebreak`). ``time_limit``: as :func:`solve_f1_bpc`.
    """
    inst.require_unit_demand("branch-and-price (bpc)")
    return _lexmin(inst, "f2", **kwargs)


# ---------------------------------------------------------------------------
# Phase 4: pareto.py integration
# ---------------------------------------------------------------------------

def make_bpc_lexmin_bounded(
    inst: Instance,
    time_limit: float | None = None,
    **bpc_kwargs,
):
    """Build a BPC-backed ``lexmin_bounded`` closure with the exact same
    shape as ``pareto._make_lexmin_bounded``'s, so ``true_pareto_frontier``/
    ``map_pareto_frontier`` can select this engine via ``engine="bpc"`` with
    zero changes to the box-search algorithm itself.

    Returns ``(lexmin_bounded, counters)``, ``counters`` holding a live
    ``"n_solves"`` entry (incremented once per ``_branch_and_price`` call)
    and ``"build_time"``, the one-off preprocessing and seeding time.

    Every solve shares one :func:`_prepare` result, one column pool and the
    list of schedules found so far (seeded by :func:`heuristic_schedules`,
    with an NSGA-II front unless ``ea_seeds=False``): the bound rows are
    aggregate, so every pooled column is valid in every box, and any earlier
    schedule inside a box's bounds is that solve's first incumbent.

    ``time_limit`` caps each ``_branch_and_price`` call (each phase), as
    ``per_solve_time_limit`` caps each Gurobi solve. A phase that runs out of
    time, or raises via ``max_nodes``/``max_iters_per_node`` in
    ``bpc_kwargs``, gives an inconclusive (``"feasible"``-status) result —
    with the best schedule found, if any — instead of aborting the whole
    search. That holds for either phase: a schedule whose tie-break phase
    failed is primary-optimal only, so it is returned as ``"feasible"`` too
    (see :func:`_lex_tiebreak`) and the box search drops the box instead of
    recording a possibly dominated point. There is no seed-retry
    equivalent: BPC's search is deterministic.

    The returned closure's contract mirrors the Gurobi engine's exactly,
    with one difference forced by the representation: instead of callables
    taking a compact model's own ``x`` (which BPC has no equivalent of),
    ``primary``/``secondary``/``bound`` are the plain strings ``"f1"``/
    ``"f2"`` — ``pareto.py``'s own ``f1_of``/``f2_of`` are defined as these
    same strings for both engines, so callers never need to know which
    engine is in use.

    ``primary``/``secondary`` need not be distinct from ``bound``'s axis in
    principle (handled generally below via ``min(...)`` combination when
    they do coincide), though in ``pareto.py``'s actual usage ``bound`` is
    always the same axis as ``secondary``.
    """
    inst.require_unit_demand("branch-and-price (bpc)")
    counters = {"n_solves": 0, "build_time": 0.0}
    # Shared by every box solve: the preprocessing, the column pool (each
    # column stays valid under any bound rows) and every schedule found so
    # far — any of them inside a box's bounds is that solve's first incumbent.
    ea_seeds = bpc_kwargs.pop("ea_seeds", True)
    t0 = time.perf_counter()
    kw = _reuse_kwargs(inst, bpc_kwargs)
    if bpc_kwargs.get("seed_schedules") is None:
        kw["seed_schedules"] = heuristic_schedules(inst, ea=ea_seeds)
    counters["build_time"] = time.perf_counter() - t0
    schedules: list[dict[int, tuple[int, int]]] = list(kw.pop("seed_schedules"))
    pool: dict = {_column_key(c): c for c in kw.pop("column_pool", None) or ()}

    def solve(w1, w2, b1, b2):
        try:
            obj, a = _branch_and_price(
                inst, w1, w2, bound1_value=b1, bound2_value=b2, time_limit=time_limit,
                seed_schedules=schedules, column_pool=list(pool.values()),
                pool_out=pool, **kw)
        except BPCTimeLimit as e:
            if e.assignment is not None:
                schedules.insert(0, e.assignment)
            raise
        if a is not None:
            schedules.insert(0, a)
        return obj, a

    def lexmin_bounded(
        primary: str, secondary: str,
        bound: str | None = None, bound_value: float | None = None,
        *, start: Solution | None = None,
    ) -> Solution:
        # Snap the box bound onto its objective's grid: the master's bound
        # rows (and the RMP integer heuristic's MIP) then never carry a
        # right-hand side a hair below an achievable value.
        b1 = grid_floor(inst, "f1", bound_value) if bound == "f1" else None
        b2 = grid_floor(inst, "f2", bound_value) if bound == "f2" else None

        w1, w2 = (1.0, 0.0) if primary == "f1" else (0.0, 1.0)
        counters["n_solves"] += 1
        try:
            obj1, assignment1 = solve(w1, w2, b1, b2)
        except BPCTimeLimit as e:
            if e.assignment is None:
                return Solution(status="feasible", objective=None, f1=None, f2=None,
                                assignment={}, completion={})
            return replace(_assignment_to_solution(inst, e.objective, e.assignment),
                           status="feasible")
        except RuntimeError:
            return Solution(status="feasible", objective=None, f1=None, f2=None,
                             assignment={}, completion={})

        if assignment1 is None:
            return _assignment_to_solution(inst, obj1, None)

        # Fix the primary at its exact optimum (the objective is recomputed
        # from the schedule, see _solve_node), as in pareto._make_lexmin_bounded.
        fix = obj1
        if primary == "f1":
            b1 = min(b1, fix) if b1 is not None else fix
        else:
            b2 = min(b2, fix) if b2 is not None else fix

        w1s, w2s = (1.0, 0.0) if secondary == "f1" else (0.0, 1.0)
        counters["n_solves"] += 1
        try:
            _obj2, assignment2 = solve(w1s, w2s, b1, b2)
        except BPCTimeLimit as e:
            fallback = e.assignment or assignment1
            return replace(_assignment_to_solution(inst, None, fallback), status="feasible")
        except RuntimeError:
            assignment2 = None
        if assignment2 is None:
            return replace(_assignment_to_solution(inst, None, assignment1), status="feasible")
        return _assignment_to_solution(inst, None, assignment2)

    return lexmin_bounded, counters
