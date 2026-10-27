"""Branch-price-and-cut (BPC) exact solver for FED-HPC — Phases 1-5 complete.

Dantzig-Wolfe reformulation of the space-time network formulation
(:class:`fedhpc.formulations.SpaceTimeFormulation`, matching
``MOFE__Multi_Objective_Scheduling_on_Federated_Enviroments.pdf`` §3): one
set-partitioning master over "instance-lifetime" columns (source-to-sink paths
in each type's space-time network ``G_m``) plus one shortest-path pricing
subproblem per instance type.

Running jobs
------------
Every public entry point first rewrites ``inst.running_jobs`` as pinned
pseudo-jobs (:func:`_absorb_running_jobs`): a running job on capacitated type
``m`` ending at ``e`` becomes a job feasible only on ``m``, only at start 0,
with ``p_occ = min(e, H)``, cost 0 and weight 0. Covering it exactly once
takes one of the ``N_m`` instance lifetimes out of service on ``[0, e)`` —
precisely the compact model's ``y_{m,-1} = N_m - n_running[m]`` boundary plus
its ghost completion at ``e``. Pseudo-jobs add nothing to f1 or f2, do not
activate a type's fixed cost, and are stripped from every returned
assignment.

The development notes below cite timings on ``tiny_*`` instances (5-25-job
subsets of the fedhpc trace). Those instances have been deleted; the tests
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

Master (RMP)
------------
A column ``p`` for type ``m`` is an ordered sequence of non-overlapping
``(job_id, start)`` pairs — one candidate full-horizon timeline for a single
instance of type ``m``. Variables ``lambda_{m,p} >= 0`` count how many
instances of type ``m`` follow pattern ``p``. Rows:

- job coverage (equality, dual ``sigma_j``): every job covered exactly once.
- fleet-size (equality, dual ``rho_m``, **one per type, including cloud**):
  ``sum_p lambda_{m,p} = N_m``. MOFE fixes ``y_{m,-1} := N_m`` and
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

**Primary branching: Ryan-Foster on job pairs.** For every pair of jobs
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

from collections import Counter
from dataclasses import dataclass, field, replace

import gurobipy as gp
from gurobipy import GRB

from . import formulations
from .data import Instance, Job
from .model import Solution, grid_floor

__all__ = [
    "Column", "RootResult", "solve_root_relaxation",
    "solve_f1_bpc", "solve_f2_bpc", "solve_weighted_sum_bpc",
    "make_bpc_lexmin_bounded",
]


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


@dataclass
class _RunningAsJobs(Instance):
    """``base`` with its running jobs as pinned pseudo-jobs ``running_ids``
    (see :func:`_absorb_running_jobs`)."""

    running_ids: frozenset[int] = frozenset()
    base: Instance | None = None

    def f2_of(self, assignment: dict[int, tuple[int, int]]) -> float:
        # A running job's type is not used by the schedule: no fixed cost.
        return self.base.f2_of(
            {jid: a for jid, a in assignment.items() if jid not in self.running_ids}
        )


def _absorb_running_jobs(inst: Instance) -> Instance:
    """Rewrite ``inst.running_jobs`` as pinned pseudo-jobs (see the module
    docstring's "Running jobs" section). Returns ``inst`` itself when no
    running job occupies capacity.

    Mirrors :meth:`Instance.build`: running jobs on unlimited types are
    ignored, ends are clipped to the horizon, and a job ending at slot 0
    never occupies anything.
    """
    if isinstance(inst, _RunningAsJobs):
        return inst
    capacitated = {m.id for m in inst.instance_types if m.capacity is not None}
    running = [(rj.type_id, min(rj.end, inst.horizon)) for rj in inst.running_jobs
               if rj.type_id in capacitated and rj.end > 0]
    if not running:
        return inst
    jobs = list(inst.jobs)
    F, T = dict(inst.F), dict(inst.T)
    p_bill, p_occ, c = dict(inst.p_bill), dict(inst.p_occ), dict(inst.c)
    c_units = dict(inst.c_units)
    next_id = max((j.id for j in inst.jobs), default=-1) + 1
    ids = []
    for mid, end in running:
        jid = next_id
        next_id += 1
        ids.append(jid)
        jobs.append(Job(id=jid, arrival=0.0, exec_time=0.0, io_volume=0.0,
                        cpu=0.0, mem=0.0, stor=0.0, weight=0.0))
        F[jid] = [mid]
        T[jid, mid] = range(0, 1)
        p_bill[jid, mid], p_occ[jid, mid], c[jid, mid] = 0.0, end, 0.0
        c_units[jid, mid] = 0
    return _RunningAsJobs(
        jobs=jobs, instance_types=inst.instance_types, horizon=inst.horizon,
        slot_size_seconds=inst.slot_size_seconds, cost_resolution=inst.cost_resolution,
        p_bill=p_bill, p_occ=p_occ, c=c, c_units=c_units, F=F, T=T,
        running_ids=frozenset(ids), base=inst,
    )


def _running_ids(inst: Instance) -> frozenset[int]:
    return inst.running_ids if isinstance(inst, _RunningAsJobs) else frozenset()


@dataclass(frozen=True)
class _Arc:
    job_id: int
    p_occ: int
    start: int
    stop: int          # exclusive, matches Instance.T's range convention
    raw_cost2: float    # inst.c[job_id, type_id]
    weight: float       # j.weight — multiplies this arc's cost1 contribution


def _build_arcs_by_type(inst: Instance) -> dict[int, list[_Arc]]:
    result: dict[int, list[_Arc]] = {itype.id: [] for itype in inst.instance_types}
    for j in inst.jobs:
        for m_id in inst.F[j.id]:
            rng = inst.T[j.id, m_id]
            result[m_id].append(_Arc(
                job_id=j.id,
                p_occ=inst.p_occ[j.id, m_id],
                start=rng.start,
                stop=rng.stop,
                raw_cost2=inst.c[j.id, m_id],
                weight=j.weight,
            ))
    return result


def _fleet_sizes(inst: Instance) -> dict[int, int]:
    """N_m per type — capacity if finite, else an n_jobs proxy for 'unlimited'.

    Matches SpaceTimeFormulation's own ``N`` dict exactly (formulations.py).
    """
    n_jobs = len(inst.jobs)
    return {
        itype.id: itype.capacity if itype.capacity is not None else n_jobs
        for itype in inst.instance_types
    }


def _fixed_cost_types(inst: Instance) -> list:
    """Instance types with a per-type fixed activation cost (U_m > 0)."""
    return [itype for itype in inst.instance_types if itype.fixed_cost]


def _heuristic_seed_columns(inst: Instance) -> list[Column]:
    """Real, valid columns from a quick greedy heuristic, to seed the very
    root of a search — never at a branch-and-price node deeper than the
    root, which already inherits real columns from its parent.

    Without this, the root RMP starts with only null columns plus big-M
    coverage artificials, which makes *every* job's dual look uniformly
    huge on iteration 1 — and that degenerate "grab as many jobs as
    possible, order barely matters" regime is exactly what defeats the
    elementary-path label-setting DP's dominance pruning (see the module
    docstring's "Elementary-path enforcement" section): with 11-12 jobs
    feasible on one type, that first pricing call did not return within a
    minute. A handful of real starting columns keeps early duals
    differentiated enough that dominance actually prunes.

    Uses ``baselines.greedy_earliest_completion`` (ignores fixed-cost/
    weight1/weight2 — irrelevant here, this only needs to be *valid*, column
    generation refines everything from here), split per type into
    non-overlapping chains via the standard greedy interval-partitioning
    algorithm (a type's several concurrent instances are one column each,
    never a single column with overlapping jobs). Jobs the heuristic leaves
    unscheduled (partial coverage on a constrained instance) simply fall
    back to their big-M artificial, same as before this seeding existed.

    With running jobs the heuristic runs on the original instance (which
    respects their occupancy) and each pseudo-job opens its own chain at 0.
    """
    from .baselines import greedy_earliest_completion

    base = inst.base if isinstance(inst, _RunningAsJobs) else inst
    assignment = dict(greedy_earliest_completion(base).assignment)
    for jid in _running_ids(inst):
        assignment[jid] = (inst.F[jid][0], 0)
    job_by_id = {j.id: j for j in inst.jobs}

    by_type: dict[int, list[tuple[int, int, int]]] = {}
    for jid, (mid, t) in assignment.items():
        end = t + inst.p_occ[jid, mid]
        by_type.setdefault(mid, []).append((t, end, jid))

    columns: list[Column] = []
    for mid, items in by_type.items():
        items.sort()
        chain_end: list[int] = []
        chain_jobs: list[list[tuple[int, int]]] = []
        for start, end, jid in items:
            for i, ce in enumerate(chain_end):
                if ce <= start:
                    chain_end[i] = end
                    chain_jobs[i].append((jid, start))
                    break
            else:
                chain_end.append(end)
                chain_jobs.append([(jid, start)])
        for jobs in chain_jobs:
            cost1 = sum(
                job_by_id[jid].weight * (t + inst.p_occ[jid, mid] - job_by_id[jid].arrival)
                for jid, t in jobs
            )
            cost2 = sum(inst.c[jid, mid] for jid, t in jobs)
            columns.append(Column(type_id=mid, jobs=tuple(jobs), cost1=cost1, cost2=cost2))
    return columns


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

    labels: list[list[_Label]] = [[] for _ in range(H + 1)]
    labels[0].append(_Label(cost=0.0, mask=0, pred=None, job_id=None, t_prev=0))

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
    _BUCKET_CAP = 200

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
            for arc in arcs:
                if not (arc.start <= t < arc.stop):
                    continue
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
    best = best_label.cost - rho_m
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

    return best, Column(type_id=type_id, jobs=tuple(jobs_seq), cost1=cost1, cost2=cost2)


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
    f1[0] = _RelaxedLabel(cost=0.0, last_job=None)

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

    for t in range(H):
        l1 = f1[t]
        if l1 is None:
            continue
        update(t + 1, _RelaxedLabel(cost=l1.cost, last_job=l1.last_job))
        l2 = f2[t]
        if l2 is not None:
            update(t + 1, _RelaxedLabel(cost=l2.cost, last_job=l2.last_job))

        for arc in arcs:
            if not (arc.start <= t < arc.stop):
                continue
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
    return min(candidates) - rho_m


# ---------------------------------------------------------------------------
# Restricted master problem
# ---------------------------------------------------------------------------

@dataclass
class _RmpState:
    coverage: dict[int, gp.Constr]
    fleet: dict[int, gp.Constr]
    bound1: gp.Constr | None   # optional f1-cap row (pareto.py's box boundary / lex-min tie-break)
    bound2: gp.Constr | None   # optional f2-cap row — both may be active simultaneously
    art: dict[int, gp.Var]
    fixed_link: dict[tuple[int, int], gp.Constr] = field(default_factory=dict)
    z: dict[int, gp.Var] = field(default_factory=dict)
    cuts: dict[frozenset[int], gp.Constr] = field(default_factory=dict)
    columns: list[Column] = field(default_factory=list)
    vars: list[gp.Var] = field(default_factory=list)


def _add_column_var(
    mdl: gp.Model,
    col: Column,
    weight1: float,
    weight2: float,
    state: _RmpState,
    name: str,
) -> gp.Var:
    coeffs = [1.0]
    constrs: list[gp.Constr] = [state.fleet[col.type_id]]
    for job_id, _start in col.jobs:
        coeffs.append(1.0)
        constrs.append(state.coverage[job_id])
        link = state.fixed_link.get((job_id, col.type_id))
        if link is not None:
            coeffs.append(1.0)
            constrs.append(link)
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
) -> tuple[gp.Model, _RmpState]:
    formulations.configure_env()
    mdl = gp.Model("bpc-rmp", env=formulations._shared_env)
    mdl.setParam("OutputFlag", 0)

    coverage = {
        j.id: mdl.addConstr(gp.LinExpr() == 1.0, name=f"cover[{j.id}]")
        for j in inst.jobs
    }
    N = _fleet_sizes(inst)
    fleet = {
        itype.id: mdl.addConstr(gp.LinExpr() == float(N[itype.id]), name=f"fleet[{itype.id}]")
        for itype in inst.instance_types
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
    # Running (pseudo-)jobs never activate a fixed cost: no link row.
    running_ids = _running_ids(inst)
    fixed_link: dict[tuple[int, int], gp.Constr] = {}
    for itype in _fixed_cost_types(inst):
        for j in inst.jobs:
            if itype.id in inst.F[j.id] and j.id not in running_ids:
                fixed_link[j.id, itype.id] = mdl.addConstr(
                    gp.LinExpr() <= 0.0, name=f"use[{j.id},{itype.id}]"
                )
    mdl.update()

    z: dict[int, gp.Var] = {}
    for itype in _fixed_cost_types(inst):
        rows = [c for (jid, mid), c in fixed_link.items() if mid == itype.id]
        coeffs = [-1.0] * len(rows)
        constrs = list(rows)
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
                       art=art, fixed_link=fixed_link, z=z, cuts=cut_rows)

    for itype in inst.instance_types:
        null_col = Column(type_id=itype.id, jobs=(), cost1=0.0, cost2=0.0)
        var = _add_column_var(mdl, null_col, weight1, weight2, state,
                               name=f"lam[{itype.id}]_null")
        state.columns.append(null_col)
        state.vars.append(var)

    for i, col in enumerate(extra_columns):
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


def _run_column_generation(
    mdl: gp.Model,
    state: _RmpState,
    inst: Instance,
    weight1: float,
    weight2: float,
    arcs_by_type: dict[int, list[_Arc]],
    arrival_of: dict[int, float],
    weight_of: dict[int, float],
    max_iters: int,
    tol: float,
    forbidden: frozenset[tuple[int, int, int]] = frozenset(),
    pinned: dict[int, tuple[int, int]] | None = None,
    differ_pairs: frozenset[tuple[int, int]] = frozenset(),
    same_pairs: frozenset[tuple[int, int]] = frozenset(),
    use_relaxed_prescreen: bool = False,
) -> int:
    """Solve -> price every type -> add improving columns -> repeat, until no
    type prices out a negative-reduced-cost column. Returns the iteration
    count. Shared by the root relaxation and every branch-and-price node.

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
        mdl.optimize()
        if mdl.Status == GRB.INFEASIBLE:
            raise _RmpInfeasible
        if mdl.Status != GRB.OPTIMAL:
            raise RuntimeError(f"BPC RMP LP not optimal: status={mdl.Status}")

        sigma = {jid: c.Pi for jid, c in state.coverage.items()}
        rho = {mid: c.Pi for mid, c in state.fleet.items()}
        nu1 = state.bound1.Pi if state.bound1 is not None else None
        nu2 = state.bound2.Pi if state.bound2 is not None else None
        theta = {key: c.Pi for key, c in state.fixed_link.items()}
        cut_duals = {triple: c.Pi for triple, c in state.cuts.items()}

        improved = False
        for itype in inst.instance_types:
            mid = itype.id
            if use_relaxed_prescreen:
                relaxed_best = _price_type_relaxed(
                    inst, mid, arcs_by_type[mid], weight1, weight2,
                    sigma, nu1, nu2, rho[mid], theta,
                    arrival_of, weight_of, forbidden, pinned,
                )
                if relaxed_best >= -tol:
                    continue
            result = _price_type(
                inst, mid, arcs_by_type[mid], weight1, weight2,
                sigma, nu1, nu2, rho[mid], theta,
                arrival_of, weight_of, tol, forbidden, pinned,
                differ_pairs, same_pairs, cut_duals,
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
) -> RootResult:
    """Column-generate the root LP relaxation of the BPC master.

    Minimises ``weight1 * f1 + weight2 * f2``, optionally with an extra
    ``f1 <= bound1_value`` and/or ``f2 <= bound2_value`` row (both may be
    active at once) — mirrors ``pareto._make_lexmin_bounded``'s
    ``bound_expr_fn``/``bound_value``, generalised to two independent caps
    since ``pareto.py``'s box search needs both simultaneously (see
    ``make_bpc_lexmin_bounded``).

    No branching: this is the LP bound only, Phase 1 of the BPC plan.
    Running jobs are absorbed as pinned pseudo-jobs (see the module
    docstring), so ``model``/``state`` cover those too.
    """
    inst = _absorb_running_jobs(inst)
    if big_m is None:
        big_m = _big_m_for(inst)

    mdl, state = _build_rmp(inst, weight1, weight2, bound1_value, bound2_value, big_m,
                             extra_columns=_heuristic_seed_columns(inst))
    arrival_of = {j.id: j.arrival for j in inst.jobs}
    weight_of = {j.id: j.weight for j in inst.jobs}
    arcs_by_type = _build_arcs_by_type(inst)

    iterations = _run_column_generation(
        mdl, state, inst, weight1, weight2,
        arcs_by_type, arrival_of, weight_of, max_iters, tol,
        use_relaxed_prescreen=use_relaxed_prescreen,
    )

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
    status: str  # "integral" | "fractional" | "infeasible"
    objective: float
    columns: list[Column]
    assignment: dict[int, tuple[int, int]] | None = None
    branch_pair: tuple[int, int] | None = None          # Ryan-Foster (job_j, job_k)
    branch_var: tuple[int, int, int] | None = None       # Phase-2 fallback (job_id, type_id, start)
    branch_type: int | None = None                       # fractional z_m — see _solve_node
    incumbent: tuple[float, dict[int, tuple[int, int]]] | None = None  # see _rmp_integer_heuristic


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
    — on tiny_large's lex-min (f2, f1) phase 2 that took > 15 min although
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
    arcs_by_type: dict[int, list[_Arc]],
    arrival_of: dict[int, float],
    weight_of: dict[int, float],
    max_iters: int,
    tol: float,
    big_m: float,
    use_relaxed_prescreen: bool = False,
    heuristic_time_limit: float | None = None,
) -> _NodeResult:
    """Column generation at one node, then its status and branching choice.

    ``heuristic_time_limit`` — when set and the node is fractional, also run
    :func:`_rmp_integer_heuristic` with that limit; its result is the
    returned ``incumbent``.
    """
    valid_seed = [
        c for c in seed_pool
        if _column_allowed(c, forbidden, pinned, differ_pairs, same_pairs)
    ]
    mdl, state = _build_rmp(inst, weight1, weight2, bound1_value, bound2_value, big_m,
                             extra_columns=valid_seed,
                             forbidden_types=forbidden_types, forced_types=forced_types,
                             cuts=cuts)
    try:
        _run_column_generation(
            mdl, state, inst, weight1, weight2,
            arcs_by_type, arrival_of, weight_of, max_iters, tol,
            forbidden, pinned, differ_pairs, same_pairs,
            use_relaxed_prescreen,
        )
    except _RmpInfeasible:
        return _NodeResult(status="infeasible", objective=float("inf"), columns=seed_pool)

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
            return _NodeResult(status="fractional", objective=mdl.ObjVal,
                                columns=state.columns, branch_type=frac_z[0])

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
    ]
    if pair_candidates:
        _dist, branch_j, branch_k = min(pair_candidates)
        result = _NodeResult(status="fractional", objective=mdl.ObjVal, columns=state.columns,
                             branch_pair=(branch_j, branch_k))
    else:
        _dist, branch_j, branch_m, branch_t = min(
            (abs(v - 0.5), jid, mid, t)
            for (jid, mid, t), v in agg.items()
            if jid in unsettled
        )
        result = _NodeResult(status="fractional", objective=mdl.ObjVal, columns=state.columns,
                             branch_var=(branch_j, branch_m, branch_t))
    if heuristic_time_limit is not None:
        result.incumbent = _rmp_integer_heuristic(
            mdl, state, inst, weight1, weight2, bound1_value, bound2_value,
            heuristic_time_limit,
        )
    return result


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
    arcs_by_type: dict[int, list[_Arc]],
    arrival_of: dict[int, float],
    weight_of: dict[int, float],
    max_iters: int,
    tol: float,
    max_cut_rounds: int,
    max_cuts_per_round: int,
    max_total_cuts: int,
    use_relaxed_prescreen: bool = False,
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
    pool = _heuristic_seed_columns(inst)
    for _round in range(max_cut_rounds):
        mdl, state = _build_rmp(inst, weight1, weight2, bound1_value, bound2_value, big_m,
                                 extra_columns=pool, cuts=cuts)
        try:
            _run_column_generation(mdl, state, inst, weight1, weight2,
                                    arcs_by_type, arrival_of, weight_of, max_iters, tol,
                                    use_relaxed_prescreen=use_relaxed_prescreen)
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
) -> tuple[float, dict[int, tuple[int, int]] | None]:
    """Exact branch-and-price. Returns ``(objective, assignment)``, or
    ``(inf, None)`` if the instance (restricted to any bound row) is
    infeasible. See the module docstring for the branching scheme.

    ``tol`` defaults looser than the root relaxation's (1e-5 vs. Phase 1's
    1e-7): with ``weight1=0`` (or any node deep in an "up"-heavy branch,
    where most jobs are pinned), f2 alone often doesn't depend on a job's
    exact start time, so many columns tie exactly — at 1e-7 that tie noise
    made column generation cycle (repeatedly re-adding near-zero-reduced-cost
    columns) instead of converging on tiny_large/tiny_xlarge. 1e-5 is still
    two orders of magnitude tighter than the ``abs=1e-4`` comparison
    tolerance the tests check f1/f2 against.

    ``use_cuts`` — separate rank-1 subset-row cuts at the root before
    branching (see ``_root_cuts_and_price``/``_separate_sr_cuts`` and the
    module docstring's "Phase 5" section). Cuts never change the optimum —
    only the bound/search order — but each *active* cut costs the pricing
    DP an extra dominance-signature bit (a label triggered on a cut and one
    that isn't become mutually non-dominating), which shrinks how much the
    elementary-path dominance can prune. On tiny_medium.json this measurably
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

    Pruning uses objective integrality: with ``weight2 == 0``, f1 =
    Σ w_j C_j − Σ w_j a_j over integer completions C_j and integral weights
    (an assumption :meth:`Instance.build` enforces), so two schedules'
    objectives differ by a multiple of ``weight1``. A node then holds a strictly better schedule only if its
    bound is at most ``best − weight1``; the 1e-3 margin absorbs the column
    generation's ``tol``-level bound error. Without it, bounds within that
    error of the optimum (e.g. 187.51663 vs 187.516667 on tiny_large) never
    prune.
    Running jobs are absorbed as pinned pseudo-jobs (see the module
    docstring); the returned assignment holds the instance's own jobs only.
    """
    inst = _absorb_running_jobs(inst)
    running_ids = _running_ids(inst)
    if big_m is None:
        big_m = _big_m_for(inst)
    prune_gap = tol
    if weight2 == 0 and weight1 > 0:
        prune_gap = max(tol, weight1 - 1e-3)
    arcs_by_type = _build_arcs_by_type(inst)
    arrival_of = {j.id: j.arrival for j in inst.jobs}
    weight_of = {j.id: j.weight for j in inst.jobs}

    cuts: frozenset[frozenset[int]] = frozenset()
    root_pool = _heuristic_seed_columns(inst)
    if use_cuts:
        feasible, cuts, root_pool = _root_cuts_and_price(
            inst, weight1, weight2, bound1_value, bound2_value, big_m,
            arcs_by_type, arrival_of, weight_of, max_iters_per_node, tol,
            max_cut_rounds, max_cuts_per_round, max_total_cuts,
            use_relaxed_prescreen,
        )
        if not feasible:
            return float("inf"), None

    Stack = list[tuple[
        frozenset[tuple[int, int, int]], dict[int, tuple[int, int]],
        frozenset[tuple[int, int]], frozenset[tuple[int, int]],
        frozenset[int], frozenset[int],
        list[Column],
    ]]
    stack: Stack = [
        (frozenset(), {}, frozenset(), frozenset(), frozenset(), frozenset(), root_pool)
    ]
    best_obj = float("inf")
    best_assignment: dict[int, tuple[int, int]] | None = None
    n_nodes = 0

    while stack:
        if n_nodes >= max_nodes:
            raise RuntimeError("BPC branch-and-price exceeded max_nodes")
        forbidden, pinned, differ_pairs, same_pairs, forbidden_types, forced_types, pool = stack.pop()
        n_nodes += 1

        node = _solve_node(
            inst, weight1, weight2, bound1_value, bound2_value,
            forbidden, pinned, differ_pairs, same_pairs, forbidden_types, forced_types,
            cuts, pool,
            arcs_by_type, arrival_of, weight_of, max_iters_per_node, tol, big_m,
            use_relaxed_prescreen,
            heuristic_time_limit if heuristic_time_limit > 0 else None,
        )
        if node.incumbent is not None and node.incumbent[0] < best_obj:
            best_obj, best_assignment = node.incumbent
        if node.status == "infeasible" or node.objective >= best_obj - prune_gap:
            continue

        if node.status == "integral":
            best_obj = node.objective
            best_assignment = node.assignment
            continue

        if node.branch_type is not None:
            m = node.branch_type
            stack.append((forbidden, pinned, differ_pairs, same_pairs,
                          forbidden_types | {m}, forced_types, node.columns))          # z_m = 0
            stack.append((forbidden, pinned, differ_pairs, same_pairs,
                          forbidden_types, forced_types | {m}, node.columns))          # z_m = 1
        elif node.branch_pair is not None:
            j, k = node.branch_pair
            stack.append((forbidden, pinned, differ_pairs | {(j, k)}, same_pairs,
                          forbidden_types, forced_types, node.columns))                # differ
            stack.append((forbidden, pinned, differ_pairs, same_pairs | {(j, k)},
                          forbidden_types, forced_types, node.columns))                # same
        else:
            jid, mid, t = node.branch_var
            stack.append((forbidden | {(jid, mid, t)}, pinned, differ_pairs, same_pairs,
                          forbidden_types, forced_types, node.columns))                # down
            stack.append((forbidden, {**pinned, jid: (mid, t)}, differ_pairs, same_pairs,
                          forbidden_types, forced_types, node.columns))                # up

    if best_assignment is not None:
        best_assignment = {jid: a for jid, a in best_assignment.items()
                           if jid not in running_ids}
    return best_obj, best_assignment


def _assignment_to_solution(inst: Instance, objective: float,
                             assignment: dict[int, tuple[int, int]] | None) -> Solution:
    if assignment is None:
        return Solution(status="infeasible", objective=None, f1=None, f2=None,
                         assignment={}, completion={})
    completion = {jid: t + inst.p_occ[jid, mid] for jid, (mid, t) in assignment.items()}
    return Solution(
        status="optimal", objective=objective,
        f1=inst.f1_of(completion), f2=inst.f2_of(assignment),
        assignment=assignment, completion=completion,
    )


def solve_weighted_sum_bpc(inst: Instance, weight1: float, weight2: float, **kwargs) -> Solution:
    """Exact branch-and-price solve of ``min weight1*f1 + weight2*f2``."""
    obj, assignment = _branch_and_price(inst, weight1, weight2, **kwargs)
    return _assignment_to_solution(inst, obj, assignment)


def _lex_tiebreak(inst: Instance, assignment1: dict[int, tuple[int, int]],
                  weight1: float, weight2: float, **kwargs) -> Solution:
    """Phase 2 of a lex-min: ``_branch_and_price(inst, weight1, weight2,
    **kwargs)`` with the bound row fixing phase 1's optimum.

    The result is ``"optimal"`` only when phase 2 itself finished. If it
    raised (``max_nodes``/``max_iters_per_node``) or came back infeasible —
    impossible in exact arithmetic, since phase 1's schedule satisfies the
    bound — phase 1's schedule is returned as ``"feasible"``: it is
    primary-optimal, but its secondary objective is unproven, so it may be
    weakly dominated and must not be trusted as a Pareto point.
    """
    try:
        _obj2, assignment2 = _branch_and_price(inst, weight1, weight2, **kwargs)
    except RuntimeError:
        assignment2 = None
    if assignment2 is None:
        return replace(_assignment_to_solution(inst, None, assignment1), status="feasible")
    return _assignment_to_solution(inst, None, assignment2)


def solve_f1_bpc(inst: Instance, **kwargs) -> Solution:
    """Lex-min (f1, f2) via branch-and-price — mirrors ``model.solve_f1``.

    Status ``"feasible"`` when the tie-break phase failed (see
    :func:`_lex_tiebreak`).
    """
    f1_star, assignment1 = _branch_and_price(inst, 1.0, 0.0, **kwargs)
    if assignment1 is None:
        return _assignment_to_solution(inst, f1_star, None)
    return _lex_tiebreak(inst, assignment1, 0.0, 1.0,
                         bound1_value=f1_star, **kwargs)


def solve_f2_bpc(inst: Instance, **kwargs) -> Solution:
    """Lex-min (f2, f1) via branch-and-price — mirrors ``model.solve_f2``.

    Status ``"feasible"`` when the tie-break phase failed (see
    :func:`_lex_tiebreak`).
    """
    f2_star, assignment1 = _branch_and_price(inst, 0.0, 1.0, **kwargs)
    if assignment1 is None:
        return _assignment_to_solution(inst, f2_star, None)
    return _lex_tiebreak(inst, assignment1, 1.0, 0.0,
                         bound2_value=f2_star, **kwargs)


# ---------------------------------------------------------------------------
# Phase 4: pareto.py integration
# ---------------------------------------------------------------------------

def make_bpc_lexmin_bounded(
    inst: Instance,
    **bpc_kwargs,
):
    """Build a BPC-backed ``lexmin_bounded`` closure with the exact same
    shape as ``pareto._make_lexmin_bounded``'s, so ``true_pareto_frontier``/
    ``map_pareto_frontier`` can select this engine via ``engine="bpc"`` with
    zero changes to the box-search algorithm itself.

    Returns ``(lexmin_bounded, counters)``, ``counters`` holding a live
    ``"n_solves"`` entry (incremented once per ``_branch_and_price`` call —
    ``"build_time"`` is always 0.0, since BPC has no single "model build"
    step comparable to the Gurobi engine's; unlike that engine this doesn't
    yet support ``per_solve_time_limit``/``n_seed_retries``-style graceful
    degradation under a wall-clock cap — a hard sub-problem currently just
    keeps running, or eventually raises via ``max_nodes``/``max_iters_per_node``
    in ``bpc_kwargs``, which this catches and reports as an inconclusive
    ("feasible"-status) result rather than letting it propagate and abort
    the whole search). That holds for either phase: a schedule whose
    tie-break phase failed is primary-optimal only, so it is returned as
    ``"feasible"`` too (see :func:`_lex_tiebreak`) and the box search drops
    the box instead of recording a possibly dominated point.

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
    counters = {"n_solves": 0, "build_time": 0.0}

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
        try:
            obj1, assignment1 = _branch_and_price(
                inst, w1, w2, bound1_value=b1, bound2_value=b2, **bpc_kwargs,
            )
        except RuntimeError:
            return Solution(status="feasible", objective=None, f1=None, f2=None,
                             assignment={}, completion={})
        counters["n_solves"] += 1

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
        return _lex_tiebreak(inst, assignment1, w1s, w2s,
                             bound1_value=b1, bound2_value=b2, **bpc_kwargs)

    return lexmin_bounded, counters
