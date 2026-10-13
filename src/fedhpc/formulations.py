"""Math formulations for the FED-HPC MIP model.

Each formulation provides a `build` method that constructs the Gurobi model and
returns (model, vars_dict).  The objectives f1/f2 are defined identically across
all formulations (they depend only on x_jmt), so they are implemented in the
abstract base class.

Available formulations
----------------------
SpaceTimeFormulation  — space-time network with flow conservation (fedhpc-1.pdf).
OccupancyFormulation  — occupancy-equality formulation (original implementation).

Objective hints
---------------
``build(inst, objective=...)`` may be told which problem the model will solve,
enabling exact objective-specific reductions (see
:class:`SpaceTimeFormulation`):

* ``None``                 — unknown; only objective-independent reductions.
* ``(a, b)``               — min ``a·f1 + b·f2`` (a, b >= 0), optionally with
                             extra upper bounds on f1 (never on f2).
* :data:`LEX_F1`           — lex-min (f1, f2), optionally with extra upper
                             bounds on f1 (never on f2).
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import Counter

import gurobipy as gp
from gurobipy import GRB

from .data import Instance

try:
    from . import _graph as _graph_ext
except ImportError:  # source-only installs retain the pure-Python fallback
    _graph_ext = None

LEX_F1 = "lex_f1"
"""Objective hint: the model solves lex-min (f1, f2) (see module docstring)."""

Objective = tuple[float, float] | str | None

# ── Shared Gurobi environment ─────────────────────────────────────────────────
# Created once via configure_env() in cli.main() before any model is built.
# Using gp.Env(empty=True) is the only reliable way to suppress the WLS licence
# banner, because it lets us set OutputFlag=0 *before* the C-level environment
# reads the licence file (gp.setParam() is too late — the banner fires during
# the first C-level init).
_shared_env: gp.Env | None = None


def configure_env(verbose: bool = False) -> None:
    """Initialise the shared Gurobi environment with the requested verbosity.

    Must be called once before any Formulation.build() invocation.
    Idempotent: a second call is a no-op.
    """
    global _shared_env
    if _shared_env is not None:
        return
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1 if verbose else 0)
    env.start()
    _shared_env = env


class XVars(dict):
    """``x[j, m, t]`` variable dict that also carries the fixed-cost binaries.

    ``z[m]`` (only for types with ``fixed_cost > 0``) rides along on the x dict
    so :meth:`Formulation.f2_expr` can charge ``U_m z_m`` while every caller
    keeps passing just ``vars["x"]``.
    """

    z: dict[int, gp.Var]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.z = {}


def add_fixed_cost_vars(mdl: gp.Model, inst: Instance, x: XVars) -> None:
    """Add ``z[m] ∈ {0,1}`` per type with ``fixed_cost > 0`` and link it to x.

    Linking uses the aggregated form ``Σ_t x[j,m,t] ≤ z[m]`` per (job, type)
    pair: one row per pair instead of one per start time, and at least as
    tight in the LP as the disaggregated ``x[j,m,t] ≤ z[m]``.
    """
    by_pair: dict[tuple[int, int], list[gp.Var]] = {}
    for (jid, mid, _t), v in x.items():
        by_pair.setdefault((jid, mid), []).append(v)
    for itype in inst.instance_types:
        if not itype.fixed_cost:
            continue
        mid = itype.id
        z = x.z[mid] = mdl.addVar(vtype=GRB.BINARY, name=f"z[{mid}]")
        for j in inst.jobs:
            if (j.id, mid) in by_pair:
                mdl.addConstr(
                    gp.quicksum(by_pair[j.id, mid]) <= z,
                    name=f"use[{j.id},{mid}]",
                )


class Formulation(ABC):
    """Abstract base class for FED-HPC MIP formulations.

    Subclasses must implement `build`; the objective expressions and solution
    extraction helpers are shared and live here.
    """

    @abstractmethod
    def build(self, inst: Instance, objective: Objective = None) -> tuple[gp.Model, dict]:
        """Construct the Gurobi model.

        ``objective`` is an optional hint about the problem the caller will
        solve (see the module docstring); formulations may ignore it.

        Returns
        -------
        mdl  : gp.Model with all variables and constraints added (not yet solved).
        vars : dict whose ``"x"`` key holds the x_jmt decision variables.
               Subclasses may include additional keys (e.g. ``"w"``, ``"K"``).
        """

    # ------------------------------------------------------------------
    # Objective expressions (common to all formulations)
    # ------------------------------------------------------------------

    # Both objectives iterate x's own keys rather than rebuilding them from
    # inst.T, so a formulation may create x for a subset of T_jm only (see
    # SpaceTimeFormulation(normal_patterns=True)).

    def f1_expr(self, inst: Instance, x: dict) -> gp.LinExpr:
        """Weighted turnaround: Σ_{j,m,t} w_j (t + p^occ_jm − a_j) x_jmt  (eq. 11;
        w_j = 1 by default)."""
        job = {j.id: j for j in inst.jobs}
        return gp.quicksum(
            job[jid].weight * (t + inst.p_occ[jid, m_id] - job[jid].arrival) * v
            for (jid, m_id, t), v in x.items()
        )

    def f2_expr(self, inst: Instance, x: dict) -> gp.LinExpr:
        """Total monetary cost: Σ_{j,m,t} c^proc_jm x_jmt  (eq. 13)
        + Σ_m U_m z_m for types with a fixed usage cost."""
        z = getattr(x, "z", {})
        missing = [m.id for m in inst.instance_types if m.fixed_cost and m.id not in z]
        if missing:
            raise ValueError(
                f"types {missing} have fixed_cost but the model has no z vars — "
                "build it with a Formulation that calls add_fixed_cost_vars"
            )
        fixed = {m.id: m.fixed_cost for m in inst.instance_types}
        return gp.quicksum(
            inst.c[jid, m_id] * v for (jid, m_id, _t), v in x.items()
        ) + gp.quicksum(fixed[mid] * zm for mid, zm in z.items())


def normal_pattern_starts(inst: Instance) -> dict[int, bytearray]:
    """Per type m, a 0/1 mask over {0..H} of the *normal patterns*: times at
    which some left-shifted schedule can start a job on m.

    Seeds are every job's earliest start slot on m and every running-job
    completion slot on m; the set is closed under ``q -> q + p^occ_jm`` for
    jobs j with ``q >= a_j`` (Kramer et al. 2020, Algorithm 1). Every job start
    in a left-shifted schedule — each job as early as capacity allows — is a
    release slot or a completion slot on its type, so restricting x to these
    times loses no optimum for any objective nondecreasing in start times
    (f1 is increasing, f2/budget are time-independent). A superset is always
    safe: processing times are grouped per distinct p^occ with the smallest
    release among the jobs sharing it.
    """
    H = inst.horizon
    masks: dict[int, bytearray] = {}
    for itype in inst.instance_types:
        mid = itype.id
        min_rel: dict[int, int] = {}          # p_occ -> earliest start slot
        seeds = set()
        for j in inst.jobs:
            if mid not in inst.F[j.id]:
                continue
            a = inst.T[j.id, mid].start
            seeds.add(a)
            p = inst.p_occ[j.id, mid]
            min_rel[p] = min(min_rel.get(p, a), a)
        seeds.update(v for (m2, v) in inst.rj_arrivals if m2 == mid)
        P = bytearray(H + 1)
        for q in seeds:
            if q <= H:
                P[q] = 1
        items = sorted(min_rel.items())
        for q in range(H + 1):
            if P[q]:
                for p, a in items:
                    if q >= a and q + p <= H:
                        P[q + p] = 1
        masks[mid] = P
    return masks


def _normal_pattern_graph_data(inst: Instance):
    """Return ``(S, nodes, loss_arcs)`` for the compressed AF graph.

    The compiled path performs the pseudo-polynomial closure and graph
    bookkeeping. The fallback preserves source-only usability and is also a
    readable reference implementation for equivalence tests.
    """
    if _graph_ext is not None:
        specs = [
            (j.id, mid, inst.T[j.id, mid].start, inst.T[j.id, mid].stop,
             inst.p_occ[j.id, mid])
            for j in inst.jobs for mid in inst.F[j.id]
        ]
        raw_s, raw_g = _graph_ext.build_normal_pattern_graph(
            inst.horizon, max(m.id for m in inst.instance_types) + 1,
            specs, list(inst.rj_arrivals),
        )
        valid_types = {m.id for m in inst.instance_types}
        S = {(jid, mid): frozenset(ts) for jid, mid, ts in raw_s}
        nodes = {mid: set(ns) for mid, ns, _arcs in raw_g if mid in valid_types}
        loss = {mid: list(arcs) for mid, _ns, arcs in raw_g if mid in valid_types}
        return S, nodes, loss

    masks = normal_pattern_starts(inst)
    S = {
        (j.id, mid): frozenset(t for t in inst.T[j.id, mid] if masks[mid][t])
        for j in inst.jobs for mid in inst.F[j.id]
    }
    H = inst.horizon
    nodes, loss = {}, {}
    for itype in inst.instance_types:
        mid = itype.id
        events = {0, H}
        events.update(inst.T[j.id, mid].start for j in inst.jobs if mid in inst.F[j.id])
        events.update(v for (m2, v) in inst.rj_arrivals if m2 == mid)
        ev = sorted(events)
        # every normal-pattern time is a node, as in _graph.cpp: events are
        # pattern seeds, so every loss arc ends on a node, and the closure
        # contains every completion of an allowed start
        ns = {0, H}
        ns.update(q for q in range(H + 1) if masks[mid][q])
        nodes[mid] = ns
        loss[mid] = [(q, next(v for v in ev if v > q)) for q in sorted(ns) if q < H]
    return S, nodes, loss


def budget_regime(inst: Instance) -> str | None:
    """``"zero"`` if B = 0, ``"free"`` if B can never bind, else ``None``.

    "Never binds" means B covers every job on its costliest feasible type plus
    every fixed cost — the worst schedule there is.
    """
    if inst.budget == 0:
        return "zero"
    worst = sum(max(inst.c[j.id, m] for m in inst.F[j.id]) for j in inst.jobs)
    worst += sum(m.fixed_cost for m in inst.instance_types)
    return "free" if inst.budget >= worst else None


def nonnegative_data(inst: Instance) -> bool:
    """Weights, costs and fixed costs all >= 0 — a hypothesis of every
    dominance reduction (and of :func:`budget_regime`'s "never binds")."""
    return (
        all(j.weight >= 0 for j in inst.jobs)
        and all(c >= 0 for c in inst.c.values())
        and all(m.fixed_cost >= 0 for m in inst.instance_types)
    )


def unbounded_types(inst: Instance) -> set[int]:
    """Types whose capacity can never bind: unlimited, or with at least as many
    instances free at t = 0 as there are jobs feasible on them (running jobs
    only ever release capacity)."""
    n_feasible = Counter(m for j in inst.jobs for m in inst.F[j.id])
    return {
        m.id for m in inst.instance_types
        if m.capacity is None
        or m.capacity - inst.n_running.get(m.id, 0) >= n_feasible[m.id]
    }


def _truncate(starts, cut: int):
    """Starts of ``starts`` strictly before ``cut`` (range stays a range)."""
    if isinstance(starts, range):
        return range(starts.start, max(starts.start, min(starts.stop, cut)))
    return frozenset(t for t in starts if t < cut)


def _first_start_reaching(score_rate: float, score_at0: float, target: float,
                          lo: int) -> float:
    """Smallest integer t >= lo with score_at0 + score_rate·t >= target
    (inf if never)."""
    if score_rate <= 0:
        return lo if score_at0 >= target else math.inf
    t = max(lo, math.ceil((target - score_at0) / score_rate))
    while t > lo and score_at0 + score_rate * (t - 1) >= target:
        t -= 1
    while score_at0 + score_rate * t < target:
        t += 1
    return t


class SpaceTimeFormulation(Formulation):
    """Space-time network formulation with flow conservation (fed-hpc.pdf).

    Implements exactly the formulation in fed-hpc.pdf (Section 1.6–1.9).

    Decision variables
    ------------------
    x[j,m,t] ∈ {0,1}   — job j starts on type m at time t  (eq. 21)
    y[m,t]   ∈ Z≥0     — instances of type m waiting during [t, t+1)  (eq. 22)

    Boundary conditions (parameters, not variables):
        y_{m,−1} := N_m  and  y_{m,H} := N_m   (eq. 14)
    where N_m = itype.capacity, or len(jobs) when capacity is None.

    Constraints
    -----------
    (18) Unique job assignment.
    (19) Flow conservation at every v ∈ T = {0, …, H}.
    (20) Global budget.

    Options (all off by default — the plain formulation above)
    ----------------------------------------------------------
    continuous_idle : y relaxed to continuous ≥ 0. Integrality is implied: with
                      x binary and integer N_m / running-job completions, flow
                      conservation fixes every y[m,t] to an integer.
    cover_ge        : assignment rows as Σ x ≥ 1 (Kramer et al. 2020's AF). No
                      optimal solution duplicates a job, since each x carries a
                      strictly positive f1 term and a nonnegative cost.
    normal_patterns : x only for starts in :func:`normal_pattern_starts` —
                      exact (see there); shrinks x and lets many flow rows
                      degenerate into pure idle-arc links.
    compressed_flow : use the paper's normal-pattern graph, including jump
                      loss arcs between release/capacity-event times.  Unlike
                      ``normal_patterns`` this removes the full-grid idle
                      variables and flow rows as well as unreachable x arcs.

    Dominance reductions (on by default, exact)
    -------------------------------------------
    Active only when the budget cannot bind (:func:`budget_regime` is
    ``"free"``, or ``"zero"``, in which case every option with a positive cost
    or fixed cost is removed first) and weights/costs are nonnegative
    (:func:`nonnegative_data`); otherwise all four are off and the plain
    model is built. The budget row is omitted whenever they are
    active. An *unbounded* type is one whose capacity never binds
    (:func:`unbounded_types`), e.g. cloud.

    collapse_unbounded  (R1) : a job on an unbounded type starts at its
                      release — later starts only raise f1 and cost is
                      time-independent — so one x per (j, m) and no flow
                      network for that type.
    pareto_unbounded    (R2) : per job, drop an unbounded option whose
                      (earliest completion, cost) is weakly dominated by a
                      zero-fixed-cost unbounded option. Objective-independent.
    rejection_deadlines (R3) : with ``objective=(a, b)``, drop any start (on
                      any type) that some unbounded option beats on both
                      completion time and score a·w_j·(C − a_j) + b·(c + U):
                      moving the job there frees capacity and does not raise
                      f1 or the scalar objective. On capacitated types this is
                      a per-job deadline.
    lex_deadlines            : with ``objective=LEX_F1``, a job either finishes
                      no later than its fastest unbounded completion C*_j, or
                      is not on a capacitated type; on unbounded types only
                      options completing at C*_j remain. Jobs with
                      w_j = 0 are exempt: a later, cheaper finish does not
                      raise f1 for them.

    R3 and the lex rule never raise f1, so callers may add f1 upper bounds;
    they are invalid under an f2 upper bound (that is a budget), so callers
    must pass ``objective=None`` then.
    """

    def __init__(self, *, continuous_idle: bool = False, cover_ge: bool = False,
                 normal_patterns: bool = False, compressed_flow: bool = False,
                 collapse_unbounded: bool = True, pareto_unbounded: bool = True,
                 rejection_deadlines: bool = True, lex_deadlines: bool = True):
        self.continuous_idle = continuous_idle
        self.cover_ge = cover_ge
        self.normal_patterns = normal_patterns
        self.compressed_flow = compressed_flow
        self.collapse_unbounded = collapse_unbounded
        self.pareto_unbounded = pareto_unbounded
        self.rejection_deadlines = rejection_deadlines
        self.lex_deadlines = lex_deadlines

    def _reduce(self, inst: Instance, S, regime: str, objective: Objective):
        """Apply R1/R2/R3/lex to the start sets; returns ``(S', applied)``
        where ``applied`` says whether an objective-specific rule ran."""
        unb = unbounded_types(inst)
        fixed = {m.id: m.fixed_cost for m in inst.instance_types}
        lex = self.lex_deadlines and objective == LEX_F1
        weighted = (self.rejection_deadlines and isinstance(objective, tuple)
                    and min(objective) >= 0)
        if weighted:
            a, b = objective
        out = {}
        for j in inst.jobs:
            Sj = {m: S[j.id, m] for m in inst.F[j.id] if S[j.id, m]}
            if regime == "zero":
                Sj = {m: ts for m, ts in Sj.items()
                      if inst.c[j.id, m] <= 0 and not fixed[m]}
            cloud = [m for m in Sj if m in unb]
            if self.collapse_unbounded:
                for m in cloud:
                    t0 = min(Sj[m])
                    Sj[m] = range(t0, t0 + 1)
            C = {m: min(Sj[m]) + inst.p_occ[j.id, m] for m in cloud}
            if self.pareto_unbounded:
                kept, best_c = [], math.inf
                for m in sorted(cloud, key=lambda m: (C[m], inst.c[j.id, m],
                                                     fixed[m] > 0, m)):
                    if inst.c[j.id, m] >= best_c:
                        continue
                    kept.append(m)
                    if not fixed[m]:
                        best_c = inst.c[j.id, m]
                cloud = kept
            if lex and cloud and j.weight > 0:
                c_star = min(C[m] for m in cloud)
                cloud = [m for m in cloud if C[m] == c_star]
                for m in Sj:
                    if m not in unb:
                        Sj[m] = _truncate(Sj[m], c_star - inst.p_occ[j.id, m] + 1)
            elif weighted and cloud:
                aw = a * j.weight

                def score_at0(m):  # score of starting on m at t = 0
                    return aw * (inst.p_occ[j.id, m] - j.arrival) + b * inst.c[j.id, m]

                # R_m: what moving the job to unbounded m costs at most
                R = {m: score_at0(m) + aw * (C[m] - inst.p_occ[j.id, m])
                     + b * fixed[m] for m in cloud}
                kept = []
                for m in sorted(cloud, key=lambda m: (R[m], C[m], m)):
                    own = R[m] - b * fixed[m]
                    if not any(C[k] <= C[m] and R[k] <= own for k in kept):
                        kept.append(m)
                cloud = kept
                for m in Sj:
                    if m in unb:
                        continue
                    p = inst.p_occ[j.id, m]
                    lo = min(Sj[m])
                    cut = min(
                        max(C[k] - p,
                            _first_start_reaching(aw, score_at0(m), R[k], lo))
                        for k in cloud
                    )
                    if cut != math.inf:
                        Sj[m] = _truncate(Sj[m], cut)
            for m, ts in Sj.items():
                if ts and (m not in unb or m in cloud):
                    out[j.id, m] = ts
        return out, lex or weighted

    def build(self, inst: Instance, objective: Objective = None) -> tuple[gp.Model, dict]:
        H = inst.horizon
        # S[j,m]: allowed start times (T_jm, or its normal-pattern subset)
        if self.normal_patterns or self.compressed_flow:
            S, pattern_nodes, pattern_loss = _normal_pattern_graph_data(inst)
        else:
            S = inst.T
        reducing = any((self.collapse_unbounded, self.pareto_unbounded,
                        self.rejection_deadlines, self.lex_deadlines))
        regime = budget_regime(inst) if reducing and nonnegative_data(inst) else None
        objective_reduced = False
        if regime is not None:
            S, objective_reduced = self._reduce(inst, S, regime, objective)
        else:
            S = {(j.id, m): S[j.id, m] for j in inst.jobs for m in inst.F[j.id]}
        # Types whose flow network is dropped (R1): no x arc touches it.
        no_flow = unbounded_types(inst) if regime is not None and self.collapse_unbounded else set()
        mdl = gp.Model("fedhpc", env=_shared_env)
        mdl.setParam("OutputFlag", 0)

        # N_m: boundary flow (eq. 14). Use job count as upper bound when unlimited.
        n_jobs = len(inst.jobs)
        N = {
            itype.id: itype.capacity if itype.capacity is not None else n_jobs
            for itype in inst.instance_types
        }

        # ---- Decision variables -------------------------------------------

        # x[j,m,t] ∈ {0,1}
        x = XVars(
            ((j.id, m_id, t), mdl.addVar(
                vtype=GRB.BINARY, name=f"x[{j.id},{m_id},{t}]"
            ))
            for j in inst.jobs
            for m_id in inst.F[j.id] if (j.id, m_id) in S
            for t in S[j.id, m_id]
        )
        flow_types = [m for m in inst.instance_types if m.id not in no_flow]

        if self.compressed_flow:
            # Paper AF loss arcs: from every normal-pattern node q to the next
            # release/capacity event.  Flow can wait across an interval in one
            # continuous variable instead of H unit idle arcs and H rows.
            loss_arcs = pattern_loss
            nodes = pattern_nodes
            y = {
                (mid, q, r): mdl.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=N[mid],
                                        name=f"y[{mid},{q},{r}]")
                for mid, arcs in loss_arcs.items() if mid not in no_flow
                for q, r in arcs
            }
            loss_in = {(mid, v): [] for mid in nodes for v in nodes[mid]}
            loss_out = {(mid, v): [] for mid in nodes for v in nodes[mid]}
            for mid, arcs in loss_arcs.items():
                if mid in no_flow:
                    continue
                for q, r in arcs:
                    loss_out[mid, q].append(y[mid, q, r])
                    loss_in[mid, r].append(y[mid, q, r])
        else:
            loss_arcs = {}
            nodes = {}
            loss_in = loss_out = {}
            # y[m,t] ∈ Z≥0 — waiting instances during [t, t+1)
            y = {
                (itype.id, t): mdl.addVar(
                    vtype=GRB.CONTINUOUS if self.continuous_idle else GRB.INTEGER,
                    lb=0, name=f"y[{itype.id},{t}]"
                )
                for itype in flow_types for t in range(H)
            }

        mdl.update()

        # ---- Constraints --------------------------------------------------

        # (18) Unique assignment: each job processed exactly once
        for j in inst.jobs:
            cover = gp.quicksum(
                x[j.id, m_id, t]
                for m_id in inst.F[j.id] if (j.id, m_id) in S
                for t in S[j.id, m_id]
            )
            mdl.addConstr(
                cover >= 1 if self.cover_ge else cover == 1,
                name=f"assign[{j.id}]",
            )

        # (19) Flow conservation for all v ∈ T = {0, …, H}:
        #   y_{m,v−1} + Σ_{j: v−p^occ_jm ∈ T_jm} x_{jm, v−p^occ_jm} + rj_arrivals[m,v]
        #             = y_{m,v} + Σ_{j: v ∈ T_jm} x_jmv
        # Boundary: y_{m,−1} = N_m − n_running[m]  (reduced by pre-occupied slots)
        #           y_{m,H}  = N_m                  (terminal: all slots idle again)
        # rj_arrivals[m,v]: running jobs that complete at v (ghost completions).
        #
        # jobs_by_type[mid] is hoisted out of the v-loop so "mid in F[j]" is
        # tested once per (job, type) instead of once per (job, type, v) — a
        # factor-of-H reduction in membership checks on large instances.
        jobs_by_type = {
            itype.id: [j for j in inst.jobs if (j.id, itype.id) in S]
            for itype in flow_types
        }
        for itype in flow_types:
            mid = itype.id
            nm = N[mid]
            # Slots available to new jobs at the very start of the horizon
            nm_available = max(0, nm - inst.n_running.get(mid, 0))
            jobs_mid = jobs_by_type[mid]
            flow_nodes = sorted(nodes[mid]) if self.compressed_flow else range(H + 1)
            for v in flow_nodes:
                if self.compressed_flow:
                    incoming_idle = gp.quicksum(loss_in[mid, v])
                    outgoing_idle = gp.quicksum(loss_out[mid, v])
                    lhs_wait = nm_available if v == 0 else incoming_idle
                    rhs_wait = nm if v == H else outgoing_idle
                else:
                    lhs_wait = nm_available if v == 0 else y[mid, v - 1]
                    rhs_wait = nm if v == H else y[mid, v]
                arriving = (
                    gp.quicksum(
                        x[j.id, mid, v - inst.p_occ[j.id, mid]]
                        for j in jobs_mid
                        if (v - inst.p_occ[j.id, mid]) in S[j.id, mid]
                    )
                    + inst.rj_arrivals.get((mid, v), 0)  # ghost completions
                )
                starting = gp.quicksum(
                    x[j.id, mid, v]
                    for j in jobs_mid
                    if v in S[j.id, mid]
                )
                mdl.addConstr(
                    lhs_wait + arriving == rhs_wait + starting,
                    name=f"flow[{mid},{v}]",
                )

        # Fixed usage costs: z[m] linked to x (no-op without fixed costs)
        add_fixed_cost_vars(mdl, inst, x)

        # (20) Global budget (includes fixed usage costs) — omitted when the
        # reductions established that it cannot bind
        if regime is None:
            mdl.addConstr(self.f2_expr(inst, x) <= inst.budget, name="budget")

        # objective_reduced: the model is valid only for the objective hint
        # it was built with (callers must not reuse it for another problem)
        return mdl, {"x": x, "y": y, "objective_reduced": objective_reduced}


class OccupancyFormulation(Formulation):
    """Occupancy-equality formulation (original implementation).

    Decision variables
    ------------------
    x[j,m,t] ∈ {0,1}  — job j starts on type m at time t
    w[m,t]   ∈ Z≥0    — number of instances of type m occupied during [t, t+1)

    Constraints
    -----------
    Unique assignment, occupancy equality, redundant upper bound, budget.
    """

    def build(self, inst: Instance, objective: Objective = None) -> tuple[gp.Model, dict]:
        H = inst.horizon
        mdl = gp.Model("fedhpc", env=_shared_env)
        mdl.setParam("OutputFlag", 0)

        # x[j,m,t] ∈ {0,1}
        x = XVars(
            ((j.id, m_id, t), mdl.addVar(
                vtype=GRB.BINARY, name=f"x[{j.id},{m_id},{t}]"
            ))
            for j in inst.jobs
            for m_id in inst.F[j.id]
            for t in inst.T[j.id, m_id]
        )

        # w[m,t] ∈ Z≥0 — occupied instances during [t, t+1)
        w = {
            (itype.id, t): mdl.addVar(
                vtype=GRB.INTEGER, lb=0, name=f"w[{itype.id},{t}]"
            )
            for itype in inst.instance_types
            for t in range(H)
        }

        mdl.update()

        # Unique assignment
        for j in inst.jobs:
            mdl.addConstr(
                gp.quicksum(
                    x[j.id, m_id, t]
                    for m_id in inst.F[j.id]
                    for t in inst.T[j.id, m_id]
                ) == 1,
                name=f"assign[{j.id}]",
            )

        # jobs_by_type[mid] hoists "mid in F[j]" out of the per-t loops below —
        # tested once per (job, type) instead of once per (job, type, t).
        jobs_by_type = {
            itype.id: [j for j in inst.jobs if itype.id in inst.F[j.id]]
            for itype in inst.instance_types
        }

        # Occupancy definition: w_mt = Σ jobs running during [t, t+1)
        # tau <= t < tau + p_occ  ⟺  tau ∈ (t − p_occ, t], intersected with the
        # feasible-start range T_jm.  Since T_jm is a contiguous range, this
        # intersection is computed directly instead of scanning all of T_jm.
        for itype in inst.instance_types:
            mid = itype.id
            jobs_mid = jobs_by_type[mid]
            for t in range(H):
                running = gp.quicksum(
                    x[j.id, mid, tau]
                    for j in jobs_mid
                    for tau in range(
                        max(inst.T[j.id, mid].start, t - inst.p_occ[j.id, mid] + 1),
                        min(inst.T[j.id, mid].stop, t + 1),
                    )
                )
                mdl.addConstr(w[mid, t] == running, name=f"occupy[{mid},{t}]")

        # Capacity / redundant upper bound on concurrent occupancy.
        # Running jobs pre-occupy some slots, so new jobs can only use the remainder.
        for itype in inst.instance_types:
            mid = itype.id
            base_ub = itype.capacity if itype.capacity is not None else len(jobs_by_type[mid])
            for t in range(H):
                # Subtract running jobs still active during [t, t+1)
                rj_occ = inst.occupied.get((mid, t), 0)
                ub = max(0, base_ub - rj_occ)
                mdl.addConstr(w[mid, t] <= ub, name=f"wbound[{mid},{t}]")

        # Fixed usage costs: z[m] linked to x (no-op without fixed costs)
        add_fixed_cost_vars(mdl, inst, x)

        # Budget (includes fixed usage costs)
        mdl.addConstr(self.f2_expr(inst, x) <= inst.budget, name="budget")

        return mdl, {"x": x, "w": w}
