"""Math formulations for the FED-HPC MIP model.

Each formulation provides a `build` method that constructs the Gurobi model and
returns (model, vars_dict).  The objectives f1/f2 are defined identically across
all formulations (they depend only on x_jmt), so they are implemented in the
abstract base class.

Available formulations
----------------------
SpaceTimeFormulation  — space-time network with flow conservation (fedhpc-1.pdf).
OccupancyFormulation  — occupancy-equality formulation (original implementation).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import gurobipy as gp
from gurobipy import GRB

from .data import Instance

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
    for itype in inst.instance_types:
        if not itype.fixed_cost:
            continue
        mid = itype.id
        z = x.z[mid] = mdl.addVar(vtype=GRB.BINARY, name=f"z[{mid}]")
        for j in inst.jobs:
            if mid in inst.F[j.id]:
                mdl.addConstr(
                    gp.quicksum(x[j.id, mid, t] for t in inst.T[j.id, mid]) <= z,
                    name=f"use[{j.id},{mid}]",
                )


class Formulation(ABC):
    """Abstract base class for FED-HPC MIP formulations.

    Subclasses must implement `build`; the objective expressions and solution
    extraction helpers are shared and live here.
    """

    @abstractmethod
    def build(self, inst: Instance) -> tuple[gp.Model, dict]:
        """Construct the Gurobi model.

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
    """

    def __init__(self, *, continuous_idle: bool = False, cover_ge: bool = False,
                 normal_patterns: bool = False):
        self.continuous_idle = continuous_idle
        self.cover_ge = cover_ge
        self.normal_patterns = normal_patterns

    def build(self, inst: Instance) -> tuple[gp.Model, dict]:
        H = inst.horizon
        # S[j,m]: allowed start times (T_jm, or its normal-pattern subset)
        if self.normal_patterns:
            masks = normal_pattern_starts(inst)
            S = {
                (j.id, m_id): frozenset(t for t in inst.T[j.id, m_id] if masks[m_id][t])
                for j in inst.jobs for m_id in inst.F[j.id]
            }
        else:
            S = inst.T
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
            for m_id in inst.F[j.id]
            for t in S[j.id, m_id]
        )

        # y[m,t] ∈ Z≥0 — waiting instances during [t, t+1), for t ∈ T\{H}  (eq. 22)
        y = {
            (itype.id, t): mdl.addVar(
                vtype=GRB.CONTINUOUS if self.continuous_idle else GRB.INTEGER,
                lb=0, name=f"y[{itype.id},{t}]"
            )
            for itype in inst.instance_types
            for t in range(H)  # t ∈ {0, …, H−1}
        }

        mdl.update()

        # ---- Constraints --------------------------------------------------

        # (18) Unique assignment: each job processed exactly once
        for j in inst.jobs:
            cover = gp.quicksum(
                x[j.id, m_id, t]
                for m_id in inst.F[j.id]
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
            itype.id: [j for j in inst.jobs if itype.id in inst.F[j.id]]
            for itype in inst.instance_types
        }
        for itype in inst.instance_types:
            mid = itype.id
            nm = N[mid]
            # Slots available to new jobs at the very start of the horizon
            nm_available = max(0, nm - inst.n_running.get(mid, 0))
            jobs_mid = jobs_by_type[mid]
            for v in range(H + 1):  # v ∈ {0, …, H}
                lhs_wait = nm_available if v == 0 else y[mid, v - 1]
                rhs_wait = nm           if v == H else y[mid, v]
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

        # (20) Global budget (includes fixed usage costs)
        mdl.addConstr(self.f2_expr(inst, x) <= inst.budget, name="budget")

        return mdl, {"x": x, "y": y}


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

    def build(self, inst: Instance) -> tuple[gp.Model, dict]:
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
