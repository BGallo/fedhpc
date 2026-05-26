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

    def f1_expr(self, inst: Instance, x: dict) -> gp.LinExpr:
        """Total turnaround: Σ_{j,m,t} (t + p^occ_jm − a_j) x_jmt  (eq. 11)."""
        return gp.quicksum(
            (t + inst.p_occ[j.id, m_id] - j.arrival) * x[j.id, m_id, t]
            for j in inst.jobs
            for m_id in inst.F[j.id]
            for t in inst.T[j.id, m_id]
        )

    def f2_expr(self, inst: Instance, x: dict) -> gp.LinExpr:
        """Total monetary cost: Σ_{j,m,t} c^proc_jm x_jmt  (eq. 13)."""
        return gp.quicksum(
            inst.c[j.id, m_id] * x[j.id, m_id, t]
            for j in inst.jobs
            for m_id in inst.F[j.id]
            for t in inst.T[j.id, m_id]
        )


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
    """

    def build(self, inst: Instance) -> tuple[gp.Model, dict]:
        H = inst.horizon
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
        x = {
            (j.id, m_id, t): mdl.addVar(
                vtype=GRB.BINARY, name=f"x[{j.id},{m_id},{t}]"
            )
            for j in inst.jobs
            for m_id in inst.F[j.id]
            for t in inst.T[j.id, m_id]
        }

        # y[m,t] ∈ Z≥0 — waiting instances during [t, t+1), for t ∈ T\{H}  (eq. 22)
        y = {
            (itype.id, t): mdl.addVar(
                vtype=GRB.INTEGER, lb=0, name=f"y[{itype.id},{t}]"
            )
            for itype in inst.instance_types
            for t in range(H)  # t ∈ {0, …, H−1}
        }

        mdl.update()

        # ---- Constraints --------------------------------------------------

        # (18) Unique assignment: each job processed exactly once
        for j in inst.jobs:
            mdl.addConstr(
                gp.quicksum(
                    x[j.id, m_id, t]
                    for m_id in inst.F[j.id]
                    for t in inst.T[j.id, m_id]
                ) == 1,
                name=f"assign[{j.id}]",
            )

        # (19) Flow conservation for all v ∈ T = {0, …, H}:
        #   y_{m,v−1} + Σ_{j: v−p^occ_jm ∈ T_jm} x_{jm, v−p^occ_jm} + rj_arrivals[m,v]
        #             = y_{m,v} + Σ_{j: v ∈ T_jm} x_jmv
        # Boundary: y_{m,−1} = N_m − n_running[m]  (reduced by pre-occupied slots)
        #           y_{m,H}  = N_m                  (terminal: all slots idle again)
        # rj_arrivals[m,v]: running jobs that complete at v (ghost completions).
        for itype in inst.instance_types:
            mid = itype.id
            nm = N[mid]
            # Slots available to new jobs at the very start of the horizon
            nm_available = max(0, nm - inst.n_running.get(mid, 0))
            for v in range(H + 1):  # v ∈ {0, …, H}
                lhs_wait = nm_available if v == 0 else y[mid, v - 1]
                rhs_wait = nm           if v == H else y[mid, v]
                arriving = (
                    gp.quicksum(
                        x[j.id, mid, v - inst.p_occ[j.id, mid]]
                        for j in inst.jobs
                        if mid in inst.F[j.id]
                        and (v - inst.p_occ[j.id, mid]) in inst.T[j.id, mid]
                    )
                    + inst.rj_arrivals.get((mid, v), 0)  # ghost completions
                )
                starting = gp.quicksum(
                    x[j.id, mid, v]
                    for j in inst.jobs
                    if mid in inst.F[j.id] and v in inst.T[j.id, mid]
                )
                mdl.addConstr(
                    lhs_wait + arriving == rhs_wait + starting,
                    name=f"flow[{mid},{v}]",
                )

        # (20) Global budget
        mdl.addConstr(
            gp.quicksum(
                inst.c[j.id, m_id] * x[j.id, m_id, t]
                for j in inst.jobs
                for m_id in inst.F[j.id]
                for t in inst.T[j.id, m_id]
            ) <= inst.budget,
            name="budget",
        )

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
        x = {
            (j.id, m_id, t): mdl.addVar(
                vtype=GRB.BINARY, name=f"x[{j.id},{m_id},{t}]"
            )
            for j in inst.jobs
            for m_id in inst.F[j.id]
            for t in inst.T[j.id, m_id]
        }

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

        # Occupancy definition: w_mt = Σ jobs running during [t, t+1)
        for itype in inst.instance_types:
            mid = itype.id
            for t in range(H):
                running = gp.quicksum(
                    x[j.id, mid, tau]
                    for j in inst.jobs if mid in inst.F[j.id]
                    for tau in inst.T[j.id, mid]
                    if tau <= t < tau + inst.p_occ[j.id, mid]
                )
                mdl.addConstr(w[mid, t] == running, name=f"occupy[{mid},{t}]")

        # Capacity / redundant upper bound on concurrent occupancy.
        # Running jobs pre-occupy some slots, so new jobs can only use the remainder.
        for itype in inst.instance_types:
            mid = itype.id
            base_ub = itype.capacity if itype.capacity is not None else sum(
                1 for j in inst.jobs if mid in inst.F[j.id]
            )
            for t in range(H):
                # Subtract running jobs still active during [t, t+1)
                rj_occ = inst.occupied.get((mid, t), 0)
                ub = max(0, base_ub - rj_occ)
                mdl.addConstr(w[mid, t] <= ub, name=f"wbound[{mid},{t}]")

        # Budget
        mdl.addConstr(
            gp.quicksum(
                inst.c[j.id, m_id] * x[j.id, m_id, t]
                for j in inst.jobs
                for m_id in inst.F[j.id]
                for t in inst.T[j.id, m_id]
            ) <= inst.budget,
            name="budget",
        )

        return mdl, {"x": x, "w": w}
