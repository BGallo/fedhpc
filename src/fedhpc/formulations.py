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
    """Space-time network formulation with flow conservation (fedhpc-1.pdf).

    Decision variables
    ------------------
    x[j,m,t] ∈ {0,1}   — job j starts on type m at time t  (eq. 19)
    w[m,t]   ∈ Z≥0     — flow on wait arc (t, t+1) of graph G_m  (eq. 20)
    K[m]     ∈ Z≥0     — total instances of type m activated  (eq. 21)

    Constraints
    -----------
    (14) Unique job assignment.
    (15) Flow conservation at internal nodes τ = 1, …, H−1.
    (16) Source boundary at τ = 0.
    (17) Sink boundary at τ = H.
    (18) Global budget.
    """

    def build(self, inst: Instance) -> tuple[gp.Model, dict]:
        H = inst.horizon
        mdl = gp.Model("fedhpc")
        mdl.setParam("OutputFlag", 0)

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

        # w[m,t] ∈ Z≥0 — idle flow on wait arc (t, t+1), t = 0, …, H−1
        w = {
            (itype.id, t): mdl.addVar(
                vtype=GRB.INTEGER, lb=0, name=f"w[{itype.id},{t}]"
            )
            for itype in inst.instance_types
            for t in range(H)
        }

        # K[m] ∈ Z≥0 — total instances of type m activated
        K = {
            itype.id: mdl.addVar(
                vtype=GRB.INTEGER, lb=0, name=f"K[{itype.id}]"
            )
            for itype in inst.instance_types
        }

        mdl.update()

        # ---- Constraints --------------------------------------------------

        # (14) Unique assignment: each job processed exactly once
        for j in inst.jobs:
            mdl.addConstr(
                gp.quicksum(
                    x[j.id, m_id, t]
                    for m_id in inst.F[j.id]
                    for t in inst.T[j.id, m_id]
                ) == 1,
                name=f"assign[{j.id}]",
            )

        # (16) Source boundary at node τ = 0:
        #   w_{m,0} + Σ_{j: 0 ∈ T_jm} x_jm0 = K_m
        for itype in inst.instance_types:
            mid = itype.id
            starting_at_0 = gp.quicksum(
                x[j.id, mid, 0]
                for j in inst.jobs
                if mid in inst.F[j.id] and 0 in inst.T[j.id, mid]
            )
            mdl.addConstr(
                w[mid, 0] + starting_at_0 == K[mid],
                name=f"source[{mid}]",
            )

        # (15) Flow conservation at internal nodes τ = 1, …, H−1:
        #   w_{m,τ−1} + Σ_{j,t: t+p^occ=τ} x_jmt  =  w_{m,τ} + Σ_{j: τ ∈ T_jm} x_jmτ
        for itype in inst.instance_types:
            mid = itype.id
            for tau in range(1, H):
                arriving = gp.quicksum(
                    x[j.id, mid, t]
                    for j in inst.jobs if mid in inst.F[j.id]
                    for t in inst.T[j.id, mid]
                    if t + inst.p_occ[j.id, mid] == tau
                )
                starting = gp.quicksum(
                    x[j.id, mid, tau]
                    for j in inst.jobs
                    if mid in inst.F[j.id] and tau in inst.T[j.id, mid]
                )
                mdl.addConstr(
                    w[mid, tau - 1] + arriving == w[mid, tau] + starting,
                    name=f"flow[{mid},{tau}]",
                )

        # (17) Sink boundary at node τ = H:
        #   w_{m,H−1} + Σ_{j,t: t+p^occ=H} x_jmt = K_m
        for itype in inst.instance_types:
            mid = itype.id
            arriving_at_H = gp.quicksum(
                x[j.id, mid, t]
                for j in inst.jobs if mid in inst.F[j.id]
                for t in inst.T[j.id, mid]
                if t + inst.p_occ[j.id, mid] == H
            )
            mdl.addConstr(
                w[mid, H - 1] + arriving_at_H == K[mid],
                name=f"sink[{mid}]",
            )

        # (18) Global budget
        mdl.addConstr(
            gp.quicksum(
                inst.c[j.id, m_id] * x[j.id, m_id, t]
                for j in inst.jobs
                for m_id in inst.F[j.id]
                for t in inst.T[j.id, m_id]
            ) <= inst.budget,
            name="budget",
        )

        # Fleet capacity: K_m ≤ capacity_m (omitted when capacity is None)
        for itype in inst.instance_types:
            if itype.capacity is not None:
                mdl.addConstr(
                    K[itype.id] <= itype.capacity,
                    name=f"cap[{itype.id}]",
                )

        return mdl, {"x": x, "w": w, "K": K}


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
        mdl = gp.Model("fedhpc")
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

        # Capacity / redundant upper bound on concurrent occupancy
        for itype in inst.instance_types:
            mid = itype.id
            ub = itype.capacity if itype.capacity is not None else sum(
                1 for j in inst.jobs if mid in inst.F[j.id]
            )
            for t in range(H):
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
