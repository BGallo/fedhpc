# FED-HPC

Multi-objective MIP scheduler for HPC jobs in federated multi-platform environments (on-premises + cloud). Minimizes total turnaround time and monetary cost jointly, subject to resource capacity, budget, and horizon constraints.

## Problem Formulation

Space-time network flow MIP formulation (non-preemptive, elastic provisioning).

### Modeling Assumptions

1. Each job is non-preemptive and runs entirely on one instance of a feasible type.
2. Each instance type `m ∈ M` represents a homogeneous VM family (e.g. `c7e`, `m6i`, on-premises cluster).
3. Time is discretized: `T = {0,1,...,H}`, `T̄ = {0,...,H-1}` (elementary intervals).
4. Deploy/provisioning adds a **time penalty only** — no monetary cost.
5. Idle time between uses of the same type is free and allowed.
6. Instance provisioning is elastic — any number of concurrent instances may be used, limited only by budget.

### Sets

| Symbol | Description |
|--------|-------------|
| `J = {1,...,n}` | Jobs |
| `M` | Instance types (e.g. `c7e`, `m6i`, on-premises cluster) |
| `T = {0,...,H}` | Discrete time points |
| `T̄ = {0,...,H-1}` | Elementary time intervals |
| `F_j ⊆ M` | Feasible types for job `j` — computed from resource requirements |

### Parameters

**Per job `j`:**
- `a_j` — arrival time
- `e_j` — base execution time
- `ι_j` — I/O volume
- `q^cpu_j`, `q^mem_j`, `q^stor_j` — CPU, memory, storage demand

**Per instance type `m`:**
- `P_m` — relative performance factor
- `τ^IO_m` — I/O time per unit volume
- `π_m` — deploy/provisioning time penalty (time only, no cost)
- `κ^VM_m` — cost per unit of billable time
- `κ^S_m` — cost per unit of billable time × storage
- `κ^IO_m` — cost per I/O unit
- `C^cpu_m`, `C^mem_m`, `C^stor_m` — CPU, memory, storage capacities

**Global:** `B` — budget.

### Derived Parameters

**Resource feasibility (eq. 1):**
```
F_j = { m ∈ M : q^cpu_j ≤ C^cpu_m,  q^mem_j ≤ C^mem_m,  q^stor_j ≤ C^stor_m }
```

**Billable time** — excludes deploy, used for cost (eq. 2):
```
p^bill_jm = e_j / P_m  +  τ^IO_m · ι_j
```

**Occupation time** — includes deploy, used for scheduling (eq. 3):
```
p^occ_jm = ⌈p^bill_jm + π_m⌉   (positive integer)
```

**Monetary cost** — based on billable time, deploy is free (eq. 4):
```
c^proc_jm = κ^IO_m · ι_j  +  κ^VM_m · p^bill_jm  +  κ^S_m · q^stor_j · p^bill_jm
```

**Feasible start times** — based on occupation time (eq. 5):
```
T_jm = { t ∈ T : t ≥ ⌈a_j⌉,  t + p^occ_jm ≤ H }
```

### Space-Time Graph (per type `m`)

One directed acyclic graph `G_m = (T, A_m)` per instance type:
- **Wait arcs** `A^wait_m`: `(t, t+1)` for `t ∈ T̄`
- **Processing arcs** `A^proc_m`: `(t, t + p^occ_jm)` for each `j ∈ J`, `t ∈ T_jm`

### Decision Variables

| Variable | Domain | Meaning |
|----------|--------|---------|
| `x_jmt` | `{0,1}` | 1 if job `j` starts on type `m` at time `t` |
| `w_mt` | `Z≥0` | number of instances of type `m` occupied during `[t, t+1)` |

### Objectives

**f1 — Minimize total turnaround (eq. 10):**
```
f1 = Σ_j Σ_{m∈F_j} Σ_{t∈T_jm} (t + p^occ_jm - a_j) · x_jmt
```

**f2 — Minimize total monetary cost (eq. 11):**
```
f2 = Σ_j Σ_{m∈F_j} Σ_{t∈T_jm} c^proc_jm · x_jmt
```

### Constraints (eqs 12–17)

1. **Unique assignment** (eq. 12):
   `Σ_{m∈F_j} Σ_{t∈T_jm} x_jmt = 1  ∀j`

2. **Occupancy definition** (eq. 13) — `w_mt` equals the number of jobs running during `[t, t+1)`:
   `w_mt = Σ_{j,τ: m∈F_j, τ∈T_jm, τ≤t<τ+p^occ_jm} x_jmτ  ∀m, t∈T̄`

3. **Redundant bound** for LP strengthening (eq. 14):
   `w_mt ≤ |{j : m ∈ F_j}|  ∀m, t∈T̄`

4. **Budget** (eq. 15):
   `Σ_j Σ_{m∈F_j} Σ_{t∈T_jm} c^proc_jm · x_jmt ≤ B`

### Multi-objective Methods

#### Weighted Sum (scalarization with range normalization)

Normalized objectives:

```
f̂1 = (f1 - f1_min) / (f1_max - f1_min)
f̂2 = f2 / f2_max
```

Scalarized objective:

```
min  α·f̂1 + (1-α)·f̂2,   α ∈ [0, 1]
```

#### ε-Constraint Method

Sweeps a discrete set of ε values to approximate the Pareto frontier, handling non-convex regions that the weighted sum cannot reach.

- **Version 1:** `min f1` subject to `f2 ≤ ε`
- **Version 2:** `min f2` subject to `f1 ≤ ε1`

## Project Structure

```
fedhpc/
├── data/
│   └── small.json            # Example instance (5 jobs, 4 machines)
├── src/
│   └── fedhpc/
│       ├── __init__.py
│       ├── model.py          # MIP model (gurobipy)
│       ├── data.py           # Instance data structures
│       ├── pareto.py         # Weighted-sum and ε-constraint solvers
│       └── cli.py            # Entry point
├── tests/
│   ├── conftest.py
│   └── test_model.py
├── pyproject.toml
└── README.md
```

## Setup

Requires a valid [Gurobi license](https://www.gurobi.com/downloads/) and [uv](https://docs.astral.sh/uv/).

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Minimise turnaround only
uv run fedhpc --instance data/small.json --method f1

# Minimise cost only
uv run fedhpc --instance data/small.json --method f2

# Weighted-sum scalarisation (α controls turnaround vs cost trade-off)
uv run fedhpc --instance data/small.json --method weighted --alpha 0.5

# ε-constraint: minimise turnaround with cost budget ε
uv run fedhpc --instance data/small.json --method epsilon --epsilon 200

# ε-constraint: minimise cost with turnaround budget ε
uv run fedhpc --instance data/small.json --method epsilon-t --epsilon 150

# Pareto frontier via weighted-sum sweep
uv run fedhpc --instance data/small.json --method pareto-ws --steps 11

# Pareto frontier via ε-constraint sweep
uv run fedhpc --instance data/small.json --method pareto-eps --steps 20

# JSON output
uv run fedhpc --instance data/small.json --method f1 --json
```

## Instance Format

Instances are JSON files with the following structure:

```json
{
  "horizon": 50,
  "budget": 5000,
  "instance_types": [
    {
      "id": 0, "kind": "on-prem",
      "perf": 1.0, "io_time": 0.0, "deploy": 0,
      "cost_io": 0.0, "cost_vm": 0.0, "cost_stor": 0.0,
      "cpu": 32, "mem": 128, "stor": 2000
    }
  ],
  "jobs": [
    {
      "id": 0, "arrival": 0, "exec_time": 10, "io_volume": 0,
      "cpu": 8, "mem": 32, "stor": 500
    }
  ]
}
```

`feasible_types` is no longer an input field — `F_j` is computed automatically from the resource constraints (CPU / mem / stor).

## Dependencies

- `gurobipy` — MIP solver
- `numpy` — numerical utilities

Managed with [uv](https://docs.astral.sh/uv/). See `pyproject.toml` for the full dependency list.

## References

FED-HPC problem formulation: `fedhpc.pdf`.# fedhpc
