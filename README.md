# FED-HPC

Multi-objective MIP scheduler for HPC jobs in federated multi-platform environments (on-premises + cloud). Minimizes total turnaround time and monetary cost jointly, subject to resource capacity and horizon constraints.

## Problem Formulation

Space-time network flow MIP formulation (non-preemptive, elastic provisioning).

### Modeling Assumptions

1. Each job is non-preemptive and runs entirely on one instance of a feasible type.
2. Each instance type `m ∈ M` represents a homogeneous VM family (e.g. `c7e`, `m6i`, on-premises cluster).
3. Time is discretized: `T = {0,1,...,H}`, `T̄ = {0,...,H-1}` (elementary intervals).
4. Deploy/provisioning adds a **time penalty only** — no monetary cost.
5. Idle time between uses of the same type is free and allowed.
6. Instance provisioning is elastic — any number of concurrent instances may be used (unlimited types). Cloud capacity is on-demand only: spot/preemptible instances (revocable mid-job) are not modelled, and the real-trace instances contain only on-demand cloud types. There is no hard budget: cost is an objective, and a cost cap is expressed as an ε-constraint on f2.
7. Capacitated (on-prem) types have `N_m` instances; jobs already running at `t = 0` (`running_jobs`) hold one instance each until they finish and are not re-optimized.
8. Job weights `w_j` are **integral** (`Instance.build()` raises otherwise). With integer completion times, any two schedules' f1 values then differ by a whole number. The exact Pareto searches (`true_pareto_frontier`, `map_pareto_frontier`) step f1 by exactly 1, and BPC pruning uses that spacing; with fractional weights both would skip Pareto points.
9. Costs live on a grid of resolution `δ` (`cost_resolution`, default `1e-4`): `build()` rounds every `c_jm` and `U_m` to a multiple of `δ`, so f2 moves in whole steps of `δ`. This is the same resolution the exact searches previously assumed implicitly through a fixed `1e-4` tolerance.

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
- `w_j` — turnaround weight (integer, default 1)

**Per instance type `m`:**
- `P_m` — relative performance factor
- `τ^IO_m` — I/O time per unit volume
- `π_m` — deploy/provisioning time penalty (time only, no cost)
- `κ^VM_m` — cost per unit of billable time
- `κ^S_m` — cost per unit of billable time × storage
- `κ^IO_m` — cost per I/O unit
- `C^cpu_m`, `C^mem_m`, `C^stor_m` — CPU, memory, storage capacities
- `N_m` — number of instances (`capacity`; unlimited when absent)
- `U_m` — fixed cost charged once if any job runs on `m` (`fixed_cost`, default 0)

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

**f1 — Minimize total weighted turnaround (eq. 10):**
```
f1 = Σ_j Σ_{m∈F_j} Σ_{t∈T_jm} w_j (t + p^occ_jm - a_j) · x_jmt
```

**f2 — Minimize total monetary cost (eq. 11):**
```
f2 = Σ_j Σ_{m∈F_j} Σ_{t∈T_jm} c^proc_jm · x_jmt  +  Σ_m U_m z_m
```

### Constraints (eqs 12–17)

1. **Unique assignment** (eq. 12):
   `Σ_{m∈F_j} Σ_{t∈T_jm} x_jmt = 1  ∀j`

2. **Occupancy definition** (eq. 13) — `w_mt` equals the number of jobs running during `[t, t+1)`:
   `w_mt = Σ_{j,τ: m∈F_j, τ∈T_jm, τ≤t<τ+p^occ_jm} x_jmτ  ∀m, t∈T̄`

3. **Redundant bound** for LP strengthening (eq. 14):
   `w_mt ≤ |{j : m ∈ F_j}|  ∀m, t∈T̄`

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

#### Exact certificates

Since f1 moves in steps of 1 and f2 in steps of `δ`, a lexicographic phase is certified optimal once its incumbent's exact value, recomputed from the schedule, is within one step of Gurobi's dual bound. `solve_f1`, `solve_f2`, the ε-constraint solves, the reference points and the Pareto searches run each phase with `MIPGap=0` and `MIPGapAbs` just under one step (`model.certify_params`). They report `"optimal"` only when certified (`model.is_certified`) and `"feasible"` otherwise. Every objective bound is written as an integer row on its grid (`Formulation.bound_constr`), so no right-hand side ever sits a rounding error away from an achievable value. Gurobi's presolve otherwise rounded `f2 ≤ 35 − 5e-5` up to 35 on integer cost data. Passing your own `MIPGap`/`MIPGapAbs` overrides the defaults, and the status then says whether the result is still proven.

Each certificate is about the schedule the solver returned, so that schedule is validated too. `Instance.check_schedule` (and `schedule_violations`) verifies the following:
- every job is assigned exactly once;
- each job's type is in `F_j`;
- each start lies in `T_jm`;
- no capacitated type exceeds its capacity in any slot, counting running jobs.

It runs on every MIP extraction (`model._extract`, which also rejects a job started twice), on every BPC result and on every EA result, and raises `ScheduleError` if the schedule is infeasible. The check is not expected to fire. It guards against integrality tolerances and decoder bugs.

#### Exact Pareto front

`pareto.true_pareto_frontier` enumerates the complete front by box splitting: about 2 solves per Pareto point. Each solve is a lexicographic minimum under a bound on the other objective. `pareto.map_pareto_frontier` bisects the objective range instead, giving broad coverage under a solve or time budget. Both rely on f1 being integer-spaced (assumption 8). `pareto.hybrid_frontier` runs NSGA-II and MOEA/D, then solves a certified lexicographic minimum under each surviving candidate's cost. Every `"optimal"` point it returns is proven Pareto-optimal, but its front is incomplete: only trade-offs near the EA candidates are visited. Each box solve runs on either the compact Gurobi model (`engine="gurobi"`) or the branch-price-and-cut engine in `bpc.py` (`engine="bpc"`). BPC handles running jobs by turning each into a pinned pseudo-job; it is practical only for instances of about a dozen jobs. `per_solve_time_limit` caps each solve phase on both engines. A phase that runs out of time leaves its box unresolved, and the search reports a partial front.

## Project Structure

```
fedhpc/
├── data/                     # Real-trace instances (see "Instances" below)
├── src/fedhpc/
│   ├── data.py               # Instance data structures and derived parameters
│   ├── formulations.py       # SpaceTime / Occupancy MIP builders + dominance reductions
│   ├── model.py              # Single-objective, weighted-sum and ε-constraint solves
│   ├── pareto.py             # Reference points, exact box-splitting / bisection fronts, hybrid
│   ├── bpc.py                # Branch-price-and-cut exact engine
│   ├── moea.py               # NSGA-II / NSGA-III / MOEA/D and weighted-sum heuristics
│   ├── baselines.py          # List-scheduling baselines (greedy, threshold bursting)
│   ├── metrics.py, viz.py    # Front indicators, plots, reports
│   ├── cg_bound.py, ilssp.py, rm_benchmark.py  # Rm||TWCT+TMUC benchmark
│   ├── cli.py                # Entry point
│   └── _ext/                 # C++ extensions (pybind11): EAs, graphs, ILS-SP, pricing
├── tests/
├── scripts/                  # Experiments and benchmarks
└── proofs/                   # Lean proofs of reduction lemmas
```

## Setup

Requires Python ≥ 3.14, a C++17 compiler with CMake, a valid [Gurobi license](https://www.gurobi.com/downloads/) and [uv](https://docs.astral.sh/uv/).

uv rebuilds the C++ extensions on the next `uv run` / `uv sync` whenever a file under `src/fedhpc/_ext/`, `CMakeLists.txt` or `pyproject.toml` changes (`[tool.uv] cache-keys`), so an edited header can't be shadowed by a stale `.so`.

```bash
# Install dependencies (builds the C++ extensions)
uv sync

# Run tests (~3.5 min)
uv run pytest

# Only what fits Gurobi's size-limited pip licence (2000 vars / constraints)
uv run pytest -m "not needs_full_license"

I=data/pos_congestion_known_runtime_offline_10min.json

# Lex-min turnaround / cost
uv run fedhpc --instance $I --method f1
uv run fedhpc --instance $I --method f2

# Weighted-sum scalarisation (α controls turnaround vs cost trade-off)
uv run fedhpc --instance $I --method weighted --alpha 0.5

# ε-constraint: minimise turnaround with cost cap ε / cost with turnaround cap ε
uv run fedhpc --instance $I --method epsilon --epsilon 200
uv run fedhpc --instance $I --method epsilon-t --epsilon 50000

# Exact Pareto front (box splitting), capped at 20 solves
uv run fedhpc --instance $I --method pareto-true --max-solves 20

# Heuristic fronts
uv run fedhpc --instance $I --method nsga2
uv run fedhpc --instance $I --method moead

# JSON output
uv run fedhpc --instance $I --method f1 --json
```

## Instances

- `data/pos_congestion_known_runtime_offline_10min.json` — 3340 jobs, 16 types (4 on-prem + 12 on-demand cloud), 1525 running jobs, 288 ten-minute slots. The stress instance: tests that exercise scale use it.
- `data/fedhpc_known_runtime_offline_10min.json`, `data/mit_supercloud_*.json` — further real-trace instances.
- `data/reference_points.json` — exact weighted-sum reference points `(f1_T, f2_T, f1_0, f2_0)` per instance, each with the SHA-256 of the instance file and the cost resolution it was computed with. Load with `fedhpc.data.reference_points(path)`, which refuses an entry whose instance or resolution has changed. Recompute with `uv run python scripts/compute_reference_points.py [INSTANCE ...]`. The experiment scripts read this table instead of hard-coding values.
- Tests that need a small instance use the *trap instances* defined in `tests/conftest.py` (7 and 12 jobs). They are small but hard. The optimum hinges on which jobs wait for a scarce fat node or for a running job to free capacity. Every list-scheduling baseline lands at least 9% (`trap_small`) and 14% (`trap_medium`) above the exact minimum turnaround. `tests/test_trap_instances.py` pins this property.

## Continuous integration

`.github/workflows/tests.yml` runs the suite on Python 3.14 on every push and pull request. With the repository secrets `GRB_WLSACCESSID`, `GRB_WLSSECRET` and `GRB_LICENSEID` set, it runs everything on that WLS licence. Without them it uses the size-limited pip licence and skips tests marked `needs_full_license`. Check that your licence terms allow CI use: academic WLS licences may not.

## Instance Format

Instances are JSON files with the following structure:

```json
{
  "horizon": 50,
  "instance_types": [
    {
      "id": 0, "kind": "on-prem",
      "perf": 1.0, "io_time": 0.0, "deploy": 0,
      "cost_io": 0.0, "cost_vm": 0.0, "cost_stor": 0.0,
      "cpu": 32, "mem": 128, "stor": 2000,
      "capacity": 4
    }
  ],
  "jobs": [
    {
      "id": 0, "arrival": 0, "exec_time": 10, "io_volume": 0,
      "cpu": 8, "mem": 32, "stor": 500, "weight": 1
    }
  ],
  "running_jobs": [
    {"id": 100, "type_id": 0, "end": 6}
  ]
}
```

`feasible_types` is no longer an input field — `F_j` is computed automatically from the resource constraints (CPU / mem / stor). Optional fields: `capacity` (absent = unlimited), `fixed_cost` per type; `weight` (an integer, default 1) per job; `running_jobs` (each holds one instance of a capacitated type from slot 0 until `end`); top-level `cost_resolution` (default `1e-4`).

## Quality indicators

`metrics.compare_fronts(front, reference)` scores a front against a reference front (the exact one, or the union of every method's points), normalised by the reference's ideal and nadir. It reports exact hits, dominated points, IGD, IGD+, GD, additive ε and the hypervolume ratio; this is the only way to compare methods. `metrics.pareto_metrics(front)` without `reference` normalises by the front's own extremes: it describes one front's shape but says nothing about its quality. Pass `reference=` to get comparable numbers.

## Dependencies

- `gurobipy` — MIP solver
- `numpy` — numerical utilities

Managed with [uv](https://docs.astral.sh/uv/). See `pyproject.toml` for the full dependency list.

## References

FED-HPC problem formulation: `MOFE__Multi_Objective_Scheduling_on_Federated_Enviroments.pdf` (not in the repository). Formulation reductions: `docs/formulation_reductions.tex`.

