# VRPSolver integration (2026-11-26)

VRPSolver (Pessoa, Sadykov, Uchoa & Vanderbeck, *Math. Programming* 183, 2020)
is the generic branch-cut-and-price solver reviewers will ask our BPC to be
compared against. This repo can now state any unit-demand fedhpc instance as a
VRPSolver model and run it.

## Status

| Piece | Where | Tested |
|---|---|---|
| Exporter (instance -> JSON model) | `src/fedhpc/vrpsolver.py`, `export_model` | yes: the exported model, solved as a compact MIP with HiGHS, matches BPC's optimum on all trap instances, the fixed-cost fixture and capped problems (`tests/test_vrpsolver.py`) |
| Julia model build | `vrpsolver/run.jl` (`--dry-run`) | yes, in the `julia:1.11` Docker image, up to Rm M8N40 (116k placements, ~2 s) |
| Solve + result check | `run.jl`, `solve_vrpsolver` | **not run**: needs the BaPCod library (below) |
| Benchmark | `scripts/vrpsolver_bench.py` | dry-run mode only |

## What running it needs

VRPSolver is **academic-use only** (BaPCod licence) and is not in the repo.

1. **Julia >= 1.11.5**, or Docker with `julia:1.11` (see below).
2. **An LP solver for BaPCod**, either one:
   - **COIN-OR CLP >= 1.17.7**: free (EPL), and enough on its own. BaPCod's
     requirements list "CPLEX 12.9 or above OR COIN-OR CLP 1.17.7 or above";
     CLP support was experimental in v0.74 (2022), improved in v0.77 (2023).
     CLP is what VRPSolverEasy uses by default.
   - **CPLEX >= 12.9** (free through the IBM Academic Initiative): faster
     according to the VRPSolverEasy paper, and still the only solver the
     VrpSolver.jl README lists. Its shared libraries must then be on
     `LD_LIBRARY_PATH`.
   - BaPCod v0.83.1 (2025-08) added a beta Gurobi interface; HiGHS is not
     supported.
3. **BaPCod shared library** `libbapcod-shared.so` (v0.84, 2025-11, is the
   latest release listed): download the source from bapcod.math.u-bordeaux.fr
   with an academic e-mail, accept the licence and build it against CLP or
   CPLEX as its `README.md` says. Point `BAPCOD_RCSP_LIB` at the `.so`.
   (The VRPSolverEasy wheel and GitHub repo do not contain a prebuilt
   library.) Untested here: whether the RCSP extension and the shared library
   build with CLP alone. The requirements page does not say otherwise.

Then, once:

```
julia --project=vrpsolver -e 'using Pkg; Pkg.instantiate()'
```

`vrpsolver/Manifest.toml` pins the versions used for the dry runs
(VrpSolver.jl from `tbulhoes/VrpSolver.jl@main`, JuMP 1.26+, JSON 0.21).

### Through Docker instead of a local Julia

`FEDHPC_JULIA` replaces the `julia` command. The repo, the temporary
directory, the LP solver (if CPLEX) and the BaPCod library must be mounted at
the same paths inside the container:

```
SP=/some/scratch; mkdir -p $SP/depot $SP/tmp
export TMPDIR=$SP/tmp
export BAPCOD_RCSP_LIB=/opt/bapcod/build/Bapcod/libbapcod-shared.so
export FEDHPC_JULIA="docker run --rm --user $(id -u):$(id -g) \
  -e JULIA_DEPOT_PATH=/depot -v $SP/depot:/depot \
  -v $PWD:$PWD -v $SP/tmp:$SP/tmp -e TMPDIR=$SP/tmp \
  -v /opt/bapcod:/opt/bapcod -e BAPCOD_RCSP_LIB=$BAPCOD_RCSP_LIB \
  -v /opt/ibm:/opt/ibm -e LD_LIBRARY_PATH=/opt/ibm/ILOG/CPLEX_Studio2211/cplex/bin/x86-64_linux \
  julia:1.11 julia"
```

(Adjust the CPLEX path to your version, or drop that line for a CLP build.) The dry run needs only the first
three `-v`/`-e` pairs.

### Commands

```
# exporter tests (always), Julia model build (needs Julia or FEDHPC_JULIA)
FEDHPC_VRPSOLVER_DRYRUN=1 uv run pytest tests/test_vrpsolver.py
# full solve tests run automatically once BAPCOD_RCSP_LIB is set
uv run python scripts/vrpsolver_bench.py --dry-run --instances rm_M2N10_1
uv run python scripts/vrpsolver_bench.py --out results/vrpsolver_bench/rm.jsonl --time-limit 600
```

```python
from fedhpc import vrpsolver
sol = vrpsolver.solve_vrpsolver(inst, 1.0, 1.0, time_limit=600)   # Solution, checked
model = vrpsolver.export_model(inst, 1.0, 0.0, bound2=500.0)       # JSON-ready dict
```

## The model (same master as `fedhpc.bpc`)

- **Graphs:** one per release group `(m, r)` of each path type: a start vertex
  `H+1`, nodes `r..H`, a start arc `H+1 -> r`, wait arcs `v -> v+1`, and one
  arc `t -> t+p_jm` per allowed start (the reduced start sets of
  `bpc._prepare`). Path multiplicity `(0, N_{m,r})`, so idle instances need no
  null column. Time is the main resource (arc consumption = duration).
- **Variables:** an integer `x` per path placement `(j, m, t)`, mapped to its
  arc in every group of type `m`; master-only binaries for placements on
  unbounded types (the static columns) and for the fixed-cost indicators
  `z_m`; an integer `u_m` per fixed-cost path type, mapped to the start arcs,
  counting its paths.
- **Master rows:**
  - coverage;
  - the optional caps on f1 and f2 (`z` included in f2);
  - the per-pair links `sum_t x_jmt <= z_m`;
  - the activation rows `u_m <= N_m z_m`. Without these the Rm root gap is
    about 13% instead of 0–2.7%, so a fair comparison needs them.
- **Packing sets:** one per job, over its arcs in all graphs (arc packing
  sets), which drive ng-routes and rank-1 cuts. The ng distance matrix is
  BPC's score (window centre and duration).
- **Branching:** expressions `y[j,m] = sum_t x_jmt` (priority 2), then `x`
  (priority 1).
- **Objective:** scaled to integers when it lies on a grid, so VRPSolver can
  round its bounds. The cutoff is the best heuristic schedule (as for BPC),
  set just above it, so VRPSolver can return a schedule of that value.
- **Results:** checked with `Instance.check_schedule` and recomputed from the
  instance before they are returned.

## Open points to check on the first real run

- Configuration: `vrpsolver/config/fedhpc.cfg` starts from the P_wjTj demo;
  `run.jl` appends `RCSPrankOneCutsMaxNumRows` (100, 0 with `--no-rank1`) and
  `GlobalTimeLimit`. The initial ng size is not passed yet (the exporter's
  `ng_size` field is unused): find the BaPCod parameter name in the
  VRPSolver parameter docs.
- Rank-1 cuts with *arc* packing sets: supported according to the API, but
  untested here; compare with `--no-rank1`.
- The status VRPSolver reports when the cutoff is already optimal
  (`found = false`, status `Optimal`?). `solve_vrpsolver` then returns
  `"infeasible"`, and the caller's cutoff schedule is the optimum. Confirm,
  and return that schedule directly.
- Not supported: multi-node jobs (`require_unit_demand`, as BPC).
