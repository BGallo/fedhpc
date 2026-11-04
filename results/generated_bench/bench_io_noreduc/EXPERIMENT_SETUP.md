# Experiment setup — `bench_io_noreduc`

Exact Gurobi optima of the fedhpc weighted-sum formulation, without dominance
reductions, against the dispatch heuristic and threshold bursting, on the
108-instance full factorial of the generated classes A and B. Results:
`../bench_io_noreduc.jsonl` (one record per instance); machine-readable
settings: `../bench_io_noreduc.setup.json`; analysis: `summary.md` and
`fig1–5` in this folder (`scripts/analyze_generated_bench.py`).

## 1. Question

How far from the exact weighted-sum optimum are the dispatch heuristic and
threshold bursting, and at what time cost? Does every part of the objective
(VM price, egress price κ^IO, transfer time τ^IO, on-prem capacity and its
initial occupancy) act on the solutions?

## 2. Instances

108 = 2 classes × 2 workloads × 3 node counts × 3 loads × 3 io levels, built
by `scripts/build_generated_instances.py` (`fedhpc.instance_gen`). An
instance is a deterministic function of its name, e.g. `B_pos_n300_lE_iM`.

### 2.1 Workloads (jobs, horizon and slot copied unchanged)

| workload | source | jobs | horizon H | slot |
|---|---|---|---|---|
| fedhpc | `data/fedhpc_known_runtime_offline_10min.json` | 964 | 289 slots | 600 s |
| pos | `data/pos_congestion_known_runtime_offline_10min.json` | 3340 | 288 slots | 600 s |

Each job has an arrival slot a_j, runtime, cores, memory and (new here) an
output volume ι_j. All jobs have weight 1 and run at perf 1 on every type.

### 2.2 Machine catalogue

On-prem types cost 0 and have κ^IO = τ^IO = 0. Every cloud type is unlimited
and on demand. Cloud types are the smallest Ice Lake instance that covers the
on-prem node's physical cores (vCPU/2) and memory. Prices are Linux on-demand
list prices (`data/cloud_prices.json`, AWS us-east-1, Azure eastus).

| on-prem | AWS ($/h) | Azure ($/h) | class A | class B |
|---|---|---|---|---|
| c16 (16 c / 128 GB) | m6i.8xlarge (1.536) | D32s_v5 (1.536) | – | pos only |
| c32 (32 c / 256 GB) | m6i.16xlarge (3.072) | D64s_v5 (3.072) | – | ✓ |
| c40 (40 c / 300 GB) | m6i.24xlarge (4.608) | D96s_v5 (4.608) | ✓ (AWS only) | ✓ |
| c64 (64 c / 480 GB) | m6i.32xlarge (6.144) | D128s_v6 (6.451) | ✓ (AWS only) | ✓ |

A type no job of the workload is placed on is dropped: no fedhpc job fits
c16, so B·fedhpc has 3 on-prem and 6 cloud types, and B·pos has 4 and 8.
Class A always has 2 and 2.

| | AWS | Azure |
|---|---|---|
| κ^IO egress ($/GB, internet, first paid tier) | 0.09 | 0.08 (routing preference: internet) |
| τ^IO transfer (slot/GB), binding disk baseline | 0.012716 (gp3 125 MiB/s) | 0.013333 (Premium SSD v2 125 MB/s) |

Volume tiers and the free 100 GB/month are ignored. τ^IO is the minimum over
source disk, network and sink (Liu et al., HPDC'17). The provider network
caps (≥ 5 Gbps) never bind.

### 2.3 Factors (one level for the whole instance)

| factor | levels | meaning |
|---|---|---|
| count N | 100, 300, 500 | nodes of **every** on-prem type |
| load | F / H / E | running jobs per on-prem type at t = 0: 0, ⌊N/2⌋, N |
| io | L / M / H | output-retention policy per job class (§2.4) |

Each running job holds one node until its end slot. End slots are drawn with
replacement from the workload's own `running_jobs` end times. The RNG is
seeded by sha256(class, workload, type, N), not by the load, so the H set is
the first half of the E set.

### 2.4 Result-data volume ι_j (`fedhpc.io_model`)

ι_j = r · 40 · (c_j/40)^0.54 · h_j · 10^(0.5 z_j). Here c_j is the core count,
h_j the runtime in hours and z_j ~ N(0,1), clipped to ±3. z_j is seeded per
(workload, job), so every io level uses the same draw (common random numbers).

The policy rates r are S = 0.02, R = 0.13 and F = 1.5 GB/core-h, measured
with OPM Flow + Darshan. A job is an ensemble member when ≥ 10 jobs share its
(user, job-name stem) (`data/job_classes_*.json`); otherwise it is a single run.

| level | ensemble | single |
|---|---|---|
| L | S | R |
| M | R | F |
| H | F | F |

The horizon guard caps ι_j at (H − ⌈a_j⌉ − e_j)/τ_max, with τ_max the
slowest cloud τ^IO in the instance. This keeps every cloud type feasible for
a job released at a_j.

The guard bound only on pos, in 3 of 3340 jobs per instance at io M and 7 at
io H. It never bound on fedhpc or at io L. Each instance records the count in
`_info.io.capped`.

### 2.5 Reproducibility check

On 2026-11-03 all 108 files were rebuilt into a scratch directory. Every
sha256 equals `data/generated/manifest.json` (3f2376…). The input hashes
are in the `.setup.json`.

## 3. Model

The model is the space-time network formulation, `SpaceTimeFormulation`, with
`collapse_unbounded`, `pareto_unbounded`, `rejection_deadlines` and
`lex_deadlines` all **off**. That is the plain network, with none of the
R1/R2/R3/lex reductions.

- f1 = Σ_j (C_j − a_j), total turnaround in slots.
- f2 = Σ_j c_jm, total cost in USD on the grid δ = 10⁻⁴.

The cost of placing job j on type m is
c_jm = κ^IO_m ι_j + κ^VM_m p^bill_jm (κ^S = 0).

- Billed slots: p^bill = e_j/P_m + τ^IO_m ι_j.
- Occupied slots: p^occ = max(1, ⌈p^bill⌉).

## 4. Methods, per instance, in this order

1. **lex f1** — `solve_f1`: phase 1 minimises f1, phase 2 minimises f2
   with f1 fixed. Output: (f1^T, f2^T).
2. **lex f2** — `solve_f2`: minimises f2, then f1. Output: f2^0 (= 0 in all
   108) and f1^0.
3. **Baselines**:
   - `greedy_earliest_completion`: completion first, then cost, least-slack-first.
   - `onprem_only_spt`: never bursts. It completed all jobs in all 108 instances.
   - `threshold_bursting(θ)`: arrival order. A job runs on-prem if its earliest
     on-prem wait is ≤ θ slots, else on the cheapest cloud type. Swept over
     θ ∈ {0, 1, 2, 3, 6, 12, 24, 48, 96}.
4. For each λ ∈ {0.25, 0.5, 0.75}:
   - **Gurobi weighted sum** — `solve_weighted_sum` minimises
     g = λ·(f1 − f1^T)/(f1^0 − f1^T) + (1 − λ)·f2/f2^T, subject to f1 ≤ f1^0 and
     f2 ≤ f2^T. On a degenerate instance (f2^T = 0) it minimises f2 subject to
     f1 ≤ f1^0 instead.
   - **Dispatch** — `weighted_solve_dispatch` with the same reference points
     and its defaults: K = 256 candidates, 60 ILS kicks of 16 flips, flags = 64,
     μ grid (mu = −1, n_mu = 1), seed 42, 4 threads.

     **Superseded in `summary.md` and the figures (2026-11-03).** Their
     dispatch results come from a re-run of the current version:
     - **Script:** `scripts/dispatch_validate.py`.
     - **Configuration:** K = 64 in batches up to K_max = 256;
       flags = 64|4096|8192 (mixed ranking, plus an EDF start without
       deadline offload); 60 kicks of 16 split over the two starts;
       kick_patience = 5; seed 42; 4 threads; 3 instances in parallel.
     - **Inputs:** the same instances, reference points and λ.
     - **Records:** `../dispatch_study/validate_fast_default.jsonl` (non-trivial)
       and `validate_fast_default_trivial.jsonl` (trivial).
     - **Not changed:** the Gurobi and threshold numbers, and this run's own
       dispatch results, which remain in `bench_io_noreduc.jsonl`.
     - **Why the version changed:** `../dispatch_study/README.md`.
   - **Threshold** — the θ from step 3 with the lowest g at this λ, chosen
     after the fact. This is an oracle choice, so it favours the baseline.

Every schedule is checked with `Instance.check_schedule`.

**Gurobi parameters:** Threads = 4, TimeLimit = 600 s, MIPGap = 0, OutputFlag = 0.
- The lex phases also set MIPGapAbs = 0.999·step (1 slot for f1, δ for f2).
- A lex result is "optimal" only when both phases are certified within one
  step of the dual bound; otherwise it is "feasible".
- TimeLimit applies to **each `optimize()` call**, so a lex solve can take
  2 × 600 s plus model build time.
- A weighted-sum solve keeps Gurobi's own status.

**Timing:** `time.perf_counter()` around each call. It is wall time and
includes model build and extraction.

## 5. Metrics

- **Trivial instance:** f2^T = 0. The fastest schedule is free and the front
  is a single point. 21 of 108 are trivial; they are reported separately and
  excluded from all quality statistics.
- **Gap:** 100·(g_method − g*)/g*, per (instance, λ), with g* the Gurobi value.
- **Paired io comparison:** ratios between io levels within a cell (class,
  workload, N, load), since the three io levels share jobs, capacity and z_j.

## 6. Execution

```
nohup nice -n 10 uv run python scripts/bench_generated.py --manifest --no-reductions \
    --workers 3 --threads 4 --out results/generated_bench/bench_io_noreduc.jsonl \
    > results/generated_bench/bench_io_noreduc.log 2>&1 &
```

- 3 instances run in parallel in a process pool, with 4 threads each:
  12 of 32 logical CPUs.
- Order: pos before fedhpc, B before A, then by name.
- Started about 14:35 and finished 19:57:01 on 2026-10-29 (−03:00), about
  5.4 h wall. 108/108 records, 0 failures.

| | |
|---|---|
| CPU | Intel Core i9-13900: 8 P-cores and 16 E-cores, 32 threads, up to 5.6 GHz. One NUMA node, no CPU pinning. |
| RAM | 62 GB. Each worker had about 9–11 GB RSS on B·pos. |
| OS | CachyOS Linux, kernel 7.3 |
| Python | CPython 3.14, uv-managed. The `.venv` has since been rebuilt for 3.14.9. |
| Solver | gurobipy 13.0.1 (uv.lock), academic WLS license |
| numpy | 2.4.4 |
| Code | fedhpc `97fb178` plus `../bench_io_noreduc.code.patch` (sha256 735bd3e3…). All edits predate the launch. |

## 7. Caveats

- **Timing noise.** There was no CPU pinning, and the CPU mixes P- and
  E-cores. Another user's LLM inference server shared the machine. Compare
  times between methods only through medians.
- **Solves without a proof of optimality.** 13 of 540 Gurobi solves hit the
  time limit:
  - 8 lex f2 (B/A·pos, N = 100, load H/E);
  - 3 lex f1 (B·pos n300 E iH, A·pos n300 H iH, B·fedhpc n100 E iH);
  - 2 weighted sums (B·pos n100 E iM and A·pos n100 E iL, both at λ = 0.25).

  A non-certified anchor rescales g for every method alike. At the two
  time-limited weighted sums, the reported dispatch gap is a lower bound.
- **An unexplained run time.** `B_pos_n300_lE_iH` lex f1 took 3194 s, beyond
  the 2 × 600 s limit plus build time. The record has no per-phase timing, so
  the cause, most likely contention, cannot be attributed.
- **Dispatch was run once,** with seed 42, so there is no variance estimate.
  The re-run was on the same machine, but not while Gurobi was running.
- **Threshold θ is an oracle choice** (§4), which favours that baseline.
- **This is not the production configuration.** Without reductions, the
  Gurobi times are an upper bound on the reduced model's times.
