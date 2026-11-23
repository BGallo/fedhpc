# MOSPACE (CCGrid submission): analysis of the review and improvement plan

*Written 2026-11-20 from `MOFE_FEDHPC_CCGRID/IEEE-conference-template-062824.tex`, `critique.md`
(the reviewer's report, which recommends a major revision), and the experiment configs and policy
code in `overseer/`.*

## Status

| Item | Status |
|---|---|
| A1 Architecture section + figure | **Drafted**: `MOFE_FEDHPC_CCGRID/sections/architecture.tex`, `figures/architecture*.tex`, `architecture_draft.pdf`. Revised so that MOSPACE is the **only scheduling authority** (central scheduler; pools only execute) |
| Slurm co-existence | **Analysed**: `slurm_integration.md`. Recommended: a holding partition plus an external dispatcher (MOSPACE decides pool and start; Slurm executes) |
| All other items | Open |

---

## 0. What the code and data show (they shape the plan)

- **Every policy uses oracle runtimes.** All configs set `overwrite_predicted_runtime_with_actual_runtime = Yes`, the baselines too. The comparison is fair, but every online result is an oracle-runtime result, and the paper never says so (critique §5.3).
- **Partition rules match within each table.** With cloud, both MOSPACE and Fed-Slurm use `ignore_partition_constraints = Yes`. Without cloud, both use `No`. That is fair, but it should be stated.
- **MOSPACE has unlimited cloud capacity** (50,000 nodes per type). The paper's Table VII compares it with baselines capped at 50–400 nodes (§6.2). A **`Fed-Slurm cloud unlimited`** run already exists and is missing from the paper:
  - **W2:** same turnaround (7,989.7 job-h). Fed-Slurm costs **$17,761**; MOSPACE at α=0.75 costs **$12,635**. At α=0.25, MOSPACE gives 8,047.6 job-h for **$6,471**. This is the strongest equal-resources result.
  - **W1:** `Fed-Slurm (on-prem first) unlimited` **ties** MOSPACE α=0.75 (1,700.7 job-h, $1,688). MOSPACE's advantage on W1 only shows at lower α. Say so openly.
- **Experiments already set up:**
  - `configs/latency_ablation`, `generate_fedhpc_latency_ablation.py`
  - `fedhpc_no_cloud_slot300_*` (a second slot size Δ)
  - `fedhpc_ga_mid_*` (the GA online)
  - `fcfs_star_wait*`, `fcfs_wise_burst_sim*` (cloud-bursting baselines)
  - `predict_runtime` (runtime prediction)
  - `startup_time_seconds` (provisioning delay)
- **Simulator ambiguity:** the configs run the Wise-Burst event loop, and a Batsim bridge exists with a parity check (`scripts/compare_engines.py`). The paper says "Batsim 5". State which engine produced each table.
- **What the policy actually does** (`fedhpc_policy.py`):
  - one decision per event timestamp
  - plan reuse, re-solving on arrival, early/late end, or a failed start
  - early starts that never delay a planned job (conservative backfilling against the plan)
  - virtual partitions (identical nodes grouped into pools)
  - the horizon H = max(48, all-on-prem list-schedule makespan) with a 900 s slot
  - with `scope = window_jobs`, jobs pending at t0 run first-come-first-served before the planner
- **Inconsistencies in the text:**
  - Section VI says Gurobi takes 71 s; Table V says 91.1 s.
  - The "257% threshold gap" has no supporting table.
  - α and λ are both used for the same weight.
  - `\code`, `\SI` and `\cref` are used without their packages; `sec:workloads` is undefined.
  - Table I has no caption or label.
  - `IEEEtran.bst` is not installed locally.

---

## Part A: Make MOSPACE a believable federation-level central scheduler (core)

- **A1. Architecture section with a figure.** *(done, revised to central authority)*
  - MOSPACE is the only scheduling authority: it owns pool and start time, and the pools execute.
  - Four components: job queue, state monitor, planner, dispatcher.
  - A table of what the platform must provide versus how the evaluation provides it.
  - Deployment on Slurm: see `slurm_integration.md`.
- **A2. Specify the online loop (§5.4, §3.2).**
  - Trigger: once per event timestamp.
  - Plan reuse and the three invalidation rules.
  - Commitment: only due jobs are started.
  - Horizon and slot size.
  - Solver budget: Gurobi `TimeLimit`, incumbent, GA as a fallback.
  - Report the decision-time distribution (p50/p95/max), the plan-reuse rate, and the latency-ablation result.
- **A3. Narrow the claims.**
  - Adopt the critique's research question (§2.1) and its "strongest defensible contribution" (§11).
  - "Incremental external expenditure" instead of "total cost" (§3.3).
  - "Exact formulation of the discretised problem" instead of "exact solution to cloud bursting" (§8.2).
  - "Single-node parallel jobs with exclusive allocation" (§2.3).
  - Rewrite the contribution bullets.
- **A4. Multi-site experiment (§2.2)**, the most important new experiment for the federation claim.
  - **Minimum:** split the production cluster into two "sites" (e.g. CLB1+CSR1 / CSR2+CSR3), each with a home-site data location and a cross-site transfer time and cost. Keep the trace's partition compatibility.
  - **Compare:** isolated sites, Slurm-multi-cluster-inspired, Slurm-federation-inspired, and MOSPACE, each with and without cloud.
  - **Stretch:** federate the production trace with MIT Supercloud.

## Part B: Controlled comparisons (critical/high)

- **B1. Equal cloud capacity (§6.2).**
  - Add the `cloud unlimited` rows.
  - Run MOSPACE with N_m capped at 50/100/200/400.
  - Report MOSPACE's peak number of concurrent cloud nodes (from the per-job CSV).
- **B2. Cost-vs-turnaround plot (§7.3).** One point per policy setting, with 5–7 α values. This replaces most of Table VII.
- **B3. Cloud-bursting baseline (§6.3).** Threshold or star-wait under the same runtimes, prices and capacity. Back up the 257% figure with it, or delete the claim.
- **B4. Document the baselines (§6.1).**
  - Rename them "Slurm multi-cluster-inspired" and "Slurm federation-inspired".
  - Describe their local policy, eligibility rules and sibling selection.
  - Explain what "faithful" means and why Fed-Slurm equals EASY in the no-cloud W1 run.
- **B5. Imperfect runtimes (§5.3).**
  - Three information settings: oracle, user walltime, and `predict_runtime`, optionally with ±x% noise.
  - Every policy gets the same information.
  - Rolling re-planning should make MOSPACE robust here, so this experiment shows a strength.

## Part C: Turn the tail-wait weakness into a feature (§7.2)

- Report the trade-off openly. Against EASY, MOSPACE gives −8% mean turnaround, but +36% (W1) and +22% (W2) in p95 wait, and a higher maximum wait.
- Add a maximum-wait variant: restrict on-prem starts to t ≤ a_j + W_max, with cloud as the fallback, or add aging weights to w_j.

## Part D: GA correctness and its independence from Gurobi (critical, §4)

- **D1.** Use a lexicographic fitness: fewest unplaced jobs, then the turnaround cap, then Z. Handle λ=0, where the cap penalty is multiplied by α₁ = 0. Define Z for partial schedules.
- **D2.** Reference points without Gurobi.
  - With unbounded cloud, f₁ˡᵉˣ and f₂ˡᵉˣ can be computed in closed form; estimate f₁⁰ heuristically.
  - Report the GA's runtime in deployment mode.
  - Show the online `fedhpc_ga_mid` runs: this is the license-free claim.
- **D3.** Precise wording.
  - "Indirect policy-based representation", "WSPT-inspired", "sampled best-improvement".
  - Delete "reduces to list scheduling".
  - Bring Algorithm 1 in line with the prose: Φ, the ⌈B/2⌉ budget, niche elitism, converged seeds.
- **D4.** Evidence.
  - Ablations: without MSXF, without converged seeds, without niche elitism, dispatch-only, serial-only.
  - At least 10 seeds on the hard W2 N=100 cases.
  - A runtime-vs-gap curve.

## Part E: Model and offline rigor

- **Discretisation and horizon (§3.2).**
  - Report Δ, H, the slot mapping, the feasibility guarantee and model sizes.
  - Compare Δ = 300 vs. 900 s.
  - Separate the solver's optimality gap from the discretisation error.
- **Formulation strength (§3.1).**
  - Root LP gap, branch-and-bound nodes, and Gurobi status and MIP gap for each run.
  - Optionally, a standard time-indexed model on a subset.
- **I/O (§3.4).**
  - Split input and output volumes.
  - Use an effective bandwidth B_eff with a sensitivity check.
  - Separate billing granularity (per second, 60 s minimum) from the slot length Δ (§3.5).
- **Heterogeneity (§3.5).** A sweep over P_m and π_m.
- **Offline confounds (§5.1–5.2).**
  - A paired test: AWS alone vs. AWS plus Azure on the same on-prem pool.
  - Specify the residual runtimes of initially running jobs (b_m(t)).
  - Check zero-cost feasibility for every instance.
- **MIT Supercloud (§7.4).** Provenance, time window, runtime distribution, filtering, initial state.
- **Model section:** add the committed load b_m(t) (the architecture section refers to it).

## Part F: Presentation (low effort, do first)

- Fix the inconsistencies listed in §0.
- Cut the general MILP introduction.
- Merge the informal N/M/O notation into the formal notation.
- Rewrite Related Work around the problem dimensions (critique §8.1).
- Add a reproducibility paragraph: versions, Gurobi settings and time limits, hardware, threads, seeds.
- Write a real Results discussion.

---

## Suggested order

| Phase | Work | New compute? |
|---|---|---|
| 1 | F, A1 *(done)*, A3, B4, oracle statement, engine statement | No |
| 2 | B1, B2, B3, A2 (decision times, latency ablation) | Mostly existing configs |
| 3 | B5, A4, C | New runs |
| 4 | D1–D2 + Table V re-run, D4, Δ / LP-gap / heterogeneity studies; optional Slurm prototype (`slurm_integration.md` §4) | New runs |

Phases 1–3 make the federated central-scheduler claim believable. Phase 4 hardens the
exact-versus-heuristic part. The critique allows the I/O split, the class confound and an
ε-constraint Pareto front to be stated as limitations instead.
