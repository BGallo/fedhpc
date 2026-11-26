# Where the BPC and LP-first solvers fit: literature, benchmarks, positioning (2026-11-26)

A companion to `research_plan_2026-11.md`. That plan ranks the thesis avenues
(A: near-integrality theory, B: exact bi-objective search, C: Rm||TWCT+TMUC).
This note answers different questions:
- which other published problems these two solvers can attack;
- which public instance sets to use;
- who the competitors are, both academic and commercial;
- where the results could be published, and what a niche industrial solver
  would need.

**How this was compiled.** Web searches on 2026-11-26. Most claims come from
abstracts, publisher pages and repository records, not full papers. Instance
sizes and results are as the authors state them. Read the papers before
citing a number.

## 1. What the solvers actually solve, in scheduling notation

In classical scheduling notation, the problem behind both engines is:

> **Heterogeneous pools of identical machines** (type m has N_m identical
> machines, N_m = ∞ allowed), **job- and type-dependent processing times and
> costs** (unrelated across types), **release dates**, objectives **Σ w_j C_j**
> and **Σ c_jm + Σ U_m** (a per-type usage cost), with **caps on either
> objective** and an **exact Pareto front**. Optional: **multi-unit (gang) jobs**
> (q_j machines at once) and running jobs that hold machines at time 0.

Each feature has a name in the literature:

| Feature here | Name in the literature | What it implies |
|---|---|---|
| Types with N_m = 1 | R \| r_j \| Σw_jC_j (unrelated machines) | BPC territory, with real root gaps |
| Types with N_m in the hundreds | "machine pools", cumulative or time-indexed capacity | LP-first territory, LP near-integral |
| Unbounded type with a cost | outsourcing, subcontracting, rejection with penalty | R1/R2 are exact outsourcing-dominance rules |
| U_m | machine usage, activation or opening cost | Chen et al. 2024 (the Rm benchmark) |
| q_j > 1 | rigid parallel jobs, multiprocessor tasks, P \| size_j \| Σw_jC_j | only LP-first handles it |
| Running jobs | machine availability / non-availability windows | release groups |
| f1 and f2 together, exact front | bi-objective scheduling, criterion-space search | box and ε-search in `pareto.py` |

## 2. Problem families that map onto the solvers

The "fit" column says how much modelling the repo would need: **as is**,
**data only** (a new instance reader), or **extension** (new rows or columns).

### 2.1 Already benchmarked: keep, and make them complete

| Problem | Reference | Fit | Notes |
|---|---|---|---|
| P \| r_j \| Σw_jC_j | Kramer, Dell'Amico, Feillet & Iori, *C&OR* 123 (2020) 105018 ([arXiv](https://arxiv.org/abs/2005.10097)) | as is | Random instances, up to 200 jobs × 10 machines solved to optimality. The set is in `data/prwc_kramer2020`. Report it in full, not as one validation table, because it is the standard comparison for arc-flow B&P. |
| Rm \|\| TWCT + TMUC (machine usage costs) | Chen, Chu, Sahli & Li, *EJOR* 316(3) (2024) 856–872 ([IDEAS](https://ideas.repec.org/a/eee/ejores/v316y2024i3p856-872.html)) | as is | Up to 225 jobs × 15 machines within 1 h in the original paper. Plan item C. |
| Same, with family setup times | *OR Spectrum* 47(1) (2025) ([IDEAS](https://ideas.repec.org/a/spr/orspec/v47y2025i1d10.1007_s00291-024-00778-8.html)) | extension: setups | Sequence-dependent setups break the "column = sequence of independent arcs" structure, since the pricer would need the previous job in its state. Not worth it unless setups become a thesis topic. |

### 2.2 Directly applicable, not yet tried (best returns)

**(a) Unrelated parallel machines, R || Σw_jC_j and R | r_j | Σw_jC_j.**

This is the most competitive arena for BPC.
- Yu, Li, Baldacci et al., *INFORMS JoC* (2026), doi
  [10.1287/ijoc.2024.0704](https://pubsonline.informs.org/doi/10.1287/ijoc.2024.0704):
  - columns built on Smith's rule (paths on a DAG), bucket-graph labelling,
    limited-memory subset-row cuts and strong branching;
  - up to 20 machines × 200 jobs, plus real JD.com instances with *rational*
    processing times;
  - code and instances at `github.com/INFORMSJoC/2024.0704`.
- Bülbül & Şen (Benders, unrelated machines): up to 1000 jobs × 8 machines in
  seconds ([PDF](https://research.sabanciuniv.edu/id/eprint/30468/1/RmTWCT_rev1.pdf)).
- Bulhões, Sadykov, Subramanian & Uchoa: a generic branch-cut-and-price for
  identical, uniform and unrelated machines with release dates and setups,
  with any objective defined over completion times; *J. Scheduling* (2020)
  ([PDF](https://www.math.u-bordeaux.fr/~rsadykov/papers/Bulhoes_etall_LOGIS18.pdf)).
- Kowalczyk & Leus, *INFORMS JoC* 30(4) (2018) 768–782: B&P with ZDD pricing
  for P||Σw_jC_j ([IJOC](https://pubsonline.informs.org/doi/fpi/10.1287/ijoc.2018.0809)).
- Kramer et al., enhanced arc-flow formulations for P||Σw_jC_j
  ([arXiv 1808.10661](https://arxiv.org/abs/1808.10661)), with "benchmark set 2"
  that Kowalczyk & Leus could not fully close.

*Honest assessment.* Without release dates, a Smith's-rule pricer is
polynomial in the number of jobs, while ours is time-indexed and pseudo-
polynomial in H. On R||Σw_jC_j we would most likely lose. Our structural edge
shows up only with **release dates, machine pools (N_m > 1), usage costs, or a
bi-objective question**. Pick the variants accordingly:
- R | r_j | Σw_jC_j built from the Yu et al. instances;
- the bi-objective version Σw_jC_j vs. cost on the same instances. No exact
  bi-objective B&P for parallel-machine scheduling turned up in the search
  (see 2.3).

**(b) Scheduling with outsourcing, rejection or order acceptance.**

This is the closest classical twin of cloud bursting, and its exact
literature is thin:
- Wang, Yin & Cheng, *Omega* 81 (2018): enhanced B&P for parallel-machine
  rescheduling with rejection. Objective: total (unweighted) completion time,
  with rejection cost as one of three criteria
  ([RePEc](https://econpapers.repec.org/RePEc:eee:jomega:v:81:y:2018:i:c:p:246-260)).
- Naderi & Roshanaei, *EJOR* 286(3) (2020): Benders for order acceptance on
  identical parallel machines
  ([IDEAS](https://ideas.repec.org/a/eee/ejores/v286y2020i3p811-827.html)).
- Mestry, Damodaran & Chen, *EJOR* 211(3) (2011): B&P for order acceptance and
  capacity planning.
- Ou, Li & Zhong, *EJOR* 329(3) (2026): outsourcing on dedicated parallel
  machines, with approximation results only
  ([RePEc](https://econpapers.repec.org/RePEc:eee:ejores:v:329:y:2026:i:3:p:798-807)).
- Maack, Meyer auf der Heide & Pukrop, "Server cloud scheduling",
  *Algorithmica* (2024): one server plus unlimited cloud machines, makespan
  and cost, approximation results
  ([arXiv](https://arxiv.org/abs/2108.02109)).

*Opportunity.* The search found no exact B&P for the weighted
P | r_j | Σw_jC_j + outsourcing cost problem, nor for its bi-objective
version. Our model *is* that problem: the cloud is an unbounded outsourcing
type, and R1/R2 are exact dominance rules for it. Making it a benchmark is
cheap: take Kramer's P|r_j|Σw_jC_j instances and add an outsourcing option
with cost ∝ p_j and turnaround = r_j + p_j + δ. The near-integrality story of
plan item A, where the cloud is the overflow that makes rounding cheap, is a
theorem about exactly this class. That makes it a cleaner audience than "HPC".

**(c) Discrete berth allocation with dynamic arrivals.**

Berths are machines, vessels are jobs with release dates and berth-dependent
handling times, and the objective is weighted service time: this is
R | r_j | Σw_jC_j.
- B&P exists: Robenek et al. (2014) for the dynamic hybrid BAP; Vacca et al.
  (2013), the first exact B&P for BAP and BAP with quay cranes
  ([EPFL PDF](https://infoscience.epfl.ch/record/196072/files/TomNitBierStef12.pdf?version=6)).
- Standard instance sets: Cordeau et al. (2005) and Buhrkal et al. (2011),
  not checked here.

*Fit:* data only for the basic discrete BAP, with capacity-1 types, so BPC.
Extensions such as berth availability windows map onto running jobs and
release groups. It opens *Transportation Science* and *TR-B* as venues.

**(d) Interval scheduling, temporal bin packing, VM placement.**

Jobs with fixed start and end times, packed on machines that cost money.
Motivated by cloud computing.
- Dell'Amico, Furini & Iori, *C&OR* 114 (2020) 104825: B&P for temporal bin
  packing ([arXiv 1902.04925](https://arxiv.org/abs/1902.04925)).
- Muir & Toriello, "Interval scheduling with economies of scale", *C&OR* 150
  (2023): B&P that proves instances with hundreds of jobs, some derived from
  cloud data ([preprint](https://optimization-online.org/?p=17547)). They also
  have a temporal bin packing paper for VM placement.
- Tactical fixed job scheduling with machine classes of different cost
  ([arXiv 1402.1881](https://arxiv.org/abs/1402.1881)).

*Fit:* extension. Fixed times mean |S_jm| = 1, which is trivial. But the
objective is a cost **per machine used**, while our U_m is **per type**. That
needs a per-column fixed cost, i.e. the activation-row idea with a cost on
every non-null column, which the pricer already supports through its
first-arc cost. Temporal bin packing also has vector capacity (CPU and
memory per bin), which our "a job holds whole instances" model does not
have. A medium-size extension with its own established benchmark sets.

**(e) Rigid parallel jobs, P | size_j | Σw_jC_j, and GPU gang scheduling.**

The multi-node extension.
- P2|size_j|Σw_jC_j is strongly NP-hard, and the complexity of the
  unweighted case was still open per a recent survey
  ([J. Sched. 2023](https://doi.org/10.1007/s10951-023-00782-w)).
- The searches found **no exact integer-programming method** for weighted
  rigid jobs with release dates. Only approximation results and the
  malleable-task B&P of Duarte & Valério de Carvalho (2010).
- Applied GPU-cluster work (Sia, CMU thesis 2025; GOGH 2025; RLTune, SoCC
  2025; Synergy) uses MILP or ILP for allocation, usually round-based and
  heuristic.

*Opportunity:* LP-first already solves the time-indexed model with q_j
coefficients exactly. "Exact weighted completion time for rigid parallel
jobs on heterogeneous pools, at trace scale" looks like an open gap. Two
caveats:
- The q_j-coefficient rows weaken the LP, so near-integrality (plan item A)
  may fail here. That is a theorem to test, not to assume.
- The GPU-systems community values online decisions. The offline optimum
  fits best as a "value of clairvoyance" baseline (plan item D).

**(f) Electric-vehicle charging with identical chargers.**

Charging stations are pools of identical machines with time windows, and
most of this literature uses time-indexed integer programs.
- A 2026 B&P on the partition-colouring view
  ([arXiv 2603.21374](https://arxiv.org/abs/2603.21374)).
- Klein & Schiffer, column generation in branch-and-bound
  ([arXiv 2201.03972](https://arxiv.org/abs/2201.03972)).

*Fit:* data only, if the objective is weighted completion or tardiness and
charging is non-preemptive. Real EV charging often has variable power, which
we don't model. Lower priority.

### 2.3 Bi-objective B&P: the methodological gap

- Parragh & Tricoire describe theirs as, to their knowledge, the first
  bi-objective branch-and-price. Their applications are location and set
  covering, not scheduling ([arXiv 1809.06823](https://arxiv.org/abs/1809.06823)).
- Bi-objective VRP B&P: Jozefowiez, Ngueveu & Glize.
- Exact criterion-space search:
  - the balanced box method, Boland, Charkhgard & Savelsbergh, *INFORMS JoC*
    27(4) (2015) 735–754 (verified);
  - warm-starting lower bound sets: Forget, Gadegaard & Nielsen, *EJOR* 302(3)
    (2022) ([IDEAS](https://ideas.repec.org/a/eee/ejores/v302y2022i3p909-924.html));
  - warm-starting scalarisations ([arXiv 2507.21933](https://arxiv.org/abs/2507.21933)).
- **No paper turned up that reuses columns across the scalarised
  subproblems**, which this repo already does (one column pool and schedule
  list per Pareto search), nor any that proves Pareto points with a
  reduced-cost certificate instead of a MIP. With the measured numbers
  (thousands of proven points at 964 and 3340 jobs), plan item B is a
  credible *INFORMS JoC* or *Math. Prog. Computation* paper. The search for
  "column reuse in criterion-space search" was thin, so check reference lists
  before claiming priority.

## 3. Benchmark instance libraries

| Library | Problem | Size | Where | Engine | In repo? |
|---|---|---|---|---|---|
| Kramer et al. 2020 PrwC | P \| r_j \| Σw_jC_j | ≤ 200 jobs exact | authors (rar in `data/prwc_kramer2020`) | BPC | yes |
| Chen et al. 2024 Rm-TMUC | Rm \|\| TWCT+TMUC | ≤ 225 × 15 | authors | BPC | yes (`rm_M*N*`) |
| Yu et al. 2026 UPMSP | R \|\| Σw_jC_j (+ JD.com real) | ≤ 200 × 20 | `github.com/INFORMSJoC/2024.0704` | BPC | **no, get it** |
| Kowalczyk & Leus 2018 / Kramer arc-flow sets | P \|\| Σw_jC_j | ≤ ~hundreds | IJOC supplement, arXiv 1808.10661 | BPC | no |
| OR-Library weighted tardiness (Pessoa et al. 2010 used P \|\| Σw_jT_j) | P \|\| Σw_jT_j | 40–100 jobs | OR-Library | needs a tardiness objective | no |
| Vallada & Ruiz 2011 | R \| s_ijk \| C_max | 50–250 × 10–30 | [RiuNet](https://riunet.upv.es/handle/10251/35412) | needs setups and makespan | no, low fit |
| Temporal bin packing (Dell'Amico et al. 2020; Muir & Toriello) | fixed intervals, machine cost | hundreds | papers | extension (d) | no |
| Discrete BAP (Cordeau et al. 2005; Buhrkal et al. 2011) | R \| r_j \| Σw_jC_j | 25–60 vessels | papers (not checked) | BPC | no |
| Parallel Workloads Archive (SWF) | HPC traces | 10⁴–10⁶ jobs | [cs.huji.ac.il](https://www.cs.huji.ac.il/labs/parallel/workload/logs.html) | generator input | no |
| MIT SuperCloud | Slurm, 460k jobs, 98k GPU | trace | [GitHub](https://github.com/MIT-AI-Accelerator/MIT-Supercloud-Dataset), AWS | LP-first | yes (`mit_supercloud_*`) |
| Alibaba GPU v2020/v2023/v2026 | GPU jobs (gang sizes) | 6k–155k GPUs | [github.com/alibaba/clusterdata](https://github.com/alibaba/clusterdata) | LP-first, multi-node | no, best q_j > 1 source |
| Google Borg 2019 | cluster cells | 8 cells, 1 month | [google/cluster-data](https://github.com/google/cluster-data) (BigQuery) | generator input | no |

Two gaps stand out:
- **The repo has no real multi-node or GPU benchmark.** `geo_gen.py` and the
  A/B generator emit single-node jobs. The Alibaba GPU traces (gang sizes,
  GPU types) are the natural source for the q_j > 1 experiments, and they
  carry the "GPU cluster" story to systems venues.
- **The PWA logs give the sizes of rigid parallel jobs directly** (SWF column
  "allocated processors"), which also yields q_j.

## 4. Competitors: academic and industrial

**Academic exact methods to beat or match on shared instances:**
- Generic BCP: **VRPSolver**, Pessoa, Sadykov, Uchoa & Vanderbeck, *Math.
  Programming* 183 (2020) 483–523
  ([PDF](https://www.math.u-bordeaux.fr/~rsadykov/papers/Pessoa_etall_MPB20.pdf)).
  It has the same ingredients as our BPC (ng-paths, limited-memory rank-1 cuts,
  packing sets, Ryan–Foster generalisation, path enumeration), packaged as a
  modelling tool. Bulhões et al. is its parallel-machine application. Any BPC
  paper will be asked "why not VRPSolver?". Our answers: machine pools with
  N_m > 1, the LP-first regime, and exact bi-objective search. On the
  capacity-1 Rm benchmarks a direct VRPSolver run would make the strongest
  baseline.
- Problem-specific: Yu et al. 2026 (UPMSP), Kowalczyk & Leus 2018, Kramer et
  al. 2020, Chen et al. 2024, Bülbül & Şen (Benders).
- Recent ML-assisted column generation, which reviewers will mention:
  - learned column prefilling for BPC
    ([arXiv 2610.21001](https://arxiv.org/abs/2610.21001));
  - transformers for improving columns in parallel machine scheduling
    ([arXiv 2410.15601](https://arxiv.org/abs/2410.15601));
  - column generation with domain-independent DP
    ([arXiv 2510.14317](https://arxiv.org/abs/2510.14317)).

**Industrial solvers, and where a niche exists:**
- **Gurobi / CPLEX on the compact model.** The project measured LP size,
  not branching, as their bottleneck at trace scale (7.6 M variables, 404 s of
  barrier per box step). LP-first wins there by 10–25× and is Gurobi-free on
  HiGHS.
- **First-order LP (PDLP / cuPDLP).** Applegate, Lubin et al. ([arXiv
  2501.07018](https://arxiv.org/abs/2501.07018)) report LPs where Gurobi barrier
  exceeds 1 TB RAM while PDLP gets 1% gaps. PDLP is in OR-Tools, and there is a
  GPU version. Two consequences:
  - it is a **threat**: a GPU PDLP on the full compact LP plus
    reduced-cost fixing could copy LP-first's trick without column
    generation;
  - it is an **opportunity**: a PDLP backend for LP-first's master, for its
    Lagrangian bound, or for the 10⁷-variable LPs of multi-day horizons.
  Benchmark it before claiming the large-LP niche.
- **CP-SAT (OR-Tools).** Mixed evidence on parallel-machine problems: CP-SAT
  ahead on wide problems with calendars and precedences
  ([arXiv 2512.11864](https://arxiv.org/abs/2512.11864)), MILP ahead on more
  MILP-like ones. Without an LP bound, CP-SAT is unlikely to prove optimality
  on 10³-job weighted-completion instances. One run on the captured eligible
  MIPs (plan §6 item 5) would settle it.

**The defensible niche for a product:**
> exact or certified-gap scheduling of 10³–10⁴ jobs on heterogeneous pools of
> identical machines with an outsourcing option, release dates and usage
> costs, including multi-unit jobs and the full cost–turnaround Pareto front.

Here the compact MILP is too big for barrier, the LP is near-integral,
outsourcing makes rounding cheap, and the decision-maker wants trade-offs,
not one weighted sum. That describes HPC/cloud bursting, GPU fleets, and
manufacturing with subcontracting. A product would also need:
1. online re-optimisation with warm starts (plan item D);
2. a modelling API that does not mention HPC (jobs, pools, outsourcing
   options, costs);
3. machine-checkable certificates (VIPR-style, or simply the dual vector plus
   the eligible-set MIP log), which is a selling point against heuristic
   schedulers;
4. solver independence (HiGHS today; PDLP next).

## 5. Venues: what fits where

| Contribution | Venue | Why |
|---|---|---|
| Near-integrality theorem (plan A) with an asymptotic gap bound under outsourcing | *Mathematical Programming*, *Math. of OR*, *IPCO* | a structural LP result with a rounding proof |
| LP-first + certificate + bi-objective search at scale (plan B), open code | *Math. Programming Computation*, *INFORMS JoC* | software and methods; IJOC requires a code repo (Yu et al. used `INFORMSJoC/…`) |
| Exact B&P for P \| r_j \| Σw_jC_j + outsourcing, bi-objective | *INFORMS JoC*, *Operations Research* (with theory), *EJOR* | open gap per 2.2(b) |
| Rm-TMUC closed (plan C) | *C&OR*, *J. Scheduling*, *EJOR* short | incremental, quick |
| Rigid jobs and GPU pools with LP-first | *Operations Research* (if theory holds), *IJOC*; *TPDS* or *SC* for the systems side | open gap per 2.2(e) |
| Berth allocation with these engines | *Transportation Science*, *TR-B* | only with a clear gain over Robenek/Vacca |

## 6. Recommended next experiments (ordered by value / cost)

1. **Get the Yu et al. 2026 UPMSP instances and code**
   (`github.com/INFORMSJoC/2024.0704`). Run BPC on R || Σw_jC_j to learn where
   we stand against the current state of the art. Then run the variants where
   we should win: with release dates, with machine pools, bi-objective.
   About a week.
2. **Outsourcing variant of Kramer's P | r_j | Σw_jC_j set.** Add a cloud type
   to every instance and compute exact fronts with BPC and LP-first. This is
   the cleanest test of the overflow rounding argument (plan A) on an
   established set. A few days.
3. **Multi-node benchmark from the Alibaba GPU v2020 trace**, or from PWA
   SWF processor counts. Gives q_j > 1 at trace scale, a test of whether
   near-integrality survives (the plan's stress test), and the first exact
   results for weighted rigid-job scheduling. About a week of data work.
4. **PDLP baseline**: the full compact LP on the 3340-job instance with
   OR-Tools PDLP, to measure the threat before claiming the niche. A day.
5. **VRPSolver on Rm-TMUC**: the baseline reviewers will ask for. A few days
   (Docker image, Julia model).
6. Optional and larger: discrete berth allocation (data only), and temporal
   bin packing (needs per-machine fixed cost and vector capacity).

## 7. Corrections and checks against `research_plan_2026-11.md`

That plan's references were from memory. The following were checked here:
- Chen, Chu, Sahli & Li, *EJOR* 316(3) 856–872 (2024): confirmed. The 2025
  *OR Spectrum* follow-up adds family setups.
- Balanced box: *INFORMS JoC* 27(4) 735–754 (2015): confirmed.
- Kowalczyk & Leus, *INFORMS JoC* 30(4) 768–782 (2018): confirmed.
- New since the plan was written, and directly relevant: Yu et al., *INFORMS
  JoC* 2026 (UPMSP BPC with lm-SRC and bucket graphs). Plan item C should
  cite and position against it. Our Rm results are on Chen's
  usage-cost variant, which Yu et al. do not address.
