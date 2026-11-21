# Data transfer between geographically distributed on-prem clusters

Design note (2026-11-19). **Level 1 implemented on branch `geo-distributed-io`
(2026-11-21)**, see §4.

## 1. How transfer is modelled today

The space-time network has no transfer flow. Transfer is folded into two
parameters per (job, type) pair in `Instance.build` (`src/fedhpc/data.py`):

```
p_bill[j,m] = e_j / P_m + τ^IO_m · ι_j
p_occ[j,m]  = ceil(p_bill + π_m)
c[j,m]      = κ^IO_m · ι_j + κ^VM_m · p_bill + κ^S_m · q^stor_j · p_bill
```

`SpaceTimeFormulation` only sees `p_occ` (how long a job arc is) and `c` (its
cost).

What this assumes:

- **Output only, one direction, one implicit "home".** `ι_j` is the result
  volume a cloud job sends home over internet egress (`io_model.py`). There is
  no input staging and no per-job data origin.
- **On-prem jobs move nothing.** `instance_gen.py` sets `io_time = cost_io = 0`
  on every on-prem type.
- **The upload runs on the VM.** `τ^IO · ι_j` is inside `p_occ`, so the
  machine stays held and billed while it uploads. Transfer therefore affects
  both f1 and f2.
- **No bandwidth contention.** The per-type networks are independent and are
  joined only by the assignment row.
- **No data passes from one job to another.** There is no precedence and no
  shared dataset.

## 2. Possible extensions, cheapest first

1. **Site-dependent parameters (Level 1).** Each on-prem type gets a `site`.
   Each job gets an origin `o_j` (where its inputs are) and a destination
   `d_j` (where its results go). Transfer then depends on the (job, type)
   pair:
   `τ_jm = τ(o_j→s(m))·ι^in_j + τ(s(m)→d_j)·ι^out_j`, and `κ_jm` likewise.
   Only the data layer changes. Cloud is the special case of a site with
   egress price > 0. Links between academic research networks are often
   `κ = 0, τ > 0`.
2. **Shared datasets (fixed-charge staging).** Add a binary `u[d,s]` for
   "dataset d is staged at site s", with cost `κ_ds·ι_d·u[d,s]` and the link
   `Σ_t x[j,m,t] ≤ u[d(j), s(m)]`. This is the same pattern as the fixed-cost
   `z_m` in `add_fixed_cost_vars`, and is relevant for ensemble members that
   share input decks.
3. **WAN links with limited capacity.** Either add per-slot capacity rows over
   active transfers, or model each link as a type with its own network plus
   precedence-linking rows. Both couple the per-type networks: the LP is no
   longer integral and the dominance reductions no longer apply.
4. **Data passed between jobs (workflows).** The time lag depends on both
   jobs' sites, so the constraint is quadratic in the assignments. That is a
   different problem class.

**Recommendation:** start with Level 1. Move to Level 3 only if WAN contention
is shown to bind; the current disk-bound estimate is about 1 Gbps per
transfer.

## 3. Level 1 with transfer folded into p_occ (no stage-in/stage-out split)

Here `τ_jm` goes into `p_occ[j,m]` and `κ_jm` into `c[j,m]`, on on-prem types
as well as cloud ones.

### Reductions: all still exact, with no code change

`p_occ` and `c` are already tables keyed by (j, m). Every proof in
`SPACETIME_REDUCTIONS.md` uses only these four facts, and all still hold:

1. f1 is non-decreasing in completion times (w ≥ 0).
2. Cost does not depend on start time (κ_jm has no time index).
3. Sending a job to an unbounded type at its release slot is always feasible
   (`unbounded_types` looks only at capacities and feasible sets).
4. Data is non-negative (`nonnegative_data`).

| Reduction | Exact? | Notes |
|---|---|---|
| R1 collapse unbounded | yes | also covers on-prem sites whose capacity can never bind |
| R2 Pareto unbounded | yes | already compares per-job (completion, cost) pairs |
| R3 rejection deadlines | yes | cut is computed from per-job `p_occ[j,m]` |
| Lex deadlines | yes | `c_star − p_occ[j,m] + 1` per job |
| normal patterns / compressed flow | yes | closure over each type's distinct `p_occ` values |
| integer bound rows, f1 grid | yes | p is still an integer, costs are still rounded to δ |

Splitting into a stage-in release date and a stage-out tail `q_jm` would
change completion to `t + p + q`. The lex rule (`C = min(Sj) + p_occ` in
`_reduce`), R3's score, `f1_expr` and `bound_constr` would all need editing
and re-proving. Folding transfer in avoids that work.

### Expected effect on solve time (not measured)

- **Columns about the same or fewer.** A job has roughly `H − r_j − p_jm`
  start slots on a capacitated type, so a longer p means fewer variables.
- **Lex and R3 cut more.** Remote sites get slower and sometimes pricier while
  the cloud options stay the same, so the deadlines move earlier. Options that
  can't beat `C*_j` are dropped entirely.
- **More congestion.** Each node is held for `e + τ`, so the instance behaves
  like one at a higher load and capacity binds in more slots. The root LP was
  tight because capacities are in the hundreds, and that argument does not
  involve p, so the root gap should stay near 0. The load levels already
  benchmarked are the best stand-in.
- **Normal patterns get weaker.** More distinct p per type make the pattern
  sets denser. They only ever shrank the model by ≤ 6 %, so the loss is
  negligible.

### What breaks

- **`_ext/weighted_dispatch.hpp`.** It was designed on the premise that every
  on-prem type is equivalent per job, so its genome doesn't choose a pool and
  lets the decoder pick one. It still stores a separate p for each option, so
  it runs correctly. But choosing a site now matters, so expect the gap to grow
  with many sites.
- **`baselines.py`.** Its docstring assumes `p_occ` does not depend on the
  type. The code reads the per-(j, m) tables, but the reasoning behind the
  baselines no longer holds.
- **The `f2_0 = 0` reference point.** It holds only if every job can run at its
  own origin site for free. Otherwise f2_0 must be computed (the heuristics
  already take an `f2_0` argument).

### Modelling cost

A compute node stays busy for the whole staging time, although on a real
cluster dedicated transfer nodes would do that work. So this approach
overstates on-prem occupancy and biases solutions toward cloud. This matters
only when staging time is a meaningful fraction of runtime. At about 8 s/GB
that is the case only for the "F" retention policy (a full restart file at
every report step).

### Suggested experiment

Build a site-aware version of one generated instance: give each on-prem type a
site and each job an origin. Solve it at λ ∈ {0.25, 0.5, 0.75}, then compare
columns, rows, build and solve time and root gap against the current instance.

## 4. Implementation (branch `geo-distributed-io`)

- **Data layer** (`src/fedhpc/data.py`). `Instance.links` (directed `Link(src,
  dst, time, cost)`) switches the site-aware mode on. `InstanceType.site`,
  `Job.origin`, `Job.dest` (default: the origin), `Job.io_in` (stage-in GB;
  `io_volume` stays the stage-out GB) and `Job.sites` (allowed sites, which
  restrict F_j). `build()` fills `io_t[j,m]` / `io_c[j,m]` and folds them into
  `p_bill` / `p_occ` / `c` exactly as §3 proposes. Without links nothing
  changes (same tables, bit for bit). In site-aware mode the per-type
  `io_time` / `cost_io` must be 0, so that no transfer is priced twice.
- **Generator** (`src/fedhpc/geo_gen.py`). Two centres S1/S2 with class-A
  pools each, plus AWS, built from the real workloads. Origins are 65/35. The
  input volume follows the output model's size law. Single runs are pinned to
  their origin. The links come from B_eff = min(read, network, write) (critique
  §3.4). The `wan` factor sets the per-flow WAN speed (0 = co-located control).
  The modes are `fed` / `iso` × cloud `C` / `N`.
- **Baselines** (`src/fedhpc/baselines.py`). `threshold_bursting` now takes the
  earliest-*completing* on-prem option, which is identical when p_occ does not
  depend on the type. The new `earliest_start_federation` is a
  Slurm-federation-inspired policy that ignores transfers.
- **Tests** (`tests/test_geo.py`). Link arithmetic, validation, the
  wait-or-move trade-off, reduction exactness on random site-aware instances
  (reduced vs plain SpaceTime, lex and weighted sum), and dispatch feasibility.
- **Experiment.** Run `scripts/bench_geo.py --set all`, then
  `scripts/geo_report.py`. The results are in `results/geo_bench/REPORT.md`.

The C++ dispatch decoder needs no change. It already scores each on-prem
option per (j, m) when a job starts, so a remote centre is chosen only when it
scores better *now*. What it cannot do is wait for the home centre in
preference to a free remote one: the decoder is non-delay.

## 5. Findings (v2 run, `results/geo_bench/REPORT.md`)

32 instances: 2 workloads × n ∈ {50, 150} nodes per type per centre; WAN set
to co-located / 10 Gb/s / 1 Gb/s; federated or isolated; with or without
cloud. Gurobi runs with MIPGap 0, 6 threads and 3600 s per solve, and with
Presolve=0 in lex phase 2. The raw records are in `bench.jsonl`. The v1 run
used the default phase-2 presolve and a 1800 s limit; it is kept as
`bench_v1_presolve_default.jsonl`.

**Formulation (critique §3.1).** With the reductions the model has 1.05M
columns (fedhpc) and 3.47M (pos); without them it has 1.57M and 5.17M, about
33% fewer. The root LP is integral, to within 2e-4%, for both min f1 and the
λ = 0.5 scalarisation on every instance. Site-dependent transfer therefore
does not weaken the relaxation.

**Transfer effect (§1).** At 1 Gb/s, f1_T moves by at most 0.2% against the
co-located control, and f1_0 and the λ = 0.5 optimum do not move at all on
the certified instances. What the optimizer changes is *where* jobs run:
inter-centre placements drop from 230 to 149 (fedhpc n50) and from 1155 to 948
(pos n150), and the remaining transfers are hidden in staging slots. The
transfer model matters for placement decisions but, at these volumes, not for
the objective values.

**Value of federating, no cloud (§2a).** The federated MILP beats the isolated
centres by 1.6–6.1% in f1, and at 1 Gb/s it reaches the co-located value. On
pos n50 the isolated centres are infeasible within the horizon, while the
federation is feasible. The transfer-blind, Slurm-inspired earliest-start
policy is better than per-centre FCFS but *worse than the isolated optimum*
(fedhpc n50: 20,792 vs 18,528; pos n150: 86,880 vs 63,591, with one job left
unplaced). Plan A4's claim therefore holds: federation pays only when the
placement is optimised.

**With cloud (§2b).** Federation lowers the cost of the fastest schedule
(f2_T) by 7–39%. At equal λ, it gives a better f1 for comparable or lower
spend on fedhpc and pos n150. On pos n50 the isolated weighted-sum points are
much slower (121,800 vs 63,381 at λ = .5), because the isolated centres cannot
absorb the load without the cloud.

**Heuristics (§3, critique §4.2).** GA-mid's mean gap is 3.96% on fedhpc,
consistent with the paper's 4.36%, and 30.5% on pos. On the certified pos
n150 instances it is 4–23%. The dispatch ILS shows similar pos gaps on the
single-site class-A twins (v1 check), so the difficulty comes from pos, not
from the two centres. In deployment mode, where GA-mid estimates its own
reference points and uses no Gurobi, it averages 4.9% on fedhpc and 35.7% on
pos. **Caveat:** GA-mid at λ = 1 without cloud is poor on pos: 139,808 vs a
MILP value of 60,318 on n150, which is worse than the earliest-start
baseline. Do not use it as the no-cloud federation result.

**End-to-end time (§4).** On fedhpc, the Gurobi pipeline (two anchors plus
three weighted sums) and the Gurobi-free pipeline are within a few tens of
seconds of each other (10–83 s vs 13–112 s). On pos n150 the Gurobi-free
pipeline is not faster (278–385 s vs 276–1090 s). On pos n50 the Gurobi
pipeline hits time limits (4.5–7.4k s), while the Gurobi-free one takes
200–870 s, but with gaps of 20–60%.

**Instance checks (§3.6).** Every job has a free option. The cheapest
schedule costs $0 everywhere except `pos_n50 … isoC` (f2_0 = $212): there, the
free capacity cannot finish every job within the horizon, so having a free
option per job is necessary but not sufficient for a free schedule.
`pos_n50 … isoN` is infeasible.

**Caveats.**
- The f1_0 anchors on pos n50 (lex phase 2) and both anchors on pos n50 fedN
  are time-limited incumbents, marked †. Their values agree to 0.01%
  (139,981–140,111) but are not certified, which explains the 5-slot
  "non-monotonicity" of 10 Gb/s against co-located. The ĝ gaps on † instances
  are measured against these anchors. The isoC row, whose f1_0 is 434,515, is
  especially unreliable.
- Input volumes are synthetic, built from the output-volume size law.
- Origins are drawn by hashing each job id (65/35), because the traces carry
  no user or site field.
- Link rates are nominal B_eff values, with no contention between concurrent
  transfers. This is Level 1 of §2.
- The dispatch decoder is non-delay, so it cannot choose to wait for the home
  centre.
