# Data transfer between geographically distributed on-prem clusters

Design note (2026-11-19). Nothing here is implemented or benchmarked yet.

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
