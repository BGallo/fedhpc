# Provider and VM-type mix in the optimum — generated instances

2026-11-03. Question: in the optima of the generated instances, does the
provider (AWS vs Azure) change meaningfully or does one dominate? Same for
VM types. Plus a description of the full compute platform.

**Short answer: neither choice moves much in the optimum.** Azure dominates
class B, and the VM size is never really a decision. The size is always the
smallest VM the job fits on, and the provider changes only for a small set of
high-I/O jobs. Class A offers only AWS, so the question only applies to
class B.

## How it was checked

The benchmark records (`../bench_io_noreduc.jsonl`) don't store which machine
each job ran on. So:

1. **Option check** (`dom.py` → `dom.json`). For every job in all 108
   instances, keep only the cloud options that aren't beaten on both cost and
   occupied slots. Cloud capacity is unlimited, so a bursted job starts at its
   release time and only those two numbers matter.
2. **Re-solves** (`mix.py` → `mix.jsonl`, table: `agg.py` → `mix_table.txt`).
   Gurobi weighted sums (λ = 0.25, 0.5, 0.75) with the stored reference points
   from `bench_io_noreduc.jsonl`, no reductions, as in that run (Threads 4,
   TimeLimit 600 s, MIPGap 0). Covered all 18 non-trivial B·fedhpc instances and
   `B_pos_n300_lE_{iL,iM,iH}`. All 63 solves were optimal. `B_pos_n100_lE_iH`
   was started but stopped before finishing (not included).

Per-job cost gaps between the two providers: `stakes.py`. Platform numbers:
`plat.py`. Scripts are run from the fedhpc root with `uv run python <script>`
(`mix.py` takes the output path as its argument).

## VM type: no real choice

All types run at the same speed (perf 1), and within one provider the transfer
time doesn't depend on size. So for every job, the smallest VM it fits on is
equally fast and strictly cheaper. In every instance, the remaining options for
a job are the same size from AWS and from Azure. The mix of VM types in the
optimum therefore just reflects which jobs burst.

One pattern from the re-solves: in B·fedhpc at loads F and H, no
32-core-sized job ever bursts. The on-prem 32/40/64-core nodes absorb them, so
what goes to the cloud is the 40-core-sized jobs.

## Provider: Azure dominates

- **16, 32 and 40-core equivalents:** the VM prices are identical, and Azure's
  egress is $0.01/GB cheaper. Azure's disk is slightly slower (125 MB/s vs
  125 MiB/s), which adds a little billed time: per GB of output, the cost
  difference Azure − AWS is −0.01 + v·0.000617 $ (v = VM $/slot ≤ 0.768), always
  negative. So Azure is cheaper for every job.
- **64-core equivalent:** Azure's D128s_v6 costs 5 % more than AWS's
  m6i.32xlarge, so AWS usually wins. Azure wins only when a job's output
  exceeds about 5.9 GB per slot of runtime.
- **The only real trade-off:** for a high-output job, AWS's faster disk can
  finish one or more slots sooner, at a higher cost. Jobs with that trade-off,
  per instance:

| io level | fedhpc (964 jobs) | pos (3340 jobs) |
|---|---|---|
| L | 2 | 9 |
| M | 9 | 53 |
| H | 85 | 290 |

If every job ran in the cloud (B··n300_lE), the all-AWS bill would exceed the
all-Azure bill by +0.2–0.3 % at io L, +2.5–3.4 % at io M and +7.4 % at io H.

### What the optima actually picked

- **B·fedhpc:** 16,846 cloud placements over 54 solves, 99.1 % of them on
  Azure.
  - AWS m6i.24xlarge was picked 83 times, only for jobs where it saves a slot.
    That number grows with λ and io: none at λ = 0.25, up to 17–18 at λ = 0.75
    with io H.
  - AWS m6i.32xlarge was picked 1–2 times per solve, for the 64-core-sized
    jobs.
- **B·pos:** 8–42 of 245–1,915 cloud jobs per solve went to AWS. The 26–33
  m6i.8xlarge picks at io L are exact ties: the job has almost no output, so
  cost and time are identical and the solver picked one arbitrarily.
- **AWS share of the cloud bill:** usually under 1 %, at most about 5.6 %. The
  exceptions (39–56 %) are λ = 0.25 solves with only 2–3 cloud jobs, where one
  64-core job carries the bill.
- **Swapping providers:** taking each optimal schedule and moving every cloud
  job to the other provider (same start):

| io level | cost (f2) change |
|---|---|
| L | +0.2 to +0.4 % |
| M | about +1 % |
| H | +2.5 to +5.4 % |

At most 22 slots of total turnaround (f1) are gained, out of tens of
thousands.

**Implication:** for the goal of exercising every part of the formulation,
the provider and VM-type dimension is close to inert. It reduces to 1 cent/GB
of egress and a 5 % gap in disk speed. To make it matter, the providers would
need to trade off: for example, cheaper VMs on one side against costlier
egress, different speeds per type, or provider quotas.

## The full platform

**Time:** 600 s slots. Horizon is 289 slots for fedhpc and 288 for pos.

**Workloads** (unchanged from the source files; all jobs have weight 1):

| | fedhpc | pos |
|---|---|---|
| jobs | 964 | 3,340 |
| arrivals | slots 0 to 10.8 | slots 0 to 24 |
| cores | 20–64 (median 40) | 1–64 (median 30) |
| memory | 150–480 GB | 7.5–480 GB |
| runtime | median 3.75 slots, max 68 | median 5.2 slots, max 268 |
| total output, io L / M / H | 2.6 / 21 / 142 TB | 16 / 147 / 640 TB |

Jobs by smallest fitting size (class B): fedhpc — c32 219, c40 740, c64 5;
pos — c16 1,341, c32 441, c40 1,532, c64 26.

**On-prem:** free, unlimited time, no I/O cost or transfer time.

- Node types: c16 (16 cores, 128 GB), c32 (32 cores, 256 GB), c40 (40 cores,
  300 GB, modelled on csr1) and c64 (64 cores, 480 GB, modelled on csr3).
- Class A has c40 and c64.
- Class B has c32, c40 and c64, plus c16 for pos only (no fedhpc job fits c16).
- Every type has the same node count: 100, 300 or 500.
- Load at t = 0 is free, half full (⌊N/2⌋ running jobs per type) or full
  (N per type). Running-job end times are drawn from each workload's real
  running jobs (median about 22 slots, long tail up to thousands of slots).
  The half-full set is the first half of the full set.

**Cloud:** on demand, unlimited capacity, no startup time, same speed as
on-prem. Each VM is the smallest Ice Lake size whose physical cores (vCPU/2)
and memory cover the on-prem node.

| on-prem | AWS (us-east-1) | Azure (eastus) |
|---|---|---|
| c16 | m6i.8xlarge $1.536/h | D32s_v5 $1.536/h |
| c32 | m6i.16xlarge $3.072/h | D64s_v5 $3.072/h |
| c40 | m6i.24xlarge $4.608/h | D96s_v5 $4.608/h |
| c64 | m6i.32xlarge $6.144/h | D128s_v6 $6.451/h |

| | AWS | Azure |
|---|---|---|
| egress | $0.09/GB | $0.08/GB |
| output transfer time | 0.012716 slot/GB (gp3, 125 MiB/s) | 0.013333 slot/GB (Premium SSD v2, 125 MB/s) |

Class A has only the AWS c40 and c64 sizes. Class B has both providers:
6 cloud types for fedhpc and 8 for pos.

**Cost and time of a cloud job:**

- Billed time = runtime + transfer time × output volume.
- Occupied slots = billed time rounded up (at least 1).
- Cost = egress price × output volume + VM price per slot × billed time.

**Output volume per job** (`fedhpc.io_model`): jobs in an ensemble (≥ 10 runs
sharing a user and job-name stem) produce 0.02 GB/core-hour at io L, 0.13 at
io M and 1.5 at io H. Single runs produce 0.13 at io L and 1.5 at io M and H.
Each job also gets a lognormal spread.

## Files

| file | content |
|---|---|
| `dom.py`, `dom.json` | per-job non-dominated cloud options, all 108 instances |
| `mix.py`, `mix.jsonl` | weighted-sum re-solves with provider/type counts, cost per type and provider-swap deltas |
| `agg.py`, `mix_table.txt` | per-solve table and totals |
| `stakes.py` | AWS − Azure cost gap per job, by size and io level |
| `plat.py` | workload, running-job and io-volume statistics |
