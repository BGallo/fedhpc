# MOSPACE alongside Slurm: how to keep scheduling authority without replacing Slurm

*Analysis, 2026-11-20. Slurm facts checked against the Slurm 26.05 docs
(slurm.conf, scontrol, power_save, job_submit_plugins, site_factor, rest). Items marked
**[verify]** came from a summary of the docs or are inferences. Check them on the target Slurm
version before relying on them.*

## 1. What "authority without substitution" means here

The split follows from what the implementation does (`fedhpc_policy.py`, and the "System
Architecture" section):

| MOSPACE must own (scheduling authority) | Slurm keeps (everything else) |
|---|---|
| **Where:** the pool $\hat m_j$ of every federation job | Submission interface (`sbatch`, `srun`, `salloc`), job IDs, the user-visible queue |
| **When:** the start time $\hat s_j$, including deliberate waiting instead of bursting | Node allocation inside a pool, launch, cgroups, MPI wire-up, signals |
| Cloud or on-prem, which VM type | Accounting (slurmdbd), QOS, limits, fair-share data |
| Re-planning on arrival, deviation, or failed start | Node health, drain/down, maintenance reservations |
| | Cloud node lifecycle: power saving with `State=CLOUD`, `ResumeProgram`/`SuspendProgram` |

The test for any integration option: **can Slurm start a federation job at a time or in a place
MOSPACE did not choose?** If it can, MOSPACE is only advisory.

## 2. Options

### O1. In-controller scheduler plugin (`SchedulerType=sched/mospace`, C)
- **Authority:** full.
- **Problems:**
  - Only `sched/backfill` and `sched/builtin` are documented scheduler types. The internal plugin API is not a stable public interface, so the plugin would break across Slurm versions.
  - The plugin runs inside slurmctld while it holds internal locks. MOSPACE solves take seconds to minutes and cannot block the controller, so the plugin would need its own asynchronous machinery.
  - A crash in the plugin brings down slurmctld.
  - Sites must rebuild Slurm to use it, which most sites will not do.
- **Verdict: reject.** It replaces Slurm's scheduler outright, the opposite of the goal.

### O2. Priority steering (`site_factor` plugin, `scontrol update Priority/Nice`, `scontrol top`)
- **Authority:** ordering only. Slurm still decides *when* (its backfill fills holes) and *where* within the partitions a job may use. A job cannot be held back on purpose to save cloud cost, and there is no cloud-or-wait decision.
- **Verdict:** not enough on its own. It is useful as a **degraded mode**: if MOSPACE is unavailable, its last priorities still shape the native schedule.

### O3. Submit-time routing (`job_submit/lua` chooses the partition or cluster, like `sbatch -M`)
- **Authority:** placement once, at submission. No control of start time and no re-planning.
- **Verdict: reject.** This is the per-job decision model the paper argues against (Slurm multi-cluster).

### O4. Publish the plan as advance reservations (`scontrol create reservation`, job `--reservation`)
- **Authority:** strong, and the plan is visible inside Slurm. Jobs still start on time if MOSPACE pauses.
- **Problems:** the policy re-solves the whole queue on every arrival. That means creating and deleting hundreds to thousands of reservations per hour on W2-like workloads. Reservations are bound to node sets, need admin rights, and clutter the system.
- **Verdict:** use selectively, if at all. One example is reserving capacity only for jobs due in the next slot, or only for large jobs, to close the dispatch race. It is not the main mechanism.

### O5. Holding partition plus an external dispatcher (**recommended**)
- **Mechanism:** federation jobs wait in a Slurm partition that never starts jobs. MOSPACE runs outside slurmctld as a privileged (operator) client. It moves each job to its pool's partition at $\hat s_j$, and Slurm then starts it.
- **Authority:** full for federation jobs. Slurm never sees a startable federation job before MOSPACE releases it, and the pool partitions only receive MOSPACE-dispatched jobs.
- **Substitution:** none. No Slurm code changes; an optional Lua script is the only addition.
- **Verdict: adopt.** The details are in Section 3.

| | O1 plugin | O2 priority | O3 routing | O4 reservations | **O5 holding partition** |
|---|---|---|---|---|---|
| Controls pool | yes | partly | at submit | yes | **yes** |
| Controls start time | yes | no | no | yes | **yes** |
| Deliberate waiting / cloud choice | yes | no | no | yes | **yes** |
| Slurm code change | rebuild | plugin (C) | Lua | none | **none (Lua optional)** |
| Solver on slurmctld critical path | yes | no | yes (Lua under locks) | no | **no** |
| Cost of re-planning | low | low | n/a | **high** | **low** |
| Failure mode | slurmctld down | native schedule | native schedule | jobs start per stale plan | **switch back to native (one command)** |

## 3. Recommended design (O5) in detail

### 3.1 Slurm configuration
```
# Federation jobs wait here. State=DOWN: submissions accepted, queued jobs never started.
PartitionName=fed      Nodes=<all on-prem pool nodes>  State=DOWN  Default=YES
# One partition per pool (identical nodes, no overlap). Only MOSPACE-dispatched jobs.
PartitionName=p_csr1   Nodes=csr1b[01-13]n[01-32]  AllowQos=mospace  State=UP
PartitionName=p_csr3   Nodes=csr3b[01-11]n[01-41]  AllowQos=mospace  State=UP
...
# One partition per cloud VM type, nodes created on demand by power saving.
NodeName=aws-c6i32-[001-400] State=CLOUD CPUs=128 ...
PartitionName=p_aws_c6i32 Nodes=aws-c6i32-[001-400] AllowQos=mospace SuspendTime=600
ResumeProgram=/opt/mospace/cloud_resume  SuspendProgram=/opt/mospace/cloud_suspend
ResumeTimeout=<above the VM boot time; this is the model's pi_m>
```
- **Holding:** `State=DOWN` on `fed` means Slurm accepts the jobs but never allocates nodes to them **[verify exact wording]**.
  - Alternative: an admin hold (`scontrol hold` sets priority to 0; a privileged `release` lifts it). It works, but users see `JobHeldAdmin` as the reason, and switching back to native scheduling then takes one command per job.
- **Pools:** one partition per pool, with no overlapping nodes, matches the model's pool abstraction. The simulator's "virtual partitions" de-duplicate overlapping Slurm partitions; a deployment makes them explicit. Restricting pool partitions to a `mospace` QOS keeps direct user submissions out, which is what makes the authority real.
- **Cloud quota:** the number of `CLOUD` nodes defined per type is the hard quota $N_m$. Spending cannot exceed what MOSPACE plans for even if a bug dispatches too much.

### 3.2 Mapping the implemented components to Slurm

| Component (paper Fig. 1) | Deployment on Slurm |
|---|---|
| **Job queue** | Pending jobs in `fed`, read through slurmrestd. $\hat e_j$ is the walltime `--time` or a predictor's estimate. $w_j$ comes from QOS or priority. $\iota_j$ comes from a user hint (e.g. `--comment`, parsed by `job_submit/lua`) or a default |
| **State monitor** | slurmrestd queries of jobs and nodes, plus end events from a jobcomp plugin (`jobcomp/kafka` or `jobcomp/script`). The same three invalidation rules: arrival, early/late end, failed start |
| **Planner** | Unchanged (MILP or GA), outside slurmctld. A long solve never blocks Slurm, and the previous plan keeps being followed meanwhile |
| **Dispatcher** | At $\hat s_j$: modify the job through slurmrestd (or `scontrol update`) to `Partition=p_<m> QOS=mospace`, optionally with `ReqNodeList`. Slurm's event-triggered scheduler then starts it. Early start works the same way, against the plan |
| **Backlog at takeover** | Jobs pending when MOSPACE starts are released first-come-first-served, as the simulator does (`fedhpc_out_of_scope_jobs = fcfs_first`) |
| **Cloud pool** | Dispatch to `p_<vmtype>` makes Slurm allocate a `CLOUD` node and run `ResumeProgram`. The node boots within `ResumeTimeout`, and idle VMs are suspended after `SuspendTime` |

### 3.3 Why Slurm starts the job when MOSPACE intends
- The main scheduler also runs "every time that any event happens which could enable a job to start (e.g. job submit, job terminate, etc.)". It is throttled by `sched_min_interval`, and also runs every `sched_interval` (default 60 s). A job modification should count as such an event **[verify; measure the dispatch latency]**.
- The target pool partition only contains MOSPACE-dispatched jobs, and the plan left the nodes free, so the job starts immediately.
- If it does not start (for example, a runtime ran over its estimate), the "failed start" rule invalidates the plan. MOSPACE re-plans and can move the job back to `fed`.

### 3.4 Graceful degradation
- A watchdog checks MOSPACE's heartbeat. On failure it runs `scontrol update PartitionName=fed State=UP`.
- `fed` covers the on-prem nodes only, so native Slurm backfill takes over the federation queue on-premises, with no uncontrolled cloud spending.
- Jobs already dispatched or running are unaffected.
- When MOSPACE comes back, `fed` goes back to `DOWN` and MOSPACE rebuilds its state from Slurm. Slurm is the source of truth and slurmrestd is stateless.

### 3.5 Coexisting with non-federation work
- **Option (a):** keep some partitions under native Slurm only. Those nodes are not in any MOSPACE pool.
- **Option (b):** let local jobs share pool nodes. MOSPACE then sees them as background load in $b_m(t)$, but its start times become best-effort. That weakens the authority claim, so the paper should assume (a).

### 3.6 Multi-site
- Each site runs its own slurmctld, and MOSPACE talks to each site's slurmrestd.
- Slurm cannot move a pending job between clusters. Dispatching a job submitted at site A to site B therefore means resubmitting it at B (new job ID, which MOSPACE tracks) and cancelling it at A.
- Slurm federation's sibling jobs might do this natively **[verify: whether siblings can be held and released per cluster]**.
- The evaluated setting (one cluster with four node groups plus cloud nodes) needs only **one slurmctld**. Slurm's power-saving docs show exactly this hybrid layout (`PartitionName=cloud Nodes=ec[0-127]`).

### 3.7 Risks and open checks
1. **Partition access on privileged update:** are a partition's `AllowQos`/`AllowAccounts` limits checked against the job owner when an operator changes the job's partition? **[verify]** If they are, set the QOS in the same update, or use `AllowAccounts=ALL` with a QOS that only MOSPACE assigns.
2. **slurmrestd job-update endpoint:** `slurm_v0043_post_job` exists. Its exact path (likely `POST /slurm/v0.0.43/job/{job_id}`) is an inference; get it from `slurmrestd --generate-openapi-spec`.
3. **Dispatch latency:** time from the partition update to the job starting, under `sched_min_interval` and `defer`. Measure it on a test cluster.
4. **Dependencies and job arrays:** Slurm still enforces `--dependency` after the move. MOSPACE should treat jobs with unmet dependencies as not yet arrived.
5. **`job_submit/lua` runs while slurmctld holds internal locks:** keep it trivial (route to `fed`, parse hints) and never call MOSPACE from it.
6. **Limits:** jobs waiting in `fed` still count against `MaxJobCount` and QOS `MaxSubmit`.

## 4. Consequences for the paper
- **The authority claim stays honest:** MOSPACE decides pool and start time, and Slurm executes. Section III (System Architecture) can add a short paragraph, "Deployment on Slurm", with the table in Section 3.2.
- **Figure:** label the dispatch arrow "move to pool partition at $\hat s_j$", and wrap the pools in a box titled "Slurm (slurmctld + slurmd): launch, accounting, power saving".
- **Optional strong addition:** a small prototype on a containerized Slurm cluster (a few slurmd containers plus fake `CLOUD` nodes whose `ResumeProgram` just sleeps for $\pi_m$). It would report dispatch latency and confirm that the planned start times are met. This would answer the CCGrid question "would this work with a real RJMS?" directly.
