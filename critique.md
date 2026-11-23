# **Peer review of MOSPACE**

Manuscript reviewed: *MOSPACE: Multi-objective Scheduler of Parallel Jobs on a Federated Computing Environment* — November 20, 2026, 11 pages

## **1. Overall assessment**

**Recommendation: Major revision, with a positive assessment of the research direction.**

MOSPACE addresses a legitimate scheduling problem at the intersection of mathematical optimization and federated HPC: assigning independently submitted jobs to heterogeneous execution resources while balancing monetary expenditure against priority-weighted turnaround time.

The paper has three substantial elements: an exact arc-flow MILP, a specialized memetic genetic algorithm, and an experimental evaluation that combines offline optimization benchmarks with trace-driven simulations.

The latest draft is stronger than the earlier versions. The addition of Slurm federation as a baseline, a structured factorial experiment, and separate evaluations with and without cloud offloading materially improves the research.

My central judgment is that the mathematical formulation is sound for the problem it actually defines. The main threats to the paper are now the correspondence between the model and the simulated environment, the validity of some GA guarantees, and the strength of the experimental evidence.

I would not recommend redesigning the core MILP or abandoning the GA. I would recommend narrowing some claims, making several assumptions explicit, correcting the comparisons where necessary, and adding a small number of carefully chosen experiments.

### **Assessment by component**

| Component                        | Assessment                                   | Priority |
| -------------------------------- | -------------------------------------------- | -------- |
| Problem motivation               | Convincing, but scope needs precision        | Medium   |
| Related work                     | Relevant, insufficiently synthesized         | Medium   |
| Arc-flow MILP                    | Sound under stated assumptions               | Low      |
| Objective normalization          | Mathematically sound, with restrictions      | Medium   |
| Resource and cost modeling       | Simplifications need validation              | High     |
| Genetic algorithm                | Credible architecture; unresolved guarantees | High     |
| Offline experiments              | Good factorial structure                     | Medium   |
| Online simulation                | Useful, but methodological gaps              | High     |
| Baseline fairness                | Not yet established                          | High     |
| Results and statistical analysis | Promising, incomplete                        | High     |
| Presentation                     | Clearly unfinished                           | Low      |

The three findings I would address before anything else are:

1. The claim that the GA is substantially faster than the exact method needs to account for how its normalization constants are obtained and whether its reported solutions satisfy the same constraints.
2. The comparison with Slurm-based approaches must distinguish the quality of the scheduling policy from the advantage of having access to different amounts of cloud capacity.
3. The numerical claims in the abstract and results disagree with the newest experimental tables.

The third issue is already demonstrable. Table V reports a mean GA gap of 4.36% and mean solver times of 91.1 seconds for Gurobi and 5.6 seconds for the GA. The abstract instead reports 5.66% and 110 seconds, while Section VI states 71 seconds for Gurobi. These values should be reconciled before submission.&#x20;

MOFE_FEDHPC_CCGRID(1).pdf

MOFE_FEDHPC_CCGRID(1).pdf



I distinguish below between a mathematical error, an unsupported assertion, a restrictive but reasonable modeling assumption, and an incomplete explanation. These require different remedies.

## **2. Problem definition and scientific contribution**

### **2.1. The research question should be stated more precisely**

Severity: Medium | Introduction, pages 1–2

The introduction establishes why federated HPC and cloud bursting matter. It is less precise about the specific research question MOSPACE answers.

There are at least two distinct questions in the manuscript. The first is whether heterogeneous resource placement and scheduling can be optimized jointly using an exact mathematical model. The second is whether this joint optimization improves practical scheduling outcomes relative to independent or hierarchical scheduling policies.

The current text sometimes treats these as one contribution. They require different evidence: mathematical and computational validation for the first, and controlled simulation comparisons for the second.

I recommend introducing an explicit research question:

> Can joint optimization of job placement and start times across fixed-capacity HPC resources and elastic cloud instances reduce priority-weighted turnaround and external execution expenditure relative to independent resource-selection and scheduling policies, while remaining computationally practical for online scheduling?

This is narrower than a general claim about solving federated HPC scheduling, but it is testable and matches the actual research.

The hypotheses could then be expressed in terms of the quality and runtime of the exact and heuristic solvers and the trade-offs observed in simulated operation. The results should make clear which hypotheses are supported, rather than treating the existence of an optimizer as evidence of better operational performance.

### **2.2. The evaluated environment is primarily hybrid HPC–cloud, not general multi-center federation**

Severity: Medium | Introduction and Section V

The mathematical model allows multiple on-premises instance types. It can also represent several geographical sites if a type is associated with a specific site.

The experimental setup, however, is based on one production HPC cluster whose hardware is divided into homogeneous node groups, plus cloud instance types. Class B adds multiple cloud providers, but it does not establish distinct independently operated on-premises HPC centers.&#x20;

MOFE_FEDHPC_CCGRID(1).pdf



This distinction matters because geographically distributed HPC centers introduce several effects that are not examined here, including different admission policies, site-specific data locations, and potentially incompatible software environments.

The experiments support claims about hybrid HPC–cloud scheduling. They do not yet establish performance across independently administered HPC centers.

There are two good ways forward.

The less expensive option is to describe the current evaluation as a hybrid HPC–cloud case study and keep the broader federated setting as the intended generalization of the formulation.

The stronger experimental option is to add a synthetic two-center scenario, with separate on-premises capacities, distinct job compatibility sets, and site-dependent transfer overheads.

I would not require a real deployment at two centers for this paper. A controlled synthetic experiment would be sufficient to show that the model handles the intended extension.

### **2.3. The meaning of a parallel job is important**

Severity: Medium | Section III-A

The formulation explicitly assumes that each job occupies one entire instance. The experimental workload description also reports zero multi-node jobs.&#x20;

MOFE_FEDHPC_CCGRID(1).pdf

MOFE_FEDHPC_CCGRID(1).pdf



This is a legitimate HPC scheduling model. A job can execute with multiple cores or processes on a single node, and it remains a parallel job.

However, it is not a model of general multi-node MPI jobs or simultaneous resource allocation across multiple independent nodes.

The distinction is not merely terminological. A multi-node gang-scheduled job would require a different resource-consumption representation.

I recommend explicitly calling the workload *single-node parallel jobs with exclusive instance allocation*. This can be stated once in the introduction and then treated as a fixed modeling assumption.

There is no need to extend the current MILP to general gang scheduling simply to satisfy a broader interpretation of the title.

## **3. Mathematical formulation**

### **3.1. The arc-flow formulation is correct for the stated scheduling problem**

Assessment: Sound | Equations (7)–(13)

I find no structural error in the flow conservation formulation.

Each instance type has a time-expanded network. The assignment constraints require exactly one processing arc per job. Flow conservation, together with \\(N_m\\) units of available flow, enforces exclusive instance occupancy throughout each job's execution interval.

The formulation represents machine idleness explicitly through waiting arcs. This is important because jobs have release dates, and intentional idleness can be optimal for weighted turnaround minimization.

The boundary conditions are also consistent: the network begins and ends with \\(N_m\\) units of flow.

The construction therefore avoids the usual pairwise disjunctive constraints needed in many machine-scheduling formulations.

However, I would distinguish validity from LP-relaxation strength.

The fact that each network is a flow network does not imply that the complete scheduling formulation has an integral LP relaxation. The job-assignment constraints couple the individual networks.

The manuscript calls the formulation *strong*, but Table V only reports integer optimization outcomes.

A useful addition would be a small computational analysis of the root relaxation:

\\[ \operatorname{Gap}\_{\mathrm{LP}} = \frac{z\_{\mathrm{IP}}-z\_{\mathrm{LP}}} {\max\\{1,|z\_{\mathrm{IP}}|\\}}. \\]

The denominator is one possible reporting convention; specify it explicitly.

Report the distribution of LP gaps, model sizes, and branch-and-bound nodes. If the relaxation is integral or nearly integral for the tested instances, that is an interesting result in its own right.

Even better, compare against a conventional time-indexed formulation on a subset of instances. This would support the argument that the arc-flow representation has computational value beyond being a valid MILP.

### **3.2. The time horizon and discretization are insufficiently specified**

Severity: High | Equations (2)–(6), Section V

The formulation requires a finite horizon \\(H\\) and a discretization interval \\(\Delta\\). These parameters are fundamental to both correctness relative to the intended continuous-time problem and computational complexity.

Yet the experimental section does not clearly specify how they are chosen.

In particular, Workload 2 contains jobs with execution times approaching 496 hours. A fine-grained time-expanded network over such a horizon can become extremely large. The reported solver times cannot be interpreted adequately without knowing the time resolution and the number of generated arcs.

I recommend reporting the physical duration of the horizon, the slot length, how raw submission times are mapped onto slots, and the procedure used to guarantee schedule feasibility.

It would also be useful to examine at least two discretization resolutions on representative instances. The purpose is not to exhaustively optimize \\(\Delta\\), but to establish whether the results are sensitive to rounding.

The paper should distinguish the optimization error from the continuous-time problem from the optimality gap of the discretized MILP. Gurobi can prove optimality for the latter without proving optimality for the former.

### **3.3. The cost model is reasonable as an incremental-cost model, but not a total-cost model**

Severity: Medium | Section III-B

The manuscript assigns zero monetary costs to all on-premises instance types.

This is economically defensible when the infrastructure is already acquired and the decision concerns incremental external expenditure. The marginal cost of an additional local job can be approximated as zero if energy and operational differences are excluded.

But the paper sometimes describes the objective as minimizing *total monetary cost*. That is stronger than the implemented model.

A more precise term is *external execution expenditure* or *incremental cloud expenditure*.

I recommend stating:

> On-premises infrastructure costs are treated as sunk costs. The monetary objective therefore measures incremental expenditure associated with external execution resources, rather than the total cost of ownership of the federation.

This preserves the validity of the analysis without requiring estimates of depreciation, electricity prices, or administrative costs.

### **3.4. The data-transfer model is the weakest physical assumption**

Severity: High | Equations (2), (5), Section V-A

The mathematical model uses one I/O volume \\(\iota_j\\), a machine-specific transfer-time coefficient \\(\tau_m^{\mathrm{IO}}\\), and a corresponding monetary rate \\(\kappa_m^{\mathrm{IO}}\\).

That is a useful first-order approximation, but it merges quantities that are economically and physically different.

For example, transferring input data to a cloud VM and returning output data from it do not generally have identical monetary charges. The model also does not represent where the data is originally stored.

I recommend separating input and output volumes:

\\[ d_j^{\mathrm{in}},\qquad d_j^{\mathrm{out}}. \\]

A slightly richer model would use

\\[ p\_{jm}^{\mathrm{raw}} = \frac{e_j^{\mathrm{raw}}}{P_m} + \tau_m^{\mathrm{in}}d_j^{\mathrm{in}} + \tau_m^{\mathrm{out}}d_j^{\mathrm{out}}, \\]

and a data-transfer cost

\\[ c\_{jm}^{\mathrm{IO}} = \kappa_m^{\mathrm{in}}d_j^{\mathrm{in}} + \kappa_m^{\mathrm{out}}d_j^{\mathrm{out}}. \\]

This retains linearity in the assignment formulation because all quantities remain precomputed for each job–machine pair.

It is not necessary to introduce a detailed network-flow model unless network contention is itself a research objective.

A more consequential issue appears in the offline-instance setup. The manuscript derives transfer times from AWS gp3 disk throughput and Azure Premium SSD v2 disk throughput.

The cited 125 MiB/s for AWS and 125 MB/s for Azure are baseline disk-throughput figures, not measurements of inter-site network bandwidth. Their official documentation confirms this distinction.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://docs.aws.amazon.com\&sz=32)

Amazon EBS+1



Disk I/O can be a bottleneck for a transfer pipeline, but it cannot be assumed to be the bottleneck without specifying the storage path and effective network throughput.

The physical model should use an effective transfer rate constrained by the slowest relevant component, or use an explicitly assumed transfer bandwidth and test sensitivity to it.

For example,

\\[ B\_{sm}^{\mathrm{eff}} = \min\left\\{ B_s^{\mathrm{read}}, B\_{sm}^{\mathrm{network}}, B_m^{\mathrm{write}} \right\\}, \\]

where \\(s\\) is the source site. For return transfers, the corresponding read and write directions are reversed.

This would turn a weakly grounded assumption into a transparent and testable one.

### **3.5. Provisioning and cloud performance are modeled but not meaningfully exercised**

Severity: Medium | Sections III-B and V-A

The formulation includes performance factors \\(P_m\\) and provisioning delays \\(\pi_m\\), but the offline setup sets

\\[ P_m=1,\qquad \pi_m=0 \\]

for all instance types.&#x20;

MOFE_FEDHPC_CCGRID(1).pdf



Consequently, the experiments do not demonstrate how the scheduler behaves when one platform executes a job faster than another or when deployment delay influences the offloading decision.

The optimization model supports these effects, but the empirical validation does not yet test them.

I would add a small sensitivity experiment with heterogeneous performance factors and provisioning delays. This need not become another full factorial experiment.

The main questions are whether the selected platform changes as deployment delay increases and whether differences in execution speed alter the cost–turnaround trade-off.

A separate issue is the billing convention. The model rounds billable duration to multiples of \\(\Delta\\), while Linux EC2 On-Demand instances are generally billed by the second with a 60-second minimum.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://aws.amazon.com\&sz=32)

Amazon Web Services, Inc.+1



If \\(\Delta\\) is larger than the actual billing increment, the model introduces artificial cost rounding. I recommend separating the scheduling time step from the provider's billing granularity.

### **3.6. The multi-objective normalization is correct, but its scope needs clarification**

Severity: Medium | Equations (16)–(20)

The lexicographic reference and zero-cost turnaround reference are mathematically well chosen under the assumption that a zero-cost feasible schedule exists.

The restriction

\\[ f_1(x)\leq f_1^0 \\]

correctly eliminates the positive-cost region dominated by the minimum-turnaround zero-cost schedule.

The limitation is that this construction depends on zero-cost feasibility and does not automatically extend to instances with jobs that can only run in the cloud.

The degenerate case where the denominators vanish is recognized in the text, which is good.

I recommend making the assumption explicit at the beginning of the normalization subsection and verifying it for every experimental instance.

A broader extension would replace the zero-cost reference with a minimum-cost lexicographic solution. I do not think that generalization is necessary for this paper if the tested instance family satisfies the current assumptions.

## **4. Genetic algorithm**

### **4.1. The fitness function is now well specified, but its penalties do not provide the claimed guarantees**

Severity: High | Equations (21)–(26)

The revised fitness construction is a substantial improvement. The derivation

\\[ Z(S)=\alpha_1f_1(S)+\alpha_2f_2(S) \\]

correctly drops a constant from the normalized MILP objective, and the individual score \\(\sigma_j(m,t)\\) is consistent with this expression.

The problem is the interpretation of the penalties.

The manuscript states that an infeasibility penalty of \\(10^9+10^6v(S)\\) guarantees that every feasible solution is preferred to every infeasible solution. That guarantee requires an upper bound on the remaining fitness terms. No such bound is provided.

The turnaround-cap penalty also has a specific edge case:

\\[ P\_{\mathrm{cap}}(S) = 10^3\alpha_1\max\\{0,f_1(S)-f_1^0\\}. \\]

When \\(\lambda=0\\), \\(\alpha_1=0\\), so the cap penalty vanishes.

I recommend a lexicographic fitness comparison rather than arbitrary penalty magnitudes. First minimize the number of unplaced jobs. Among complete schedules, first prefer schedules satisfying the cap. Among schedules satisfying both requirements, minimize the scalarized objective.

That is mathematically more transparent and avoids numerical scaling problems. If additive penalties are retained, their claimed dominance should be justified through instance-specific objective bounds.

The paper also needs to define how \\(Z(S)\\) is computed for a partial schedule. The per-job score is only defined for an assigned job.

### **4.2. The normalization constants create a dependency on exact optimization**

Severity: High | Equation (22)

The GA requires the reference values \\(f_1^0\\), \\(f_1^{\mathrm{lex}}\\), and \\(f_2^{\mathrm{lex}}\\).

These are solutions to auxiliary optimization problems. If Gurobi is used to obtain them for each instance, the GA runtime reported in Table V excludes a potentially substantial computational expense.

This directly affects the stated motivation of using the GA as a faster alternative to the commercial exact solver.

I recommend distinguishing two modes of operation.

In the benchmark mode, exact reference values can be provided to both methods to ensure that their objectives are directly comparable.

In the deployment mode, the GA should use independently available normalization estimates, cached reference values, or analytical bounds. The resulting scalarization should be reported with its actual coefficients.

This distinction would preserve a fair optimization benchmark without overstating the GA's independence from Gurobi.

### **4.3. The indirect representation is appropriate**

Assessment: Positive | Section IV-A.3

The chromosome

\\[ \chi=(\chi_1,\ldots,\chi_n;\pi,\delta,\eta) \\]

is a sensible design.

A gene does not directly encode a particular execution instance or start time. Instead, it constrains offloading behavior, while the strategy genes control the construction of a complete schedule.

This reduces the search-space representation and allows one gene modification to produce substantial changes in scheduling decisions through decoding.

The mechanism is defensible, but the text should call it an *indirect policy-based representation*, rather than saying that every chromosome fixes the platform decision of every job. Auto genes do not fix that decision.

Another useful clarification concerns the heuristic seeds. Mapping a preconstructed schedule to cloud/auto genes discards its original machine assignments and start times. The decoder may therefore produce a different schedule. These are heuristic-derived chromosomes, not necessarily preserved heuristic schedules.

### **4.4. The two decoders are justified but incompletely specified**

Severity: Medium | Section IV-A.5

The use of both chronological dispatch and serial schedule generation is a good decision.

A time-driven non-delay decoder tends to avoid unnecessary idleness. A serial decoder can prioritize jobs with later release times and intentionally reserve earlier capacity, which can improve weighted turnaround.

The difference is mathematically important rather than merely an implementation detail.

However, the serial decoder is described too briefly. The manuscript does not establish precisely how its offload tests interact with its earliest feasible placement decisions.

I recommend specifying a shared placement procedure with clearly defined inputs and outputs. The procedure should determine the best feasible on-premises assignment, compare it with the cloud alternative when permitted by the chromosome, enforce the horizon, and either commit an assignment or report failure.

The text currently says that both decoders return complete feasible schedules, despite also defining unplaced jobs. The statement should instead say that they return a complete schedule when all jobs are placed, and otherwise return a partial schedule with an explicit infeasibility count.

The assumption that cloud capacity is nonbinding is now stated, which addresses an earlier concern. Still, it applies to every decoded assignment only if the implementation permits sufficient cloud instances for all simultaneously offloaded jobs.

### **4.5. The scheduling-theory interpretation should be more rigorous**

Severity: Medium | Sections IV-A.4–6

The priority rule derived from Smith's weighted-shortest-processing-time index is a reasonable heuristic.

However, Smith's optimality theorem does not extend directly to unrelated parallel machines with release dates. Describe the rule as WSPT-inspired.

The indifference deadline is similarly a heuristic estimate. It compares individual local and cloud scores, but does not account for the downstream effect of occupying scarce local capacity.

The local search is also described as best improvement, even though it evaluates only a sampled shortlist of the flip neighborhood.

A more accurate name is *sampled best-improvement local search*. A chromosome at which the procedure stops is not necessarily a local optimum of the full neighborhood.

One mathematical statement needs direct correction: fixing the offloaded jobs does not reduce exact on-premises scheduling to list scheduling. It leaves a combinatorial scheduling problem, which your decoders approximate.

None of these issues requires changing the algorithm. They require more precise statements about what the heuristics are guaranteed to do.

### **4.6. The GA's design choices need ablation evidence**

Severity: Medium | Sections IV-A.7–9

The GA contains several specialized mechanisms: heuristic-derived seeds, intensively refined seeds, multi-step crossover fusion, niche elitism, and iterated local search.

These are defensible, but the current experiments only evaluate the full algorithm.

Ablation experiments should determine which mechanisms materially contribute to the result.

I would prioritize comparison of the full algorithm against versions without MSXF, without the converged seeds, and without niche elitism. I would also compare the dispatch-only and serial-only decoders.

The niche-elitism policy particularly deserves examination. With a population of 16 and 11 priority rules, preserving one individual from each rule can leave relatively few positions for new offspring. Whether that diversity mechanism improves convergence is an empirical question.

The current three random seeds per instance are useful preliminary evidence, but they provide limited information about run-to-run stability. At least a representative subset of difficult instances should receive more repetitions, with confidence intervals or dispersion statistics.

Finally, Algorithm 1 should be synchronized with the prose. It still uses \\(G\\) instead of \\(\Phi\\), applies the wrong offspring local-search budget, and omits niche elitism and the converged initial seeds.

## **5. Experimental design**

The experimental section has improved considerably. The factorial benchmark and the new Slurm comparisons are valuable additions.

This is also where I see the largest remaining threats to the validity of the conclusions.

### **5.1. The offline factorial experiment is well structured, but some factors are confounded**

Severity: High | Section V-A, Table IV

The experiment uses 108 instances constructed from two machine classes, two workloads, three node-count levels, three initial-load levels, and three I/O levels.

This is a reasonable design. It permits analysis of how algorithm performance changes with capacity, congestion, and data-transfer requirements.

However, Class A and Class B differ along two dimensions simultaneously.

Class A has two on-premises instance types with one cloud equivalent each. Class B has four on-premises types with both AWS and Azure equivalents.

Because \\(N\\) is defined as the number of nodes of every on-premises type, Class B also has twice as many local nodes at each value of \\(N\\), assuming all types have positive capacity.&#x20;

MOFE_FEDHPC_CCGRID(1).pdf



Consequently, differences between the two classes cannot be attributed specifically to multi-cloud provider choice or increased platform heterogeneity. They also reflect the difference in total local capacity.

This is a confounding variable.

I recommend a paired comparison where the same on-premises resource pool is evaluated first with AWS alone and then with AWS and Azure. A separate experiment can vary the number and diversity of local instance types.

That would allow you to distinguish the benefit of an additional cloud provider from the benefit of additional local computational resources.

The current factorial experiment can remain. It simply should not be interpreted as a controlled isolation of those effects.

### **5.2. The initial-load factor requires a temporal specification**

Severity: High | Table IV

The current load factor specifies whether zero, half, or all local nodes are occupied at \\(t=0\\).

But initial occupancy is not sufficient to characterize scheduling congestion.

Consider two instances where every machine is occupied initially. In one, all running jobs finish after one time step. In the other, they remain occupied for several hours.

These systems have identical initial utilization and radically different scheduling opportunities.

The paper does not adequately specify how residual execution times are assigned to the initial jobs in the offline experiments.

I recommend describing the initial occupation process through a time-dependent capacity function,

\\[ b_m(t)= \text{number of instances of type }m \text{ reserved by previously committed jobs at }t. \\]

Then the available capacity is

\\[ N_m^{\mathrm{avail}}(t)=N_m-b_m(t). \\]

Specify whether residual execution times are sampled from the production trace, derived from a snapshot, or generated synthetically.

This matters for reproducibility and for the interpretation of the load factor.

### **5.3. The difference between offline and online information must be explicit**

Severity: High | Sections III and V

This is one of the most important experimental questions.

The offline model has access to each job's release time, resource requirements, and processing duration. In trace-driven simulation, actual processing durations are available retrospectively from the job log.

A real online scheduler does not generally know the exact execution duration of a newly submitted job. It has a requested walltime, a prediction, or another estimate.

The manuscript does not clearly establish which information MOSPACE and the Slurm baselines receive.

If MOSPACE uses actual future processing durations while the Slurm implementations use requested walltimes, MOSPACE has an informational advantage. If the optimizer also receives jobs before their submission times, the advantage is even larger.

I recommend defining two evaluation regimes.

The first is an oracle benchmark, where exact job durations and the full job set are available. This measures the quality of the scheduling methods under perfect information.

The second is an operational benchmark, where only submitted jobs are visible and processing durations are estimated from information available at decision time.

Both are valid experiments, but they answer different questions.

An especially useful sensitivity experiment would introduce controlled runtime-estimation errors. This would establish whether the optimization remains useful when the execution-time information is imperfect.

### **5.4. Scheduler computation time can affect the validity of online claims**

Severity: High | Sections V–VI

Table V reports an average GA runtime of 5.6 seconds and an average Gurobi runtime of 91.1 seconds.

Workload 1 contains 964 submissions over 1 hour and 48 minutes, corresponding to approximately one new job every 6.7 seconds on average.

Workload 2 contains 3,349 submissions over four hours, or approximately one job every 4.3 seconds on average.

If MOSPACE recomputes a complete schedule after every arrival, its computation latency becomes relevant to whether it can keep up with the submission process.

The paper does not say enough about the frequency of optimization calls, whether scheduling computations consume simulation time, or how jobs are treated while a scheduling decision is pending.

I recommend reporting the distribution of scheduler decision times in the online simulations, including the 95th percentile and maximum.

Explain whether the scheduler is invoked on every arrival, periodically, or after a batch of events. Also state whether a solution can be interrupted and whether the incumbent is used when a scheduling deadline expires.

If optimization is instantaneous from Batsim's perspective, state that the simulation neglects decision latency. That is acceptable for an initial study, but operational conclusions must be qualified.

## **6. Baseline correctness and fairness**

### **6.1. The Slurm-based comparisons require implementation details**

Severity: High | Tables VI–VII

The addition of FCFS, EASY, Multi-cluster Slurm, and Federated Slurm is valuable. These baselines make the paper much more relevant to actual HPC scheduling practice.

However, Multi-cluster Slurm and Federated Slurm are operational mechanisms, not single, completely specified scheduling policies.

The official documentation distinguishes multi-cluster job submission from federated scheduling, in which sibling jobs are replicated and the participating clusters independently attempt scheduling with coordination.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://slurm.schedmd.com\&sz=32)

Multi-Cluster Operation+1



The manuscript needs to explain exactly how these mechanisms were implemented in Batsim.

In particular, which local scheduling algorithm is used? How are cluster eligibility and estimated start times computed? How does federation-level coordination select the winning execution? How are cloud instances represented?

There is also an important nomenclature distinction.

If the simulations reproduce Slurm's relevant allocation semantics through an independent EDC implementation, I would call them *Slurm multi-cluster-inspired* and *Slurm federation-inspired* policies unless their behavioral equivalence has been validated.

If actual Slurm behavior is reproduced through a faithful simulation or integration, document the scheduler configuration and validation procedure.

The distinction protects the paper from an objection that the results compare MOSPACE against simplified substitutes rather than against the actual Slurm mechanisms.

### **6.2. The cloud-enabled baseline comparison does not use equal resource constraints**

Severity: High | Table VII

Table VII compares MOSPACE, whose cloud capacity is modeled as unlimited, with Slurm-based strategies whose cloud capacity is varied between 50 and 400 nodes.

This is not a controlled comparison of scheduling policy quality because the policies operate with different capacity constraints.

The comparison remains informative: it shows the outcomes of several practical provisioning configurations. But it does not isolate the benefit of the optimizer.

I recommend adding a comparison under equal cloud-capacity constraints, or at least reporting how much concurrent cloud capacity MOSPACE actually uses.

A cost-matched comparison would also be valuable. For each baseline configuration, identify the MOSPACE solution with comparable monetary expenditure and compare the corresponding turnaround.

The present results are still interesting. For example, in Workload 2, MOSPACE at the lowest reported preference weight has lower turnaround and lower expenditure than the 400-node federated Slurm configuration.

That is a meaningful empirical observation. It becomes much stronger if the resource and information assumptions are made equivalent.

### **6.3. A relevant cloud-bursting baseline is missing from the latest reported tables**

Severity: Medium

The related work section discusses Ambati et al.'s waiting policies and Starburst, and the abstract promises comparison with a state-of-the-art cloud-bursting strategy.

However, the newest Tables VI and VII report Slurm-based baselines. They do not present the earlier Wise-Burst or threshold results in a corresponding comparison table.

Section VI still discusses a threshold-based strategy with a reported gap of 257%, but the evidence for that figure is not present in Table V.

I recommend either restoring a clearly specified cloud-bursting baseline or removing unsupported comparison claims from the abstract and results.

A threshold waiting policy is particularly appropriate because it is conceptually close to the offloading decision that MOSPACE optimizes.

The comparison should use the same job durations, prices, resource limits, and information availability.

## **7. Results: what the data support**

### **7.1. The exact-versus-heuristic results are useful, but the mean hides difficult cases**

Severity: Medium | Table V

The latest table shows the GA achieving an overall average gap of 4.36%, with a mean time of 5.6 seconds, compared with 91.1 seconds for Gurobi.

This is encouraging evidence that the heuristic obtains reasonably close solutions much faster on the tested instances.

However, the difficult instances have substantially larger gaps.

GA mean optimality gap by workload, class and node count

Values reported in Table V of the manuscript. Lower is better.

0%5%10%15%20%A · W1 · 100A · W1 · 300A · W1 · 500A · W2 · 100A · W2 · 300A · W2 · 500B · W1 · 100B · W1 · 300B · W1 · 500B · W2 · 100B · W2 · 300B · W2 · 500

The gaps reach 16.59% and 17.34% for the most constrained W2 configurations with \\(N=100\\).

This pattern is scientifically interesting. It suggests that the GA's main difficulty is scheduling under severe resource contention rather than simply the number of jobs.

I would investigate whether the difficulty comes from chromosome expressiveness, decoder limitations, or insufficient local-search effort.

A runtime-versus-quality curve on these difficult cases would be more informative than adding many easy instances.

The table's definition of best-known solution also needs clarification. Gurobi obtains the best reported solution in all 324 scalarized cases, but that fact alone does not certify optimality. Report the actual solver termination status and certified MIP gap.

For the GA, specify whether the reported gap is the average over three runs or the gap of the best run. Clarify the meaning of its 103 best-known-solution matches.

### **7.2. The no-cloud results show a meaningful mean-versus-tail trade-off**

Severity: High | Table VI

This is one of the most important observations in the new manuscript, and it deserves explicit discussion.

Without cloud offloading, MOSPACE reduces average turnaround relative to EASY on both production workloads. However, it also increases the upper tail of waiting time.

Workload 1

# **−8.0%**

Mean turnaround vs EASY

## **+35.7%**

95th-percentile waiting time

Workload 2

# **−7.8%**

Mean turnaround vs EASY

## **+22.2%**

95th-percentile waiting time

Maximum waiting time also increases in both workloads: from 1.60 to 2.44 hours for W1 and from 4.39 to 6.15 hours for W2.

This is not a contradiction. Minimizing total turnaround does not guarantee that every job's waiting time improves.

The paper should acknowledge this distributional trade-off.

An appropriate interpretation would be:

> MOSPACE reduces mean turnaround and bounded slowdown relative to EASY, but its stronger emphasis on aggregate completion-time performance increases upper-tail waiting times in the evaluated workloads. This result indicates that improvements in system-level efficiency do not necessarily translate into uniform improvements across individual jobs.

A practical extension would be to impose maximum-wait constraints or introduce aging in the online priority weights.

I would not require the paper to solve fairness optimization in addition to its present objectives. But the adverse tail behavior should be reported prominently rather than hidden behind improvements in the mean.

### **7.3. The cloud-enabled results are promising but do not establish a complete Pareto frontier**

Severity: Medium | Table VII

With cloud offloading, MOSPACE reaches very low waiting times. The resulting total turnaround approaches the sum of execution durations.

This is consistent with a model in which cloud capacity is effectively unlimited and offloaded jobs start without waiting.

The interesting question is therefore not whether sufficiently expensive cloud bursting can eliminate queues. It is how much expenditure MOSPACE requires to achieve a given reduction in turnaround.

Table VII contains useful data for that question, but its presentation does not make the trade-off sufficiently clear.

I recommend a cost-versus-turnaround scatter plot with one point per policy configuration. Use consistent workload-specific axes and distinguish the algorithms through marker types.

For the exact method, include solutions across more than three preference weights if feasible.

The manuscript already acknowledges that weighted sums recover only supported Pareto-optimal solutions. A small \\(\varepsilon\\)-constraint experiment would be useful if you intend to characterize the Pareto frontier more completely.

It is not necessary if the goal is only to compare several practically relevant trade-offs. In that case, describe them as sampled operating points rather than the complete frontier.

A notation correction is also necessary: Table VII varies \\(\alpha=0.25,0.5,0.75\\), but Section III defines the preference parameter as \\(\lambda\\), while \\(\alpha_1\\) and \\(\alpha_2\\) are derived coefficients. Use the same parameter consistently.

### **7.4. The MIT Supercloud experiment lacks methodological description**

Severity: Medium | Table VI

Table VI includes a third workload labeled MIT Supercloud with 917 jobs.

That is a valuable addition because it expands the evaluation beyond the two reservoir-simulation workloads.

However, the setup section describes W1 and W2 but does not provide equivalent information about the MIT Supercloud trace.

I would expect a short paragraph specifying its provenance, submission window, execution-time distribution, resource configuration, initial system state, and any filtering or transformation applied to the jobs.

If the workload originates from a publicly available trace, cite its source and describe the transformations necessary to reproduce the experiment.

The result itself is worth retaining. MOSPACE's mean turnaround is substantially lower than EASY's on this workload, while its maximum wait increases. This reinforces the need to discuss the mean-versus-tail trade-off.

## **8. Related work and novelty**

### **8.1. The coverage is relevant, but the discussion is too descriptive**

Severity: Medium | Section II

The selected literature is relevant. Kramer et al. and Chen et al. establish mathematical connections to machine scheduling. Brotcorne et al. provides a multi-cloud optimization comparison. Ambati et al. and Starburst provide operational cloud-bursting comparisons. Wang et al. provides a related federated scheduling architecture.

The new Slurm discussion is appropriate and improves the section.

The problem is that the section spends considerable space describing individual papers while giving comparatively little attention to the precise mathematical differences between their problem assumptions and yours.

I recommend organizing the synthesis around the dimensions that actually matter for novelty: release dates, heterogeneous processing times, exclusive instance allocation, cloud prices, and data-transfer overheads.

Do not rely on broad statements such as no previous study jointly optimizes all these features unless the literature search supports that claim comprehensively.

A narrower novelty claim is safer and more useful.

One possible replacement paragraph is:

The existing literature addresses several components of the scheduling problem considered here. Exact formulations for identical and unrelated parallel machines provide methods for minimizing completion-time objectives under different machine and job assumptions, while multi-cloud resource-provisioning models address the trade-off between execution cost and resource availability. Cloud-bursting systems instead emphasize operational policies for deciding when workloads should use external capacity. MOSPACE combines these perspectives by jointly selecting execution resources and job start times in a centralized optimization model with release dates, priority-weighted turnaround, heterogeneous instance characteristics, transfer overheads, and incremental cloud expenditure. Its contribution is therefore the integrated formulation and computational evaluation of this particular scheduling setting, rather than a general solution to federated resource management.

This wording better distinguishes the contribution without overstating it.

### **8.2. The exact-solution claim requires qualification**

The related work section describes MOSPACE as providing an exact solution to cloud bursting.

The MILP provides an exact solution to a discretized deterministic scheduling problem under its modeling assumptions.

It does not solve cloud bursting in general, which can involve uncertainty, data locality, instance acquisition decisions, failures, network contention, and independent local scheduling policies.

I recommend replacing claims of an exact solution to cloud bursting with *an exact formulation for the considered hybrid scheduling problem*.

That is a strong contribution and does not require a broader claim.

## **9. Presentation and reproducibility**

Severity: Low to Medium | Throughout the manuscript

The writing quality is uneven. The mathematical sections are generally more precise than the surrounding prose. Several parts of the introduction and related work contain grammatical errors, incomplete sentences, or unnecessarily repetitive explanations.

Section III also introduces \\(N\\), \\(M\\), and \\(O\\) informally as counts of clusters, clouds, and jobs before defining \\(J\\) and \\(M\\) differently in the formal model. That introductory paragraph should be simplified rather than adding a second notation scheme.

The general introduction to MILP in Section IV occupies substantial space but contributes little to understanding MOSPACE. I would shorten it and allocate the space to decoder pseudocode or experimental methodology.

The references section also appears before Tables V–VII because of LaTeX float placement. The results tables should ideally appear near their discussion, and Section VI needs substantive interpretation rather than a short summary of old numerical values.

Reproducibility would improve substantially with a concise specification of the software versions, Gurobi settings, time limits, hardware, parallel worker counts, random seeds, and data-generation procedure. For the Batsim experiments, include the EDC scheduling policy settings and the exact information available at each decision.

Source code or a reduced anonymized benchmark generator would be valuable, especially if the production trace cannot be released.

## **10. Recommended revision priorities**

I would allocate the remaining work according to scientific risk rather than the order in which the problems appear.

| Priority | Revision                                                                     | Main benefit                                              |
| -------- | ---------------------------------------------------------------------------- | --------------------------------------------------------- |
| Critical | Verify the GA's feasibility and cap handling; explain normalization overhead | Establishes validity of exact-versus-heuristic comparison |
| Critical | Specify online information availability and scheduler timing                 | Establishes operational realism                           |
| High     | Make the Slurm comparisons resource- and information-consistent              | Strengthens causal interpretation                         |
| High     | Correct the I/O performance model and distinguish input from egress          | Improves physical and economic validity                   |
| High     | Explain initial residual loads, time discretization, and horizon             | Makes the computational experiments reproducible          |
| High     | Discuss the adverse tail-waiting results                                     | Prevents misleading performance conclusions               |
| Medium   | Add GA ablations and a difficult-instance runtime study                      | Explains why the algorithm performs well or poorly        |
| Medium   | Isolate provider diversity from total local capacity                         | Removes factorial confounding                             |
| Medium   | Add heterogeneous performance and provisioning sensitivity                   | Tests generality of the modeled effects                   |
| Medium   | Specify MIT Supercloud and missing cloud-bursting baselines                  | Improves completeness                                     |
| Low      | Reconcile numerical claims and restructure presentation                      | Makes the paper internally consistent                     |

## **11. Final reviewer assessment**

I see a credible paper here, with a clear route to a substantially stronger submission.

The arc-flow formulation is the most technically defensible part. Its feasibility logic is sound, and the normalization scheme is more carefully constructed than in many applied scheduling papers.

The genetic algorithm is not a generic use of evolutionary optimization. Its indirect representation, schedule-generating decoders, and local-search operators are adapted to the structure of the problem. That is an appropriate algorithmic contribution, provided the manuscript establishes that its reported solutions are admissible for the same optimization model used by the exact solver.

The experimental work is more ambitious than in the earlier drafts. The 108-instance factorial experiment and the trace-driven comparison with Slurm policies provide a useful foundation. The reported performance improvements are plausible and interesting. They do not yet support all of the manuscript's broader claims because of the unresolved resource-capacity comparisons, model assumptions, and incomplete operational specification.

The adverse upper-tail waiting-time results are not a reason to reject the approach. They are evidence of a genuine scheduling trade-off and should become part of the analysis.

I would recommend major revision at this stage, not because the research direction is unsound, but because several conclusions require stronger methodological support than the current manuscript provides.

### **What I would not require before submission**

I would not require an implementation on a real multi-center federation, an exact model of network contention, multi-node gang scheduling, or a complete solution to online fairness. Those are reasonable subjects for subsequent research.

For this paper, the more important goal is to define the current abstraction precisely, verify that both solvers respect it, and establish which empirical improvements persist under controlled comparisons.

The strongest defensible contribution would then be:

> MOSPACE provides a valid time-expanded arc-flow formulation for joint placement and scheduling of exclusive-instance parallel jobs in hybrid HPC–cloud environments, together with a specialized memetic approximation method. Under the specified deterministic resource and cost assumptions, the exact model provides certified reference solutions, while the heuristic offers a faster alternative. Trace-driven experiments examine how centralized optimization affects turnaround, expenditure, and the distribution of waiting times relative to established scheduling policies.

That claim is narrower than the current general language about solving federated HPC scheduling. It is also more rigorous and easier to substantiate.

My assessment is that the paper's remaining work is predominantly experimental validation, formal clarification, and scientific communication. The fundamental mathematical approach is sufficiently developed that I would concentrate effort on these issues rather than pursuing a new formulation.
