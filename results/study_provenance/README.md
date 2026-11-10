# Study provenance of the `fedhpc` and `pos` workloads

Regenerate with `python3 scripts/study_provenance.py` (system Python; ~30 s).
Outputs: `summary.json` (every number below), `studies_<workload>.csv` (one row per
study, anonymised), `submission_raster.{pdf,png}`.

## Claim

Almost every job in both workloads (98–99 % in `fedhpc`, 95 % in `pos`) is a member of an
**ensemble study**: one reservoir model run many times with varied parameters by an
optimisation, history-matching or uncertainty tool. A handful of studies make up most of
each window: 5 studies hold 85 % of `fedhpc`, and 5 CMOST projects hold 79 % of `pos`.
So the jobs are not independent submissions from many users.

The claim is **"a few ensemble studies"**, not "one study". The largest single study is
34 % of `fedhpc` and 24 % of `pos` (26 % at project level).

## How studies are identified (provenance, not name similarity)

The earlier argument (`build_job_classes.py`) grouped jobs by `(user, job-name stem)`.
That is a heuristic. Here a study is read from the folder layout each launcher writes,
recorded by Slurm in `work_dir` (and `script` for `fedhpc`):

| Tool | Folder written by the tool | Study | Job name | Evidence in the batch script |
|---|---|---|---|---|
| CMG **CMOST** (history matching / optimisation / UA) | `<project>.cmpd/<study>.cmsd` | the `.cmsd` folder | `<study>_<experiment#>` (robust optimisation: `<study>_<exp#>_<sub>_<realization#>`) | `cmgsimrun @<study>.cmsd/<study>_<exp#>.resp` and a generated `cmgrun<k>.s` |
| **BR-Kalman** (ensemble history matching, Petrobras in-house) | `<case>_RUN_<iteration>/<case>_<member#>` | `<case>`, with each `_RUN_k` as one assimilation iteration | `<case>_<member#>` = member folder | header `## BR-Kalman Version 1.0.x`, QoS `cmg_brkalman`; each member runs the same deck in its own folder |
| none (standalone) | anything else | `work_dir` + name stem | free-form, usually `*.dat` | — |

For CMOST, `.cmpd` and `.cmsd` follow CMG's naming for the data folders of a project (`.cmp`)
and a study (`.cms`). This is from the logs and CMG practice; I could not find a public
copy of the CMOST manual to cite. BR-Kalman's identity rests only on the version header in
the scripts. Its iterate-over-the-same-ensemble structure matches ES-MDA (Emerick & Reynolds,
2013), but the logs do not state which algorithm it uses.

## Evidence

T2–T6 test the provenance grouping against information it was **not** built from.
"Null" is a permutation test with 2000 label shuffles that keep the group sizes. All
p-values are at the floor, 1/2001 ≈ 0.0005, except where noted.

| Test | `fedhpc` (964 jobs, 11 users) | `pos` (3340 jobs, 33 users) |
|---|---|---|
| **T1 Coverage:** jobs from CMOST / BR-Kalman | 48.7 % / 50.6 % → **99.3 %** (97.4 % of core-h) | 86.7 % / 8.7 % → **95.4 %** (83.1 % of core-h) |
| T1 jobs in studies of ≥ 10 jobs | 98.1 % | 95.0 % |
| T1 top-5 studies (jobs / core-h) | 84.8 % / 91.8 % | 57.2 % / 65.5 % (top-5 projects: 79.0 % of jobs) |
| T1 effective number of studies, 1/HHI (jobs / core-h) | 5.2 / 2.9 | 9.9 / 7.8 (6.1 projects) |
| **T2 Naming:** job name = `<study>_<run#>` | 100 % | 99.2 % |
| T2 run# vs submit order (Spearman ρ, median) | CMOST 1.00, BR-Kalman 1.00 | CMOST 1.00 (min 0.88); BR-Kalman 0.19 (reruns, see below) |
| T2 reruns (same run# in the same iteration) | 0 % | 2.7 % |
| **T3 Agreement with the name heuristic:** ARI (name groups vs studies) | 0.93 | 0.93 |
| T3 ensemble/single label agreement | 98.8 % | 99.0 % |
| **T4 Same request:** jobs asking for their study's usual request (null) | **100 %** (77 %) | **99.9 %** (45 %) |
| **T5 Runtime:** η² of log runtime explained by study (null) | **0.97** (0.008) | **0.79** (0.010) |
| T5 runtime CV within study (median) vs pooled | 0.16 vs 1.65 | 0.07 vs 2.27 |
| **T6 Submit time:** within-study std, min (null) | **23.6** (37.5), window 108 min | **44.7** (68.3), window 240 min |
| **T7 Script templates:** studies with exactly one normalised template | 9 / 9 | n/a (no scripts in this snapshot) |
| **T8 2024 log:** total 2024 jobs per study, median (max) | 2578 (26 847) | 100 (32 930) |
| T8 share of each study inside the window (median) | 2.2 % | 34 % |
| T8 active days per study (median) | 9.3 | 0.5 |

What each test shows:

- **T1:** almost every job carries a tool's study folder. The concentration measures show
  that each window is effectively 3–10 studies, against 11 and 33 users.
- **T2:** names are generated, not typed. Experiment numbers rise strictly with submission
  time, as a launcher that numbers experiments in order would produce. The two exceptions
  are not counter-evidence:
  - The low ρ for BR-Kalman in `pos` comes from two BR-Kalman studies that resubmit
    members into the same `RUN_k` folder.
  - Those resubmissions are 78 of the 84 reruns, the 2.7 % in `pos`. The other 6 are two
    CMOST studies re-running one experiment 3 times each (including the base case
    `00001`).
- **T3:** the earlier name-only heuristic recovers the provenance grouping almost exactly.
  The previous ensemble/single labels (`data/job_classes_*.json`) therefore stand. The
  disagreements:
  - 12 + 14 jobs belong to small tool studies that fall under the size-10 threshold.
  - 20 `pos` jobs are standalone runs that share a name stem.
- **T4:** ensemble members are the same model, so they ask for the same resources. The only
  exceptions in `pos` are 3 base-case re-runs of one CMOST study with a different core
  count.
- **T5:** knowing the study explains 79–97 % of the variance in log runtime. Members of a
  study have nearly identical runtimes (median CV 0.07–0.16).
- **T6:** each study submits in a tight burst. Studies overlap in time, though (see
  `submission_raster.png`), so a test of whether consecutive submissions belong to the same
  study is weak for `pos`: 0.107 against a null of 0.101, p ≈ 0.1. The spread test above is
  the right one.
- **T7:** all members of a study run the same batch script once paths and numbers are
  removed.
- **T8:** the window is a slice of a larger campaign. A `fedhpc` study runs for about 9 days
  and the window catches about 2 % of it. `pos` studies are shorter: a 15-study CMOST project
  of about 100 runs each mostly completes inside the window.

**These windows are typical of the cluster.** Across the whole 2024 log (2.75 M jobs),
CMOST + BR-Kalman account for **77 % of jobs and 70 % of core-hours**: 40.5 % + 36.6 % of
jobs, from about 1350 CMOST and 890 BR-Kalman studies. The windows sit somewhat above that
average (95–99 %) because they are busy periods, and busy periods are made by ensembles.

## Consequences for the experiments

- **Ensemble flurries are the workload itself.** Tsafrir & Feitelson (IPDPS 2006) call
  short bursts of repetitive jobs from one user *flurries*, and recommend filtering them
  out because they destabilise simulation results. That advice targets anomalies. Here
  ensembles are 77 % of the year's jobs, so filtering them would remove the workload. The
  sensitivity they describe does apply, though: a few studies decide each window. Results
  should be reported per workload with that in mind, and claims should not be generalised
  from one window.
- **Modelling per class is justified.** Request, runtime and data volume are shared within
  a study (T4, T5), so the class-based egress model (`fedhpc.io_model`: ensemble members vs
  single runs) is grounded in provenance, not only in names.
- **Batch arrival is real.** Within a study, jobs arrive in a burst with identical
  requests. This is why ties in slots and priorities are so common in these instances
  (see the SPT tie-break issue).

## Limitations

- **`.cmsd` and BR-Kalman semantics** are inferred from the logs (folder layout, generated
  names, script headers), not from vendor documentation.
- **`pos` has no batch scripts**, so T7 covers `fedhpc` only.
- **BR-Kalman study keys** in the 2024 log are any `*_RUN_<k>` parent folder. A few
  non-BR-Kalman folders could match that pattern, which would inflate the 36.6 % base rate
  slightly.
- **Instance jobs** are matched to Slurm records by `build_job_classes.match` (by job id for
  `fedhpc`; by order, runtime and cores for `pos`). Any mismatch raises an error.

## References

- Tsafrir, D., Feitelson, D. G. *Instability in parallel job scheduling simulation: the role of workload flurries.* IPDPS 2006. <https://www.cse.huji.ac.il/~feit/papers/Butterfly06IPDPS.pdf>
- Feitelson, D. G. *Workload Modeling for Computer Systems Performance Evaluation.* Cambridge University Press, 2015.
- Emerick, A. A., Reynolds, A. C. *Ensemble smoother with multiple data assimilation.* Computers & Geosciences 55, 2013.
- Hubert, L., Arabie, P. *Comparing partitions.* Journal of Classification 2, 1985 (adjusted Rand index).
