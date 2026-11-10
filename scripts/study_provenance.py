"""Which study each instance job belongs to, and the evidence for it.

Claim under test: most jobs of both source workloads (``fedhpc``: 2024-11-25
snapshot, ``pos``: 2024-01-30 congestion snapshot) are *members of a few
ensemble studies*: many runs of one reservoir model with varied parameters,
launched by an optimisation / history-matching / uncertainty tool. They are not
independent submissions.

Primary identification is by *tool provenance*, not by job-name similarity.
Each launcher writes its jobs into its own folder layout:

* CMG CMOST: ``<project>.cmpd/<study>.cmsd``. One ``.cmsd`` folder is one
  CMOST study. Its jobs are named ``<study>_<run#>`` and submitted through
  ``cmgsimrun @<study>.cmsd/<study>_<run#>.resp``.
* BR-Kalman (ensemble history matching): ``<case>_RUN_<iter>/<case>_<member>``,
  QoS ``cmg_brkalman``. Every member runs the same deck. One ``<case>`` is one
  study, and each ``_RUN_<iter>`` is one ensemble iteration.

Everything else is labelled *standalone* (study = work_dir + job-name stem).

That grouping is then tested against evidence it was not built from:

  T1  coverage and concentration (jobs and core-hours, effective no. of studies)
  T2  internal consistency: job name = study name + run number; run numbers vs
      submission order
  T3  agreement with the name-only heuristic of build_job_classes.py
  T4  identical resource request inside a study
  T5  runtime homogeneity: eta^2 of log runtime by study
  T6  submission contiguity: runs test on the study labels in submit order
  T7  one batch-script template per study (fedhpc only: pos has no scripts)
  T8  the full 2024 log: each study's whole size and lifetime, and the share of
      all 2024 jobs that come from these two tools

T4-T6 are compared with a permutation null that keeps the study sizes and
shuffles the labels. Only aggregates and anonymised labels are written
(no user names, paths or job names): results/study_provenance/.
Run with the system python3 (needs pandas, scipy, pyarrow, matplotlib).
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "study_provenance"
LOG2024 = Path.home() / "Documentos" / "slurm" / "combined_data_2024_1_2024_12_stripped.parquet"
RNG = np.random.default_rng(20261105)
N_PERM = 2000

_spec = importlib.util.spec_from_file_location("bjc", ROOT / "scripts" / "build_job_classes.py")
bjc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bjc)

CMSD = re.compile(r"^(.*?/([^/]+)\.cmsd)(?:/|$)")
BRK_ITER = re.compile(r"^(.*)_RUN_(\d+)$")


# ---------------------------------------------------------------- provenance
def provenance(work_dir: str, job_name: str, qos: str = "") -> dict:
    wd = str(work_dir).rstrip("/")
    m = CMSD.match(wd)
    if m:
        return dict(tool="cmost", study=m.group(1), base=m.group(2), iteration=m.group(1))
    parent, _, leaf = wd.rpartition("/")
    b = BRK_ITER.match(parent)
    if b or "brkalman" in str(qos):
        case = b.group(1) if b else parent
        return dict(tool="brkalman", study=case, base=case.rpartition("/")[2],
                    iteration=parent, leaf=leaf)
    return dict(tool="standalone", study=f"{wd}::{bjc.stem(job_name)}", base=bjc.stem(job_name),
                iteration=f"{wd}::{bjc.stem(job_name)}")


def load_workload(workload: str) -> pd.DataFrame:
    inst_path, rec_path = bjc.SOURCES[workload]
    inst = json.loads(inst_path.read_text())
    rows = bjc.match(workload, inst, bjc.load_records(rec_path))
    df = pd.DataFrame(rows)
    p = pd.DataFrame([provenance(r["work_dir"], r["job_name"], r.get("qos", "")) for r in rows])
    df = pd.concat([df.reset_index(drop=True), p], axis=1)
    df["submit"] = pd.to_datetime(df["@submit"])
    df["core_h"] = df.elapsed * df.total_cpus / 3600
    df["name_group"] = list(zip(df.username, df.job_name.map(bjc.stem)))
    df["heur_ensemble"] = json.loads((ROOT / "data" / f"job_classes_{workload}.json").read_text())["ensemble"]
    return df


# ---------------------------------------------------------------- statistics
def ari(a, b) -> float:
    """Adjusted Rand index (Hubert & Arabie 1985)."""
    ct = pd.crosstab(pd.Series(a, dtype=str).values, pd.Series(b, dtype=str).values).values
    c2 = lambda x: (x * (x - 1) / 2).sum()
    n = ct.sum()
    s_ij, s_a, s_b = c2(ct), c2(ct.sum(1)), c2(ct.sum(0))
    exp = s_a * s_b / (n * (n - 1) / 2)
    return float((s_ij - exp) / (0.5 * (s_a + s_b) - exp))


def eta2(y: np.ndarray, g: np.ndarray) -> float:
    """Share of variance of y explained by group g (one-way ANOVA eta^2)."""
    d = pd.DataFrame(dict(y=y, g=g))
    m = d.groupby("g").y.transform("mean")
    return float(((m - d.y.mean()) ** 2).sum() / ((d.y - d.y.mean()) ** 2).sum())


def modal_request_share(req: np.ndarray, g: np.ndarray) -> float:
    """Share of jobs (in groups of >= 2) that ask for their group's most common request."""
    d = pd.DataFrame(dict(r=req, g=g))
    d = d[d.groupby("g").r.transform("size") >= 2]
    return float(d.groupby("g").r.agg(lambda x: x.value_counts().iloc[0]).sum() / len(d))


def submit_tightness(t_min: np.ndarray, g: np.ndarray) -> float:
    """Minus the job-weighted mean within-group std of submit time (minutes); larger = tighter."""
    d = pd.DataFrame(dict(t=t_min, g=g))
    d = d[d.groupby("g").t.transform("size") >= 2]
    return -float(d.groupby("g").t.transform("std").mean())


def perm_test(stat, g: np.ndarray, *args) -> dict:
    """Observed statistic and a label-permutation null (sizes kept)."""
    obs = stat(*args, g)
    null = np.array([stat(*args, RNG.permutation(g)) for _ in range(N_PERM)])
    return dict(observed=round(obs, 4), null_mean=round(float(null.mean()), 4),
                null_p99=round(float(np.quantile(null, 0.99)), 4),
                p_value=float((1 + (null >= obs).sum()) / (N_PERM + 1)))


def run_numbers(df: pd.DataFrame) -> dict:
    """T2: job name is study name + run number; run numbers follow submission order."""
    # The run id is everything after "<study>_": CMOST experiment number, possibly
    # followed by a realization (robust optimisation: <study>_<exp>_<sub>_<real>);
    # BR-Kalman member number. A repeated id in one iteration is a rerun.
    ok, rhos, reruns, n_id = [], [], 0, 0
    for _, s in df[df.tool != "standalone"].groupby("study"):
        base = s.base.iloc[0]
        pat = re.compile(rf"^{re.escape(base)}_(\d+)((?:_[A-Za-z]\w*?_\d+)?)(?:\.dat)?$")
        ms = [pat.match(str(n)) for n in s.job_name]
        if s.tool.iloc[0] == "brkalman":   # member folder is the job name
            ms = [m if (m and str(l) == str(n)) else None for m, l, n in zip(ms, s.leaf, s.job_name)]
        ok += [m is not None for m in ms]
        good = [(i, m) for i, m in enumerate(ms) if m]
        ids = pd.Series([(s.iteration.iloc[i], m.group(1), m.group(2)) for i, m in good])
        reruns += int(ids.duplicated().sum())
        n_id += len(ids)
        if len(good) >= 10:
            exp = [int(m.group(1)) for _, m in good]
            sub = s.submit.iloc[[i for i, _ in good]].astype("int64")
            rhos.append((s.tool.iloc[0], stats.spearmanr(exp, sub).statistic))
    r = pd.DataFrame(rhos, columns=["tool", "rho"])
    return dict(name_is_study_plus_run=round(float(np.mean(ok)), 4),
                rerun_share=round(reruns / max(n_id, 1), 4),
                spearman_run_vs_submit={t: dict(median=round(float(x.rho.median()), 3),
                                                min=round(float(x.rho.min()), 3), studies=len(x))
                                        for t, x in r.groupby("tool")})


def script_templates(df: pd.DataFrame) -> dict | None:
    """T7: number of distinct batch-script templates per study."""
    s = df[df.script.notna() & (df.script.astype(str) != "nan")]
    if s.empty:
        return None

    def norm(row):
        t = str(row.script)
        for v in sorted({str(row.job_name), str(row.work_dir), str(row.get("leaf", ""))}, key=len, reverse=True):
            if v and v != "nan":
                t = t.replace(v, "<X>")
        return re.sub(r"\d+", "<n>", t)

    s = s.assign(tpl=s.apply(norm, axis=1))
    per = s[s.tool != "standalone"].groupby("study").tpl.nunique()
    return dict(jobs_with_script=len(s), tool_studies=len(per),
                studies_with_one_template=int((per == 1).sum()),
                templates_per_tool={t: int(v) for t, v in s.groupby("tool").tpl.nunique().items()})


# ---------------------------------------------------------------- 2024 log
def log2024() -> pd.DataFrame:
    d = pd.read_parquet(LOG2024, columns=["@submit", "work_dir", "job_name", "elapsed", "tres_cpu"])
    wd = d.work_dir.astype(str).str.rstrip("/")
    cm = wd.str.extract(CMSD.pattern)[0]
    parent = wd.str.rpartition("/")[0]
    br = parent.str.extract(BRK_ITER.pattern)
    d["tool"] = np.where(cm.notna(), "cmost", np.where(br[0].notna(), "brkalman", "standalone"))
    d["study"] = cm.fillna(br[0])
    d["iteration"] = cm.fillna(parent.where(br[0].notna()))
    d["submit"] = pd.to_datetime(d["@submit"])
    d["core_h"] = d.elapsed * pd.to_numeric(d.tres_cpu, errors="coerce") / 3600
    return d


def study_lifetimes(df: pd.DataFrame, log: pd.DataFrame) -> pd.DataFrame:
    keys = df[df.tool != "standalone"].groupby("study").agg(
        tool=("tool", "first"), in_window=("jobid", "size"), core_h_window=("core_h", "sum"))
    g = log[log.study.isin(keys.index)].groupby("study").agg(
        jobs_2024=("study", "size"), first=("submit", "min"), last=("submit", "max"),
        iterations=("iteration", "nunique"), core_h_2024=("core_h", "sum"))
    t = keys.join(g)
    t["active_days"] = (t["last"] - t["first"]).dt.total_seconds() / 86400
    return t.sort_values("in_window", ascending=False)


# ---------------------------------------------------------------- report
def analyse(workload: str, log: pd.DataFrame | None) -> tuple[dict, pd.DataFrame]:
    df = load_workload(workload)
    n, ch = len(df), df.core_h.sum()
    res: dict = dict(workload=workload, jobs=n, core_hours=round(ch), users=df.username.nunique(),
                     window=[str(df.submit.min()), str(df.submit.max())])

    # T1 coverage / concentration
    tool = df.groupby("tool").agg(jobs=("jobid", "size"), core_h=("core_h", "sum"), studies=("study", "nunique"))
    tool["job_share"] = (tool.jobs / n).round(4)
    tool["core_h_share"] = (tool.core_h / ch).round(4)
    res["T1_tools"] = tool.drop(columns="core_h").to_dict("index")
    st = df.groupby("study").agg(tool=("tool", "first"), jobs=("jobid", "size"), core_h=("core_h", "sum"),
                                 user=("username", "first")).sort_values("jobs", ascending=False)
    p_j, p_c = st.jobs / n, st.core_h / ch
    res["T1_concentration"] = dict(
        studies=len(st), studies_ge10=int((st.jobs >= 10).sum()),
        top1_jobs=round(float(p_j.iloc[0]), 4), top3_jobs=round(float(p_j.iloc[:3].sum()), 4),
        top5_jobs=round(float(p_j.iloc[:5].sum()), 4),
        top5_core_h=round(float(p_c.sort_values(ascending=False).iloc[:5].sum()), 4),
        effective_studies_jobs=round(float(1 / (p_j ** 2).sum()), 2),
        effective_studies_core_h=round(float(1 / (p_c ** 2).sum()), 2),
        jobs_in_studies_ge10=round(float(st.jobs[st.jobs >= 10].sum() / n), 4))

    # T2 naming and run numbers
    res["T2_naming"] = run_numbers(df)

    # T3 agreement with the name-only heuristic
    prov_ens = (df.tool != "standalone").astype(int)
    h = np.array(df.heur_ensemble)
    res["T3_name_heuristic"] = dict(
        ari_name_groups_vs_studies=round(ari(df.name_group.astype(str), df.study), 4),
        ensemble_label_agreement=round(float((prov_ens == h).mean()), 4),
        heuristic_ensemble_but_standalone=int(((h == 1) & (prov_ens == 0)).sum()),
        tool_member_but_heuristic_single=int(((h == 0) & (prov_ens == 1)).sum()))

    # T4-T6 against a permutation null. Groups: ensemble iterations (CMOST study,
    # BR-Kalman RUN_k); standalone jobs each form their own group.
    g = np.where(df.tool != "standalone", df.iteration, "solo::" + df.index.astype(str)).astype(object)
    ens = (df.tool != "standalone").values
    req = df.tres_req.astype(str).values
    logrt = np.log10(df.elapsed.clip(lower=1).values.astype(float))
    res["T4_modal_request"] = perm_test(modal_request_share, g[ens], req[ens])
    res["T5_runtime_eta2"] = perm_test(eta2, g[ens], logrt[ens])
    within = df[ens].groupby(g[ens]).elapsed.agg(lambda x: x.std() / x.mean() if len(x) > 1 else np.nan)
    res["T5_runtime_cv"] = dict(within_study_median=round(float(within.median()), 3),
                                pooled=round(float(df[ens].elapsed.std() / df[ens].elapsed.mean()), 3))
    t_min = ((df.submit - df.submit.min()).dt.total_seconds() / 60).values
    t6 = perm_test(submit_tightness, g[ens], t_min[ens])
    res["T6_submit_std_min"] = dict(observed=-t6["observed"], null_mean=-t6["null_mean"],
                                    null_p01=-t6["null_p99"], p_value=t6["p_value"],
                                    window=round(float(t_min.max()), 1))

    # concentration one level up: CMOST project (.cmpd) holding several studies
    proj = np.where(df.tool == "cmost", df.study.str.extract(r"^(.*?\.cmpd)")[0].fillna(df.study), df.study)
    pp = pd.Series(proj).value_counts() / n
    res["T1_projects"] = dict(projects=len(pp), top1_jobs=round(float(pp.iloc[0]), 4),
                              top5_jobs=round(float(pp.iloc[:5].sum()), 4),
                              effective_projects_jobs=round(float(1 / (pp ** 2).sum()), 2))

    # T7 script templates
    res["T7_script_templates"] = script_templates(df)

    # sensitivity of the earlier size-threshold rule
    res["threshold_sensitivity"] = {
        k: round(float((df.groupby("study").jobid.transform("size") >= k).mean()), 4) for k in (2, 5, 10, 20, 50)}

    # T8 lifetime in the 2024 log
    life = None
    if log is not None:
        life = study_lifetimes(df, log)
        res["T8_lifetimes"] = dict(
            studies=len(life),
            window_share_of_study_median=round(float((life.in_window / life.jobs_2024).median()), 4),
            jobs_2024_median=int(life.jobs_2024.median()), jobs_2024_max=int(life.jobs_2024.max()),
            active_days_median=round(float(life.active_days.median()), 1),
            iterations_brkalman=life[life.tool == "brkalman"].iterations.astype(int).tolist())

    # anonymised per-study table
    users = {u: f"U{i + 1}" for i, u in enumerate(df.username.value_counts().index)}
    tab = st.reset_index()
    tab["label"] = [f"{workload[0].upper()}{i + 1}" for i in range(len(tab))]
    tab["user"] = tab.user.map(users)
    tab["req"] = tab.study.map(df.groupby("study").tres_req.agg(lambda x: "/".join(sorted(set(
        re.search(r"cpu=(\d+)", s).group(1) + "c" for s in x)))))
    tab["iterations"] = tab.study.map(df.groupby("study").iteration.nunique())
    tab["median_runtime_min"] = tab.study.map(df.groupby("study").elapsed.median() / 60).round(1)
    if life is not None:
        tab = tab.join(life[["jobs_2024", "active_days", "iterations"]].rename(
            columns={"iterations": "iterations_2024"}), on="study")
    tab["job_share"] = (tab.jobs / n).round(4)
    tab["core_h_share"] = (tab.core_h / ch).round(4)
    tab = tab.drop(columns=["study"])
    return res, tab


def base_rate(log: pd.DataFrame) -> dict:
    t = log.groupby("tool").agg(jobs=("tool", "size"), core_h=("core_h", "sum"))
    return dict(jobs=int(len(log)),
                job_share={k: round(float(v), 4) for k, v in (t.jobs / t.jobs.sum()).items()},
                core_h_share={k: round(float(v), 4) for k, v in (t.core_h / t.core_h.sum()).items()},
                studies={k: int(v) for k, v in log.groupby("tool").study.nunique().items()})


def figure(tabs: dict[str, pd.DataFrame], dfs: dict[str, pd.DataFrame], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = dict(cmost="#2a6fdb", brkalman="#d9822b", standalone="#8a8a8a")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for ax, (w, df) in zip(axes, dfs.items()):
        rank = {s: i for i, s in enumerate(df.study.value_counts().index)}
        y = df.study.map(rank)
        t = (df.submit - df.submit.min()).dt.total_seconds() / 60
        for tool, c in col.items():
            m = df.tool == tool
            ax.scatter(t[m], y[m], s=4, c=c, label=f"{tool} ({m.sum()} jobs)", alpha=.7, lw=0)
        ax.set_ylim(min(30, len(rank)) - .5, -.5)
        ax.set_xlabel("minutes since first submission")
        ax.set_ylabel("study (rank by job count; top 30)")
        ax.set_title(f"{w}: {len(df)} jobs, {len(rank)} studies")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=8, markerscale=3, frameon=False)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-log", action="store_true", help="skip the 2024-log checks (T8)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log = None if args.no_log or not LOG2024.exists() else log2024()
    summary, tabs, dfs = {}, {}, {}
    for w in ("fedhpc", "pos"):
        summary[w], tabs[w] = analyse(w, log)
        dfs[w] = load_workload(w)
        tabs[w].to_csv(OUT / f"studies_{w}.csv", index=False)
    if log is not None:
        summary["base_rate_2024"] = base_rate(log)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    figure(tabs, dfs, OUT / "submission_raster.pdf")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
