"""Visualisation and reporting helpers for FED-HPC solutions."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no X11 connection required
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from .data import Instance
from .model import Solution

# ── Gantt rendering limits ────────────────────────────────────────────────────
_MAX_FIG_HEIGHT_IN = 50.0   # inches; taller figures are downscaled
_MAX_FIG_WIDTH_IN  = 80.0
_LABEL_JOB_LIMIT   = 200    # above this, skip per-bar text (unreadable anyway)
_LEGEND_JOB_LIMIT  = 80     # above this, omit the legend entirely

# ── Text report width ─────────────────────────────────────────────────────────
_W = 72


# ──────────────────────────────────────────────────────────────────────────────
# Statistics
# ──────────────────────────────────────────────────────────────────────────────

def _stat_block(values: list[float]) -> dict:
    """Return {avg, min, max, total} for a list of floats."""
    if not values:
        return {"avg": 0.0, "min": 0.0, "max": 0.0, "total": 0.0}
    return {
        "avg":   sum(values) / len(values),
        "min":   min(values),
        "max":   max(values),
        "total": sum(values),
    }


def compute_stats(sol: Solution, inst: Instance) -> dict:
    """Compute per-job and system-level statistics from a Solution + Instance.

    Returns a dict with:
      n_scheduled, n_total
      wait_time, run_time, turnaround, bounded_slowdown
          — each a stat block {avg, min, max, total}
      per_job  — list of per-job dicts sorted by job id:
                   {id, type_id, kind, arrival, start, end,
                    wait, run, turnaround, bounded_slowdown, cost}
      system   — {f1, f2, onprem_jobs, cloud_jobs, onprem_cost, cloud_cost,
                  onprem_util_pct}

    Bounded slowdown: BS(j) = max(1, turnaround / max(exec_time, 1)).
    On-prem utilisation: (sum of p_occ slots on on-prem) / (total on-prem
    capacity × horizon) × 100.
    """
    n_total = len(inst.jobs)
    empty   = {"avg": 0.0, "min": 0.0, "max": 0.0, "total": 0.0}

    if not sol.assignment:
        return {
            "n_scheduled": 0, "n_total": n_total,
            "wait_time": empty, "run_time": empty,
            "turnaround": empty, "bounded_slowdown": empty,
            "per_job": [],
            "system": {
                "f1": sol.f1 or 0.0, "f2": sol.f2 or 0.0,
                "onprem_jobs": 0, "cloud_jobs": 0,
                "onprem_cost": 0.0, "cloud_cost": 0.0,
                "onprem_util_pct": 0.0,
            },
        }

    job_map  = {j.id: j for j in inst.jobs}
    type_map = {m.id: m for m in inst.instance_types}

    wait_times:  list[float] = []
    run_times:   list[float] = []
    turnarounds: list[float] = []
    bslowdowns:  list[float] = []

    onprem_jobs  = 0;  cloud_jobs  = 0
    onprem_cost  = 0.0; cloud_cost  = 0.0
    onprem_slots = 0.0
    per_job: list[dict] = []

    for jid, (mid, t_start) in sorted(sol.assignment.items()):
        j          = job_map[jid]
        m          = type_map[mid]
        t_end      = sol.completion[jid]
        runtime    = float(t_end - t_start)
        wait       = float(t_start) - j.arrival
        turnaround = float(t_end)   - j.arrival
        bsd        = max(1.0, turnaround / max(j.exec_time, 1.0))
        cost       = inst.c.get((jid, mid), 0.0)

        wait_times.append(wait)
        run_times.append(runtime)
        turnarounds.append(turnaround)
        bslowdowns.append(bsd)

        if m.kind == "on-prem":
            onprem_jobs  += 1
            onprem_cost  += cost
            onprem_slots += runtime
        else:
            cloud_jobs   += 1
            cloud_cost   += cost

        per_job.append({
            "id": jid, "type_id": mid, "kind": m.kind,
            "arrival": j.arrival, "start": t_start, "end": t_end,
            "wait": wait, "run": runtime,
            "turnaround": turnaround, "bounded_slowdown": bsd,
            "cost": cost,
        })

    onprem_capacity = sum(
        m.capacity * inst.horizon
        for m in inst.instance_types
        if m.kind == "on-prem" and m.capacity is not None
    )
    # inst.occupied counts running-job occupancy per (type, slot); add it so the
    # numerator reflects *all* on-prem usage, not just the newly scheduled jobs.
    running_slots = sum(inst.occupied.values())
    onprem_util_pct = (
        100.0 * (onprem_slots + running_slots) / onprem_capacity
        if onprem_capacity > 0 else 0.0
    )

    return {
        "n_scheduled":    len(sol.assignment),
        "n_total":        n_total,
        "wait_time":        _stat_block(wait_times),
        "run_time":         _stat_block(run_times),
        "turnaround":       _stat_block(turnarounds),
        "bounded_slowdown": _stat_block(bslowdowns),
        "per_job":          per_job,
        "system": {
            "f1":              sol.f1 or 0.0,
            "f2":              sol.f2 or 0.0,
            "onprem_jobs":     onprem_jobs,
            "cloud_jobs":      cloud_jobs,
            "onprem_cost":     onprem_cost,
            "cloud_cost":      cloud_cost,
            "onprem_util_pct": onprem_util_pct,
            "n_running_jobs":  len(inst.running_jobs),
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Shared text-report building blocks
# ──────────────────────────────────────────────────────────────────────────────

def _rule(char: str = "═") -> str:
    return char * _W


def _stat_table_lines(st: dict, slot_secs: float = 1.0) -> list[str]:
    """Four-row metric table used by both console summary and the schedule file.

    All slot-based values are multiplied by *slot_secs* before display so that
    the table always shows wall-clock seconds.
    """
    lines = [
        f"  {'Metric':<30} {'Avg':>11} {'Min':>11} {'Max':>11} {'Total':>12}",
        f"  {_rule('-')[:71]}",
    ]

    def row(label: str, blk: dict, unit: str = "", total: bool = True) -> str:
        lbl = f"{label} {unit}".strip()
        scale = slot_secs
        tot = f"{blk['total'] * scale:>12.1f}" if total else f"{'—':>12}"
        return (
            f"  {lbl:<30}"
            f" {blk['avg'] * scale:>11.1f}"
            f" {blk['min'] * scale:>11.1f}"
            f" {blk['max'] * scale:>11.1f}"
            f" {tot}"
        )

    lines += [
        row("Wait time",        st["wait_time"],        "(s)"),
        row("Run time",         st["run_time"],          "(s)"),
        row("Turnaround time",  st["turnaround"],        "(s)"),
        row("Bounded slowdown", st["bounded_slowdown"],  "   ", total=False),
    ]
    return lines


def _system_lines(sys: dict, slot_secs: float = 1.0) -> list[str]:
    f1_s = sys["f1"] * slot_secs
    lines = [
        f"  f1  total turnaround  : {f1_s:>15.1f}  s",
        f"  f2  total cost        : {sys['f2']:>15.2f}  $",
        f"  On-prem  jobs / cost  : {sys['onprem_jobs']:>6}  /  {sys['onprem_cost']:>12.2f}  $",
        f"  Cloud    jobs / cost  : {sys['cloud_jobs']:>6}  /  {sys['cloud_cost']:>12.2f}  $",
        f"  On-prem utilisation   : {sys['onprem_util_pct']:>14.2f}  %",
    ]
    if sys.get("n_running_jobs", 0):
        lines.append(
            f"  Running jobs (t=0)    : {sys['n_running_jobs']:>14}  pre-occupied slots"
        )
    return lines


def _timing_lines(sol: "Solution") -> list[str]:
    """Return solver timing lines, or an empty list if not available."""
    from .model import Solution  # local import to avoid circular dependency
    lines = []
    if sol.build_time is not None or sol.solve_time is not None:
        bt = f"{sol.build_time:.2f} s" if sol.build_time is not None else "—"
        st = f"{sol.solve_time:.2f} s" if sol.solve_time is not None else "—"
        total = (sol.build_time or 0.0) + (sol.solve_time or 0.0)
        lines += [
            f"  Pre-processing (MIP build) : {bt:>10}",
            f"  Gurobi solve               : {st:>10}",
            f"  Total solver wall-clock    : {total:.2f} s",
        ]
    return lines


# ──────────────────────────────────────────────────────────────────────────────
# Console summary (aggregate only — no per-job listing)
# ──────────────────────────────────────────────────────────────────────────────

def format_summary(
    sol: Solution,
    inst: Instance,
    title: str = "FED-HPC Schedule",
) -> str:
    """Return a human-readable one-screen summary for stdout.

    Shows only aggregate statistics.  Per-job details are written to the
    machine-schedule file by save_machine_schedule().
    """
    lines = [f"{title}  [{sol.status}]", _rule(), ""]

    if sol.status not in ("optimal", "feasible"):
        lines.append(f"  No solution available (solver status: {sol.status}).")
        return "\n".join(lines)

    slot_secs = inst.slot_size_seconds
    st  = compute_stats(sol, inst)
    sys = st["system"]
    n   = st["n_scheduled"]
    u   = st["n_total"] - n
    un  = f", {u} unscheduled" if u else ""

    lines += [f"JOB STATISTICS  ({n} scheduled{un})", _rule("─")]
    lines += _stat_table_lines(st, slot_secs)
    lines += ["", "SYSTEM STATISTICS", _rule("─")]
    lines += _system_lines(sys, slot_secs)

    timing = _timing_lines(sol)
    if timing:
        lines += ["", "SOLVER TIMING", _rule("─")]
        lines += timing

    lines.append("")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Gantt chart
# ──────────────────────────────────────────────────────────────────────────────

def _assign_lanes(
    jobs: list[tuple[int, int, int]],
) -> tuple[list[tuple[int, int, int, int]], int]:
    """Greedy interval-graph colouring: assign concurrent jobs to separate lanes.

    Returns (list of (t_start, t_end, jid, lane_index), total_lanes).
    """
    lane_ends: list[int] = []
    result: list[tuple[int, int, int, int]] = []
    for t_start, t_end, jid in sorted(jobs):
        for i, end in enumerate(lane_ends):
            if t_start >= end:
                lane_ends[i] = t_end
                result.append((t_start, t_end, jid, i))
                break
        else:
            result.append((t_start, t_end, jid, len(lane_ends)))
            lane_ends.append(t_end)
    return result, max(len(lane_ends), 1)


def save_gantt(
    sol: Solution,
    inst: Instance,
    path: str | Path = "gantt.png",
    *,
    title: str = "FED-HPC Schedule",
) -> Path:
    """Save a Gantt chart of *sol* as a PNG image.

    Each section of the Y axis is one machine type; concurrent jobs on the same
    machine are stacked in separate lanes so bars never overlap.  Bars are
    coloured by job; text labels and the legend are suppressed for large instances
    to keep the file size tractable.
    """
    path     = Path(path)
    type_map = {m.id: m for m in inst.instance_types}

    job_ids  = sorted(sol.assignment)
    n_jobs   = len(job_ids)
    colours  = plt.colormaps["tab10"].resampled(max(n_jobs, 1))
    colour_of = {jid: colours(i) for i, jid in enumerate(job_ids)}

    # Group assignments by machine type
    by_type: dict[int, list[tuple[int, int, int]]] = {m.id: [] for m in inst.instance_types}
    for jid, (mid, t_start) in sol.assignment.items():
        by_type[mid].append((t_start, sol.completion[jid], jid))

    machine_ids = [m.id for m in inst.instance_types]

    # Lane assignments per machine
    lanes_for: dict[int, list[tuple[int, int, int, int]]] = {}
    n_lanes:   dict[int, int] = {}
    for mid in machine_ids:
        assigned, nl = _assign_lanes(by_type[mid])
        lanes_for[mid] = assigned
        n_lanes[mid]   = nl

    # Cumulative Y base for each machine band
    sub_h = 0.8
    gap   = 0.5
    machine_base: dict[int, float] = {}
    y = 0.0
    for mid in machine_ids:
        machine_base[mid] = y
        y += n_lanes[mid] * sub_h + gap
    total_height = y - gap

    raw_h  = max(3.0, total_height * 0.8 + 1.5)
    raw_w  = max(10.0, inst.horizon * 0.4)
    scale  = min(1.0, _MAX_FIG_HEIGHT_IN / raw_h, _MAX_FIG_WIDTH_IN / raw_w)
    fig_h  = raw_h * scale
    fig_w  = raw_w * scale
    dpi    = max(72, int(150 * scale))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    show_labels = n_jobs <= _LABEL_JOB_LIMIT

    for mid in machine_ids:
        base = machine_base[mid]
        for t_start, t_end, jid, lane in lanes_for[mid]:
            bar_center = base + (lane + 0.5) * sub_h
            width      = t_end - t_start
            ax.barh(
                bar_center, width, left=t_start,
                height=sub_h * 0.85,
                color=colour_of[jid], edgecolor="black", linewidth=0.5,
            )
            if show_labels:
                ax.text(
                    t_start + width / 2, bar_center, f"j{jid}",
                    ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold",
                )
        sep_y = base + n_lanes[mid] * sub_h + gap / 2
        ax.axhline(sep_y, color="grey", linewidth=0.5, linestyle="--")

    ytick_pos    = [machine_base[mid] + n_lanes[mid] * sub_h / 2 for mid in machine_ids]
    ytick_labels = [f"Type {mid} ({type_map[mid].kind})" for mid in machine_ids]
    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(ytick_labels)
    ax.set_ylim(-gap / 2, total_height + gap / 2)
    ax.set_xlabel("Time (slots)")
    ax.set_xlim(0, inst.horizon)
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    if n_jobs <= _LEGEND_JOB_LIMIT:
        ncols = max(1, math.ceil(n_jobs / 8))
        legend_patches = [
            mpatches.Patch(color=colour_of[jid], label=f"Job {jid}")
            for jid in job_ids
        ]
        ax.legend(handles=legend_patches, ncols=ncols, loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Machine schedule text file (full detail — per-job and per-type)
# ──────────────────────────────────────────────────────────────────────────────

def save_machine_schedule(
    sol: Solution,
    inst: Instance,
    path: str | Path = "machine_schedule.txt",
    *,
    title: str = "FED-HPC Schedule",
) -> Path:
    """Write a detailed schedule report to *path*.

    Structure
    ---------
    1. Summary header  — same aggregate stats shown on the console.
    2. Job-centric view — one row per job with all per-job KPIs:
       arrival, start, end, wait, run-time, turnaround, bounded slowdown, cost.
    3. Machine-centric view — one block per instance type with jobs listed in
       start-time order and peak-concurrency noted.
    """
    path = Path(path)

    sep  = _rule()
    thin = _rule("─")

    # ── header / summary ──────────────────────────────────────────────────────
    lines: list[str] = [title, sep, ""]

    if sol.status not in ("optimal", "feasible"):
        lines.append(f"  No solution available (solver status: {sol.status}).")
        path.write_text("\n".join(lines))
        return path

    slot_secs = inst.slot_size_seconds
    st  = compute_stats(sol, inst)
    sys = st["system"]
    n   = st["n_scheduled"]
    u   = st["n_total"] - n
    un  = f", {u} unscheduled" if u else ""

    lines += [f"JOB STATISTICS  ({n} scheduled{un})", thin]
    lines += _stat_table_lines(st, slot_secs)
    lines += ["", "SYSTEM STATISTICS", thin]
    lines += _system_lines(sys, slot_secs)

    timing = _timing_lines(sol)
    if timing:
        lines += ["", "SOLVER TIMING", thin]
        lines += timing

    lines += ["", sep, ""]

    # ── job-centric view ──────────────────────────────────────────────────────
    lines += ["JOB-CENTRIC VIEW  (all times in seconds)", thin]

    # Column header
    h = (
        f"  {'Job':>5}  {'Type':>4}  {'Kind':<8}"
        f"  {'Arrival(s)':>10}  {'Start(s)':>9}  {'End(s)':>9}"
        f"  {'Wait(s)':>9}  {'Run(s)':>9}  {'Turnaround(s)':>14}"
        f"  {'BSlowdown':>9}  {'Cost':>10}"
    )
    lines.append(h)
    lines.append(
        f"  {'─'*5}  {'─'*4}  {'─'*8}"
        f"  {'─'*10}  {'─'*9}  {'─'*9}"
        f"  {'─'*9}  {'─'*9}  {'─'*14}"
        f"  {'─'*9}  {'─'*10}"
    )

    for rec in st["per_job"]:
        lines.append(
            f"  {rec['id']:>5}  {rec['type_id']:>4}  {rec['kind']:<8}"
            f"  {rec['arrival'] * slot_secs:>10.1f}"
            f"  {rec['start']   * slot_secs:>9.1f}"
            f"  {rec['end']     * slot_secs:>9.1f}"
            f"  {rec['wait']    * slot_secs:>9.1f}"
            f"  {rec['run']     * slot_secs:>9.1f}"
            f"  {rec['turnaround'] * slot_secs:>14.1f}"
            f"  {rec['bounded_slowdown']:>9.3f}  {rec['cost']:>10.4f}"
        )
    lines.append("")

    # ── machine-centric view ──────────────────────────────────────────────────
    type_map = {m.id: m for m in inst.instance_types}

    by_type: dict[int, list[tuple[int, int, int]]] = {m.id: [] for m in inst.instance_types}
    for jid, (mid, t_start) in sol.assignment.items():
        by_type[mid].append((t_start, sol.completion[jid], jid))

    # Index running jobs by type for the summary section
    rj_by_type: dict[int, list] = {}
    for rj in inst.running_jobs:
        rj_by_type.setdefault(rj.type_id, []).append(rj)

    lines += ["MACHINE-CENTRIC VIEW  (all times in seconds)", thin]

    for mid in sorted(by_type):
        m          = type_map[mid]
        cap        = m.capacity if m.capacity is not None else "unlimited"
        jobs_on_m  = sorted(by_type[mid])
        rj_list    = rj_by_type.get(mid, [])
        lines.append(f"  Type {mid}  [{m.kind}]  cpu={m.cpu}  (capacity: {cap})")

        # Running-jobs summary (initial state)
        if rj_list:
            rj_ends = sorted(rj.end for rj in rj_list)
            n_rj    = len(rj_list)
            median_end = rj_ends[n_rj // 2]
            n_clipped  = sum(1 for e in rj_ends if e >= inst.horizon)
            e0  = rj_ends[0]  * slot_secs
            e1  = rj_ends[-1] * slot_secs
            med = median_end  * slot_secs
            lines.append(
                f"    running at t=0: {n_rj} jobs  "
                f"(end range: {e0:.0f}–{e1:.0f} s, "
                f"median: {med:.0f} s"
                + (f", {n_clipped} extend past horizon" if n_clipped else "")
                + ")"
            )

        if not jobs_on_m:
            lines.append("    (no new jobs assigned)")
        else:
            lines.append(
                f"    {'Job':>5}  {'Start(s)':>9}  {'End(s)':>9}  {'Duration(s)':>11}"
            )
            for t_start, t_end, jid in jobs_on_m:
                lines.append(
                    f"    {jid:>5}"
                    f"  {t_start * slot_secs:>9.1f}"
                    f"  {t_end   * slot_secs:>9.1f}"
                    f"  {(t_end - t_start) * slot_secs:>11.1f}"
                )
            # Peak concurrency (new jobs only)
            slots: dict[int, int] = {}
            for t_start, t_end, _ in jobs_on_m:
                for t in range(t_start, t_end):
                    slots[t] = slots.get(t, 0) + 1
            peak = max(slots.values()) if slots else 0
            lines.append(f"    peak concurrency (new jobs): {peak}")

        lines.append("")

    path.write_text("\n".join(lines))
    return path
