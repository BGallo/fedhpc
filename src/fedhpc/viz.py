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

# ── Graph rendering limits ────────────────────────────────────────────────────
_MAX_GRAPH_HORIZON = 80     # skip space-time graph if horizon exceeds this

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
    """Compute per-job, per-partition, and system-level statistics.

    Returns a dict with:
      n_scheduled, n_total
      wait_time, run_time, turnaround, bounded_slowdown
          — each a stat block {avg, min, max, total}
      per_job       — list of per-job dicts sorted by job id:
                        {id, type_id, kind, arrival, start, end,
                         wait, run, turnaround, bounded_slowdown, cost}
      per_partition — list of per-partition dicts (one per InstanceType):
                        {type_id, kind, cpu, mem, stor, capacity,
                         jobs, cost, util_pct,
                         wait_time, run_time, turnaround, bounded_slowdown}
                      util_pct is None for elastic (unlimited) partitions.
      system        — {f1, f2, onprem_jobs, cloud_jobs, onprem_cost, cloud_cost,
                       onprem_util_pct}

    Bounded slowdown: BS(j) = max(1, turnaround / max(exec_time, 1)).
    On-prem utilisation: (sum of p_occ slots on on-prem) / (total on-prem
    capacity × horizon) × 100.
    Partition utilisation: same formula restricted to that partition's capacity.
    """
    n_total = len(inst.jobs)
    empty   = {"avg": 0.0, "min": 0.0, "max": 0.0, "total": 0.0}

    empty_partition = [
        {
            "type_id": m.id, "kind": m.kind,
            "cpu": m.cpu, "mem": m.mem, "stor": m.stor,
            "capacity": m.capacity,
            "jobs": 0, "cost": 0.0,
            "util_pct": 0.0 if m.capacity is not None else None,
            "wait_time": empty, "run_time": empty,
            "turnaround": empty, "bounded_slowdown": empty,
        }
        for m in inst.instance_types
    ]

    if not sol.assignment:
        return {
            "n_scheduled": 0, "n_total": n_total,
            "wait_time": empty, "run_time": empty,
            "turnaround": empty, "bounded_slowdown": empty,
            "per_job": [],
            "per_partition": empty_partition,
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

    # Per-partition accumulators keyed by type_id
    part_acc: dict[int, dict] = {
        m.id: {"jobs": 0, "cost": 0.0, "slots": 0.0,
                "wait": [], "run": [], "turnaround": [], "bsd": []}
        for m in inst.instance_types
    }

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

        pa = part_acc[mid]
        pa["jobs"]  += 1
        pa["cost"]  += cost
        pa["slots"] += runtime
        pa["wait"].append(wait)
        pa["run"].append(runtime)
        pa["turnaround"].append(turnaround)
        pa["bsd"].append(bsd)

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

    # Build per-partition stats
    per_partition: list[dict] = []
    for m in inst.instance_types:
        pa = part_acc[m.id]
        if m.capacity is not None:
            cap_slots = m.capacity * inst.horizon
            rj_slots  = sum(v for (tid, _t), v in inst.occupied.items() if tid == m.id)
            util_pct: float | None = (
                100.0 * (pa["slots"] + rj_slots) / cap_slots if cap_slots > 0 else 0.0
            )
        else:
            util_pct = None  # elastic cloud partition — utilisation is undefined
        per_partition.append({
            "type_id":        m.id,
            "kind":           m.kind,
            "cpu":            m.cpu,
            "mem":            m.mem,
            "stor":           m.stor,
            "capacity":       m.capacity,
            "jobs":           pa["jobs"],
            "cost":           pa["cost"],
            "util_pct":       util_pct,
            "wait_time":      _stat_block(pa["wait"]),
            "run_time":       _stat_block(pa["run"]),
            "turnaround":     _stat_block(pa["turnaround"]),
            "bounded_slowdown": _stat_block(pa["bsd"]),
        })

    return {
        "n_scheduled":    len(sol.assignment),
        "n_total":        n_total,
        "wait_time":        _stat_block(wait_times),
        "run_time":         _stat_block(run_times),
        "turnaround":       _stat_block(turnarounds),
        "bounded_slowdown": _stat_block(bslowdowns),
        "per_job":          per_job,
        "per_partition":    per_partition,
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


def _partition_lines(per_partition: list[dict], slot_secs: float = 1.0) -> list[str]:
    """Per-partition summary block: one entry per InstanceType (partition)."""

    def _prow(label: str, blk: dict, unit: str = "", total: bool = True) -> str:
        lbl = f"{label} {unit}".strip()
        tot = f"{blk['total'] * slot_secs:>12.1f}" if total else f"{'—':>12}"
        return (
            f"    {lbl:<20}"
            f" {blk['avg'] * slot_secs:>11.1f}"
            f" {blk['min'] * slot_secs:>11.1f}"
            f" {blk['max'] * slot_secs:>11.1f}"
            f" {tot}"
        )

    lines: list[str] = []
    for pp in per_partition:
        cap  = pp["capacity"] if pp["capacity"] is not None else "unlimited"
        util = f"{pp['util_pct']:.1f} %" if pp["util_pct"] is not None else "N/A (elastic)"
        lines.append(
            f"  Partition {pp['type_id']}  [{pp['kind']}]"
            f"  cpu={pp['cpu']}  mem={pp['mem']}  stor={pp['stor']}"
            f"  cap={cap}"
        )
        lines.append(
            f"    jobs: {pp['jobs']:>4}  cost: {pp['cost']:>12.2f} $"
            f"  utilisation: {util}"
        )
        if pp["jobs"] > 0:
            lines.append(
                f"    {'Metric':<20}"
                f" {'Avg':>11} {'Min':>11} {'Max':>11} {'Total':>12}"
            )
            lines.append(f"    {'─'*57}")
            lines += [
                _prow("Wait time",        pp["wait_time"],        "(s)"),
                _prow("Run time",         pp["run_time"],          "(s)"),
                _prow("Turnaround",       pp["turnaround"],        "(s)"),
                _prow("Bounded slowdown", pp["bounded_slowdown"],  "   ", total=False),
            ]
        lines.append("")
    return lines


def format_pareto_metrics(metrics: dict, ranked: list, inst: "Instance") -> str:
    """Format Pareto quality metrics as a human-readable block.

    *ranked* must be the same list passed to pareto_metrics, sorted by f1
    ascending, so that knee/minimax ranks are reported correctly.
    """
    slot_secs = inst.slot_size_seconds

    def _rank(sol) -> int | None:
        for i, s in enumerate(ranked):
            if s is sol:
                return i + 1
        return None

    n        = metrics["cardinality"]
    hv       = metrics["hypervolume"]
    igd      = metrics["igd"]
    r2       = metrics["r2"]
    spread   = metrics["spread"]
    coverage = metrics["coverage"]
    knee     = metrics["knee_point"]
    regret   = metrics["regret"]

    W = 50    # label column width
    lines = ["PARETO QUALITY METRICS", _rule("─"), ""]

    def _mrow(label: str, value: str, note: str = "") -> str:
        return f"  {label:<{W}} {value}  {note}"

    lines += [
        _mrow("Cardinality    (# non-dominated pts)",     f"{n}"),
        _mrow("Hypervolume    (norm, ref=(1.1,1.1))",
              f"{hv:.6f}", "↑ higher is better"),
        _mrow("IGD            (vs uniform diag ref)",
              f"{igd:.6f}", "↓ lower  is better"),
        _mrow("R2             (Tchebycheff, 101 weights)",
              f"{r2:.6f}", "↓ lower  is better"),
        _mrow("Spread (Δ)     (uniformity of spacing)",
              f"{spread:.6f}", "↓ lower  is better"),
        _mrow("Coverage       (HV / ref-box area)",
              f"{coverage * 100:.2f} %", "↑ higher is better"),
        "",
    ]

    if knee is not None and knee.f1 is not None:
        rk = _rank(knee)
        tag = f"  (point #{rk})" if rk is not None else ""
        lines.append(
            f"  Knee point     (max ⊥ dist from extreme line)"
            f"  f1={knee.f1 * slot_secs:.1f} s   f2={knee.f2:.4f} ${tag}"
        )

    mm = regret.get("minimax_solution")
    if mm is not None and mm.f1 is not None:
        rk = _rank(mm)
        tag = f"  (point #{rk})" if rk is not None else ""
        lines += [
            f"  Min regret     (minimax Chebyshev, norm.)"
            f"  {regret['min']:.4f}  →  f1={mm.f1 * slot_secs:.1f} s   f2={mm.f2:.4f} ${tag}",
            f"  Max regret     (worst-case point, norm.)"
            f"  {regret['max']:.4f}",
        ]

    lines.append("")
    return "\n".join(lines)


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
    lines += ["", "PARTITION STATISTICS", _rule("─")]
    lines += _partition_lines(st["per_partition"], slot_secs)

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

    # Running jobs: use negative IDs to avoid collision with new job IDs.
    # They run from t=0 until rj.end (clipped to horizon).
    _RJ_ID_OFFSET = 10_000_000
    rj_by_type: dict[int, list[tuple[int, int, int]]] = {m.id: [] for m in inst.instance_types}
    for rj in inst.running_jobs:
        if rj.type_id in rj_by_type:
            end = min(rj.end, inst.horizon)
            rj_by_type[rj.type_id].append((0, end, -(rj.id + _RJ_ID_OFFSET)))

    machine_ids = [m.id for m in inst.instance_types]

    # Lane assignments per machine — running jobs are included so new jobs don't overlap them.
    lanes_for: dict[int, list[tuple[int, int, int, int]]] = {}
    n_lanes:   dict[int, int] = {}
    for mid in machine_ids:
        combined, nl = _assign_lanes(rj_by_type[mid] + by_type[mid])
        lanes_for[mid] = combined
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

    _running_legend_added = False
    for mid in machine_ids:
        base = machine_base[mid]
        for t_start, t_end, jid, lane in lanes_for[mid]:
            bar_center = base + (lane + 0.5) * sub_h
            width      = t_end - t_start
            is_running = jid < 0
            if is_running:
                label_kw = {"label": "Running (busy)"} if not _running_legend_added else {}
                ax.barh(
                    bar_center, width, left=t_start,
                    height=sub_h * 0.85,
                    color="red", edgecolor="darkred", linewidth=0.5,
                    alpha=0.65, hatch="//", **label_kw,
                )
                _running_legend_added = True
                if show_labels:
                    real_id = -(jid + _RJ_ID_OFFSET)
                    ax.text(
                        t_start + width / 2, bar_center, f"r{real_id}",
                        ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold",
                    )
            else:
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

    has_running = any(rj_by_type[mid] for mid in machine_ids)
    if n_jobs <= _LEGEND_JOB_LIMIT:
        ncols = max(1, math.ceil(n_jobs / 8))
        legend_patches = [
            mpatches.Patch(color=colour_of[jid], label=f"Job {jid}")
            for jid in job_ids
        ]
        if has_running:
            legend_patches.insert(
                0,
                mpatches.Patch(facecolor="red", edgecolor="darkred",
                               alpha=0.65, hatch="//", label="Running (busy)"),
            )
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
    lines += ["", "PARTITION STATISTICS", thin]
    lines += _partition_lines(st["per_partition"], slot_secs)

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


# ──────────────────────────────────────────────────────────────────────────────
# Space-time network graph
# ──────────────────────────────────────────────────────────────────────────────

def save_spacetime_graph(
    inst: Instance,
    sol: Solution | None = None,
    path: str | Path = "spacetime_graph.png",
    *,
    title: str = "Space-Time Network",
    show_feasible_arcs: bool = False,
    show_boundary_flow: bool = False,
) -> list[Path]:
    """Save one space-time network PNG per instance type, all in the same folder.

    Files are named ``<stem>_type{mid}<suffix>`` next to *path*.
    Each file shows one machine type with:
      - Black dots at every integer time step t = 0 … H (network nodes),
        each labelled with its time index T directly on the node.
      - Silver horizontal arrows for idle-capacity arcs (y[m,t] flow).
      - Coloured arched arrows for job assignment arcs x[j,m,t]:
          bold + labelled when selected in *sol*;
          faint when *show_feasible_arcs* is True (all feasible arcs shown).
      - ``N=`` source/sink labels at t=0 and t=H when *show_boundary_flow* is True.

    Returns an empty list when H > _MAX_GRAPH_HORIZON (too dense to render legibly).
    """
    H = inst.horizon
    if H > _MAX_GRAPH_HORIZON:
        return []

    from matplotlib.patches import FancyArrowPatch

    base      = Path(path)
    stem      = base.stem
    suffix    = base.suffix
    out_dir   = base.parent

    job_list  = sorted(inst.jobs, key=lambda j: j.id)
    n_jobs    = len(job_list)

    colours   = plt.colormaps["tab10"].resampled(max(n_jobs, 1))
    colour_of = {j.id: colours(i) for i, j in enumerate(job_list)}

    selected: set[tuple[int, int, int]] = set()
    if sol and sol.assignment:
        for jid, (mid, t_start) in sol.assignment.items():
            selected.add((jid, mid, t_start))

    saved: list[Path] = []

    for itype in inst.instance_types:
        mid = itype.id

        # ── idle-flow weights: y[m,t] = capacity − jobs running at t ──────────
        cap_int = itype.capacity  # None for cloud (infinite)
        if cap_int is not None:
            jobs_at: list[int] = [0] * (H + 1)
            if sol and sol.assignment:
                for jid, (m, ts) in sol.assignment.items():
                    if m == mid:
                        p_j = inst.p_occ[jid, mid]
                        for tt in range(ts, min(ts + p_j, H + 1)):
                            jobs_at[tt] += 1
            for rj in inst.running_jobs:
                if rj.type_id == mid:
                    for tt in range(0, min(rj.end, H + 1)):
                        jobs_at[tt] += 1
            idle_flow: list[int | None] = [max(0, cap_int - jobs_at[t]) for t in range(H + 1)]
        else:
            idle_flow = [None] * (H + 1)  # infinite capacity — don't label

        fig_w = max(14.0, H * 0.35)
        fig, ax = plt.subplots(1, 1, figsize=(fig_w, 3.6))

        # ── nodes (labelled with time index) ─────────────────────────────────
        ax.scatter(range(H + 1), [0.0] * (H + 1), s=40, color="black", zorder=5)
        for t in range(H + 1):
            ax.text(t, -0.30, str(t), ha="center", va="top",
                    fontsize=6, color="black", zorder=6)

        # ── idle-capacity arcs (horizontal) ───────────────────────────────────
        for t in range(H):
            ax.annotate(
                "", xy=(t + 1, 0.0), xytext=(t, 0.0),
                arrowprops=dict(arrowstyle="-|>", color="silver", lw=1.2),
            )
            w = idle_flow[t]
            if w is not None:
                ax.text(
                    t + 0.5, -0.12, str(w),
                    ha="center", va="top", fontsize=6,
                    color="gray", zorder=6,
                )

        # ── job arcs ──────────────────────────────────────────────────────────
        arc_entries: list[tuple[int, int, int, bool]] = []  # (jid, t_start, p_occ, is_selected)
        for j in job_list:
            if mid not in inst.F[j.id]:
                continue
            p = inst.p_occ[j.id, mid]
            for t in inst.T[j.id, mid]:
                is_sel = (j.id, mid, t) in selected
                if not show_feasible_arcs and sol is not None and not is_sel:
                    continue
                arc_entries.append((j.id, t, p, is_sel))

        # Assign lanes (greedy interval colouring) to avoid arc overlap.
        sorted_arcs = sorted(arc_entries, key=lambda a: (a[1], a[1] + a[2]))
        lane_ends: list[int] = []
        arcs_with_lanes: list[tuple[int, int, int, bool, int]] = []
        for jid, t_start, p, is_sel in sorted_arcs:
            t_end = t_start + p
            placed = False
            for li, le in enumerate(lane_ends):
                if t_start >= le:
                    lane_ends[li] = t_end
                    arcs_with_lanes.append((jid, t_start, p, is_sel, li))
                    placed = True
                    break
            if not placed:
                arcs_with_lanes.append((jid, t_start, p, is_sel, len(lane_ends)))
                lane_ends.append(t_end)

        max_lane = max((lane for *_, lane in arcs_with_lanes), default=0)

        for jid, t_start, p, is_sel, lane in arcs_with_lanes:
            c   = colour_of[jid]
            lw  = 2.2 if is_sel else 0.8
            alp = 1.0 if is_sel else 0.22

            target_peak = 0.4 + lane * 0.3
            rad = -2.0 * target_peak / max(p, 1)

            patch = FancyArrowPatch(
                (t_start, 0.0), (t_start + p, 0.0),
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>",
                color=c, lw=lw, alpha=alp, zorder=4,
                transform=ax.transData,
            )
            ax.add_patch(patch)

            if is_sel and n_jobs <= _LABEL_JOB_LIMIT:
                label_x = t_start + p / 2
                label_y = target_peak + 0.05
                ax.text(
                    label_x, label_y, f"j{jid} [p={p}]",
                    ha="center", va="bottom", fontsize=7,
                    color=c, fontweight="bold",
                )
            elif not is_sel and n_jobs <= _LABEL_JOB_LIMIT:
                # Feasible-but-unselected arcs: show duration as faint weight
                label_x = t_start + p / 2
                label_y = target_peak + 0.02
                ax.text(
                    label_x, label_y, str(p),
                    ha="center", va="bottom", fontsize=5,
                    color=c, alpha=0.4,
                )

        # ── boundary labels (source / sink flow) ──────────────────────────────
        cap     = itype.capacity if itype.capacity is not None else "∞"
        if show_boundary_flow:
            n_running = inst.n_running.get(mid, 0)
            n_avail   = (itype.capacity or n_jobs) - n_running
            # For unlimited capacity N_end must equal N_start (no net consumption).
            right_n   = n_avail if itype.capacity is None else cap
            ax.text(-0.8, 0.0, f"N={n_avail}", ha="right", va="center",
                    fontsize=8, color="dimgray")
            ax.text(H + 0.8, 0.0, f"N={right_n}", ha="left", va="center",
                    fontsize=8, color="dimgray")

        y_top = max(0.6, 0.4 + max_lane * 0.3 + 0.45)
        ax.set_xlim(-1.5, H + 1.5)
        ax.set_ylim(-0.55, y_top)
        ax.axhline(0, color="black", lw=0.5, zorder=1)
        ax.yaxis.set_visible(False)
        ax.xaxis.set_visible(False)
        for spine in ("top", "left", "right", "bottom"):
            ax.spines[spine].set_visible(False)
        ax.grid(axis="x", linestyle=":", alpha=0.3)

        # ── legend ────────────────────────────────────────────────────────────
        if n_jobs <= _LEGEND_JOB_LIMIT:
            ncols = max(1, math.ceil(n_jobs / 8))
            handles = [
                mpatches.Patch(color=colour_of[j.id], label=f"Job {j.id}")
                for j in job_list
            ]
            fig.legend(handles=handles, ncols=ncols, loc="lower center",
                       fontsize=8, bbox_to_anchor=(0.5, 0.0))
            fig.subplots_adjust(bottom=0.06 + 0.025 * math.ceil(n_jobs / ncols))

        cap_str = f"cap={cap}"
        fig.suptitle(
            f"{title} — Type {mid}  [{itype.kind}]  perf={itype.perf}  {cap_str}",
            fontsize=11, fontweight="bold",
        )
        fig.tight_layout(rect=(0, 0.04, 1, 1))

        out_path = out_dir / f"{stem}_type{mid}{suffix}"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        out_pdf = out_path.with_suffix(".pdf")
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)
        saved.append(out_pdf)

    saved.extend(
        _save_spacetime_graph_gv(
            inst, sol, base,
            title=title,
            show_feasible_arcs=show_feasible_arcs,
            show_boundary_flow=show_boundary_flow,
        )
    )
    return saved


# ──────────────────────────────────────────────────────────────────────────────
# Space-time network — Graphviz / DOT renderer
# ──────────────────────────────────────────────────────────────────────────────

def _rgba_to_hex(rgba) -> str:
    r, g, b, _ = rgba
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def _save_spacetime_graph_gv(
    inst: Instance,
    sol: Solution | None,
    base_path: Path,
    *,
    title: str,
    show_feasible_arcs: bool,
    show_boundary_flow: bool,
) -> list[Path]:
    """Render one Graphviz SVG space-time graph per instance type.

    Files are saved as ``<stem>_type{mid}_gv.svg`` next to *base_path*.
    The graph shares the same arc-visibility rules as the matplotlib version:
    selected arcs are bold and coloured; feasible-only arcs are dashed and
    faint, shown only when *show_feasible_arcs* is True.
    """
    import graphviz

    H       = inst.horizon
    stem    = base_path.stem
    out_dir = base_path.parent

    job_list  = sorted(inst.jobs, key=lambda j: j.id)
    n_jobs    = len(job_list)

    colours   = plt.colormaps["tab10"].resampled(max(n_jobs, 1))
    colour_of = {j.id: colours(i) for i, j in enumerate(job_list)}

    selected: set[tuple[int, int, int]] = set()
    if sol and sol.assignment:
        for jid, (mid_s, t_start) in sol.assignment.items():
            selected.add((jid, mid_s, t_start))

    saved: list[Path] = []

    for itype in inst.instance_types:
        mid     = itype.id
        cap_int = itype.capacity
        cap_str = str(cap_int) if cap_int is not None else "∞"

        # ── idle-flow weights (same logic as matplotlib renderer) ─────────────
        if cap_int is not None:
            jobs_at: list[int] = [0] * (H + 1)
            if sol and sol.assignment:
                for jid, (m, ts) in sol.assignment.items():
                    if m == mid:
                        p_j = inst.p_occ[jid, mid]
                        for tt in range(ts, min(ts + p_j, H + 1)):
                            jobs_at[tt] += 1
            for rj in inst.running_jobs:
                if rj.type_id == mid:
                    for tt in range(0, min(rj.end, H + 1)):
                        jobs_at[tt] += 1
            idle_flow: list[int | None] = [max(0, cap_int - jobs_at[t]) for t in range(H + 1)]
        else:
            idle_flow = [None] * (H + 1)

        # ── Digraph setup ─────────────────────────────────────────────────────
        gv = graphviz.Digraph(
            graph_attr={
                "rankdir":  "LR",
                "label":    (f"{title} — Type {mid}  [{itype.kind}]"
                             f"  perf={itype.perf}  cap={cap_str}"),
                "labelloc": "t",
                "fontsize": "13",
                "splines":  "spline",
                "nodesep":  "0.6",
                "ranksep":  "0.4",
                "bgcolor":  "white",
            },
            node_attr={
                "shape":     "circle",
                "style":     "filled",
                "fillcolor": "black",
                "fontcolor": "white",
                "fontsize":  "9",
                "width":     "0.28",
                "height":    "0.28",
                "fixedsize": "true",
            },
            edge_attr={"fontsize": "7"},
        )

        # ── nodes ─────────────────────────────────────────────────────────────
        for t in range(H + 1):
            gv.node(f"t{t}", label=str(t))

        # ── idle-capacity arcs ────────────────────────────────────────────────
        for t in range(H):
            w   = idle_flow[t]
            lbl = str(w) if w is not None else ""
            gv.edge(f"t{t}", f"t{t + 1}", label=lbl,
                    color="silver", fontcolor="gray",
                    arrowsize="0.6", penwidth="1.2")

        # ── job arcs ──────────────────────────────────────────────────────────
        for j in job_list:
            if mid not in inst.F[j.id]:
                continue
            p = inst.p_occ[j.id, mid]
            for t in inst.T[j.id, mid]:
                is_sel = (j.id, mid, t) in selected
                if not show_feasible_arcs and sol is not None and not is_sel:
                    continue
                c_hex = _rgba_to_hex(colour_of[j.id])
                if is_sel:
                    gv.edge(f"t{t}", f"t{t + p}",
                            label=f"j{j.id} [p={p}]",
                            color=c_hex, fontcolor=c_hex,
                            penwidth="2.5",
                            constraint="false", weight="0")
                else:
                    faint = f"{c_hex}38"  # ~22 % opacity (graphviz #rrggbbaa)
                    gv.edge(f"t{t}", f"t{t + p}",
                            label=str(p),
                            color=faint, fontcolor=faint,
                            penwidth="0.8", style="dashed",
                            constraint="false", weight="0")

        # ── boundary flow labels ──────────────────────────────────────────────
        if show_boundary_flow:
            n_running = inst.n_running.get(mid, 0)
            cap       = itype.capacity if itype.capacity is not None else "∞"
            n_avail   = (itype.capacity or n_jobs) - n_running
            right_n   = n_avail if itype.capacity is None else cap
            gv.node("t0",     label="0",    xlabel=f"N={n_avail}")
            gv.node(f"t{H}", label=str(H), xlabel=f"N={right_n}")

        # ── render ────────────────────────────────────────────────────────────
        out_stem = str(out_dir / f"{stem}_type{mid}_gv")
        gv.render(out_stem, format="svg", cleanup=False)
        gv.render(out_stem, format="pdf", cleanup=True)
        saved.append(Path(f"{out_stem}.svg"))
        saved.append(Path(f"{out_stem}.pdf"))

    return saved


# ──────────────────────────────────────────────────────────────────────────────
# Feasibility / assignment bipartite graph
# ──────────────────────────────────────────────────────────────────────────────

def save_feasibility_graph(
    inst: Instance,
    sol: Solution | None = None,
    path: str | Path = "feasibility_graph.png",
    *,
    title: str = "Job–Machine Feasibility Graph",
) -> Path:
    """Save a bipartite graph of jobs vs. machine types as PNG.

    Left nodes  — jobs (coloured circles).
    Right nodes — machine types (squares; blue=on-prem, orange=cloud).
    Light gray edges — feasible (j, m) pairs where m ∈ F_j.
    Bold coloured edges — assigned pair from *sol* (same colour as the job node).
    """
    jobs  = sorted(inst.jobs,           key=lambda j: j.id)
    types = sorted(inst.instance_types, key=lambda m: m.id)
    n_j   = len(jobs)
    n_m   = len(types)

    colours   = plt.colormaps["tab10"].resampled(max(n_j, 1))
    colour_of = {j.id: colours(i) for i, j in enumerate(jobs)}

    assigned: dict[int, int] = {}
    if sol and sol.assignment:
        assigned = {jid: mid for jid, (mid, _) in sol.assignment.items()}

    # Evenly spread both columns over the same vertical range [0, n_j-1].
    job_y: dict[int, float] = {j.id: float(n_j - 1 - i) for i, j in enumerate(jobs)}
    if n_m == 1:
        type_y: dict[int, float] = {types[0].id: (n_j - 1) / 2.0}
    else:
        step = (n_j - 1) / (n_m - 1)
        type_y = {m.id: float(n_j - 1 - i * step) for i, m in enumerate(types)}

    fig_h = max(5.0, max(n_j, n_m) * 0.65 + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_h))

    kind_colour = {"on-prem": "#1976D2", "cloud": "#F57C00"}

    # ── edges ─────────────────────────────────────────────────────────────────
    for j in jobs:
        for mid in inst.F[j.id]:
            is_asgn = assigned.get(j.id) == mid
            ax.plot(
                [0.0, 1.0], [job_y[j.id], type_y[mid]],
                color=colour_of[j.id] if is_asgn else "lightgray",
                lw=2.2 if is_asgn else 0.8,
                alpha=1.0 if is_asgn else 0.6,
                zorder=2,
            )

    # ── job nodes (left) ──────────────────────────────────────────────────────
    for j in jobs:
        ax.scatter(0.0, job_y[j.id], s=260, color=colour_of[j.id],
                   zorder=5, edgecolors="black", lw=0.8)
        ax.text(-0.04, job_y[j.id], f"j{j.id}",
                ha="right", va="center", fontsize=9)

    # ── machine type nodes (right) ─────────────────────────────────────────────
    for m in types:
        c   = kind_colour.get(m.kind, "gray")
        cap = m.capacity if m.capacity is not None else "∞"
        ax.scatter(1.0, type_y[m.id], s=260, color=c, marker="s",
                   zorder=5, edgecolors="black", lw=0.8)
        ax.text(
            1.04, type_y[m.id],
            f"Type {m.id}  ({m.kind}, cap={cap})\nperf={m.perf}",
            ha="left", va="center", fontsize=8,
        )

    # ── column labels ─────────────────────────────────────────────────────────
    y_top = float(n_j - 1)
    ax.text(0.0, y_top + 0.5, "Jobs",          ha="center", fontsize=10, fontweight="bold")
    ax.text(1.0, y_top + 0.5, "Instance Types", ha="center", fontsize=10, fontweight="bold")

    # ── legend ────────────────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(color="#1976D2", label="On-prem"),
        mpatches.Patch(color="#F57C00", label="Cloud"),
        plt.Line2D([0], [0], color="lightgray", lw=0.8, label="Feasible"),
    ]
    if assigned:
        handles.append(plt.Line2D([0], [0], color="black", lw=2.2, label="Assigned"))
    ax.legend(handles=handles, loc="lower right", fontsize=9)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlim(-0.5, 1.9)
    ax.set_ylim(-0.8, y_top + 1.0)
    ax.axis("off")

    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path
