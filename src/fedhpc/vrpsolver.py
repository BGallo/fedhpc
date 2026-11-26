"""VRPSolver (Pessoa, Sadykov, Uchoa & Vanderbeck 2020) as an external engine.

VRPSolver is a generic branch-cut-and-price solver (ng-routes, limited-memory
rank-1 cuts, bucket-graph labelling, path enumeration, strong branching). It is
the baseline reviewers ask a problem-specific BPC to beat, so this module states
our problem in its terms and runs it, for head-to-head comparisons with
:mod:`fedhpc.bpc`. See ``docs/vrpsolver_integration.md``.

It is **academic-use only** and is not shipped with the repo. Running it needs
Julia >= 1.11.5, CPLEX >= 12.9 and the BaPCod shared library (the path in
``BAPCOD_RCSP_LIB``); :func:`available` says whether this machine has them.
Exporting and checking the model need none of that.

The model is the BPC master of ``fedhpc.bpc`` on the same space-time networks:

- one VRPSolver graph per release group ``(m, r)`` of each path type, on the
  nodes ``r..H`` with wait arcs ``v -> v+1`` and a processing arc
  ``t -> t + p_jm`` per allowed start; its path multiplicity is
  ``(0, N_{m,r})``, so idle instances need no null column;
- one integer variable ``x`` per path placement ``(j, m, t)``, mapped to that
  arc in every group of type ``m``;
- master-only binaries for the placements on unbounded types (the static
  columns) and for the fixed-cost indicators ``z_m``;
- master rows: coverage, the optional caps on f1 and f2, and the per-pair
  fixed-cost links ``sum_t x_jmt <= z_m``;
- one arc packing set per job (its arcs in all graphs), so ng-routes and rank-1
  cuts work per job; the ng distance matrix is the score of
  :func:`fedhpc.bpc._ensure_ng`.

- BPC's activation rows ``#paths of m <= N_m z_m`` for fixed-cost path types:
  each graph gets a start arc mapped to an integer ``u_m`` that counts its
  paths, and the master has ``u_m <= N_m z_m``. On the Rm instances these
  rows are what closes most of the root gap, so a comparison needs them.

Start sets are the exact reductions of :func:`fedhpc.bpc._prepare`, as in BPC.

The objective is scaled so its coefficients are integers when the objective
lies on a grid (:func:`fedhpc.bpc._objective_step`), which lets VRPSolver round
its bounds. A schedule's value is ``scale * vrp_objective + offset``.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from . import bpc
from .data import Instance
from .model import Solution

FORMAT = "fedhpc-vrpsolver/1"
JULIA_PROJECT = Path(__file__).resolve().parents[2] / "vrpsolver"

__all__ = ["export_model", "solve_exported_mip", "available", "solve_vrpsolver", "FORMAT"]


def _integral(values: list[float]) -> bool:
    return all(abs(v - round(v)) <= 1e-6 * max(1.0, abs(v)) for v in values)


def export_model(
    inst: Instance,
    weight1: float,
    weight2: float,
    bound1: float | None = None,
    bound2: float | None = None,
    *,
    prep: "bpc._Prep | None" = None,
    reduce: bool = True,
    ng_size: int = 8,
    cutoff: float | None = None,
    seed_schedules: list[dict[int, tuple[int, int]]] | None = None,
) -> dict:
    """The VRPSolver model of ``min weight1·f1 + weight2·f2`` s.t. the optional
    caps, as a JSON-ready dict (format :data:`FORMAT`).

    ``cutoff`` — an upper bound on the objective (in original units). Without
    it, the best of ``seed_schedules`` (default: :func:`bpc.heuristic_schedules`)
    within the caps is used, if any. Job and type ids are the instance's; job
    *indices* (``j``) are 0-based positions in ``jobs``.
    """
    inst.require_unit_demand("VRPSolver export")
    prep = prep or bpc._prepare(inst, reduce)
    H = inst.horizon
    jobs = list(inst.jobs)
    jpos = {j.id: i for i, j in enumerate(jobs)}
    offset = -weight1 * sum(j.weight * j.arrival for j in jobs)

    def c1(jid: int, mid: int, t: int) -> float:
        j = jobs[jpos[jid]]
        return j.weight * (t + inst.p_occ[jid, mid] - j.arrival)

    # objective without the constant -w1·Σ w_j a_j, then scaled to integers
    raw: list[float] = []
    arcs_by_type: dict[int, list[tuple[int, int, int, float, float, float]]] = {}
    for m in prep.path_types:
        rows = []
        for arc in prep.arcs_by_type[m]:
            for t in sorted(arc.starts):
                if 0 <= t and t + arc.p_occ <= H:
                    o = weight1 * arc.weight * (t + arc.p_occ) + weight2 * arc.raw_cost2
                    raw.append(o)
                    rows.append((jpos[arc.job_id], t, arc.p_occ, o, c1(arc.job_id, m, t),
                                 arc.raw_cost2))
        arcs_by_type[m] = rows
    statics = []
    for col in prep.static_columns:
        (jid, t), = col.jobs
        j = jobs[jpos[jid]]
        o = weight1 * j.weight * (t + inst.p_occ[jid, col.type_id]) + weight2 * col.cost2
        raw.append(o)
        statics.append((jpos[jid], col.type_id, t, o, col.cost1, col.cost2))
    fixed = [(m.id, m.fixed_cost, weight2 * m.fixed_cost)
             for m in inst.instance_types if m.fixed_cost]
    raw += [o for *_r, o in fixed]

    step = bpc._objective_step(inst, weight1, weight2)
    scale = step if step > 0 and _integral([o / step for o in raw]) else 1.0
    integral = scale != 1.0 or _integral(raw)

    def sc(o: float) -> float:
        v = o / scale
        return float(round(v)) if integral else v

    # ng distances: window centre and duration on each path type (bpc._ensure_ng)
    in_paths = sorted({row[0] for rows in arcs_by_type.values() for row in rows})
    ps_of = {j: k for k, j in enumerate(in_paths)}
    dist = {}
    for m, rows in arcs_by_type.items():
        key: dict[int, tuple[float, int]] = {}
        lo: dict[int, int] = {}
        hi: dict[int, int] = {}
        for j, t, p, *_ in rows:
            lo[j] = min(lo.get(j, t), t)
            hi[j] = max(hi.get(j, t), t)
            key[j] = (0.0, p)
        for j in key:
            key[j] = ((lo[j] + hi[j] + key[j][1]) / 2.0, key[j][1])
        far = 4.0 * H + 1.0
        dist[m] = [[0.0 if a == b else
                    (abs(key[a][0] - key[b][0]) + abs(key[a][1] - key[b][1])
                     if a in key and b in key else far)
                    for b in in_paths] for a in in_paths]

    if cutoff is None:
        best = math.inf
        for a in (seed_schedules if seed_schedules is not None
                  else bpc.heuristic_schedules(inst)):
            v = bpc._schedule_objective(inst, a, weight1, weight2, bound1, bound2)
            if v is not None:
                best = min(best, v)
        cutoff = best if math.isfinite(best) else None

    return {
        "format": FORMAT,
        "name": getattr(inst, "name", "") or "",
        "horizon": H,
        "jobs": [{"id": j.id, "weight": j.weight, "arrival": j.arrival} for j in jobs],
        "objective": {"weight1": weight1, "weight2": weight2, "scale": scale,
                      "offset": offset, "integral": integral},
        "bound1": bound1,
        "bound2": bound2,
        "path_types": [
            {"type": m,
             "releases": sorted([r, n] for r, n in prep.releases[m].items()),
             "arcs": [[j, t, p, sc(o), a1, a2] for j, t, p, o, a1, a2 in arcs_by_type[m]],
             "dist": dist[m]}
            for m in prep.path_types
        ],
        "statics": [[j, m, t, sc(o), a1, a2] for j, m, t, o, a1, a2 in statics],
        "fixed": [{"type": m, "cost": U, "obj": sc(o)} for m, U, o in fixed],
        "packing_jobs": in_paths,
        "ng_size": ng_size,
        # strictly above the incumbent, so VRPSolver may return a solution of
        # that value (its cutoff prunes nodes whose bound reaches it)
        "cutoff": None if cutoff is None else (cutoff - offset) / scale + 1e-6 * max(
            1.0, abs(cutoff)) + (0.5 if integral else 0.0),
    }


def model_value(model: dict, placements: list[tuple[int, int, int]],
                types_used: set[int]) -> float:
    """Objective of a schedule (``(job index, type, start)``) in original units."""
    obj = model["objective"]
    by = {}
    for pt in model["path_types"]:
        for j, t, p, o, *_ in pt["arcs"]:
            by[j, pt["type"], t] = o
    for j, m, t, o, *_ in model["statics"]:
        by[j, m, t] = o
    v = sum(by[k] for k in placements)
    v += sum(f["obj"] for f in model["fixed"] if f["type"] in types_used)
    return obj["scale"] * v + obj["offset"]


def solve_exported_mip(model: dict, time_limit: float | None = None) -> dict:
    """Solve an exported model as the equivalent compact time-indexed MIP with
    HiGHS: same placements, objective, caps and fixed-cost links, and each
    graph's flow replaced by its capacity rows (busy placements of type ``m``
    in slot ``u`` <= instances released by ``u``; Proposition 5.1 of
    ``docs/branch_and_price_formulation.tex``). A check of the export's
    semantics that needs no VRPSolver.

    Returns ``{"status", "objective", "assignment"}`` with the objective in
    original units and the assignment keyed by job id.
    """
    import highspy

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    if time_limit is not None:
        h.setOptionValue("time_limit", float(time_limit))
    inf = highspy.kHighsInf
    cols = []          # (job index, type, start, c1, c2)
    cost = []
    for pt in model["path_types"]:
        for j, t, p, o, a1, a2 in pt["arcs"]:
            cols.append((j, pt["type"], t, p, a1, a2))
            cost.append(o)
    for j, m, t, o, a1, a2 in model["statics"]:
        cols.append((j, m, t, None, a1, a2))
        cost.append(o)
    nx = len(cols)
    zidx = {f["type"]: nx + k for k, f in enumerate(model["fixed"])}
    cost += [f["obj"] for f in model["fixed"]]
    n = len(cost)
    for c in cost:
        h.addVar(0.0, 1.0)
    h.changeColsCost(n, list(range(n)), cost)
    h.changeColsIntegrality(n, list(range(n)), [highspy.HighsVarType.kInteger] * n)

    def row(lo, hi, idx, val):
        if idx:
            h.addRow(lo, hi, len(idx), idx, val)

    njobs = len(model["jobs"])
    by_job = [[] for _ in range(njobs)]
    for k, (j, *_r) in enumerate(cols):
        by_job[j].append(k)
    for j in range(njobs):
        row(1.0, 1.0, by_job[j], [1.0] * len(by_job[j]))
    if model["bound1"] is not None:
        row(-inf, model["bound1"], list(range(nx)), [c[4] for c in cols])
    if model["bound2"] is not None:
        idx = list(range(nx)) + list(zidx.values())
        val = [c[5] for c in cols] + [f["cost"] for f in model["fixed"]]
        row(-inf, model["bound2"], idx, val)
    for m, zk in zidx.items():
        per_job: dict[int, list[int]] = {}
        for k, c in enumerate(cols):
            if c[1] == m:
                per_job.setdefault(c[0], []).append(k)
        for ks in per_job.values():
            row(-inf, 0.0, ks + [zk], [1.0] * len(ks) + [-1.0])
    H = model["horizon"]
    for pt in model["path_types"]:
        m = pt["type"]
        cap = [0] * H
        for r, cnt in pt["releases"]:
            for u in range(max(r, 0), H):
                cap[u] += cnt
        busy: list[list[int]] = [[] for _ in range(H)]
        for k, c in enumerate(cols):
            if c[1] == m and c[3] is not None:
                for u in range(c[2], c[2] + c[3]):
                    busy[u].append(k)
        for u in range(H):
            row(-inf, float(cap[u]), busy[u], [1.0] * len(busy[u]))
    h.run()
    status = h.modelStatusToString(h.getModelStatus())
    if h.getInfo().primal_solution_status != 2:
        return {"status": status, "objective": None, "assignment": None}
    xs = h.getSolution().col_value
    obj = model["objective"]
    assignment = {}
    for k, (j, m, t, *_r) in enumerate(cols):
        if xs[k] > 0.5:
            assignment[model["jobs"][j]["id"]] = (m, t)
    value = obj["scale"] * h.getInfo().objective_function_value + obj["offset"]
    return {"status": status, "objective": value, "assignment": assignment}


# ---------------------------------------------------------------------------
# Running VRPSolver
# ---------------------------------------------------------------------------

def _julia_cmd() -> list[str]:
    return os.environ.get("FEDHPC_JULIA", "julia").split()


def available() -> tuple[bool, str]:
    """Whether VRPSolver can run here, with the reason when it cannot."""
    lib = os.environ.get("BAPCOD_RCSP_LIB")
    if not lib:
        return False, "BAPCOD_RCSP_LIB is not set (BaPCod shared library)"
    if not Path(lib).is_file():
        return False, f"BAPCOD_RCSP_LIB={lib} does not exist"
    if shutil.which(_julia_cmd()[0]) is None:
        return False, f"{_julia_cmd()[0]} not found (set FEDHPC_JULIA)"
    return True, ""


def run_julia(model: dict, *, dry_run: bool = False, time_limit: float | None = None,
              rank1_cuts: bool = True, workdir: str | None = None,
              julia_cmd: list[str] | None = None) -> dict:
    """Write ``model`` to JSON, run ``vrpsolver/run.jl`` on it and return its
    JSON result. ``dry_run`` builds the VRPSolver model and stops before
    calling BaPCod, so it needs neither CPLEX nor the BaPCod library."""
    tmp = workdir or tempfile.mkdtemp(prefix="fedhpc-vrp-")
    src, out = Path(tmp) / "model.json", Path(tmp) / "result.json"
    src.write_text(json.dumps(model))
    cmd = [*(julia_cmd or _julia_cmd()), f"--project={JULIA_PROJECT}",
           str(JULIA_PROJECT / "run.jl"), str(src), str(out)]
    if dry_run:
        cmd.append("--dry-run")
    if time_limit is not None:
        cmd += ["--time-limit", str(int(math.ceil(time_limit)))]
    if not rank1_cuts:
        cmd.append("--no-rank1")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out.exists():
        raise RuntimeError(f"VRPSolver run failed ({proc.returncode}):\n"
                           f"{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}")
    res = json.loads(out.read_text())
    res["log"] = proc.stdout
    return res


def solve_vrpsolver(inst: Instance, weight1: float, weight2: float,
                    bound1: float | None = None, bound2: float | None = None, *,
                    time_limit: float | None = None, rank1_cuts: bool = True,
                    stats: dict | None = None, **export_kw) -> Solution:
    """``min weight1·f1 + weight2·f2`` s.t. the caps, solved by VRPSolver.

    The schedule is checked with :meth:`Instance.check_schedule` and its
    objective recomputed from the instance. Status ``"optimal"`` when
    VRPSolver proved it, ``"feasible"`` when it stopped with a schedule,
    ``"infeasible"`` when it proved there is none within the cutoff (then the
    cutoff schedule, if any, is optimal: the caller passed it), else
    ``"unknown"``.
    """
    ok, why = available()
    if not ok:
        raise RuntimeError(f"VRPSolver is not available: {why}")
    t0 = time.perf_counter()
    model = export_model(inst, weight1, weight2, bound1, bound2, **export_kw)
    res = run_julia(model, time_limit=time_limit, rank1_cuts=rank1_cuts)
    if stats is not None:
        stats.update({k: v for k, v in res.items() if k != "log"})
        stats["wall_s"] = time.perf_counter() - t0
    if not res.get("found"):
        status = "infeasible" if res.get("status") == "Optimal" else "unknown"
        return Solution(status=status, objective=None, f1=None, f2=None,
                        assignment={}, completion={})
    jobs = model["jobs"]
    assignment = {jobs[j]["id"]: (m, t) for j, m, t in res["placements"]}
    inst.check_schedule(assignment)
    completion = {jid: t + inst.p_occ[jid, mid] for jid, (mid, t) in assignment.items()}
    f1, f2 = inst.f1_of(completion), inst.f2_of(assignment)
    return Solution(
        status="optimal" if res.get("status") == "Optimal" else "feasible",
        objective=weight1 * f1 + weight2 * f2, f1=f1, f2=f2,
        assignment=assignment, completion=completion,
        solve_time=time.perf_counter() - t0,
    )
