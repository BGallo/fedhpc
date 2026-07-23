"""Multi-objective evolutionary algorithms for FED-HPC (NSGA-II and MOEA/D).

These provide heuristic Pareto-frontier approximations that trade optimality
guarantees for much lower wall-clock time than exact MIP-based methods.

Both algorithms encode solutions as a direct assignment vector (one (m, t)
slot per job) and evaluate f1 / f2 identically to the MIP formulations.
Constraint violations (capacity, budget) are handled via constrained-dominance
ranking (NSGA-II) or a large Tchebycheff penalty (MOEA/D).

The returned ``list[Solution]`` is drop-in compatible with
``pareto.true_pareto_frontier`` and ``pareto.hybrid_frontier``.
Status is set to ``"heuristic"`` instead of ``"optimal"``.
"""
from __future__ import annotations

import sys

try:
    from . import _moea as _ext
    _HAS_EXT = True
except ImportError:
    _HAS_EXT = False

from .data import Instance
from .model import Solution


def _require_ext() -> None:
    if not _HAS_EXT:
        raise ImportError(
            "fedhpc._moea C++ extension is not built. "
            "Reinstall the package with a C++ compiler: uv pip install -e ."
        )


def _job_slots(inst: Instance) -> list[list[tuple[int, int, int, float, float]]]:
    """Flatten Instance into per-job slot tables for the C++ layer.

    Each slot is (type_id, start, p_occ, f1_contrib, cost).
    f1_contrib = start + p_occ - arrival is precomputed to avoid recomputing
    it on every evaluation inside the tight C++ inner loop.
    """
    slots: list[list[tuple[int, int, int, float, float]]] = []
    for j in inst.jobs:
        job_slots: list[tuple[int, int, int, float, float]] = []
        for m_id in inst.F[j.id]:
            p_occ = inst.p_occ[j.id, m_id]
            cost  = inst.c[j.id, m_id]
            for t in inst.T[j.id, m_id]:
                job_slots.append((m_id, t, p_occ, float(t + p_occ - j.arrival), cost))
        slots.append(job_slots)
    return slots


def _type_cap(inst: Instance) -> list[int]:
    """Capacity list indexed by type_id. -1 = unlimited."""
    max_id = max(m.id for m in inst.instance_types)
    cap = [-1] * (max_id + 1)
    for m in inst.instance_types:
        if m.capacity is not None:
            cap[m.id] = m.capacity
    return cap


def _init_occ(inst: Instance) -> list[tuple[int, int, int]]:
    """Running-job occupancy as (type_id, t, count) triples."""
    return [(m, t, cnt) for (m, t), cnt in inst.occupied.items()]


def _to_solutions(
    inst: Instance,
    raw: list[tuple[list[tuple[int, int]], float, float]],
) -> list[Solution]:
    out: list[Solution] = []
    for assignment_raw, f1, f2 in raw:
        assignment: dict[int, tuple[int, int]] = {}
        completion: dict[int, int] = {}
        for j, (m_id, t) in zip(inst.jobs, assignment_raw):
            assignment[j.id] = (m_id, t)
            completion[j.id] = t + inst.p_occ[j.id, m_id]
        out.append(Solution(
            status="heuristic",
            objective=None,
            f1=f1,
            f2=f2,
            assignment=assignment,
            completion=completion,
        ))
    return out


# ── Profile display ───────────────────────────────────────────────────────────

_W = 72  # report width

def _print_profile(prof: dict, *, algorithm: str, label: str = "") -> None:
    """Print a per-phase timing breakdown to stderr.

    *prof* is the dict returned by the C++ binding.  Algorithm-specific phase
    keys are detected by name so this function works for both NSGA-II and MOEA/D.
    """
    sep   = "─" * _W
    thin  = "─" * (_W - 4)
    n_gen = int(prof.get("n_gen", 0))
    total = prof.get("total_ms", 0.0)

    # Phases reported as (label, total_ms, per_gen_ms_or_None)
    if algorithm == "nsga2":
        phases: list[tuple[str, float, float | None]] = [
            ("Initial eval (seeds + random)",
             prof.get("init_eval_ms", 0.0), None),
            ("NDS + rank (parent pop)",
             prof.get("nds_total_ms", 0.0),
             prof.get("nds_avg_ms")),
            ("Crowding distance",
             prof.get("crowding_total_ms", 0.0),
             prof.get("crowding_avg_ms")),
            ("Offspring generate + eval  ← hot path",
             prof.get("offspring_total_ms", 0.0),
             prof.get("offspring_avg_ms")),
            ("Combine NDS + crowding + select",
             prof.get("combine_select_total_ms", 0.0),
             prof.get("combine_select_avg_ms")),
            ("Extract + Pareto filter",
             prof.get("extract_ms", 0.0), None),
        ]
    elif algorithm == "nsga3":
        phases = [
            ("Initial eval (seeds + random)",
             prof.get("init_eval_ms", 0.0), None),
            ("NDS + rank (parent pop, for mating)",
             prof.get("rank_nds_total_ms", 0.0),
             prof.get("rank_nds_avg_ms")),
            ("Offspring generate + eval  ← hot path",
             prof.get("offspring_total_ms", 0.0),
             prof.get("offspring_avg_ms")),
            ("Combine NDS + normalize + niching select",
             prof.get("combine_select_total_ms", 0.0),
             prof.get("combine_select_avg_ms")),
            ("Extract + Pareto filter",
             prof.get("extract_ms", 0.0), None),
        ]
    else:  # moead
        phases = [
            ("Initial eval (seeds + random)",
             prof.get("init_eval_ms", 0.0), None),
            ("Offspring + eval + ideal update  ← hot path",
             prof.get("offspring_total_ms", 0.0),
             prof.get("offspring_avg_ms")),
            ("Neighbourhood replacement (sequential)",
             prof.get("replacement_total_ms", 0.0),
             prof.get("replacement_avg_ms")),
            ("Extract + Pareto filter",
             prof.get("extract_ms", 0.0), None),
        ]

    hdr_tag = f" {label}" if label else ""
    print(f"\n{algorithm.upper()} Profile{hdr_tag}  [{n_gen} gen]", file=sys.stderr)
    print(sep, file=sys.stderr)

    col_w = 46
    print(
        f"  {'Phase':<{col_w}}  {'Total (ms)':>10}  {'Per-gen (ms)':>12}  {'Share':>6}",
        file=sys.stderr,
    )
    print(f"  {thin}", file=sys.stderr)

    for name, t_ms, per_ms in phases:
        share = 100.0 * t_ms / total if total > 0 else 0.0
        per_str = f"{per_ms:>12.3f}" if per_ms is not None else f"{'—':>12}"
        print(
            f"  {name:<{col_w}}  {t_ms:>10.2f}  {per_str}  {share:>5.1f} %",
            file=sys.stderr,
        )

    print(f"  {thin}", file=sys.stderr)
    print(
        f"  {'Total':<{col_w}}  {total:>10.2f}  {'—':>12}  {'100.0':>5} %",
        file=sys.stderr,
    )
    print(sep, file=sys.stderr)


# ── Public API ────────────────────────────────────────────────────────────────

def nsga2_frontier(
    inst: Instance,
    *,
    pop_size: int = 100,
    n_gen: int = 200,
    seed: int = 42,
    n_threads: int = 0,
    p_mut_start: float = -1.0,
    p_mut_end: float = -1.0,
    crossover_kind: int = 0,
    tourn_k: int = 2,
    profile: bool = False,
) -> list[Solution]:
    """Approximate the Pareto frontier via NSGA-II.

    Parameters
    ----------
    inst           : built fedhpc Instance.
    pop_size       : population size (individuals per generation).
    n_gen          : number of generations.
    seed           : RNG seed for reproducibility.
    n_threads      : OpenMP thread count; 0 = use all available cores.
    p_mut_start    : mutation-rate schedule start. ``<0`` (default) resolves to
                     the fixed formula rate ``2/n_jobs``.
    p_mut_end      : mutation-rate schedule end. ``<0`` (default) holds the
                     resolved start rate constant (no annealing). When both are
                     given, the rate is linearly annealed start→end across
                     generations.
    crossover_kind : 0 = two-point (default), 1 = uniform (per-gene coin flip).
    tourn_k        : mating-tournament size. 2 (default) = original binary
                     tournament; larger values increase selection pressure.
    profile        : if True, print a per-phase timing breakdown to stderr.

    Returns
    -------
    Feasible non-dominated solutions (status="heuristic"). Not guaranteed
    globally optimal — use MIP-based pareto.py for exact results.
    """
    _require_ext()
    raw, prof = _ext.nsga2(
        n_jobs         = len(inst.jobs),
        budget         = inst.budget,
        job_slots      = _job_slots(inst),
        type_cap       = _type_cap(inst),
        init_occ       = _init_occ(inst),
        pop_size       = pop_size,
        n_gen          = n_gen,
        seed           = seed,
        n_threads      = n_threads,
        p_mut_start    = p_mut_start,
        p_mut_end      = p_mut_end,
        crossover_kind = crossover_kind,
        tourn_k        = tourn_k,
    )
    if profile:
        _print_profile(prof, algorithm="nsga2",
                       label=f"pop={pop_size}  threads={n_threads or 'all'}")
    return _to_solutions(inst, raw)


def nsga3_frontier(
    inst: Instance,
    *,
    pop_size: int = 100,
    n_divisions: int = 99,
    n_gen: int = 200,
    seed: int = 42,
    n_threads: int = 0,
    p_mut_start: float = -1.0,
    p_mut_end: float = -1.0,
    crossover_kind: int = 0,
    tourn_k: int = 2,
    profile: bool = False,
) -> list[Solution]:
    """Approximate the Pareto frontier via NSGA-III (Deb & Jain 2014).

    NSGA-III replaces NSGA-II's crowding-distance diversity mechanism with
    structured Das-Dennis reference points on the objective unit hyperplane.
    Each generation the combined pool is normalized, individuals are associated
    with their nearest reference point, and survival selection from the critical
    front uses niching counts to ensure even reference-point coverage.

    Parameters
    ----------
    inst        : built fedhpc Instance.
    pop_size    : population size.  Should equal ``n_divisions + 1`` for best
                  reference-point coverage (default 100 matches default 99 divisions).
    n_divisions : number of divisions along each objective axis for the Das-Dennis
                  reference-point lattice.  Produces ``n_divisions + 1`` reference
                  points for the 2-objective case.
    n_gen       : number of generations.
    seed        : RNG seed for reproducibility.
    n_threads   : OpenMP thread count; 0 = use all available cores.
    p_mut_start : mutation-rate schedule start. ``<0`` (default) resolves to
                  the fixed formula rate ``2/n_jobs``.
    p_mut_end   : mutation-rate schedule end. ``<0`` (default) holds the
                  resolved start rate constant (no annealing). When both are
                  given, the rate is linearly annealed start→end across
                  generations.
    crossover_kind : 0 = two-point (default), 1 = uniform (per-gene coin flip).
    tourn_k     : mating-tournament size. 2 (default) = original binary
                  tournament; larger values increase selection pressure.
    profile     : if True, print a per-phase timing breakdown to stderr.

    Returns
    -------
    Feasible non-dominated solutions (status="heuristic"). Not guaranteed
    globally optimal — use MIP-based pareto.py for exact results.
    """
    _require_ext()
    raw, prof = _ext.nsga3(
        n_jobs         = len(inst.jobs),
        budget         = inst.budget,
        job_slots      = _job_slots(inst),
        type_cap       = _type_cap(inst),
        init_occ       = _init_occ(inst),
        pop_size       = pop_size,
        n_divisions    = n_divisions,
        n_gen          = n_gen,
        seed           = seed,
        n_threads      = n_threads,
        p_mut_start    = p_mut_start,
        p_mut_end      = p_mut_end,
        crossover_kind = crossover_kind,
        tourn_k        = tourn_k,
    )
    if profile:
        _print_profile(prof, algorithm="nsga3",
                       label=f"pop={pop_size}  divs={n_divisions}  threads={n_threads or 'all'}")
    return _to_solutions(inst, raw)


def moead_frontier(
    inst: Instance,
    *,
    n_weights: int = 100,
    n_gen: int = 200,
    neighborhood_size: int = 20,
    seed: int = 42,
    n_threads: int = 0,
    max_replace: int = 5,
    p_mut_start: float = -1.0,
    p_mut_end: float = -1.0,
    crossover_kind: int = 0,
    archive_size: int = 20,
    profile: bool = False,
) -> list[Solution]:
    """Approximate the Pareto frontier via MOEA/D (Tchebycheff decomposition).

    Parameters
    ----------
    inst              : built fedhpc Instance.
    n_weights         : number of weight vectors (= population size).
    n_gen             : number of generations.
    neighborhood_size : |T| nearest weight vectors used for mating/replacement.
    seed              : RNG seed for reproducibility.
    n_threads         : OpenMP thread count; 0 = use all available cores.
    max_replace       : cap on how many neighbours a single child may overwrite
                        per generation (Zhang & Li's ``nr``). Default 5, chosen
                        via A/B benchmark against the original unbounded
                        replacement (ΔHV positive across small/medium/large
                        instances; see ablation.py). ``<=0`` disables the cap
                        and reproduces the original unbounded replacement.
    p_mut_start       : mutation-rate schedule start. ``<0`` (default) resolves
                        to the fixed formula rate ``1/n_jobs``.
    p_mut_end         : mutation-rate schedule end. ``<0`` (default) holds the
                        resolved start rate constant (no annealing). When both
                        are given, the rate is linearly annealed start→end
                        across generations.
    crossover_kind    : 0 = two-point (default), 1 = uniform (per-gene coin flip).
    archive_size      : bounded external elitist archive capacity, merged into
                        the returned front at extraction. Recovers points that
                        bounded neighbourhood replacement (max_replace) can
                        overwrite and lose. Default 20, chosen via A/B
                        benchmark (consistent small ΔHV gain and higher
                        cardinality across small/medium/large; plateaus at 20,
                        see ablation.py). ``<=0`` disables it.
    profile           : if True, print a per-phase timing breakdown to stderr.

    Returns
    -------
    Feasible non-dominated solutions (status="heuristic"). Not guaranteed
    globally optimal — use MIP-based pareto.py for exact results.
    """
    _require_ext()
    raw, prof = _ext.moead(
        n_jobs            = len(inst.jobs),
        budget            = inst.budget,
        job_slots         = _job_slots(inst),
        type_cap          = _type_cap(inst),
        init_occ          = _init_occ(inst),
        n_weights         = n_weights,
        n_gen             = n_gen,
        neighborhood_size = neighborhood_size,
        seed              = seed,
        n_threads         = n_threads,
        max_replace       = max_replace,
        p_mut_start       = p_mut_start,
        p_mut_end         = p_mut_end,
        crossover_kind    = crossover_kind,
        archive_size      = archive_size,
    )
    if profile:
        _print_profile(prof, algorithm="moead",
                       label=f"weights={n_weights}  T={neighborhood_size}  threads={n_threads or 'all'}")
    return _to_solutions(inst, raw)
