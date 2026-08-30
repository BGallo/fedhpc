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


def _prune_dominated_cloud_types(
    inst: Instance, j_id: int, type_risk: list[float], cap_of: dict[int, int | None],
) -> list[int]:
    """Light pre-pruning: drop unlimited-capacity (cloud) types strictly
    dominated — same or worse cost, occupation time, and revocation risk —
    by another *unlimited-capacity* type feasible for the same job.

    Deliberately restricted to unlimited-capacity types. Finite-capacity
    (on-prem) types are independent, concurrently-usable pools — e.g. two
    on-prem groups can both be at their own cap at once, so "dominating" one
    pool with another would silently shrink total available on-prem capacity
    (jobs that need both pools simultaneously would become infeasible even
    though nothing was actually redundant). Unlimited types have no such
    coupling: capacity is never checked for them (see ga_common.hpp
    evaluate()), so a strictly-dominated one can never be worth choosing.

    Only affects the heuristic MOEA/D + NSGA search space built here — the
    exact MIP formulations (model.py/formulations.py) always see the full
    Instance.F, unpruned.
    """
    feasible = inst.F[j_id]
    unlimited = [m for m in feasible if cap_of[m] is None]
    dominated: set[int] = set()
    for m in unlimited:
        for m2 in unlimited:
            if m2 == m:
                continue
            c1, c2 = inst.c[j_id, m], inst.c[j_id, m2]
            p1, p2 = inst.p_occ[j_id, m], inst.p_occ[j_id, m2]
            r1, r2 = type_risk[m], type_risk[m2]
            if c2 <= c1 and p2 <= p1 and r2 <= r1 and (c2 < c1 or p2 < p1 or r2 < r1):
                dominated.add(m)
                break
    return [m for m in feasible if m not in dominated]


def _job_slots(inst: Instance) -> list[list[tuple[int, int, int, float, float]]]:
    """Flatten Instance into per-job slot tables for the C++ layer.

    Each slot is (type_id, start, p_occ, f1_contrib, cost).
    f1_contrib = start + p_occ - arrival is precomputed to avoid recomputing
    it on every evaluation inside the tight C++ inner loop.

    Per-job feasible types are lightly pruned first (see
    _prune_dominated_cloud_types) — this only shrinks the heuristic search
    space, never the exact MIP's.
    """
    type_risk = _type_risk(inst)
    cap_of = {m.id: m.capacity for m in inst.instance_types}
    slots: list[list[tuple[int, int, int, float, float]]] = []
    for j in inst.jobs:
        job_slots: list[tuple[int, int, int, float, float]] = []
        for m_id in _prune_dominated_cloud_types(inst, j.id, type_risk, cap_of):
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


def _type_risk(inst: Instance) -> list[float]:
    """Revocation-risk list indexed by type_id. 0.0 = no revocation risk."""
    max_id = max(m.id for m in inst.instance_types)
    risk = [0.0] * (max_id + 1)
    for m in inst.instance_types:
        risk[m.id] = m.revocation_risk
    return risk


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
            ("Periodic local search",
             prof.get("local_search_total_ms", 0.0), None),
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
            ("Periodic local search",
             prof.get("local_search_total_ms", 0.0), None),
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
            ("Periodic local search",
             prof.get("local_search_total_ms", 0.0), None),
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
    pop_size: int = 400,
    n_gen: int = 200,
    seed: int = 42,
    n_threads: int = 0,
    p_mut_start: float = -1.0,
    p_mut_end: float = -1.0,
    crossover_kind: int = 0,
    tourn_k: int = 2,
    local_search_interval: int = -1,
    sched_repair: int = 1,
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
    local_search_interval : generations between periodic capacity-repair /
                     cost-descent local search on the population (see
                     ga_common.hpp's ``local_search``). ``<0`` (default)
                     resolves to ~10 applications spread across the run;
                     ``0`` disables it, restoring seeds+final-polish-only
                     behaviour. Always applied to the heuristic seeds and the
                     final population regardless of this setting.
    sched_repair    : ``1`` (default) runs the earliest-feasible SPT
                     list-scheduling repair (``schedule_repair`` in
                     ga_common.hpp) on the heuristic seeds and the final
                     population, right after ``local_search``: it keeps each
                     job's type (cost is start-independent) and left-shifts
                     starts to the earliest capacity-feasible slot, jobs taken
                     shortest-processing-time-first. Closes the scheduling-
                     subproblem gap coordinate descent can't, especially at the
                     low-cost / high-turnaround corner. Chosen as the default
                     via A/B benchmark (see scripts/ab_sched_repair.py);
                     ``0`` = pre-change behaviour, byte-for-byte. ``2`` also
                     lets the decoder spread each job's schedule across every
                     equal-price type (e.g. multiple free on-prem pools),
                     which further improves NSGA-II / NSGA-III at the
                     cost-minimal end but regresses MOEA/D — opt-in only.
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
        type_risk      = _type_risk(inst),
        init_occ       = _init_occ(inst),
        pop_size       = pop_size,
        n_gen          = n_gen,
        seed           = seed,
        n_threads      = n_threads,
        p_mut_start    = p_mut_start,
        p_mut_end      = p_mut_end,
        crossover_kind = crossover_kind,
        tourn_k        = tourn_k,
        local_search_interval = local_search_interval,
        sched_repair    = sched_repair,
    )
    if profile:
        _print_profile(prof, algorithm="nsga2",
                       label=f"pop={pop_size}  threads={n_threads or 'all'}")
    return _to_solutions(inst, raw)


def nsga3_frontier(
    inst: Instance,
    *,
    pop_size: int = 400,
    n_divisions: int = 399,
    n_gen: int = 200,
    seed: int = 42,
    n_threads: int = 0,
    p_mut_start: float = -1.0,
    p_mut_end: float = -1.0,
    crossover_kind: int = 0,
    tourn_k: int = 2,
    local_search_interval: int = -1,
    sched_repair: int = 1,
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
                  reference-point coverage (default 400 matches default 399 divisions).
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
    local_search_interval : generations between periodic capacity-repair /
                  cost-descent local search on the population; see
                  nsga2_frontier for the convention.
    sched_repair : ``1`` (default) SPT list-scheduling repair on seeds + final
                  population; ``0`` = pre-change; ``2`` adds equal-price pool balancing.
                  See ``nsga2_frontier``.
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
        type_risk      = _type_risk(inst),
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
        local_search_interval = local_search_interval,
        sched_repair    = sched_repair,
    )
    if profile:
        _print_profile(prof, algorithm="nsga3",
                       label=f"pop={pop_size}  divs={n_divisions}  threads={n_threads or 'all'}")
    return _to_solutions(inst, raw)


def moead_frontier(
    inst: Instance,
    *,
    n_weights: int = 400,
    n_gen: int = 200,
    neighborhood_size: int = 20,
    seed: int = 42,
    n_threads: int = 0,
    max_replace: int = 5,
    p_mut_start: float = -1.0,
    p_mut_end: float = -1.0,
    crossover_kind: int = 0,
    archive_size: int = 20,
    local_search_interval: int = -1,
    sched_repair: int = 1,
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
    local_search_interval : generations between periodic capacity-repair /
                        cost-descent local search on the population; see
                        nsga2_frontier for the convention.
    sched_repair : ``1`` (default) SPT list-scheduling repair on seeds + final
                        population; ``0`` = pre-change; ``2`` adds equal-price pool balancing.
                        See ``nsga2_frontier``.
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
        type_risk         = _type_risk(inst),
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
        local_search_interval = local_search_interval,
        sched_repair    = sched_repair,
    )
    if profile:
        _print_profile(prof, algorithm="moead",
                       label=f"weights={n_weights}  T={neighborhood_size}  threads={n_threads or 'all'}")
    return _to_solutions(inst, raw)


def _weighted_raw(
    inst: Instance, w1: float, w2: float, f1_cap: float, *,
    pop_size: int, n_gen: int, seed: int, n_threads: int,
    ls_moves: int, restart_patience: int, shortlist: int,
) -> tuple[list[tuple[int, int]], float, float, float, int]:
    _require_ext()
    return _ext.weighted(
        n_jobs          = len(inst.jobs),
        budget          = inst.budget,
        job_slots       = _job_slots(inst),
        type_cap        = _type_cap(inst),
        type_risk       = _type_risk(inst),
        init_occ        = _init_occ(inst),
        w1              = w1,
        w2              = w2,
        f1_cap          = f1_cap,
        pop_size        = pop_size,
        n_gen           = n_gen,
        seed            = seed,
        n_threads       = n_threads,
        ls_moves        = ls_moves,
        restart_patience= restart_patience,
        shortlist       = shortlist,
    )


def heuristic_weighted_reference_points(
    inst: Instance, *, seed: int = 42, n_threads: int = 0,
    pop_size: int = 24, n_gen: int = 40, ls_moves: int = 6,
    restart_patience: int = 6, shortlist: int = 24,
) -> tuple[float, float, float]:
    """Heuristic estimates of the three weighted-sum reference points.

    Mirrors ``pareto._reference_points`` (which solves them exactly with
    Gurobi) using the memetic metaheuristic instead:

    - ``f1_T``  ≈ minimum total turnaround        (weights w1=1, w2=0)
    - ``f2_T``  ≈ minimum cost at that turnaround  (w1 huge, w2=1, f1 capped at f1_T)
    - ``f1_0``  ≈ minimum turnaround at zero cost  (w1=1, w2 huge)

    These are upper bounds on / close approximations of the true reference
    points — good enough to scale the weighted-sum objective for
    :func:`weighted_solve`, not a replacement for the exact values.
    """
    kw = dict(pop_size=pop_size, n_gen=n_gen, seed=seed, n_threads=n_threads,
              ls_moves=ls_moves, restart_patience=restart_patience, shortlist=shortlist)
    _, f1_T, _, _, _ = _weighted_raw(inst, 1.0, 0.0, 1e18, **kw)
    _, f1_at_cap, f2_T, _, _ = _weighted_raw(inst, 1e6, 1.0, f1_T + 0.5, **kw)
    _, f1_0, _, _, _ = _weighted_raw(inst, 1.0, 1e6, 1e18, **kw)
    return float(f1_T), float(f2_T), float(f1_0)


def weighted_solve(
    inst: Instance,
    lam: float,
    *,
    f1_T: float | None = None,
    f2_T: float | None = None,
    f1_0: float | None = None,
    pop_size: int = 24,
    n_gen: int = 40,
    seed: int = 42,
    n_threads: int = 0,
    ls_moves: int = 6,
    restart_patience: int = 6,
    shortlist: int = 24,
) -> Solution:
    """Heuristically minimise ``lam·f̂1 + (1−lam)·f̂2`` (the single-objective
    weighted-sum scalarisation), returning one feasible ``Solution``.

    Solves the *same* normalised objective as :func:`model.solve_weighted_sum`
    — ``(lam/(f1_0−f1_T))·f1 + ((1−lam)/f2_T)·f2`` subject to ``f1 ≤ f1_0`` —
    but with a memetic metaheuristic (see ``_ext/weighted.hpp``): a small
    population of per-job type-assignment vectors, each decoded to a schedule
    by the SPT list-scheduling repair, refined by a greedy scalar type-flip
    local search, with two-point crossover for recombination and ILS
    perturbation kicks to escape local optima.

    Parameters
    ----------
    lam              : trade-off weight in ``[0, 1]`` (1 → pure turnaround).
    f1_T, f2_T, f1_0 : weighted-sum reference points (see
                       ``pareto._reference_points``). If any is ``None`` all
                       three are estimated via
                       :func:`heuristic_weighted_reference_points`.
    pop_size, n_gen  : memetic-GA population and generation count.
    ls_moves         : max improving type-flips per local-search call
                       (``·3`` for the final polish).
    restart_patience : generations without improvement before an ILS kick
                       (``0`` disables kicks).
    shortlist        : type-flip candidates decoded exactly per local-search
                       step (ranked first by an O(1) Δobjective estimate).
    seed, n_threads  : determinism / OpenMP controls (bit-for-bit stable for a
                       fixed ``(seed, n_threads)``).

    Returns
    -------
    A single feasible ``Solution`` with ``status="heuristic"``.
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must be in [0, 1]")

    if f1_T is None or f2_T is None or f1_0 is None:
        f1_T, f2_T, f1_0 = heuristic_weighted_reference_points(
            inst, seed=seed, n_threads=n_threads, pop_size=pop_size, n_gen=n_gen,
            ls_moves=ls_moves, restart_patience=restart_patience, shortlist=shortlist,
        )

    degenerate = f1_0 <= f1_T + 1e-8 or f2_T <= 1e-8
    if degenerate:
        w1, w2, f1_cap = 1e-6, 1.0, f1_0
    else:
        w1 = lam / (f1_0 - f1_T)
        w2 = (1.0 - lam) / f2_T
        f1_cap = f1_0

    asgn, f1, f2, _g, _ls = _weighted_raw(
        inst, w1, w2, f1_cap, pop_size=pop_size, n_gen=n_gen, seed=seed,
        n_threads=n_threads, ls_moves=ls_moves, restart_patience=restart_patience,
        shortlist=shortlist,
    )
    return _to_solutions(inst, [(asgn, f1, f2)])[0]


def time_seeds(inst: Instance) -> list[dict]:
    """Diagnostic: time each deterministic heuristic seed individually.

    Times construction, initial evaluation, and local_search() repair
    separately for each of the 8 heuristic seeds (greedy-time, greedy-cost,
    no-wait, full-burst, fixed-wait-25%, fixed-wait-50%,
    star-wait, list-schedule), on the given instance. Not used by
    nsga2_frontier/nsga3_frontier/moead_frontier — those construct and
    repair the same seeds internally without this per-seed breakdown.

    Returns
    -------
    List of per-seed dicts: name, construct_ms, eval_ms, repair_ms,
    f1_before/f2_before/cv_before (post initial evaluate, pre-repair),
    f1_after/f2_after/cv_after (post local_search + re-evaluate).
    """
    _require_ext()
    return _ext.time_seeds(
        n_jobs    = len(inst.jobs),
        budget    = inst.budget,
        job_slots = _job_slots(inst),
        type_cap  = _type_cap(inst),
        type_risk = _type_risk(inst),
        init_occ  = _init_occ(inst),
    )
