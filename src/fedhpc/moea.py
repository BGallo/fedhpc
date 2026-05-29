"""Multi-objective evolutionary algorithms for FED-HPC (NSGA-II and MOEA/D).

These provide heuristic Pareto-frontier approximations that trade optimality
guarantees for much lower wall-clock time than exact MIP-based methods.

Both algorithms encode solutions as a direct assignment vector (one (m, t)
slot per job) and evaluate f1 / f2 identically to the MIP formulations.
Constraint violations (capacity, budget) are handled via constrained-dominance
ranking (NSGA-II) or a large Tchebycheff penalty (MOEA/D).

The returned ``list[Solution]`` is drop-in compatible with
``pareto.weighted_sum_frontier`` and ``pareto.epsilon_constraint_frontier``.
Status is set to ``"heuristic"`` instead of ``"optimal"``.
"""
from __future__ import annotations

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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def nsga2_frontier(
    inst: Instance,
    *,
    pop_size: int = 100,
    n_gen: int = 200,
    seed: int = 42,
    n_threads: int = 0,
) -> list[Solution]:
    """Approximate the Pareto frontier via NSGA-II.

    Parameters
    ----------
    inst       : built fedhpc Instance.
    pop_size   : population size (individuals per generation).
    n_gen      : number of generations.
    seed       : RNG seed for reproducibility.
    n_threads  : OpenMP thread count; 0 = use all available cores.

    Returns
    -------
    Feasible non-dominated solutions (status="heuristic"). Not guaranteed
    globally optimal — use MIP-based pareto.py for exact results.
    """
    _require_ext()
    raw = _ext.nsga2(
        n_jobs    = len(inst.jobs),
        budget    = inst.budget,
        job_slots = _job_slots(inst),
        type_cap  = _type_cap(inst),
        init_occ  = _init_occ(inst),
        pop_size  = pop_size,
        n_gen     = n_gen,
        seed      = seed,
        n_threads = n_threads,
    )
    return _to_solutions(inst, raw)


def moead_frontier(
    inst: Instance,
    *,
    n_weights: int = 100,
    n_gen: int = 200,
    neighborhood_size: int = 20,
    seed: int = 42,
    n_threads: int = 0,
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

    Returns
    -------
    Feasible non-dominated solutions (status="heuristic"). Not guaranteed
    globally optimal — use MIP-based pareto.py for exact results.
    """
    _require_ext()
    raw = _ext.moead(
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
    )
    return _to_solutions(inst, raw)
