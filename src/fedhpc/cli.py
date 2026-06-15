"""Command-line interface for FED-HPC."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .data import Instance
from .formulations import OccupancyFormulation, SpaceTimeFormulation, configure_env
from .metrics import metrics_to_serialisable, pareto_metrics
from .model import Solution, solve_epsilon_cost, solve_epsilon_turnaround, solve_f1, solve_f2, solve_weighted_sum
from .pareto import hybrid_frontier, true_pareto_frontier
from .viz import (
    compute_stats, format_pareto_metrics, format_summary,
    save_feasibility_graph, save_gantt, save_machine_schedule, save_spacetime_graph,
)

_FORMULATIONS = {
    "spacetime": SpaceTimeFormulation,
    "occupancy": OccupancyFormulation,
}

_EA_METHODS     = {"nsga2", "nsga3", "moead"}
_HYBRID_METHODS = {"hybrid"}
_MIP_METHODS    = {"f1", "f2", "weighted", "epsilon", "epsilon-t", "pareto-true"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fedhpc",
        description=(
            "FED-HPC: multi-objective MIP scheduler for HPC jobs in federated environments.\n\n"
            "Exact MIP methods (require Gurobi):\n"
            "  f1, f2, weighted, epsilon, epsilon-t, pareto-true\n\n"
            "Heuristic evolutionary methods (no licence required):\n"
            "  nsga2 — NSGA-II with constrained-dominance ranking\n"
            "  nsga3 — NSGA-III with reference-point-based selection (Deb & Jain 2014)\n"
            "  moead — MOEA/D with Tchebycheff decomposition\n\n"
            "Hybrid (requires both C++ extension and Gurobi):\n"
            "  hybrid — EA exploration + Gurobi ε-constraint verification\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--instance", required=True, metavar="FILE",
                   help="Path to instance JSON file.")
    p.add_argument("--method", default="weighted",
                   choices=sorted(_MIP_METHODS | _EA_METHODS | _HYBRID_METHODS),
                   help=(
                       "Solving method (default: weighted).\n"
                       "\n"
                       "  Exact MIP (require Gurobi):\n"
                       "    f1          — lex-min turnaround then cost (cheapest best schedule)\n"
                       "    f2          — lex-min cost then turnaround (best schedule at min cost)\n"
                       "    weighted    — weighted-sum scalarisation (needs --alpha)\n"
                       "    epsilon     — ε-constraint min f1 s.t. f2 ≤ ε (needs --epsilon)\n"
                       "    epsilon-t   — ε-constraint min f2 s.t. f1 ≤ ε (needs --epsilon)\n"
                       "    pareto-true — complete true Pareto front (exact, one solve per point)\n"
                       "\n"
                       "  Heuristic (no licence, parallelised with OpenMP):\n"
                       "    nsga2       — NSGA-II (pop_size × n_gen generations)\n"
                       "    nsga3       — NSGA-III with reference-point selection\n"
                       "    moead       — MOEA/D with Tchebycheff decomposition\n"
                       "\n"
                       "  Hybrid (requires C++ extension + Gurobi):\n"
                       "    hybrid      — NSGA-II + MOEA/D exploration, then Gurobi\n"
                       "                  ε-constraint verification at each EA cost level\n"
                   ))

    # ── MIP / hybrid options ──────────────────────────────────────────────────
    mip = p.add_argument_group("MIP / hybrid options (exact methods and hybrid)")
    mip.add_argument("--formulation", default="spacetime",
                     choices=list(_FORMULATIONS),
                     help=(
                         "MIP formulation used by exact and hybrid methods (default: spacetime):\n"
                         "  spacetime  — space-time network with flow conservation\n"
                         "  occupancy  — occupancy-equality formulation\n"
                     ))
    mip.add_argument("--alpha", type=float, default=0.5,
                     help="Weight for f1 in [0, 1] (used by 'weighted' and 'pareto-ws'). Default: 0.5.")
    mip.add_argument("--epsilon", type=float, default=None,
                     help="ε bound for epsilon-constraint methods.")
    mip.add_argument("--steps", type=int, default=20,
                     help="Number of frontier points for Pareto sweep methods. Default: 20.")
    mip.add_argument("--time-limit", type=float, default=None,
                     help=(
                         "Gurobi time limit in seconds. For exact methods: per solve. "
                         "For hybrid: per ε-constraint solve (one per EA candidate). "
                         "Default: no limit (run to optimality)."
                     ))
    mip.add_argument("--mip-gap", type=float, default=1e-4,
                     help="Gurobi MIPGap (used by exact and hybrid methods). Default: 1e-4.")

    # ── EA / hybrid options ───────────────────────────────────────────────────
    ea = p.add_argument_group("Evolutionary algorithm options (nsga2 / nsga3 / moead / hybrid)")
    ea.add_argument("--pop-size", type=int, default=200, metavar="N",
                    help=(
                        "Population size for NSGA-II / NSGA-III, number of weight vectors for MOEA/D, "
                        "or EA population for each algorithm in hybrid. Default: 200. "
                        "For NSGA-III, should equal --n-divisions + 1."
                    ))
    ea.add_argument("--n-divisions", type=int, default=199, metavar="P",
                    help=(
                        "NSGA-III: number of divisions for the Das-Dennis reference-point lattice. "
                        "Produces P+1 reference points; set to pop_size - 1 for best coverage. "
                        "Default: 199 (matches default pop_size=200)."
                    ))
    ea.add_argument("--n-gen", type=int, default=300, metavar="N",
                    help="Number of EA generations (nsga2, moead, hybrid). Default: 300.")
    ea.add_argument("--neighborhood-size", type=int, default=20, metavar="T",
                    help="MOEA/D neighbourhood size |T| (moead, hybrid). Default: 20.")
    ea.add_argument("--seed", type=int, default=42,
                    help=(
                        "RNG seed for reproducibility. "
                        "For hybrid, NSGA-II uses this seed and MOEA/D uses seed+1. "
                        "Default: 42."
                    ))
    ea.add_argument("--n-threads", type=int, default=0, metavar="N",
                    help=(
                        "OpenMP thread count for evolutionary algorithms (nsga2, moead, hybrid). "
                        "0 = use all available cores (default)."
                    ))

    # ── general options ───────────────────────────────────────────────────────
    p.add_argument("--output-dir", default=".", metavar="DIR",
                   help="Directory for Gantt PNG and machine-schedule files. Default: current dir.")
    p.add_argument("--verbose", action="store_true",
                   help=(
                       "Show detailed progress. MIP methods: Gurobi solver log. "
                       "EA methods: run parameters. "
                       "hybrid: phase progress, per-solve Gurobi log, and timing."
                   ))
    p.add_argument("--json", action="store_true",
                   help="Output results as JSON.")
    p.add_argument("--graph", action="store_true",
                   help=(
                       "Save graph visualizations alongside the Gantt chart:\n"
                       "  spacetime_graph.png — space-time network diagram\n"
                       "  feasibility_graph.png — job–machine bipartite graph\n"
                       "(skipped automatically when the horizon exceeds 80 slots)"
                   ))
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    try:
        inst = Instance.from_file(args.instance)
    except Exception as e:
        print(f"Error loading instance: {e}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    method  = args.method

    # ── evolutionary (heuristic) methods — no Gurobi needed ──────────────────

    if method in _EA_METHODS:
        _run_ea(args, inst, method, out_dir)
        return

    # ── hybrid method — needs both C++ extension and Gurobi ──────────────────

    if method in _HYBRID_METHODS:
        _run_hybrid(args, inst, out_dir)
        return

    # ── MIP methods — need Gurobi ─────────────────────────────────────────────

    formulation = _FORMULATIONS[args.formulation]()
    configure_env(verbose=args.verbose)

    gurobi_params: dict = {"MIPGap": args.mip_gap}
    if args.time_limit is not None:
        gurobi_params["TimeLimit"] = args.time_limit
    if not args.verbose:
        gurobi_params["OutputFlag"] = 0

    if method == "f1":
        sol = solve_f1(inst, formulation=formulation, **gurobi_params)
        _print_single(sol, inst, args.json, title=f"FED-HPC — {method}")
        _save_outputs(sol, inst, out_dir, stem=method, graph=args.graph)

    elif method == "f2":
        sol = solve_f2(inst, formulation=formulation, **gurobi_params)
        _print_single(sol, inst, args.json, title=f"FED-HPC — {method}")
        _save_outputs(sol, inst, out_dir, stem=method, graph=args.graph)

    elif method == "weighted":
        from .pareto import _reference_points
        f1_T, f2_T, f1_0 = _reference_points(inst, formulation=formulation, **gurobi_params)
        sol = solve_weighted_sum(
            inst, args.alpha,
            f1_T=f1_T, f2_T=f2_T, f1_0=f1_0,
            formulation=formulation,
            **gurobi_params,
        )
        _print_single(sol, inst, args.json,
                      title=f"FED-HPC — {method} (α={args.alpha})")
        _save_outputs(sol, inst, out_dir, stem=method, graph=args.graph)

    elif method == "epsilon":
        if args.epsilon is None:
            print("--epsilon is required for method 'epsilon'.", file=sys.stderr)
            sys.exit(1)
        sol = solve_epsilon_cost(inst, args.epsilon, formulation=formulation, **gurobi_params)
        _print_single(sol, inst, args.json,
                      title=f"FED-HPC — {method} (ε={args.epsilon})")
        _save_outputs(sol, inst, out_dir, stem=method, graph=args.graph)

    elif method == "epsilon-t":
        if args.epsilon is None:
            print("--epsilon is required for method 'epsilon-t'.", file=sys.stderr)
            sys.exit(1)
        sol = solve_epsilon_turnaround(inst, args.epsilon, formulation=formulation, **gurobi_params)
        _print_single(sol, inst, args.json,
                      title=f"FED-HPC — {method} (ε={args.epsilon})")
        _save_outputs(sol, inst, out_dir, stem=method, graph=args.graph)

    elif method == "pareto-true":
        solutions = true_pareto_frontier(
            inst, formulation=formulation, verbose=args.verbose, **gurobi_params
        )
        _print_pareto(solutions, inst, args.json, method=method)
        _save_pareto_outputs(solutions, inst, out_dir, stem=method)


def _run_ea(args: argparse.Namespace, inst: Instance, method: str, out_dir: Path) -> None:
    """Dispatch to nsga2_frontier, nsga3_frontier, or moead_frontier and print results."""
    try:
        from .moea import moead_frontier, nsga2_frontier, nsga3_frontier
    except ImportError as e:
        print(f"Evolutionary methods require the fedhpc C++ extension: {e}", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        extra = ""
        if method == "moead":
            extra = f"  T={args.neighborhood_size}"
        elif method == "nsga3":
            extra = f"  divs={args.n_divisions}"
        print(
            f"Running {method.upper()}  pop={args.pop_size}  gen={args.n_gen}"
            + extra
            + f"  seed={args.seed}"
            + (f"  threads={args.n_threads}" if args.n_threads else "  threads=all"),
            file=sys.stderr,
        )

    if method == "nsga2":
        solutions = nsga2_frontier(
            inst,
            pop_size  = args.pop_size,
            n_gen     = args.n_gen,
            seed      = args.seed,
            n_threads = args.n_threads,
            profile   = args.verbose,
        )
    elif method == "nsga3":
        solutions = nsga3_frontier(
            inst,
            pop_size    = args.pop_size,
            n_divisions = args.n_divisions,
            n_gen       = args.n_gen,
            seed        = args.seed,
            n_threads   = args.n_threads,
            profile     = args.verbose,
        )
    else:  # moead
        solutions = moead_frontier(
            inst,
            n_weights         = args.pop_size,
            n_gen             = args.n_gen,
            neighborhood_size = args.neighborhood_size,
            seed              = args.seed,
            n_threads         = args.n_threads,
            profile           = args.verbose,
        )

    _print_pareto(solutions, inst, args.json, method=method)
    _save_pareto_outputs(solutions, inst, out_dir, stem=method)


def _run_hybrid(args: argparse.Namespace, inst: Instance, out_dir: Path) -> None:
    """Run the hybrid EA + Gurobi frontier and print results."""
    try:
        from .moea import moead_frontier, nsga2_frontier  # noqa: F401
    except ImportError as e:
        print(f"hybrid requires the fedhpc C++ extension: {e}", file=sys.stderr)
        sys.exit(1)

    configure_env(verbose=args.verbose)
    formulation = _FORMULATIONS[args.formulation]()

    if args.verbose:
        print(
            f"Running HYBRID  pop={args.pop_size}  gen={args.n_gen}"
            f"  T={args.neighborhood_size}  seed={args.seed}"
            + (f"  threads={args.n_threads}" if args.n_threads else "  threads=all")
            + (f"  time_limit={args.time_limit}s" if args.time_limit is not None else "  time_limit=∞")
            + f"  mip_gap={args.mip_gap}",
            file=sys.stderr,
        )

    solutions = hybrid_frontier(
        inst,
        pop_size          = args.pop_size,
        n_gen             = args.n_gen,
        neighborhood_size = args.neighborhood_size,
        seed              = args.seed,
        n_threads         = args.n_threads,
        formulation       = formulation,
        time_limit        = args.time_limit,
        mip_gap           = args.mip_gap,
        verbose           = args.verbose,
    )

    _print_pareto(solutions, inst, args.json, method="hybrid")
    _save_pareto_outputs(solutions, inst, out_dir, stem="hybrid")


# ── Console output helpers ────────────────────────────────────────────────────

def _print_single(
    sol: Solution,
    inst: Instance,
    as_json: bool,
    title: str = "FED-HPC Schedule",
) -> None:
    if as_json:
        print(json.dumps(_sol_to_dict(sol, inst), indent=2))
    else:
        print(format_summary(sol, inst, title=title))


def _print_pareto(
    solutions: list[Solution],
    inst: Instance,
    as_json: bool,
    method: str = "pareto",
) -> None:
    _PRINTABLE = ("optimal", "feasible", "heuristic")
    ranked = sorted(solutions, key=lambda s: s.f1 or 0)
    n      = len(ranked)

    # Compute quality metrics on the ranked front.
    m = pareto_metrics(ranked)

    if as_json:
        out = {
            "front":   [_sol_to_dict(s, inst) for s in ranked],
            "metrics": metrics_to_serialisable(m, ranked),
        }
        print(json.dumps(out, indent=2))
        return

    from .viz import _rule
    print(f"FED-HPC Pareto Frontier — {method}  [{n} non-dominated point(s)]")
    print(_rule())
    print()

    # Build sets of flagged solution objects for in-table annotation.
    knee_sol  = m["knee_point"]
    mm_sol    = m["regret"]["minimax_solution"]

    def _flags(sol: Solution) -> str:
        tag = ""
        if sol is knee_sol:
            tag += "K"
        if sol is mm_sol:
            tag += "R"
        return f"{tag:<2}"

    slot_secs = inst.slot_size_seconds
    hdr = (
        f"  {'Pt':>3}  {'Status':<10}"
        f"  {'f1 (turnaround s)':>18}  {'f2 (cost $)':>12}"
        f"  {'Sched':>6}  {'OnPrem':>7}  {'Cloud':>6}"
        f"  {'AvgWait(s)':>11}  {'AvgRun(s)':>10}  {'AvgTA(s)':>10}  {'AvgBS':>7}"
        f"  {'':>2}"
    )
    print(hdr)
    print(f"  {_rule('-')[:len(hdr) - 2]}")

    for i, sol in enumerate(ranked, start=1):
        flags = _flags(sol)
        if sol.status not in _PRINTABLE:
            print(
                f"  {i:>3}  {sol.status:<10}"
                f"  {'—':>18}  {'—':>12}"
                f"  {'—':>6}  {'—':>7}  {'—':>6}"
                f"  {'—':>11}  {'—':>10}  {'—':>10}  {'—':>7}"
                f"  {flags}"
            )
            continue

        st  = compute_stats(sol, inst)
        sys = st["system"]
        print(
            f"  {i:>3}  {sol.status:<10}"
            f"  {sys['f1'] * slot_secs:>18.1f}  {sys['f2']:>12.2f}"
            f"  {st['n_scheduled']:>6}  {sys['onprem_jobs']:>7}  {sys['cloud_jobs']:>6}"
            f"  {st['wait_time']['avg']    * slot_secs:>11.1f}"
            f"  {st['run_time']['avg']     * slot_secs:>10.1f}"
            f"  {st['turnaround']['avg']   * slot_secs:>10.1f}"
            f"  {st['bounded_slowdown']['avg']:>7.3f}"
            f"  {flags}"
        )

    print(f"  {'K'} = knee point   {'R'} = min-regret solution")
    print()
    print(format_pareto_metrics(m, ranked, inst))


# ── File-output helpers ───────────────────────────────────────────────────────

def _save_outputs(
    sol: Solution,
    inst: Instance,
    out_dir: Path,
    stem: str,
    *,
    graph: bool = False,
) -> None:
    """Save Gantt PNG, machine-schedule text, and optionally graph visualizations."""
    if sol.status not in ("optimal", "feasible", "heuristic"):
        return
    gantt_path = save_gantt(
        sol, inst,
        out_dir / f"{stem}_gantt.png",
        title=f"FED-HPC Schedule [{stem}]",
    )
    sched_path = save_machine_schedule(
        sol, inst,
        out_dir / f"{stem}_machine_schedule.txt",
        title=f"FED-HPC Machine Schedule [{stem}]",
    )
    print(f"Gantt chart      → {gantt_path}", file=sys.stderr)
    print(f"Machine schedule → {sched_path}", file=sys.stderr)

    if graph:
        st_paths = save_spacetime_graph(
            inst, sol,
            out_dir / f"{stem}_spacetime_graph.png",
            title=f"Space-Time Network [{stem}]",
        )
        if st_paths:
            for p in st_paths:
                print(f"Space-time graph → {p}", file=sys.stderr)
        else:
            print(
                "Space-time graph skipped (horizon > 80 slots).",
                file=sys.stderr,
            )
        feas_path = save_feasibility_graph(
            inst, sol,
            out_dir / f"{stem}_feasibility_graph.png",
            title=f"Job–Machine Feasibility [{stem}]",
        )
        print(f"Feasibility graph → {feas_path}", file=sys.stderr)


def _save_pareto_outputs(
    solutions: list[Solution],
    inst: Instance,
    out_dir: Path,
    stem: str,
) -> None:
    """Save a single CSV with the Pareto frontier values (f1, f2, status)."""
    import csv

    ranked = sorted(
        (s for s in solutions if s.status in ("optimal", "feasible", "heuristic")),
        key=lambda s: s.f1 or 0,
    )
    csv_path = out_dir / f"{stem}_frontier.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["rank", "f1_turnaround", "f2_cost", "status"])
        for i, sol in enumerate(ranked, start=1):
            writer.writerow([i, sol.f1, sol.f2, sol.status])
    print(f"Frontier CSV     → {csv_path}", file=sys.stderr)


# ── JSON serialisation ────────────────────────────────────────────────────────

def _sol_to_dict(sol: Solution, inst: Instance) -> dict:
    st = compute_stats(sol, inst)
    per_job = st.pop("per_job")

    assignment_detail = {
        str(jid): {"type_id": type_id, "start_time": start_time, "end_time": sol.completion[jid]}
        for jid, (type_id, start_time) in sol.assignment.items()
    }

    return {
        "status":        sol.status,
        "objective":     sol.objective,
        "f1_turnaround": sol.f1,
        "f2_cost":       sol.f2,
        "statistics":    st,
        "per_job":       per_job,
        "assignment":    assignment_detail,
    }
