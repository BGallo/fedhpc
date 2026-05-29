"""Command-line interface for FED-HPC."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .data import Instance
from .formulations import OccupancyFormulation, SpaceTimeFormulation, configure_env
from .model import Solution, solve_epsilon_cost, solve_epsilon_turnaround, solve_f1, solve_f2, solve_weighted_sum
from .pareto import epsilon_constraint_frontier, weighted_sum_frontier
from .viz import (
    compute_stats, format_summary,
    save_feasibility_graph, save_gantt, save_machine_schedule, save_spacetime_graph,
)

_FORMULATIONS = {
    "spacetime": SpaceTimeFormulation,
    "occupancy": OccupancyFormulation,
}

_EA_METHODS = {"nsga2", "moead"}
_MIP_METHODS = {"f1", "f2", "weighted", "epsilon", "epsilon-t", "pareto-ws", "pareto-eps"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fedhpc",
        description=(
            "FED-HPC: multi-objective MIP scheduler for HPC jobs in federated environments.\n\n"
            "Exact MIP methods (require Gurobi):\n"
            "  f1, f2, weighted, epsilon, epsilon-t, pareto-ws, pareto-eps\n\n"
            "Heuristic evolutionary methods (no licence required):\n"
            "  nsga2 — NSGA-II with constrained-dominance ranking\n"
            "  moead — MOEA/D with Tchebycheff decomposition\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--instance", required=True, metavar="FILE",
                   help="Path to instance JSON file.")
    p.add_argument("--method", default="weighted",
                   choices=sorted(_MIP_METHODS | _EA_METHODS),
                   help=(
                       "Solving method (default: weighted).\n"
                       "\n"
                       "  Exact MIP (require Gurobi):\n"
                       "    f1          — minimise total turnaround only\n"
                       "    f2          — minimise total cost only\n"
                       "    weighted    — weighted-sum scalarisation (needs --alpha)\n"
                       "    epsilon     — ε-constraint min f1 s.t. f2 ≤ ε (needs --epsilon)\n"
                       "    epsilon-t   — ε-constraint min f2 s.t. f1 ≤ ε (needs --epsilon)\n"
                       "    pareto-ws   — Pareto frontier via weighted-sum sweep\n"
                       "    pareto-eps  — Pareto frontier via ε-constraint sweep\n"
                       "\n"
                       "  Heuristic (no licence, parallelised with OpenMP):\n"
                       "    nsga2       — NSGA-II (pop_size × n_gen generations)\n"
                       "    moead       — MOEA/D with Tchebycheff decomposition\n"
                   ))

    # ── MIP options ───────────────────────────────────────────────────────────
    mip = p.add_argument_group("MIP options (exact methods only)")
    mip.add_argument("--formulation", default="spacetime",
                     choices=list(_FORMULATIONS),
                     help=(
                         "MIP formulation (default: spacetime):\n"
                         "  spacetime  — space-time network with flow conservation\n"
                         "  occupancy  — occupancy-equality formulation\n"
                     ))
    mip.add_argument("--alpha", type=float, default=0.5,
                     help="Weight for f1 in [0, 1] (used by 'weighted' and 'pareto-ws'). Default: 0.5.")
    mip.add_argument("--epsilon", type=float, default=None,
                     help="ε bound for epsilon-constraint methods.")
    mip.add_argument("--steps", type=int, default=20,
                     help="Number of frontier points for Pareto sweep. Default: 20.")
    mip.add_argument("--time-limit", type=float, default=300.0,
                     help="Gurobi time limit per solve in seconds. Default: 300.")
    mip.add_argument("--mip-gap", type=float, default=1e-4,
                     help="Gurobi MIPGap. Default: 1e-4.")

    # ── EA options ────────────────────────────────────────────────────────────
    ea = p.add_argument_group("Evolutionary algorithm options (nsga2 / moead)")
    ea.add_argument("--pop-size", type=int, default=200, metavar="N",
                    help=(
                        "Population size for NSGA-II, or number of weight vectors for MOEA/D. "
                        "Default: 200."
                    ))
    ea.add_argument("--n-gen", type=int, default=300, metavar="N",
                    help="Number of generations. Default: 300.")
    ea.add_argument("--neighborhood-size", type=int, default=20, metavar="T",
                    help="MOEA/D neighbourhood size |T|. Default: 20.")
    ea.add_argument("--seed", type=int, default=42,
                    help="RNG seed for reproducibility. Default: 42.")
    ea.add_argument("--n-threads", type=int, default=0, metavar="N",
                    help=(
                        "OpenMP thread count for evolutionary algorithms. "
                        "0 = use all available cores (default)."
                    ))

    # ── general options ───────────────────────────────────────────────────────
    p.add_argument("--output-dir", default=".", metavar="DIR",
                   help="Directory for Gantt PNG and machine-schedule files. Default: current dir.")
    p.add_argument("--verbose", action="store_true",
                   help="Show Gurobi log output (MIP methods) or algorithm progress (EA methods).")
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

    # ── MIP methods — need Gurobi ─────────────────────────────────────────────

    formulation = _FORMULATIONS[args.formulation]()
    configure_env(verbose=args.verbose)

    gurobi_params: dict = {
        "TimeLimit": args.time_limit,
        "MIPGap":    args.mip_gap,
    }
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

    elif method == "pareto-ws":
        solutions = weighted_sum_frontier(
            inst, n_points=args.steps, formulation=formulation, **gurobi_params
        )
        _print_pareto(solutions, inst, args.json, method=method)
        _save_pareto_outputs(solutions, inst, out_dir, stem=method, graph=args.graph)

    elif method == "pareto-eps":
        solutions = epsilon_constraint_frontier(
            inst, n_points=args.steps, formulation=formulation, **gurobi_params
        )
        _print_pareto(solutions, inst, args.json, method=method)
        _save_pareto_outputs(solutions, inst, out_dir, stem=method, graph=args.graph)


def _run_ea(args: argparse.Namespace, inst: Instance, method: str, out_dir: Path) -> None:
    """Dispatch to nsga2_frontier or moead_frontier and print results."""
    try:
        from .moea import moead_frontier, nsga2_frontier
    except ImportError as e:
        print(f"Evolutionary methods require the fedhpc C++ extension: {e}", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(
            f"Running {method.upper()}  pop={args.pop_size}  gen={args.n_gen}"
            + (f"  T={args.neighborhood_size}" if method == "moead" else "")
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
        )
    else:  # moead
        solutions = moead_frontier(
            inst,
            n_weights         = args.pop_size,
            n_gen             = args.n_gen,
            neighborhood_size = args.neighborhood_size,
            seed              = args.seed,
            n_threads         = args.n_threads,
        )

    _print_pareto(solutions, inst, args.json, method=method)
    _save_pareto_outputs(solutions, inst, out_dir, stem=method)


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

    if as_json:
        print(json.dumps([_sol_to_dict(s, inst) for s in ranked], indent=2))
        return

    from .viz import _rule
    print(f"FED-HPC Pareto Frontier — {method}  [{n} non-dominated point(s)]")
    print(_rule())
    print()

    slot_secs = inst.slot_size_seconds
    hdr = (
        f"  {'Pt':>3}  {'Status':<10}"
        f"  {'f1 (turnaround s)':>18}  {'f2 (cost $)':>12}"
        f"  {'Sched':>6}  {'OnPrem':>7}  {'Cloud':>6}"
        f"  {'AvgWait(s)':>11}  {'AvgRun(s)':>10}  {'AvgTA(s)':>10}  {'AvgBS':>7}"
    )
    print(hdr)
    print(f"  {_rule('-')[:len(hdr) - 2]}")

    for i, sol in enumerate(ranked, start=1):
        if sol.status not in _PRINTABLE:
            print(
                f"  {i:>3}  {sol.status:<10}"
                f"  {'—':>18}  {'—':>12}"
                f"  {'—':>6}  {'—':>7}  {'—':>6}"
                f"  {'—':>11}  {'—':>10}  {'—':>10}  {'—':>7}"
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
        )

    print()


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
        st_path = save_spacetime_graph(
            inst, sol,
            out_dir / f"{stem}_spacetime_graph.png",
            title=f"Space-Time Network [{stem}]",
        )
        if st_path:
            print(f"Space-time graph → {st_path}", file=sys.stderr)
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
    *,
    graph: bool = False,
) -> None:
    ranked = sorted(
        (s for s in solutions if s.status in ("optimal", "feasible", "heuristic")),
        key=lambda s: s.f1 or 0,
    )
    for i, sol in enumerate(ranked, start=1):
        _save_outputs(sol, inst, out_dir, stem=f"{stem}_{i:02d}", graph=graph)


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
