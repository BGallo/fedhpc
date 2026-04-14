"""Command-line interface for FED-HPC."""

from __future__ import annotations

import argparse
import json
import sys

from .data import Instance
from .formulations import OccupancyFormulation, SpaceTimeFormulation
from .model import solve_epsilon_cost, solve_epsilon_turnaround, solve_f1, solve_f2, solve_weighted_sum
from .pareto import epsilon_constraint_frontier, weighted_sum_frontier

_FORMULATIONS = {
    "spacetime": SpaceTimeFormulation,
    "occupancy": OccupancyFormulation,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fedhpc",
        description="FED-HPC: multi-objective MIP scheduler for HPC jobs in federated environments.",
    )
    p.add_argument("--instance", required=True, metavar="FILE",
                   help="Path to instance JSON file.")
    p.add_argument("--method", default="weighted",
                   choices=["f1", "f2", "weighted", "epsilon", "epsilon-t",
                             "pareto-ws", "pareto-eps"],
                   help=(
                       "Solving method:\n"
                       "  f1          — minimise turnaround only\n"
                       "  f2          — minimise cost only\n"
                       "  weighted    — weighted-sum scalarisation (needs --alpha)\n"
                       "  epsilon     — ε-constraint min f1 s.t. f2≤ε (needs --epsilon)\n"
                       "  epsilon-t   — ε-constraint min f2 s.t. f1≤ε (needs --epsilon)\n"
                       "  pareto-ws   — Pareto frontier via weighted-sum sweep\n"
                       "  pareto-eps  — Pareto frontier via ε-constraint sweep\n"
                   ))
    p.add_argument("--formulation", default="spacetime",
                   choices=list(_FORMULATIONS),
                   help=(
                       "MIP formulation to use (default: spacetime):\n"
                       "  spacetime  — space-time network with flow conservation (fedhpc-1.pdf)\n"
                       "  occupancy  — occupancy-equality formulation\n"
                   ))
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Weight for f1 in [0,1] (used by 'weighted' and 'pareto-ws'). Default: 0.5.")
    p.add_argument("--epsilon", type=float, default=None,
                   help="ε bound for epsilon-constraint methods.")
    p.add_argument("--steps", type=int, default=20,
                   help="Number of points for Pareto frontier sweep. Default: 20.")
    p.add_argument("--time-limit", type=float, default=300.0,
                   help="Gurobi time limit in seconds. Default: 300.")
    p.add_argument("--mip-gap", type=float, default=1e-4,
                   help="Gurobi MIPGap. Default: 1e-4.")
    p.add_argument("--verbose", action="store_true",
                   help="Show Gurobi log output.")
    p.add_argument("--json", action="store_true",
                   help="Output results as JSON.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    try:
        inst = Instance.from_file(args.instance)
    except Exception as e:
        print(f"Error loading instance: {e}", file=sys.stderr)
        sys.exit(1)

    formulation = _FORMULATIONS[args.formulation]()

    gurobi_params: dict = {
        "TimeLimit": args.time_limit,
        "MIPGap": args.mip_gap,
    }
    if not args.verbose:
        gurobi_params["OutputFlag"] = 0

    method = args.method

    # ---- Single-solution methods ------------------------------------------

    if method == "f1":
        sol = solve_f1(inst, formulation=formulation, **gurobi_params)
        _print_single(sol, args.json)

    elif method == "f2":
        sol = solve_f2(inst, formulation=formulation, **gurobi_params)
        _print_single(sol, args.json)

    elif method == "weighted":
        from .pareto import _anchor_points
        f1_min, f1_max, _, f2_max = _anchor_points(inst, formulation=formulation, **gurobi_params)
        sol = solve_weighted_sum(
            inst, args.alpha,
            f1_min=f1_min, f1_max=f1_max, f2_max=f2_max,
            formulation=formulation,
            **gurobi_params,
        )
        _print_single(sol, args.json)

    elif method == "epsilon":
        if args.epsilon is None:
            print("--epsilon is required for method 'epsilon'.", file=sys.stderr)
            sys.exit(1)
        sol = solve_epsilon_cost(inst, args.epsilon, formulation=formulation, **gurobi_params)
        _print_single(sol, args.json)

    elif method == "epsilon-t":
        if args.epsilon is None:
            print("--epsilon is required for method 'epsilon-t'.", file=sys.stderr)
            sys.exit(1)
        sol = solve_epsilon_turnaround(inst, args.epsilon, formulation=formulation, **gurobi_params)
        _print_single(sol, args.json)

    # ---- Pareto frontier methods ------------------------------------------

    elif method == "pareto-ws":
        solutions = weighted_sum_frontier(
            inst, n_points=args.steps, formulation=formulation, **gurobi_params
        )
        _print_pareto(solutions, args.json)

    elif method == "pareto-eps":
        solutions = epsilon_constraint_frontier(
            inst, n_points=args.steps, formulation=formulation, **gurobi_params
        )
        _print_pareto(solutions, args.json)


def _print_single(sol, as_json: bool) -> None:
    if as_json:
        print(json.dumps(_sol_to_dict(sol), indent=2))
    else:
        print(sol)


def _print_pareto(solutions: list, as_json: bool) -> None:
    if as_json:
        print(json.dumps([_sol_to_dict(s) for s in solutions], indent=2))
    else:
        print(f"Pareto frontier: {len(solutions)} non-dominated point(s)\n")
        for i, s in enumerate(sorted(solutions, key=lambda s: s.f1 or 0)):
            print(f"--- Point {i + 1} ---")
            print(s)
            print()


def _sol_to_dict(sol) -> dict:
    assignment_detail = {}
    for jid, (type_id, start_time) in sol.assignment.items():
        assignment_detail[str(jid)] = {
            "type_id": type_id,
            "start_time": start_time,
        }
    return {
        "status": sol.status,
        "objective": sol.objective,
        "f1_turnaround": sol.f1,
        "f2_cost": sol.f2,
        "assignment": assignment_detail,
        "completion": {str(k): v for k, v in sol.completion.items()},
    }
