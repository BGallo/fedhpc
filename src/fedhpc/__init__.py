"""fedhpc: FED-HPC scheduling models, engines and the fedhpc.mip solver.

The public names below are imported on first use (PEP 562): ``import
fedhpc.mip`` must not pay for the CLI, the plotting stack (matplotlib, ~0.2 s)
and the metaheuristics, since every benchmark run starts a fresh process.
"""
from __future__ import annotations

import importlib

_EXPORTS = {
    "main": ".cli",
    "Instance": ".data",
    "InstanceType": ".data",
    "Job": ".data",
    "RunningJob": ".data",
    "pareto_metrics": ".metrics",
    "heuristic_weighted_reference_points": ".moea",
    "heuristic_weighted_sum_reference": ".moea",
    "moead_frontier": ".moea",
    "nsga2_frontier": ".moea",
    "weighted_solve": ".moea",
    "hybrid_frontier": ".pareto",
    "map_pareto_frontier": ".pareto",
    "true_pareto_frontier": ".pareto",
    "save_gantt": ".viz",
    "save_machine_schedule": ".viz",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    mod = _EXPORTS.get(name)
    if mod is None:
        raise AttributeError(f"module 'fedhpc' has no attribute {name!r}")
    value = getattr(importlib.import_module(mod, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
