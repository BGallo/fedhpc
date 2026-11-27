"""fedhpc.mip: a branch-and-cut MIP solver on fedlp.

    from fedhpc.mip import read_mps, solve
    result = solve(read_mps("model.mps.gz"), time_limit=60)

Stages, one module each: ``problem`` (the MIP), ``mps`` (reader),
``presolve``, ``propagate`` (domain propagation), ``lp`` (the fedlp
relaxation), ``cuts`` (Gomory, covers, pool), ``heuristics``, ``branching``
(reliability pseudocosts) and ``solver`` (root cut loop, tree search).
"""
from .mps import read_mps
from .problem import MipProblem
from .solver import MipOptions, MipResult, MipSolver, solve

__all__ = ["MipOptions", "MipProblem", "MipResult", "MipSolver", "read_mps", "solve"]
