# FED-HPC EA improvement — brief for a coding agent

Full implementation spec + benchmark: Artifact "FED-HPC Evolutionary Solver"
<https://claude.ai/code/artifact/5bc984c3-b0d2-4f5b-9b67-b17fd9238a0a>

Paste the prompt below into a coding-agent session with this repo checked out.

---

You are working in the FED-HPC repository (multi-objective MIP + evolutionary
scheduler for federated HPC jobs). Your task: improve the evolutionary solver so
its Pareto-front approximation on the large congested instance measurably
approaches the proven-optimal front, without breaking determinism, feasibility,
or the test suite.

CONTEXT
- EA code: src/fedhpc/_ext/ga_common.hpp (encoding, evaluate, seeds, operators,
  local_search), nsga2.hpp, nsga3.hpp, moead.hpp, moea.cpp (pybind11 bindings);
  Python wrappers in src/fedhpc/moea.py.
- Encoding: integer vector, gene j indexes job_slots[j] = feasible (type,start)
  slots. Objectives f1 = total turnaround (slots), f2 = total cost. Single
  scalar constraint violation cv (capacity grid + budget); feasible iff cv==0.
- Exact reference: src/fedhpc/pareto.py true_pareto_frontier / map_pareto_frontier
  (exact box-splitting, one proven-optimal MIP point per solve).

THE PROBLEM
On data/fedhpc_known_runtime_offline_10min.json (964 jobs, 29 types, ~7.6M
job-slots) the EA never reaches the proven front. Scored against the 40 proven-
optimal points in pareto_runs/*.json (load via _solution_from_dict + _filter_
dominated): at pop=400/n_gen=2000 NSGA-II gets IGD 0.083, eps+ 0.110, 0 exact
hits, ~87% of its points strictly dominated by a proven point. Pushing to
pop=800/n_gen=6000 only reaches IGD 0.051 -- a structural ceiling, not a budget
limit. Correctness is intact: no EA point ever dominates a proven point, all are
feasible. The small bundled instances (smallest/small/xlarge) are already solved
exactly and must stay that way.

LIKELY CAUSES (verify or discard -- do not assume)
1. Mutation rate ~1.5e-3 mutates ~1 gene/gen, resampled UNIFORMLY over a
   thousands-long slot list -> almost never a useful start time. No move biased
   toward nearby starts, alternative types, or decongesting a specific slot.
2. local_search is coordinate descent over a per-job shortlist -- it polishes,
   it cannot find the global reallocation the cost-minimal corner needs (nearly
   all jobs load-balanced across the horizon on 1-2 free on-prem pools).
3. Crossover is index-level; no notion of a transferable "good block" (a
   decongested window, a coherent on-prem packing).
4. Heavy job/slot symmetry (same structure that makes the exact MIP hard here).
5. MOEA/D drops the turnaround-minimal extreme (eps+ 0.24 at high budget).

DELIVERABLE
- A focused, A/B-measurable change: a new operator behind a default-off param,
  or a new default with the old path still reachable. New params MUST keep their
  default a byte-for-byte match to current behaviour (see the existing
  max_replace / archive_size / p_mut_* / local_search_interval contracts and
  their tests).
- Candidate directions worth trying (your call): start-time-local mutation
  (Gaussian / +-k slots on the current start, same type); a targeted
  decongestion move (pick an over-subscribed (type,slot), relocate one
  contending job to its cheapest slack slot); a horizon-shift / block crossover
  that transplants a coherent time window; capacity-aware load-balancing seeds
  for the cost-minimal basin; a path-relinking pass between the current front's
  neighbours; restart / hypermutation when a subproblem stalls.

CONSTRAINTS (hard)
- Determinism: output bit-for-bit stable for fixed (seed, n_threads). New
  randomness comes from per-thread RNGs seeded before the parallel region.
- Every returned Solution has cv==0 and objectives matching a fresh recompute.
- uv run pytest tests/test_moea.py  (182 tests) stays green.
- Rebuild after any _ext/ change:  uv sync --reinstall-package fedhpc
  (the installed .so goes stale against source).

MEASURE (report before/after)
- POP=400 NGEN=2000 uv run python scripts/compare_moea_vs_known_front.py
  -> IGD, eps+, exact-hit count, dominated fraction for all three algorithms.
- uv run python scripts/compare_moea_vs_true.py
  -> small/medium/large instances must not regress.
Report the table, the exact change, and which hypotheses it confirmed or killed.
