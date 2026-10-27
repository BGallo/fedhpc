/**
 * FED-HPC multi-objective scheduling — pybind11 module binding.
 *
 * Algorithm implementations live in separate headers:
 *   ga_common.hpp — shared types, evaluation, operators, Pareto utilities
 *   nsga2.hpp     — NSGA-II with constrained-dominance ranking
 *   nsga3.hpp     — NSGA-III with reference-point-based selection (Deb & Jain 2014)
 *   moead.hpp     — MOEA/D with normalised Tchebycheff decomposition
 *
 * All three algorithms return (ResultList, Profile).  The bindings below
 * expose this as a Python tuple (solutions, profile_dict) where profile_dict
 * maps phase names to wall-clock milliseconds.
 */

#include "nsga2.hpp"
#include "nsga3.hpp"
#include "moead.hpp"
#include "weighted.hpp"
#include "weighted_brkga.hpp"
#include "weighted_dispatch.hpp"
#include "weighted_memetic.hpp"
#include "sgs_algos.hpp"

// Convert a Profile (vector<pair<string,double>>) to a Python dict.
static py::dict profile_to_dict(const Profile& prof) {
    py::dict d;
    for (auto& [k, v] : prof)
        d[py::str(k)] = v;
    return d;
}

PYBIND11_MODULE(_moea, m) {
    m.doc() = "NSGA-II and MOEA/D for FED-HPC multi-objective scheduling (C++/OpenMP)";

#ifdef _OPENMP
    m.attr("openmp_enabled") = true;
    m.attr("max_threads")    = omp_get_max_threads();
#else
    m.attr("openmp_enabled") = false;
    m.attr("max_threads")    = 1;
#endif

    m.def("nsga2",
        [](int n_jobs,
           const std::vector<std::vector<std::tuple<int,int,int,double,double,double>>>& job_slots,
           const std::vector<int>& type_cap,
           const std::vector<std::tuple<int,int,int>>& init_occ,
           int pop_size, int n_gen, int seed, int n_threads,
           double p_mut_start, double p_mut_end, int crossover_kind, int tourn_k,
           int local_search_interval, int sched_repair, int ablate,
           const std::vector<std::vector<int>>& extra_seeds) {
            auto [results, prof] = run_nsga2(
                build_problem(n_jobs, job_slots, type_cap, init_occ, ablate),
                pop_size, n_gen, seed, n_threads, p_mut_start, p_mut_end, crossover_kind,
                tourn_k, local_search_interval, sched_repair, ablate, extra_seeds);
            return py::make_tuple(results, profile_to_dict(prof));
        },
        py::arg("n_jobs"),
        py::arg("job_slots"),
        py::arg("type_cap"),
        py::arg("init_occ"),
        py::arg("pop_size")       = 100,
        py::arg("n_gen")          = 200,
        py::arg("seed")           = 42,
        py::arg("n_threads")      = 0,
        py::arg("p_mut_start")    = -1.0,
        py::arg("p_mut_end")      = -1.0,
        py::arg("crossover_kind") = 0,
        py::arg("tourn_k")        = 2,
        py::arg("local_search_interval") = -1,
        py::arg("sched_repair")   = 0,
        py::arg("ablate")         = 0,
        py::arg("extra_seeds")    = std::vector<std::vector<int>>{},
        "NSGA-II Pareto frontier approximation (parallel).\n\n"
        "p_mut_start/p_mut_end linearly anneal the mutation rate across\n"
        "generations; <0 (default) resolves to the fixed formula rate 2/n_jobs.\n"
        "crossover_kind: 0 = two-point (default), 1 = uniform.\n"
        "tourn_k: mating-tournament size; 2 (default) = original binary tournament.\n"
        "local_search_interval: generations between periodic congestion-repair /\n"
        "cost-descent local search on the population; <0 (default) resolves to\n"
        "~10 applications spread across the run, 0 disables (seeds+final only).\n"
        "Returns (solutions, profile_dict) where profile_dict maps phase names\n"
        "to wall-clock milliseconds.  n_threads=0 lets OpenMP choose."
    );

    m.def("nsga3",
        [](int n_jobs,
           const std::vector<std::vector<std::tuple<int,int,int,double,double,double>>>& job_slots,
           const std::vector<int>& type_cap,
           const std::vector<std::tuple<int,int,int>>& init_occ,
           int pop_size, int n_divisions, int n_gen, int seed, int n_threads,
           double p_mut_start, double p_mut_end, int crossover_kind, int tourn_k,
           int local_search_interval, int sched_repair, int ablate,
           const std::vector<std::vector<int>>& extra_seeds) {
            auto [results, prof] = run_nsga3(
                build_problem(n_jobs, job_slots, type_cap, init_occ, ablate),
                pop_size, n_divisions, n_gen, seed, n_threads, p_mut_start, p_mut_end,
                crossover_kind, tourn_k, local_search_interval, sched_repair, ablate, extra_seeds);
            return py::make_tuple(results, profile_to_dict(prof));
        },
        py::arg("n_jobs"),
        py::arg("job_slots"),
        py::arg("type_cap"),
        py::arg("init_occ"),
        py::arg("pop_size")       = 100,
        py::arg("n_divisions")    = 99,
        py::arg("n_gen")          = 200,
        py::arg("seed")           = 42,
        py::arg("n_threads")      = 0,
        py::arg("p_mut_start")    = -1.0,
        py::arg("p_mut_end")      = -1.0,
        py::arg("crossover_kind") = 0,
        py::arg("tourn_k")        = 2,
        py::arg("local_search_interval") = -1,
        py::arg("sched_repair")   = 0,
        py::arg("ablate")         = 0,
        py::arg("extra_seeds")    = std::vector<std::vector<int>>{},
        "NSGA-III Pareto frontier approximation (parallel, reference-point selection).\n\n"
        "pop_size should equal n_divisions + 1 for best reference-point coverage.\n"
        "p_mut_start/p_mut_end linearly anneal the mutation rate across\n"
        "generations; <0 (default) resolves to the fixed formula rate 2/n_jobs.\n"
        "crossover_kind: 0 = two-point (default), 1 = uniform.\n"
        "tourn_k: mating-tournament size; 2 (default) = original binary tournament.\n"
        "local_search_interval: generations between periodic congestion-repair /\n"
        "cost-descent local search on the population; <0 (default) resolves to\n"
        "~10 applications spread across the run, 0 disables (seeds+final only).\n"
        "Returns (solutions, profile_dict) where profile_dict maps phase names\n"
        "to wall-clock milliseconds.  n_threads=0 lets OpenMP choose."
    );

    m.def("moead",
        [](int n_jobs,
           const std::vector<std::vector<std::tuple<int,int,int,double,double,double>>>& job_slots,
           const std::vector<int>& type_cap,
           const std::vector<std::tuple<int,int,int>>& init_occ,
           int n_weights, int n_gen, int neighborhood_size,
           int seed, int n_threads, int max_replace,
           double p_mut_start, double p_mut_end, int crossover_kind, int archive_size,
           int local_search_interval, int sched_repair, int scalar_ls_interval, int ablate,
           const std::vector<std::vector<int>>& extra_seeds) {
            auto [results, prof] = run_moead(
                build_problem(n_jobs, job_slots, type_cap, init_occ, ablate),
                n_weights, n_gen, neighborhood_size, seed, n_threads, max_replace,
                p_mut_start, p_mut_end, crossover_kind, archive_size, local_search_interval,
                sched_repair, scalar_ls_interval, ablate, extra_seeds);
            return py::make_tuple(results, profile_to_dict(prof));
        },
        py::arg("n_jobs"),
        py::arg("job_slots"),
        py::arg("type_cap"),
        py::arg("init_occ"),
        py::arg("n_weights")         = 100,
        py::arg("n_gen")             = 200,
        py::arg("neighborhood_size") = 20,
        py::arg("seed")              = 42,
        py::arg("n_threads")         = 0,
        py::arg("max_replace")       = -1,
        py::arg("p_mut_start")       = -1.0,
        py::arg("p_mut_end")         = -1.0,
        py::arg("crossover_kind")    = 0,
        py::arg("archive_size")      = 0,
        py::arg("local_search_interval") = -1,
        py::arg("sched_repair")      = 0,
        py::arg("scalar_ls_interval") = 0,
        py::arg("ablate")           = 0,
        py::arg("extra_seeds")      = std::vector<std::vector<int>>{},
        "MOEA/D Pareto frontier approximation (parallel, Tchebycheff decomposition).\n\n"
        "max_replace caps how many neighbours a single child may overwrite per\n"
        "generation (Zhang & Li's nr); <=0 disables the cap (original behaviour).\n"
        "p_mut_start/p_mut_end linearly anneal the mutation rate across\n"
        "generations; <0 (default) resolves to the fixed formula rate 1/n_jobs.\n"
        "crossover_kind: 0 = two-point (default), 1 = uniform.\n"
        "archive_size: bounded elitist archive merged in at extraction;\n"
        "0 (default) disables it (original extraction path, unchanged).\n"
        "local_search_interval: generations between periodic congestion-repair /\n"
        "cost-descent local search on the population; <0 (default) resolves to\n"
        "~10 applications spread across the run, 0 disables (seeds+final only).\n"
        "scalar_ls_interval: *scalarised* local search (weighted-sum type-flip\n"
        "hill climb along each subproblem's own weight direction) — makes\n"
        "trade-off moves the dominance-only local search cannot. 0 (raw-binding\n"
        "default) disables it, output byte-for-byte unchanged. <0: strong\n"
        "two-pass final-population polish only, budget |value|, polished points\n"
        "emitted alongside the unpolished ones (pure upside). >0: in-loop pass\n"
        "every N gens + the final polish. Deterministic for fixed\n"
        "(seed, n_threads). moea.moead_frontier defaults to -30.\n"
        "Returns (solutions, profile_dict) where profile_dict maps phase names\n"
        "to wall-clock milliseconds.  n_threads=0 lets OpenMP choose."
    );

    m.def("weighted",
        [](int n_jobs,
           const std::vector<std::vector<std::tuple<int,int,int,double,double,double>>>& job_slots,
           const std::vector<int>& type_cap,
           const std::vector<std::tuple<int,int,int>>& init_occ,
           double w1, double w2, double f1_cap,
           int pop_size, int n_gen, int seed, int n_threads,
           int ls_moves, int restart_patience, int shortlist, int ablate,
           const std::vector<std::vector<int>>& extra_seeds, int xover_mode, int mut_mode,
           int decode_order, int fbi_passes, int ls_mode) {
            auto [asgn, f1, f2, g, ls_calls] = run_weighted(
                build_problem(n_jobs, job_slots, type_cap, init_occ, ablate),
                w1, w2, f1_cap, pop_size, n_gen, seed, n_threads,
                ls_moves, restart_patience, shortlist, ablate, extra_seeds, xover_mode, mut_mode,
                decode_order, fbi_passes, ls_mode);
            return py::make_tuple(asgn, f1, f2, g, ls_calls);
        },
        py::arg("n_jobs"),
        py::arg("job_slots"),
        py::arg("type_cap"),
        py::arg("init_occ"),
        py::arg("w1"),
        py::arg("w2"),
        py::arg("f1_cap")           = 1e18,
        py::arg("pop_size")         = 24,
        py::arg("n_gen")            = 40,
        py::arg("seed")             = 42,
        py::arg("n_threads")        = 0,
        py::arg("ls_moves")         = 6,
        py::arg("restart_patience") = 6,
        py::arg("shortlist")        = 24,
        py::arg("ablate")           = 0,
        py::arg("extra_seeds")      = std::vector<std::vector<int>>{},
        py::arg("xover_mode")       = 0,
        py::arg("mut_mode")        = 0,
        py::arg("decode_order")    = 0,
        py::arg("fbi_passes")      = 0,
        py::arg("ls_mode")         = 0,
        "Single-objective memetic metaheuristic: minimise w1*f1 + w2*f2 with a\n"
        "soft cap f1 <= f1_cap. Memetic GA over per-job type assignments with an\n"
        "SPT list-scheduling decoder + greedy scalar type-flip local search and\n"
        "ILS perturbation kicks on stagnation. xover_mode: 0 = two-point,\n"
        "1 = none (pure ILS), 2 = Multi-Step Crossover Fusion (scalar-guided walk\n"
        "from one parent toward the other; best on both 10-min instances). Raw\n"
        "binding default 0; moea.weighted_solve defaults to 2.\n"
        "decode_order: 0 (this raw binding's default) = SPT list-scheduling\n"
        "decoder only; 1 = also try an Extract-from-Preempt re-decode\n"
        "(preemptive-SRPT completion order, the release-date-aware 2-approx order)\n"
        "per individual; 2 = instead try a WSPT/Smith-rule re-decode for\n"
        "weighted-completion objectives. Both keep the better f1;\n"
        "moea.weighted_solve defaults to 1.\n"
        "fbi_passes: >0 applies that many forward-backward improvement (double\n"
        "justification) passes per individual, keeping the better f1. Both keep\n"
        "f2 fixed, so g is monotone; both are off by default (byte-for-byte\n"
        "unchanged). Deterministic for fixed (seed, n_threads).\n"
        "ls_mode: 0 = type-flip local search; 1 = on-prem/cloud split flip with\n"
        "half est-ranked / half random candidates.\n"
        "Returns (assignment, f1, f2, g, n_ls_calls)."
    );

    m.def("weighted_dispatch",
        [](int H,
           const std::vector<std::tuple<double, double,
               std::vector<std::tuple<int, int, double, int, int>>>>& jobs,
           const std::vector<int>& type_cap,
           const std::vector<std::tuple<int,int,int>>& init_occ,
           double w1, double w2, double f1_cap, int seed, int n_threads,
           int K, int max_iters, int n_kicks, int kick_size, int flags, int ls, double mu, int n_mu) {
            const DProblem P = build_dproblem(H, jobs, type_cap, init_occ);
            DispatchResult R = run_weighted_dispatch(P, w1, w2, f1_cap, seed, n_threads,
                                                     K, max_iters, n_kicks, kick_size, flags, ls, mu, n_mu);
            py::dict d;
            d["decodes"] = R.st.decodes; d["ls_moves"] = R.st.ls_moves; d["kicks"] = R.st.kicks;
            d["iters"] = R.st.iters; d["decode_ms"] = R.st.decode_ms; d["total_ms"] = R.total_ms;
            d["cv"] = R.best.cv; d["n_cloud"] = R.best.n_cloud;
            d["n_auto_off"] = R.best.n_auto_off; d["pen_g"] = R.best.pen_g;
            d["trace"] = R.st.trace; d["mu"] = R.st.mu;
            return py::make_tuple(R.asgn, R.best.f1, R.best.f2, R.best.g, d);
        },
        py::arg("H"), py::arg("jobs"), py::arg("type_cap"), py::arg("init_occ"),
        py::arg("w1"), py::arg("w2"), py::arg("f1_cap") = 1e18,
        py::arg("seed") = 42, py::arg("n_threads") = 0,
        py::arg("K") = 64, py::arg("max_iters") = 100000,
        py::arg("n_kicks") = 0, py::arg("kick_size") = 8,
        py::arg("flags") = 0, py::arg("ls") = 1, py::arg("mu") = 0.0, py::arg("n_mu") = 1,
        "Weighted-sum solver on the time-driven dispatch decoder (see\n"
        "weighted_dispatch.hpp). jobs[j] = (arrival, weight, [(type, p_occ,\n"
        "cost, earliest_start, latest_start), ...]). flags: DFlag bitmask\n"
        "(1 no deadline offload, 2/4 pool tie-break most-free/best-fit,\n"
        "8 random LS candidates, 16 no recall moves, 32 no ILS kicks,\n"
        "64 half est-ranked / half random LS candidates, 256 serial SGS\n"
        "decoder, 512 best of dispatch and serial SGS).\n"
        "mu: dispatch priority mix (0 = WSPT; < 0 = search from the n_mu best of a\n"
        "grid by all-auto decode; n_kicks is split across them).\n"
        "Returns (assignment, f1, f2, g, stats)."
    );

    m.def("weighted_memetic",
        [](int n_jobs,
           const std::vector<std::vector<std::tuple<int,int,int,double,double,double>>>& job_slots,
           const std::vector<int>& type_cap,
           const std::vector<std::tuple<int,int,int>>& init_occ,
           int H,
           const std::vector<std::tuple<double, double,
               std::vector<std::tuple<int, int, double, int, int>>>>& jobs,
           double w1, double w2, double f1_cap, int pop_size, int n_gen, int seed,
           int n_threads, int ls_moves, int restart_patience, int shortlist,
           int xover_mode, int flags, double mu, int ablate, int adapt) {
            const Problem prob = build_problem(n_jobs, job_slots, type_cap, init_occ, ablate);
            const DProblem P = build_dproblem(H, jobs, type_cap, init_occ);
            MemeticResult R = run_weighted_memetic(prob, P, w1, w2, f1_cap, pop_size, n_gen, seed,
                                                   n_threads, ls_moves, restart_patience, shortlist,
                                                   xover_mode, flags, mu, ablate, adapt);
            py::dict d;
            d["decodes"] = R.decodes; d["ls_calls"] = R.ls_calls; d["kicks"] = R.kicks;
            d["total_ms"] = R.total_ms; d["cv"] = R.best.cv; d["n_cloud"] = R.best.n_cloud;
            d["n_auto_off"] = R.best.n_auto_off; d["pen_g"] = R.best.pen_g;
            d["trace"] = R.trace; d["mu"] = R.mu; d["deadline"] = R.dl; d["serial"] = R.dec;
            d["mu_hist"] = R.mu_hist; d["deadline_hist"] = R.dl_hist; d["serial_hist"] = R.dec_hist;
            return py::make_tuple(R.asgn, R.best.f1, R.best.f2, R.best.g, d);
        },
        py::arg("n_jobs"), py::arg("job_slots"), py::arg("type_cap"),
        py::arg("init_occ"), py::arg("H"), py::arg("jobs"),
        py::arg("w1"), py::arg("w2"), py::arg("f1_cap") = 1e18,
        py::arg("pop_size") = 32, py::arg("n_gen") = 60, py::arg("seed") = 42,
        py::arg("n_threads") = 0, py::arg("ls_moves") = 8, py::arg("restart_patience") = 6,
        py::arg("shortlist") = 24, py::arg("xover_mode") = 2, py::arg("flags") = 64,
        py::arg("mu") = -1.0, py::arg("ablate") = 0, py::arg("adapt") = 0,
        "The weighted() memetic GA on the dispatch decoder (weighted_memetic.hpp):\n"
        "auto/cloud/on-prem genome, dispatch decoder, dispatch_ls local search.\n"
        "job_slots/type_cap/... as for weighted() (heuristic seeds); H/jobs as for\n"
        "weighted_dispatch(). xover_mode 0 = two-point, 1 = none, 2 = MSXF.\n"
        "flags: DFlag bitmask (128 = weighted.hpp's improving-estimate shortlist).\n"
        "mu: dispatch priority (< 0 = best of a grid by all-auto decode).\n"
        "flags 256 = serial SGS decoder, 512 = best of dispatch and serial SGS.\n"
        "adapt: MAdapt bitmask of per-individual settings (1 mu grid index,\n"
        "2 deadline offload, 4 dispatch/serial decoder, 8 setting moves in LS).\n"
        "Returns (assignment, f1, f2, g, stats)."
    );

    m.def("weighted_brkga",
        [](int n_jobs,
           const std::vector<std::vector<std::tuple<int,int,int,double,double,double>>>& job_slots,
           const std::vector<int>& type_cap,
           const std::vector<std::tuple<int,int,int>>& init_occ,
           double w1, double w2, double f1_cap,
           int pop_size, int n_gen, int seed, int n_threads, int ablate,
           const std::vector<std::vector<int>>& extra_seeds,
           int use_delay, int local_polish) {
            auto [asgn, f1, f2, g, decodes] = run_weighted_brkga(
                build_problem(n_jobs, job_slots, type_cap, init_occ, ablate),
                w1, w2, f1_cap, pop_size, n_gen, seed, n_threads,
                ablate, extra_seeds, use_delay, local_polish);
            return py::make_tuple(asgn, f1, f2, g, decodes);
        },
        py::arg("n_jobs"),
        py::arg("job_slots"),
        py::arg("type_cap"),
        py::arg("init_occ"),
        py::arg("w1"),
        py::arg("w2"),
        py::arg("f1_cap")        = 1e18,
        py::arg("pop_size")      = 48,
        py::arg("n_gen")         = 100,
        py::arg("seed")          = 42,
        py::arg("n_threads")     = 0,
        py::arg("ablate")        = 0,
        py::arg("extra_seeds")   = std::vector<std::vector<int>>{},
        py::arg("use_delay")     = 1,
        py::arg("local_polish")  = 2,
        "Weighted-sum solver, random-key / serial-SGS encoding (options B+C).\n"
        "Chromosome = 3*n_jobs random keys (priority | delay | type); decoder is\n"
        "a serial schedule generation scheme driven by the evolved priority\n"
        "order, with a per-job delay floor (parameterized-active schedules,\n"
        "Mendes-Goncalves-Resende 2005). BRKGA population management (elite /\n"
        "mutant / biased-uniform crossover). use_delay=0 disables the delay\n"
        "block (pure serial SGS = option B only). local_polish=1 runs the\n"
        "type-flip local search on each decoded schedule. Deterministic for\n"
        "fixed (seed, n_threads). Returns (assignment, f1, f2, g, n_decodes)."
    );

    // ── Priority-key + non-delay-SGS representation (sgs_algos.hpp) ───────────
    // Standard-literature chromosome alternative: a job-processing permutation
    // + per-job type-choice index, decoded via a non-delay Schedule Generation
    // Scheme, instead of ga_common.hpp's job_slots-index encoding. See
    // sgs_common.hpp's file comment for the full rationale.

    m.def("nsga2_sgs",
        [](int n_jobs, int horizon,
           const std::vector<double>& arrival,
           const std::vector<std::vector<std::tuple<int,int,int,int,int,double>>>& cand,
           const std::vector<std::tuple<int,int,int>>& init_occ,
           int pop_size, int n_gen, int seed, int n_threads,
           double p_mut_start, double p_mut_end, int crossover_kind, int tourn_k,
           int local_search_interval) {
            auto [results, prof] = run_nsga2_sgs(
                build_sgs_problem(n_jobs, horizon, arrival, cand, init_occ),
                pop_size, n_gen, seed, n_threads, p_mut_start, p_mut_end,
                crossover_kind, tourn_k, local_search_interval);
            return py::make_tuple(results, profile_to_dict(prof));
        },
        py::arg("n_jobs"), py::arg("horizon"),
        py::arg("arrival"), py::arg("cand"), py::arg("init_occ"),
        py::arg("pop_size") = 100, py::arg("n_gen") = 200, py::arg("seed") = 42,
        py::arg("n_threads") = 0, py::arg("p_mut_start") = -1.0, py::arg("p_mut_end") = -1.0,
        py::arg("crossover_kind") = 0, py::arg("tourn_k") = 2,
        py::arg("local_search_interval") = -1,
        "NSGA-II on the priority-key + non-delay-SGS representation "
        "(see sgs_common.hpp). Same dominance/crowding selection as nsga2(), "
        "different chromosome and decoder. local_search_interval: generations "
        "between periodic local_search_sgs() passes on survivors (-1 = auto, "
        "max(1, n_gen/10)); a final polish always runs before extraction."
    );

    m.def("nsga3_sgs",
        [](int n_jobs, int horizon,
           const std::vector<double>& arrival,
           const std::vector<std::vector<std::tuple<int,int,int,int,int,double>>>& cand,
           const std::vector<std::tuple<int,int,int>>& init_occ,
           int pop_size, int n_divisions, int n_gen, int seed, int n_threads,
           double p_mut_start, double p_mut_end, int crossover_kind, int tourn_k,
           int local_search_interval) {
            auto [results, prof] = run_nsga3_sgs(
                build_sgs_problem(n_jobs, horizon, arrival, cand, init_occ),
                pop_size, n_divisions, n_gen, seed, n_threads, p_mut_start, p_mut_end,
                crossover_kind, tourn_k, local_search_interval);
            return py::make_tuple(results, profile_to_dict(prof));
        },
        py::arg("n_jobs"), py::arg("horizon"),
        py::arg("arrival"), py::arg("cand"), py::arg("init_occ"),
        py::arg("pop_size") = 100, py::arg("n_divisions") = 99, py::arg("n_gen") = 200,
        py::arg("seed") = 42, py::arg("n_threads") = 0,
        py::arg("p_mut_start") = -1.0, py::arg("p_mut_end") = -1.0,
        py::arg("crossover_kind") = 0, py::arg("tourn_k") = 2,
        py::arg("local_search_interval") = -1,
        "NSGA-III on the priority-key + non-delay-SGS representation "
        "(see sgs_common.hpp). Same reference-point niching as nsga3(), "
        "different chromosome and decoder. local_search_interval: see nsga2_sgs()."
    );

    m.def("moead_sgs",
        [](int n_jobs, int horizon,
           const std::vector<double>& arrival,
           const std::vector<std::vector<std::tuple<int,int,int,int,int,double>>>& cand,
           const std::vector<std::tuple<int,int,int>>& init_occ,
           int n_weights, int n_gen, int neighborhood_size,
           int seed, int n_threads, int max_replace,
           double p_mut_start, double p_mut_end, int crossover_kind, int archive_size,
           int local_search_interval) {
            auto [results, prof] = run_moead_sgs(
                build_sgs_problem(n_jobs, horizon, arrival, cand, init_occ),
                n_weights, n_gen, neighborhood_size, seed, n_threads, max_replace,
                p_mut_start, p_mut_end, crossover_kind, archive_size, local_search_interval);
            return py::make_tuple(results, profile_to_dict(prof));
        },
        py::arg("n_jobs"), py::arg("horizon"),
        py::arg("arrival"), py::arg("cand"), py::arg("init_occ"),
        py::arg("n_weights") = 100, py::arg("n_gen") = 200, py::arg("neighborhood_size") = 20,
        py::arg("seed") = 42, py::arg("n_threads") = 0, py::arg("max_replace") = -1,
        py::arg("p_mut_start") = -1.0, py::arg("p_mut_end") = -1.0,
        py::arg("crossover_kind") = 0, py::arg("archive_size") = 0,
        py::arg("local_search_interval") = -1,
        "MOEA/D on the priority-key + non-delay-SGS representation "
        "(see sgs_common.hpp). Same Tchebycheff decomposition + bounded "
        "neighbourhood replacement as moead(), different chromosome and "
        "decoder; local_search_interval: see nsga2_sgs(). No scalar_ls_interval "
        "polish (no SGS analogue)."
    );

    m.def("time_seeds",
        [](int n_jobs,
           const std::vector<std::vector<std::tuple<int,int,int,double,double,double>>>& job_slots,
           const std::vector<int>& type_cap,
           const std::vector<std::tuple<int,int,int>>& init_occ) {
            struct SeedTiming {
                std::string name;
                double construct_ms, eval_ms, repair_ms;
                double f1_before, f2_before, cv_before;
                double f1_after,  f2_after,  cv_after;
            };
            std::vector<SeedTiming> timings;
            {
                py::gil_scoped_release release;
                Problem prob = build_problem(n_jobs, job_slots, type_cap, init_occ);
                EvalWorkspace ws;
                ws.reset(prob.n_types, prob.max_slot);

                auto record = [&](const char* name, auto ctor) {
                    const auto t0 = Clock::now();
                    Individual ind = ctor();
                    const double construct_ms = ms_since(t0);

                    const auto t1 = Clock::now();
                    evaluate(ind, prob, ws);
                    const double eval_ms = ms_since(t1);
                    const double f1_before = ind.f1, f2_before = ind.f2, cv_before = ind.cv;

                    const auto t2 = Clock::now();
                    local_search(ind, prob, ws);
                    evaluate(ind, prob, ws);
                    const double repair_ms = ms_since(t2);

                    timings.push_back({name, construct_ms, eval_ms, repair_ms,
                                        f1_before, f2_before, cv_before,
                                        ind.f1, ind.f2, ind.cv});
                };

                record("greedy_time",   [&]{ return make_greedy(prob, false); });
                record("greedy_cost",   [&]{ return make_greedy(prob, true); });
                record("no_wait",       [&]{ return make_no_wait(prob); });
                record("full_burst",    [&]{ return make_full_burst(prob); });
                record("fixed_wait_25", [&]{ return make_fixed_wait(prob, 0.25); });
                record("fixed_wait_50", [&]{ return make_fixed_wait(prob, 0.50); });
                record("star_wait",     [&]{ return make_star_wait(prob); });
                record("list_schedule", [&]{ return make_list_schedule(prob); });
            }

            py::list out;
            for (auto& t : timings) {
                py::dict d;
                d["name"]         = t.name;
                d["construct_ms"] = t.construct_ms;
                d["eval_ms"]      = t.eval_ms;
                d["repair_ms"]    = t.repair_ms;
                d["f1_before"]    = t.f1_before;
                d["f2_before"]    = t.f2_before;
                d["cv_before"]    = t.cv_before;
                d["f1_after"]     = t.f1_after;
                d["f2_after"]     = t.f2_after;
                d["cv_after"]     = t.cv_after;
                out.append(d);
            }
            return out;
        },
        py::arg("n_jobs"),
        py::arg("job_slots"),
        py::arg("type_cap"),
        py::arg("init_occ"),
        "Diagnostic: time each deterministic heuristic seed's construction,\n"
        "initial evaluation, and local_search repair individually (not used by\n"
        "the main algorithms). Returns a list of per-seed dicts with timings\n"
        "(ms) and f1/f2/constraint-violation before and after repair."
    );
}
