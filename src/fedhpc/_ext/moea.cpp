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
        [](int n_jobs, double budget,
           const std::vector<std::vector<std::tuple<int,int,int,double,double>>>& job_slots,
           const std::vector<int>& type_cap,
           const std::vector<double>& type_risk,
           const std::vector<std::tuple<int,int,int>>& init_occ,
           int pop_size, int n_gen, int seed, int n_threads,
           double p_mut_start, double p_mut_end, int crossover_kind, int tourn_k,
           int local_search_interval) {
            auto [results, prof] = run_nsga2(
                build_problem(n_jobs, budget, job_slots, type_cap, type_risk, init_occ),
                pop_size, n_gen, seed, n_threads, p_mut_start, p_mut_end, crossover_kind,
                tourn_k, local_search_interval);
            return py::make_tuple(results, profile_to_dict(prof));
        },
        py::arg("n_jobs"),
        py::arg("budget"),
        py::arg("job_slots"),
        py::arg("type_cap"),
        py::arg("type_risk"),
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
        [](int n_jobs, double budget,
           const std::vector<std::vector<std::tuple<int,int,int,double,double>>>& job_slots,
           const std::vector<int>& type_cap,
           const std::vector<double>& type_risk,
           const std::vector<std::tuple<int,int,int>>& init_occ,
           int pop_size, int n_divisions, int n_gen, int seed, int n_threads,
           double p_mut_start, double p_mut_end, int crossover_kind, int tourn_k,
           int local_search_interval) {
            auto [results, prof] = run_nsga3(
                build_problem(n_jobs, budget, job_slots, type_cap, type_risk, init_occ),
                pop_size, n_divisions, n_gen, seed, n_threads, p_mut_start, p_mut_end,
                crossover_kind, tourn_k, local_search_interval);
            return py::make_tuple(results, profile_to_dict(prof));
        },
        py::arg("n_jobs"),
        py::arg("budget"),
        py::arg("job_slots"),
        py::arg("type_cap"),
        py::arg("type_risk"),
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
        [](int n_jobs, double budget,
           const std::vector<std::vector<std::tuple<int,int,int,double,double>>>& job_slots,
           const std::vector<int>& type_cap,
           const std::vector<double>& type_risk,
           const std::vector<std::tuple<int,int,int>>& init_occ,
           int n_weights, int n_gen, int neighborhood_size,
           int seed, int n_threads, int max_replace,
           double p_mut_start, double p_mut_end, int crossover_kind, int archive_size,
           int local_search_interval) {
            auto [results, prof] = run_moead(
                build_problem(n_jobs, budget, job_slots, type_cap, type_risk, init_occ),
                n_weights, n_gen, neighborhood_size, seed, n_threads, max_replace,
                p_mut_start, p_mut_end, crossover_kind, archive_size, local_search_interval);
            return py::make_tuple(results, profile_to_dict(prof));
        },
        py::arg("n_jobs"),
        py::arg("budget"),
        py::arg("job_slots"),
        py::arg("type_cap"),
        py::arg("type_risk"),
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
        "Returns (solutions, profile_dict) where profile_dict maps phase names\n"
        "to wall-clock milliseconds.  n_threads=0 lets OpenMP choose."
    );

    m.def("time_seeds",
        [](int n_jobs, double budget,
           const std::vector<std::vector<std::tuple<int,int,int,double,double>>>& job_slots,
           const std::vector<int>& type_cap,
           const std::vector<double>& type_risk,
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
                Problem prob = build_problem(n_jobs, budget, job_slots, type_cap, type_risk, init_occ);
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
        py::arg("budget"),
        py::arg("job_slots"),
        py::arg("type_cap"),
        py::arg("type_risk"),
        py::arg("init_occ"),
        "Diagnostic: time each deterministic heuristic seed's construction,\n"
        "initial evaluation, and local_search repair individually (not used by\n"
        "the main algorithms). Returns a list of per-seed dicts with timings\n"
        "(ms) and f1/f2/constraint-violation before and after repair."
    );
}
