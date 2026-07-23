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
           const std::vector<std::tuple<int,int,int>>& init_occ,
           int pop_size, int n_gen, int seed, int n_threads) {
            auto [results, prof] = run_nsga2(
                build_problem(n_jobs, budget, job_slots, type_cap, init_occ),
                pop_size, n_gen, seed, n_threads);
            return py::make_tuple(results, profile_to_dict(prof));
        },
        py::arg("n_jobs"),
        py::arg("budget"),
        py::arg("job_slots"),
        py::arg("type_cap"),
        py::arg("init_occ"),
        py::arg("pop_size")   = 100,
        py::arg("n_gen")      = 200,
        py::arg("seed")       = 42,
        py::arg("n_threads")  = 0,
        "NSGA-II Pareto frontier approximation (parallel).\n\n"
        "Returns (solutions, profile_dict) where profile_dict maps phase names\n"
        "to wall-clock milliseconds.  n_threads=0 lets OpenMP choose."
    );

    m.def("nsga3",
        [](int n_jobs, double budget,
           const std::vector<std::vector<std::tuple<int,int,int,double,double>>>& job_slots,
           const std::vector<int>& type_cap,
           const std::vector<std::tuple<int,int,int>>& init_occ,
           int pop_size, int n_divisions, int n_gen, int seed, int n_threads) {
            auto [results, prof] = run_nsga3(
                build_problem(n_jobs, budget, job_slots, type_cap, init_occ),
                pop_size, n_divisions, n_gen, seed, n_threads);
            return py::make_tuple(results, profile_to_dict(prof));
        },
        py::arg("n_jobs"),
        py::arg("budget"),
        py::arg("job_slots"),
        py::arg("type_cap"),
        py::arg("init_occ"),
        py::arg("pop_size")    = 100,
        py::arg("n_divisions") = 99,
        py::arg("n_gen")       = 200,
        py::arg("seed")        = 42,
        py::arg("n_threads")   = 0,
        "NSGA-III Pareto frontier approximation (parallel, reference-point selection).\n\n"
        "pop_size should equal n_divisions + 1 for best reference-point coverage.\n"
        "Returns (solutions, profile_dict) where profile_dict maps phase names\n"
        "to wall-clock milliseconds.  n_threads=0 lets OpenMP choose."
    );

    m.def("moead",
        [](int n_jobs, double budget,
           const std::vector<std::vector<std::tuple<int,int,int,double,double>>>& job_slots,
           const std::vector<int>& type_cap,
           const std::vector<std::tuple<int,int,int>>& init_occ,
           int n_weights, int n_gen, int neighborhood_size,
           int seed, int n_threads, int max_replace) {
            auto [results, prof] = run_moead(
                build_problem(n_jobs, budget, job_slots, type_cap, init_occ),
                n_weights, n_gen, neighborhood_size, seed, n_threads, max_replace);
            return py::make_tuple(results, profile_to_dict(prof));
        },
        py::arg("n_jobs"),
        py::arg("budget"),
        py::arg("job_slots"),
        py::arg("type_cap"),
        py::arg("init_occ"),
        py::arg("n_weights")         = 100,
        py::arg("n_gen")             = 200,
        py::arg("neighborhood_size") = 20,
        py::arg("seed")              = 42,
        py::arg("n_threads")         = 0,
        py::arg("max_replace")       = -1,
        "MOEA/D Pareto frontier approximation (parallel, Tchebycheff decomposition).\n\n"
        "max_replace caps how many neighbours a single child may overwrite per\n"
        "generation (Zhang & Li's nr); <=0 disables the cap (original behaviour).\n"
        "Returns (solutions, profile_dict) where profile_dict maps phase names\n"
        "to wall-clock milliseconds.  n_threads=0 lets OpenMP choose."
    );
}
