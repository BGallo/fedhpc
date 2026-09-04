/**
 * FED-HPC — NSGA-II / NSGA-III / MOEA-D driver loops for the priority-key +
 * non-delay-SGS representation (sgs_common.hpp).
 *
 * These are near-duplicates of run_nsga2/run_nsga3/run_moead in
 * nsga2.hpp/nsga3.hpp/moead.hpp, with only the representation-specific calls
 * swapped (evaluate -> evaluate_sgs, crossover -> crossover_sgs, mutate ->
 * mutate_sgs, make_seeds -> make_heuristic_seeds_sgs, extract_assignment ->
 * extract_assignment_sgs). The dominance/crowding/niching/decomposition math
 * itself — fast_nds, crowding_distance, tournament(_k), the NSGA-III
 * reference-point functions, and MOEA/D's weight vectors + Tchebycheff
 * scalar + neighbourhood replacement — is reused verbatim from those headers
 * (it only ever touches Individual.f1/f2/cv/rank/crowding), not
 * re-implemented.
 *
 * Each driver now also runs local_search_sgs() (sgs_common.hpp) periodically
 * on the survivor population, every `local_search_interval` generations
 * (default -1 -> max(1, n_gen/10), same auto-interval convention as
 * nsga2.hpp/nsga3.hpp/moead.hpp) plus once as a final polish before
 * extraction — mirroring exactly how those headers use local_search().
 *
 * Still deliberately omitted relative to the job_slots-index drivers:
 * schedule_repair() (no SGS analogue), the ablation bitmask, extra_seeds/
 * lp_seeds, and MOEA/D's scalar_ls_interval polish (built around job_slots
 * type-flip moves).
 */
#pragma once

#include "sgs_common.hpp"
#include "nsga2.hpp"   // fast_nds, crowding_distance, tournament(_k)
#include "nsga3.hpp"   // make_reference_points_2d, tournament3(_k), nsga3_*, niching_select

// ── NSGA-II (SGS representation) ──────────────────────────────────────────────

inline std::pair<ResultList, Profile>
run_nsga2_sgs(const SgsProblem& prob, int pop_size, int n_gen, int seed, int n_threads,
             double p_mut_start = -1.0, double p_mut_end = -1.0, int crossover_kind = 0,
             int tourn_k = 2, int local_search_interval = -1) {
    py::gil_scoped_release release;
    set_num_threads(n_threads);
    const auto t_total = Clock::now();

    std::mt19937 rng(seed);
    const auto mut_sched = make_mutation_schedule(p_mut_start, p_mut_end, 2.0 / prob.n_jobs);
    const int ls_interval = (local_search_interval < 0)
        ? std::max(1, n_gen / 10) : local_search_interval;

    auto h_seeds = make_heuristic_seeds_sgs(prob);
    const int n_hs = static_cast<int>(h_seeds.size());

    std::vector<Individual> pop(pop_size);
    for (int k = 0; k < n_hs && k < pop_size; k++) pop[k] = h_seeds[k];

    std::vector<uint32_t> init_seeds(pop_size);
    for (auto& s : init_seeds) s = rng();

#ifdef _OPENMP
    #pragma omp parallel
    {
        SgsWorkspace ws; ws.reset(prob);
        #pragma omp for schedule(static)
        for (int k = 0; k < pop_size; k++) {
            if (k >= n_hs) { std::mt19937 lrng(init_seeds[k]); pop[k] = make_random_sgs(prob, lrng); }
            evaluate_sgs(pop[k], prob, ws);
        }
    }
#else
    {
        SgsWorkspace ws; ws.reset(prob);
        for (int k = 0; k < pop_size; k++) {
            if (k >= n_hs) { std::mt19937 lrng(init_seeds[k]); pop[k] = make_random_sgs(prob, lrng); }
            evaluate_sgs(pop[k], prob, ws);
        }
    }
#endif

    std::vector<Individual> offspring(pop_size);
    std::vector<Individual> combined;
    combined.reserve(2 * pop_size);
    std::vector<uint32_t> child_seeds(pop_size);

    for (int gen = 0; gen < n_gen; gen++) {
        auto fronts = fast_nds(pop);
        assign_ranks(pop, fronts);
        for (auto& f : fronts) crowding_distance(pop, f);

        for (auto& s : child_seeds) s = rng();
        const double p_mut_gen = mut_sched.at(gen, n_gen);

#ifdef _OPENMP
        #pragma omp parallel
        {
            SgsWorkspace ws; ws.reset(prob);
            #pragma omp for schedule(static)
            for (int k = 0; k < pop_size; k++) {
                std::mt19937 lrng(child_seeds[k]);
                std::uniform_int_distribution<int> ri(0, pop_size - 1);
                const Individual& p1 = (tourn_k == 2)
                    ? tournament(pop[ri(lrng)], pop[ri(lrng)]) : tournament_k(pop, tourn_k, lrng);
                const Individual& p2 = (tourn_k == 2)
                    ? tournament(pop[ri(lrng)], pop[ri(lrng)]) : tournament_k(pop, tourn_k, lrng);
                offspring[k] = crossover_sgs(p1, p2, lrng, crossover_kind, prob.n_jobs);
                mutate_sgs(offspring[k], prob, p_mut_gen, lrng);
                evaluate_sgs(offspring[k], prob, ws);
            }
        }
#else
        {
            SgsWorkspace ws; ws.reset(prob);
            for (int k = 0; k < pop_size; k++) {
                std::mt19937 lrng(child_seeds[k]);
                std::uniform_int_distribution<int> ri(0, pop_size - 1);
                const Individual& p1 = (tourn_k == 2)
                    ? tournament(pop[ri(lrng)], pop[ri(lrng)]) : tournament_k(pop, tourn_k, lrng);
                const Individual& p2 = (tourn_k == 2)
                    ? tournament(pop[ri(lrng)], pop[ri(lrng)]) : tournament_k(pop, tourn_k, lrng);
                offspring[k] = crossover_sgs(p1, p2, lrng, crossover_kind, prob.n_jobs);
                mutate_sgs(offspring[k], prob, p_mut_gen, lrng);
                evaluate_sgs(offspring[k], prob, ws);
            }
        }
#endif

        combined.clear();
        for (auto& x : pop)       combined.push_back(std::move(x));
        for (auto& x : offspring) combined.push_back(std::move(x));

        auto cf = fast_nds(combined);
        assign_ranks(combined, cf);
        for (auto& f : cf) crowding_distance(combined, f);

        pop.clear();
        for (auto& f : cf) {
            const int remaining = pop_size - static_cast<int>(pop.size());
            if (remaining <= 0) break;
            if ((int)f.size() <= remaining) {
                for (int i : f) pop.push_back(std::move(combined[i]));
            } else {
                std::sort(f.begin(), f.end(), [&](int a, int b) {
                    return combined[a].crowding > combined[b].crowding;
                });
                for (int k = 0; k < remaining; k++) pop.push_back(std::move(combined[f[k]]));
                break;
            }
        }

        // ── Periodic local search (see sgs_common.hpp's local_search_sgs) ──
        if (ls_interval > 0 && (gen + 1) % ls_interval == 0) {
            std::vector<uint32_t> ls_seeds(pop.size());
            for (auto& s : ls_seeds) s = rng();
#ifdef _OPENMP
            #pragma omp parallel
            {
                SgsWorkspace ws; ws.reset(prob);
                #pragma omp for schedule(static)
                for (int k = 0; k < (int)pop.size(); k++) {
                    std::mt19937 lrng(ls_seeds[k]);
                    local_search_sgs(pop[k], prob, ws, lrng, /*max_passes=*/1);
                }
            }
#else
            {
                SgsWorkspace ws; ws.reset(prob);
                for (int k = 0; k < (int)pop.size(); k++) {
                    std::mt19937 lrng(ls_seeds[k]);
                    local_search_sgs(pop[k], prob, ws, lrng, 1);
                }
            }
#endif
        }
    }

    // Final polish before ranking — see nsga2.hpp's extraction step.
    {
        std::vector<uint32_t> ls_seeds(pop.size());
        for (auto& s : ls_seeds) s = rng();
        SgsWorkspace ws_polish; ws_polish.reset(prob);
        for (int k = 0; k < (int)pop.size(); k++) {
            std::mt19937 lrng(ls_seeds[k]);
            local_search_sgs(pop[k], prob, ws_polish, lrng, /*max_passes=*/3);
        }
    }

    auto fronts = fast_nds(pop);
    assign_ranks(pop, fronts);
    ResultList results;
    SgsWorkspace ws_extract; ws_extract.reset(prob);
    for (auto& ind : pop)
        if (ind.rank == 0 && ind.cv == 0.0)
            results.emplace_back(extract_assignment_sgs(ind, prob, ws_extract), ind.f1, ind.f2);
    auto filtered = pareto_filter(std::move(results));

    Profile profile = {
        {"n_gen", static_cast<double>(n_gen)},
        {"total_ms", ms_since(t_total)},
    };
    return {std::move(filtered), std::move(profile)};
}

// ── NSGA-III (SGS representation) ─────────────────────────────────────────────

inline std::pair<ResultList, Profile>
run_nsga3_sgs(const SgsProblem& prob, int pop_size, int n_divisions, int n_gen,
             int seed, int n_threads, double p_mut_start = -1.0, double p_mut_end = -1.0,
             int crossover_kind = 0, int tourn_k = 2, int local_search_interval = -1) {
    py::gil_scoped_release release;
    set_num_threads(n_threads);
    const auto t_total = Clock::now();

    std::mt19937 rng(seed);
    const auto mut_sched = make_mutation_schedule(p_mut_start, p_mut_end, 2.0 / prob.n_jobs);
    const int ls_interval = (local_search_interval < 0)
        ? std::max(1, n_gen / 10) : local_search_interval;
    const auto refs = make_reference_points_2d(n_divisions);
    const int  R    = (int)refs.size();

    auto h_seeds = make_heuristic_seeds_sgs(prob);
    const int n_hs = static_cast<int>(h_seeds.size());

    std::vector<Individual> pop(pop_size);
    for (int k = 0; k < n_hs && k < pop_size; k++) pop[k] = h_seeds[k];

    std::vector<uint32_t> init_seeds(pop_size);
    for (auto& s : init_seeds) s = rng();

#ifdef _OPENMP
    #pragma omp parallel
    {
        SgsWorkspace ws; ws.reset(prob);
        #pragma omp for schedule(static)
        for (int k = 0; k < pop_size; k++) {
            if (k >= n_hs) { std::mt19937 lrng(init_seeds[k]); pop[k] = make_random_sgs(prob, lrng); }
            evaluate_sgs(pop[k], prob, ws);
        }
    }
#else
    {
        SgsWorkspace ws; ws.reset(prob);
        for (int k = 0; k < pop_size; k++) {
            if (k >= n_hs) { std::mt19937 lrng(init_seeds[k]); pop[k] = make_random_sgs(prob, lrng); }
            evaluate_sgs(pop[k], prob, ws);
        }
    }
#endif

    std::vector<Individual> offspring(pop_size);
    std::vector<Individual> combined;
    combined.reserve(2 * pop_size);
    std::vector<uint32_t> child_seeds(pop_size);

    for (int gen = 0; gen < n_gen; gen++) {
        {
            auto fronts_pop = fast_nds(pop);
            assign_ranks(pop, fronts_pop);
        }

        for (auto& s : child_seeds) s = rng();
        const double p_mut_gen = mut_sched.at(gen, n_gen);

#ifdef _OPENMP
        #pragma omp parallel
        {
            SgsWorkspace ws; ws.reset(prob);
            #pragma omp for schedule(static)
            for (int k = 0; k < pop_size; k++) {
                std::mt19937 lrng(child_seeds[k]);
                std::uniform_int_distribution<int> ri(0, pop_size - 1);
                const Individual& p1 = (tourn_k == 2)
                    ? tournament3(pop[ri(lrng)], pop[ri(lrng)], lrng) : tournament3_k(pop, tourn_k, lrng);
                const Individual& p2 = (tourn_k == 2)
                    ? tournament3(pop[ri(lrng)], pop[ri(lrng)], lrng) : tournament3_k(pop, tourn_k, lrng);
                offspring[k] = crossover_sgs(p1, p2, lrng, crossover_kind, prob.n_jobs);
                mutate_sgs(offspring[k], prob, p_mut_gen, lrng);
                evaluate_sgs(offspring[k], prob, ws);
            }
        }
#else
        {
            SgsWorkspace ws; ws.reset(prob);
            for (int k = 0; k < pop_size; k++) {
                std::mt19937 lrng(child_seeds[k]);
                std::uniform_int_distribution<int> ri(0, pop_size - 1);
                const Individual& p1 = (tourn_k == 2)
                    ? tournament3(pop[ri(lrng)], pop[ri(lrng)], lrng) : tournament3_k(pop, tourn_k, lrng);
                const Individual& p2 = (tourn_k == 2)
                    ? tournament3(pop[ri(lrng)], pop[ri(lrng)], lrng) : tournament3_k(pop, tourn_k, lrng);
                offspring[k] = crossover_sgs(p1, p2, lrng, crossover_kind, prob.n_jobs);
                mutate_sgs(offspring[k], prob, p_mut_gen, lrng);
                evaluate_sgs(offspring[k], prob, ws);
            }
        }
#endif

        combined.clear();
        for (auto& x : pop)       combined.push_back(std::move(x));
        for (auto& x : offspring) combined.push_back(std::move(x));

        auto cf = fast_nds(combined);
        assign_ranks(combined, cf);

        int filled = 0;
        int critical_fi = (int)cf.size();
        for (int fi = 0; fi < (int)cf.size(); fi++) {
            if (filled + (int)cf[fi].size() > (size_t)pop_size) { critical_fi = fi; break; }
            filled += (int)cf[fi].size();
        }
        const int needed = pop_size - filled;

        pop.clear();
        if (needed == 0 || critical_fi == (int)cf.size()) {
            for (int fi = 0; fi < (int)cf.size() && (int)pop.size() < pop_size; fi++)
                for (int i : cf[fi]) {
                    if ((int)pop.size() >= pop_size) break;
                    pop.push_back(std::move(combined[i]));
                }
        } else {
            std::vector<int> all_indices;
            all_indices.reserve(filled + (int)cf[critical_fi].size());
            for (int fi = 0; fi < critical_fi; fi++)
                for (int i : cf[fi]) all_indices.push_back(i);
            for (int i : cf[critical_fi]) all_indices.push_back(i);

            auto [z1, z2] = nsga3_ideal(combined, all_indices);
            auto [a1, a2] = nsga3_intercepts(combined, all_indices, z1, z2);
            auto all_data = nsga3_associate(combined, all_indices, z1, z2, a1, a2, refs);

            const int n_S  = filled;
            const int n_Fl = (int)cf[critical_fi].size();
            auto sel_from_fl = niching_select(all_data, n_S, n_Fl, R, needed, rng);

            for (int fi = 0; fi < critical_fi; fi++)
                for (int i : cf[fi]) pop.push_back(std::move(combined[i]));
            for (int k : sel_from_fl)
                pop.push_back(std::move(combined[cf[critical_fi][k]]));
        }
        if ((int)pop.size() > pop_size) pop.resize(pop_size);

        // ── Periodic local search (see sgs_common.hpp's local_search_sgs) ──
        if (ls_interval > 0 && (gen + 1) % ls_interval == 0) {
            std::vector<uint32_t> ls_seeds(pop.size());
            for (auto& s : ls_seeds) s = rng();
#ifdef _OPENMP
            #pragma omp parallel
            {
                SgsWorkspace ws; ws.reset(prob);
                #pragma omp for schedule(static)
                for (int k = 0; k < (int)pop.size(); k++) {
                    std::mt19937 lrng(ls_seeds[k]);
                    local_search_sgs(pop[k], prob, ws, lrng, /*max_passes=*/1);
                }
            }
#else
            {
                SgsWorkspace ws; ws.reset(prob);
                for (int k = 0; k < (int)pop.size(); k++) {
                    std::mt19937 lrng(ls_seeds[k]);
                    local_search_sgs(pop[k], prob, ws, lrng, 1);
                }
            }
#endif
        }
    }

    // Final polish before ranking — see nsga2.hpp's extraction step.
    {
        std::vector<uint32_t> ls_seeds(pop.size());
        for (auto& s : ls_seeds) s = rng();
        SgsWorkspace ws_polish; ws_polish.reset(prob);
        for (int k = 0; k < (int)pop.size(); k++) {
            std::mt19937 lrng(ls_seeds[k]);
            local_search_sgs(pop[k], prob, ws_polish, lrng, /*max_passes=*/3);
        }
    }

    auto fronts = fast_nds(pop);
    assign_ranks(pop, fronts);
    ResultList results;
    SgsWorkspace ws_extract; ws_extract.reset(prob);
    for (auto& ind : pop)
        if (ind.rank == 0 && ind.cv == 0.0)
            results.emplace_back(extract_assignment_sgs(ind, prob, ws_extract), ind.f1, ind.f2);
    auto filtered = pareto_filter(std::move(results));

    Profile profile = {
        {"n_gen", static_cast<double>(n_gen)},
        {"total_ms", ms_since(t_total)},
    };
    return {std::move(filtered), std::move(profile)};
}

// ── MOEA/D (SGS representation) ───────────────────────────────────────────────

inline std::pair<ResultList, Profile>
run_moead_sgs(const SgsProblem& prob, int n_weights, int n_gen, int T_size,
             int seed, int n_threads, int max_replace = -1,
             double p_mut_start = -1.0, double p_mut_end = -1.0, int crossover_kind = 0,
             int archive_size = 0, int local_search_interval = -1) {
    py::gil_scoped_release release;
    set_num_threads(n_threads);
    const auto t_total = Clock::now();

    std::mt19937 rng(seed);
    const auto mut_sched = make_mutation_schedule(p_mut_start, p_mut_end, 1.0 / prob.n_jobs);
    const int ls_interval = (local_search_interval < 0)
        ? std::max(1, n_gen / 10) : local_search_interval;

    constexpr double W_EPS = 1e-4;
    std::vector<std::pair<double,double>> W(n_weights);
    for (int i = 0; i < n_weights; i++) {
        const double lam = (n_weights > 1) ? (double)i / (n_weights - 1) : 0.5;
        W[i] = {lam + W_EPS, 1.0 - lam + W_EPS};
    }

    const int T = std::min(T_size, n_weights);
    std::vector<std::vector<int>> neigh(n_weights);
    for (int i = 0; i < n_weights; i++) {
        std::vector<std::pair<double,int>> d;
        d.reserve(n_weights);
        for (int j = 0; j < n_weights; j++) {
            const double dl = W[i].first  - W[j].first;
            const double dm = W[i].second - W[j].second;
            d.emplace_back(dl*dl + dm*dm, j);
        }
        std::partial_sort(d.begin(), d.begin() + T, d.end());
        for (int k = 0; k < T; k++) neigh[i].push_back(d[k].second);
    }

    auto h_seeds = make_heuristic_seeds_sgs(prob);
    const int n_hs = static_cast<int>(h_seeds.size());

    std::vector<Individual> pop(n_weights);
    for (int i = 0; i < n_hs && i < n_weights; i++) pop[i] = h_seeds[i];

    std::vector<uint32_t> init_seeds(n_weights);
    for (auto& s : init_seeds) s = rng();

    double z1 = std::numeric_limits<double>::infinity(), z2 = std::numeric_limits<double>::infinity();
    double n1 = -std::numeric_limits<double>::infinity(), n2 = -std::numeric_limits<double>::infinity();

#ifdef _OPENMP
    #pragma omp parallel reduction(min : z1, z2) reduction(max : n1, n2)
    {
        SgsWorkspace ws; ws.reset(prob);
        #pragma omp for schedule(static)
        for (int i = 0; i < n_weights; i++) {
            if (i >= n_hs) { std::mt19937 lrng(init_seeds[i]); pop[i] = make_random_sgs(prob, lrng); }
            evaluate_sgs(pop[i], prob, ws);
            if (pop[i].cv == 0.0) {
                if (pop[i].f1 < z1) z1 = pop[i].f1;
                if (pop[i].f2 < z2) z2 = pop[i].f2;
                if (pop[i].f1 > n1) n1 = pop[i].f1;
                if (pop[i].f2 > n2) n2 = pop[i].f2;
            }
        }
    }
#else
    {
        SgsWorkspace ws; ws.reset(prob);
        for (int i = 0; i < n_weights; i++) {
            if (i >= n_hs) { std::mt19937 lrng(init_seeds[i]); pop[i] = make_random_sgs(prob, lrng); }
            evaluate_sgs(pop[i], prob, ws);
            if (pop[i].cv == 0.0) {
                z1 = std::min(z1, pop[i].f1); z2 = std::min(z2, pop[i].f2);
                n1 = std::max(n1, pop[i].f1); n2 = std::max(n2, pop[i].f2);
            }
        }
    }
#endif
    if (!std::isfinite(z1)) z1 = 0.0;
    if (!std::isfinite(z2)) z2 = 0.0;
    if (!std::isfinite(n1)) n1 = z1 + 1.0;
    if (!std::isfinite(n2)) n2 = z2 + 1.0;

    auto scalar = [&](const Individual& ind, int i) noexcept -> double {
        if (ind.cv > 0.0) return 1e18 + ind.cv * 1e6;
        const double lam1 = W[i].first, lam2 = W[i].second;
        const double r1 = std::max(n1 - z1, 1.0);
        const double r2 = std::max(n2 - z2, 1.0);
        return std::max(lam1 * (ind.f1 - z1) / r1, lam2 * (ind.f2 - z2) / r2);
    };

    std::vector<Individual> children(n_weights);
    std::vector<uint32_t>   child_seeds(n_weights);
    std::vector<Individual> archive;
    if (archive_size > 0) archive.reserve(archive_size + 1);

    for (int gen = 0; gen < n_gen; gen++) {
        for (auto& s : child_seeds) s = rng();
        double dz1 = z1, dz2 = z2, dn1 = n1, dn2 = n2;
        const double p_mut_gen = mut_sched.at(gen, n_gen);

#ifdef _OPENMP
        #pragma omp parallel reduction(min : dz1, dz2) reduction(max : dn1, dn2)
        {
            SgsWorkspace ws; ws.reset(prob);
            #pragma omp for schedule(static)
            for (int i = 0; i < n_weights; i++) {
                std::mt19937 lrng(child_seeds[i]);
                std::uniform_int_distribution<int> rn(0, T - 1);
                const int a = neigh[i][rn(lrng)];
                const int b = neigh[i][rn(lrng)];
                children[i] = crossover_sgs(pop[a], pop[b], lrng, crossover_kind, prob.n_jobs);
                mutate_sgs(children[i], prob, p_mut_gen, lrng);
                evaluate_sgs(children[i], prob, ws);
                if (children[i].cv == 0.0) {
                    if (children[i].f1 < dz1) dz1 = children[i].f1;
                    if (children[i].f2 < dz2) dz2 = children[i].f2;
                    if (children[i].f1 > dn1) dn1 = children[i].f1;
                    if (children[i].f2 > dn2) dn2 = children[i].f2;
                }
            }
        }
#else
        {
            SgsWorkspace ws; ws.reset(prob);
            for (int i = 0; i < n_weights; i++) {
                std::mt19937 lrng(child_seeds[i]);
                std::uniform_int_distribution<int> rn(0, T - 1);
                const int a = neigh[i][rn(lrng)];
                const int b = neigh[i][rn(lrng)];
                children[i] = crossover_sgs(pop[a], pop[b], lrng, crossover_kind, prob.n_jobs);
                mutate_sgs(children[i], prob, p_mut_gen, lrng);
                evaluate_sgs(children[i], prob, ws);
                if (children[i].cv == 0.0) {
                    dz1 = std::min(dz1, children[i].f1); dz2 = std::min(dz2, children[i].f2);
                    dn1 = std::max(dn1, children[i].f1); dn2 = std::max(dn2, children[i].f2);
                }
            }
        }
#endif
        z1 = dz1; z2 = dz2; n1 = dn1; n2 = dn2;

        if (archive_size > 0)
            for (const auto& c : children) archive_insert(archive, c, archive_size);

        const bool capped = max_replace > 0 && max_replace < T;
        std::vector<int> order;
        if (capped) order.resize(T);
        for (int i = 0; i < n_weights; i++) {
            if (!capped) {
                for (int j : neigh[i])
                    if (scalar(children[i], j) <= scalar(pop[j], j)) pop[j] = children[i];
                continue;
            }
            std::copy(neigh[i].begin(), neigh[i].end(), order.begin());
            std::shuffle(order.begin(), order.end(), rng);
            int replaced = 0;
            for (int j : order) {
                if (replaced >= max_replace) break;
                if (scalar(children[i], j) <= scalar(pop[j], j)) { pop[j] = children[i]; replaced++; }
            }
        }

        // ── Periodic local search (see sgs_common.hpp's local_search_sgs) ──
        if (ls_interval > 0 && (gen + 1) % ls_interval == 0) {
            std::vector<uint32_t> ls_seeds(pop.size());
            for (auto& s : ls_seeds) s = rng();
#ifdef _OPENMP
            #pragma omp parallel
            {
                SgsWorkspace ws; ws.reset(prob);
                #pragma omp for schedule(static)
                for (int i = 0; i < (int)pop.size(); i++) {
                    std::mt19937 lrng(ls_seeds[i]);
                    local_search_sgs(pop[i], prob, ws, lrng, /*max_passes=*/1);
                }
            }
#else
            {
                SgsWorkspace ws; ws.reset(prob);
                for (int i = 0; i < (int)pop.size(); i++) {
                    std::mt19937 lrng(ls_seeds[i]);
                    local_search_sgs(pop[i], prob, ws, lrng, 1);
                }
            }
#endif
        }
    }

    // Final polish before extraction — see nsga2.hpp's extraction step.
    {
        std::vector<uint32_t> ls_seeds(pop.size() + archive.size());
        for (auto& s : ls_seeds) s = rng();
        SgsWorkspace ws_polish; ws_polish.reset(prob);
        size_t si = 0;
        for (auto& ind : pop) {
            std::mt19937 lrng(ls_seeds[si++]);
            local_search_sgs(ind, prob, ws_polish, lrng, /*max_passes=*/3);
        }
        for (auto& ind : archive) {
            std::mt19937 lrng(ls_seeds[si++]);
            local_search_sgs(ind, prob, ws_polish, lrng, /*max_passes=*/3);
        }
    }

    ResultList results;
    SgsWorkspace ws_extract; ws_extract.reset(prob);
    for (auto& ind : pop)
        if (ind.cv == 0.0)
            results.emplace_back(extract_assignment_sgs(ind, prob, ws_extract), ind.f1, ind.f2);
    if (archive_size > 0)
        for (auto& ind : archive)
            if (ind.cv == 0.0)
                results.emplace_back(extract_assignment_sgs(ind, prob, ws_extract), ind.f1, ind.f2);
    auto filtered = pareto_filter(std::move(results));

    Profile profile = {
        {"n_gen", static_cast<double>(n_gen)},
        {"total_ms", ms_since(t_total)},
    };
    return {std::move(filtered), std::move(profile)};
}
