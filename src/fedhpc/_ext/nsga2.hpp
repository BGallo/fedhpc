/**
 * FED-HPC — NSGA-II with constrained-dominance ranking.
 *
 * fast_nds():
 *   The O(N²) dominance-computation phase is parallelised over rows p.
 *   Each row writes only to dominated_by[p] / dom_count[p] — no data races.
 *   Front propagation remains sequential (dependency chain).
 *
 * run_nsga2():
 *   All pop_size children per generation are generated and evaluated in parallel.
 *   Seeds are drawn from the master RNG before the parallel region so results
 *   are deterministic for any fixed (seed, n_threads) pair.
 *
 * Profiling:
 *   Returns a Profile (vector of (name, ms) pairs) alongside the solutions.
 *   Phases timed per generation (summed + per-gen average reported):
 *     init_eval_ms      — heuristic seeds + random init + evaluation
 *     nds_ms            — fast_nds + assign_ranks on the parent population
 *     crowding_ms       — crowding_distance on all parent fronts
 *     offspring_ms      — parallel offspring generation + evaluation
 *     combine_select_ms — NDS + crowding + survival selection on the merged pool
 *     extract_ms        — final NDS + result extraction + Pareto filter
 */
#pragma once

#include "ga_common.hpp"

// ── Non-dominated sort ────────────────────────────────────────────────────────

// O(N²) with parallelised dominance-computation phase.
inline std::vector<std::vector<int>> fast_nds(const std::vector<Individual>& pop) {
    const int N = static_cast<int>(pop.size());
    std::vector<int>              dom_count(N, 0);
    std::vector<std::vector<int>> dominated_by(N);

    // Each row p writes only to its own dominated_by[p] and dom_count[p].
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 8)
#endif
    for (int p = 0; p < N; p++) {
        for (int q = 0; q < N; q++) {
            if (p == q) continue;
            if (pop[p].dominates(pop[q]))
                dominated_by[p].push_back(q);
            else if (pop[q].dominates(pop[p]))
                dom_count[p]++;
        }
    }

    std::vector<std::vector<int>> fronts(1);
    for (int p = 0; p < N; p++)
        if (dom_count[p] == 0) fronts[0].push_back(p);

    for (int fi = 0; !fronts[fi].empty(); fi++) {
        fronts.emplace_back();
        for (int p : fronts[fi])
            for (int q : dominated_by[p])
                if (--dom_count[q] == 0) fronts[fi + 1].push_back(q);
    }
    if (fronts.back().empty()) fronts.pop_back();
    return fronts;
}

inline void assign_ranks(std::vector<Individual>& pop,
                          const std::vector<std::vector<int>>& fronts) {
    for (int r = 0; r < (int)fronts.size(); r++)
        for (int i : fronts[r]) pop[i].rank = r;
}

inline void crowding_distance(std::vector<Individual>& pop,
                               const std::vector<int>& front) {
    const int sz = static_cast<int>(front.size());
    for (int i : front) pop[i].crowding = 0.0;
    if (sz <= 2) {
        for (int i : front) pop[i].crowding = std::numeric_limits<double>::infinity();
        return;
    }

    auto process_obj = [&](auto get_f) {
        std::vector<int> order = front;
        std::sort(order.begin(), order.end(),
                  [&](int a, int b) { return get_f(pop[a]) < get_f(pop[b]); });
        pop[order.front()].crowding = std::numeric_limits<double>::infinity();
        pop[order.back() ].crowding = std::numeric_limits<double>::infinity();
        const double range = get_f(pop[order.back()]) - get_f(pop[order.front()]);
        if (range < 1e-12) return;
        for (int k = 1; k < sz - 1; k++)
            pop[order[k]].crowding +=
                (get_f(pop[order[k+1]]) - get_f(pop[order[k-1]])) / range;
    };
    process_obj([](const Individual& x) { return x.f1; });
    process_obj([](const Individual& x) { return x.f2; });
}

inline const Individual& tournament(const Individual& a,
                                     const Individual& b) noexcept {
    if (a.rank < b.rank) return a;
    if (b.rank < a.rank) return b;
    return (a.crowding >= b.crowding) ? a : b;
}

// ── NSGA-II main loop ─────────────────────────────────────────────────────────

inline std::pair<ResultList, Profile>
run_nsga2(const Problem& prob, int pop_size, int n_gen, int seed, int n_threads) {
    py::gil_scoped_release release;
    set_num_threads(n_threads);

    const auto t_total = Clock::now();

    std::mt19937 rng(seed);
    const double p_mut = 2.0 / prob.n_jobs;

    // ── Initial population ────────────────────────────────────────────────────

    const auto t_init = Clock::now();

    std::vector<uint32_t> init_seeds(pop_size);
    for (auto& s : init_seeds) s = rng();

    auto h_seeds   = make_heuristic_seeds(prob);
    const int n_hs = static_cast<int>(h_seeds.size());

    std::vector<Individual> pop(pop_size);
    for (int k = 0; k < n_hs && k < pop_size; k++)
        pop[k] = h_seeds[k];

#ifdef _OPENMP
    #pragma omp parallel
    {
        EvalWorkspace ws;
        ws.reset(prob.n_types, prob.max_slot);
        #pragma omp for schedule(static)
        for (int k = 0; k < pop_size; k++) {
            if (k >= n_hs) {
                std::mt19937 lrng(init_seeds[k]);
                pop[k] = make_random(prob, lrng);
            }
            evaluate(pop[k], prob, ws);
        }
    }
#else
    {
        EvalWorkspace ws;
        ws.reset(prob.n_types, prob.max_slot);
        for (int k = 0; k < pop_size; k++) {
            if (k >= n_hs) {
                std::mt19937 lrng(init_seeds[k]);
                pop[k] = make_random(prob, lrng);
            }
            evaluate(pop[k], prob, ws);
        }
    }
#endif

    const double init_eval_ms = ms_since(t_init);

    std::vector<Individual> offspring(pop_size);
    std::vector<Individual> combined;
    combined.reserve(2 * pop_size);
    std::vector<uint32_t> child_seeds(pop_size);

    double nds_ms = 0.0, crowding_ms = 0.0,
           offspring_ms = 0.0, combine_select_ms = 0.0;

    for (int gen = 0; gen < n_gen; gen++) {
        // ── Non-dominated sort + crowding ─────────────────────────────────────

        auto t0 = Clock::now();
        auto fronts = fast_nds(pop);
        assign_ranks(pop, fronts);
        nds_ms += ms_since(t0);

        auto t1 = Clock::now();
        for (auto& f : fronts) crowding_distance(pop, f);
        crowding_ms += ms_since(t1);

        // ── Offspring generation — parallel ───────────────────────────────────

        for (auto& s : child_seeds) s = rng();

        const auto t2 = Clock::now();

#ifdef _OPENMP
        #pragma omp parallel
        {
            EvalWorkspace ws;
            ws.reset(prob.n_types, prob.max_slot);
            #pragma omp for schedule(static)
            for (int k = 0; k < pop_size; k++) {
                std::mt19937 lrng(child_seeds[k]);
                std::uniform_int_distribution<int> ri(0, pop_size - 1);
                const Individual& p1 = tournament(pop[ri(lrng)], pop[ri(lrng)]);
                const Individual& p2 = tournament(pop[ri(lrng)], pop[ri(lrng)]);
                offspring[k] = crossover(p1, p2, lrng);
                mutate(offspring[k], prob, p_mut, lrng);
                evaluate(offspring[k], prob, ws);
            }
        }
#else
        {
            EvalWorkspace ws;
            ws.reset(prob.n_types, prob.max_slot);
            for (int k = 0; k < pop_size; k++) {
                std::mt19937 lrng(child_seeds[k]);
                std::uniform_int_distribution<int> ri(0, pop_size - 1);
                const Individual& p1 = tournament(pop[ri(lrng)], pop[ri(lrng)]);
                const Individual& p2 = tournament(pop[ri(lrng)], pop[ri(lrng)]);
                offspring[k] = crossover(p1, p2, lrng);
                mutate(offspring[k], prob, p_mut, lrng);
                evaluate(offspring[k], prob, ws);
            }
        }
#endif

        offspring_ms += ms_since(t2);

        // ── Combine + environmental selection ────────────────────────────────

        const auto t3 = Clock::now();

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
                for (int k = 0; k < remaining; k++)
                    pop.push_back(std::move(combined[f[k]]));
                break;
            }
        }

        combine_select_ms += ms_since(t3);
    }

    // ── Extract Pareto front ──────────────────────────────────────────────────

    const auto t_extract = Clock::now();

    auto fronts = fast_nds(pop);
    assign_ranks(pop, fronts);
    ResultList results;
    for (auto& ind : pop)
        if (ind.rank == 0 && ind.cv == 0.0)
            results.emplace_back(extract_assignment(ind, prob), ind.f1, ind.f2);
    auto filtered = pareto_filter(std::move(results));

    const double extract_ms = ms_since(t_extract);
    const double total_ms   = ms_since(t_total);
    const double gen_d      = std::max(n_gen, 1);

    Profile profile = {
        {"n_gen",                  static_cast<double>(n_gen)},
        {"total_ms",               total_ms},
        {"init_eval_ms",           init_eval_ms},
        {"nds_total_ms",           nds_ms},
        {"nds_avg_ms",             nds_ms         / gen_d},
        {"crowding_total_ms",      crowding_ms},
        {"crowding_avg_ms",        crowding_ms    / gen_d},
        {"offspring_total_ms",     offspring_ms},
        {"offspring_avg_ms",       offspring_ms   / gen_d},
        {"combine_select_total_ms",combine_select_ms},
        {"combine_select_avg_ms",  combine_select_ms / gen_d},
        {"extract_ms",             extract_ms},
    };

    return {std::move(filtered), std::move(profile)};
}
