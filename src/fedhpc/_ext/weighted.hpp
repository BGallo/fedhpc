/**
 * FED-HPC — single-objective (weighted-sum) memetic metaheuristic.
 *
 * Solves   min  w1·f1 + w2·f2      (+ soft cap  f1 ≤ f1_cap)
 * i.e. the same scalarised objective as model.solve_weighted_sum, but
 * heuristically. The Python layer converts the paper's
 * (λ, f1_T, f2_T, f1_0) reference-point form into (w1, w2, f1_cap).
 *
 * Why a memetic GA rather than a plain GA
 * --------------------------------------
 * The problem almost decomposes: f2 (cost) is a function of the *type vector*
 * alone, and for a fixed type vector f1 is minimised by an SPT list schedule
 * (schedule_repair). So the real search space is "one type per job", and a
 * strong local search over single type-reassignments — decoding each candidate
 * with schedule_repair — does most of the work. The GA layer supplies
 * diversification (type-block crossover) and escapes local optima (ILS-style
 * perturbation kicks on stagnation).
 *
 * Determinism: bit-for-bit stable for a fixed (seed, n_threads). Per-thread
 * RNGs are seeded from the master RNG before every parallel region.
 */
#pragma once

#include "ga_common.hpp"

// Scalarised fitness (lower is better). Infeasible individuals get a large
// additive penalty so selection still orders them sensibly; f1 above the
// Pareto-region cap is softly penalised on the same scale as the objective.
struct WeightedScalar {
    double w1, w2, f1_cap, cap_pen;

    double operator()(const Individual& ind) const noexcept {
        double g = w1 * ind.f1 + w2 * ind.f2;
        if (ind.f1 > f1_cap) g += cap_pen * (ind.f1 - f1_cap);
        if (ind.cv > 0.0)    g += 1e9 + 1e6 * ind.cv;
        return g;
    }
};

// Greedy type-flip descent on the weighted scalar. Single-objective, so there
// is exactly one improving direction: this genuinely hill-climbs to a local
// optimum of g over the (type-vector × SPT-schedule) neighbourhood.
//
// Each step: shortlist single job→type reassignments by an O(1) estimate of Δg
// (Δf2 exact, Δf1 ≈ Δp_occ), decode the best `shortlist` of them exactly with
// schedule_repair + local_search, take the one that most reduces g, repeat up
// to `max_moves` times. Leaves ind holding a decoded, evaluated solution.
inline bool weighted_local_search(Individual& ind, const Problem& prob,
                                  EvalWorkspace& ws, const WeightedScalar& G,
                                  int max_moves, int shortlist) {
    schedule_repair(ind, prob, ws);
    local_search(ind, prob, ws);
    schedule_repair(ind, prob, ws);
    evaluate(ind, prob, ws);
    double g = G(ind);
    bool improved = false;

    struct Cand { double est; int j, k; };
    std::vector<Cand> cand;

    for (int mv = 0; mv < max_moves; mv++) {
        cand.clear();
        for (int j = 0; j < prob.n_jobs; j++) {
            const SlotInfo& s = prob.job_slots[j][ind.genes[j]];
            for (const auto& sp : prob.job_type_span[j]) {
                if (sp[0] == s.type_id) continue;
                const SlotInfo& a0 = prob.job_slots[j][sp[1]];
                const double est = G.w1 * (a0.p_occ - s.p_occ)
                                 + G.w2 * (a0.cost  - s.cost);
                if (est < -1e-12) cand.push_back({est, j, sp[1]});
            }
        }
        if (cand.empty()) break;
        const int K = std::min<int>((int)cand.size(), shortlist);
        std::partial_sort(cand.begin(), cand.begin() + K, cand.end(),
                          [](const Cand& a, const Cand& b) { return a.est < b.est; });

        double best_g = g;
        Individual best;
        bool found = false;
        for (int c = 0; c < K; c++) {
            Individual w = ind;
            w.genes[cand[c].j] = cand[c].k;
            schedule_repair(w, prob, ws);
            local_search(w, prob, ws);
            evaluate(w, prob, ws);
            const double gw = G(w);
            if (gw < best_g - 1e-12) { best_g = gw; best = std::move(w); found = true; }
        }
        if (!found) break;
        ind = std::move(best);
        g = best_g;
        improved = true;
    }
    return improved;
}

// Result: (assignment, f1, f2, g, n_local_search_calls).
using WeightedResult = std::tuple<std::vector<std::tuple<int,int>>, double, double, double, int>;

inline WeightedResult
run_weighted(const Problem& prob, double w1, double w2, double f1_cap,
             int pop_size, int n_gen, int seed, int n_threads,
             int ls_moves, int restart_patience, int shortlist) {
    py::gil_scoped_release release;
    set_num_threads(n_threads);

    std::mt19937 rng(seed);
    WeightedScalar G{w1, w2, f1_cap,
                     (w1 > 0.0 ? w1 : (w2 > 0.0 ? w2 : 1.0)) * 1e3};
    int ls_calls = 0;

    const double p_mut = 2.0 / std::max(prob.n_jobs, 1);

    // ── Initial population: repaired heuristic seeds + weighted-random ───────
    std::vector<uint32_t> init_seeds(pop_size);
    for (auto& s : init_seeds) s = rng();

    auto h_seeds = make_heuristic_seeds(prob);
    const int n_hs = std::min<int>((int)h_seeds.size(), pop_size);

    std::vector<Individual> pop(pop_size);
    for (int k = 0; k < n_hs; k++) pop[k] = h_seeds[k];

#ifdef _OPENMP
    #pragma omp parallel
    {
        EvalWorkspace ws; ws.reset(prob.n_types, prob.max_slot);
        int local_calls = 0;
        #pragma omp for schedule(dynamic, 1)
        for (int k = 0; k < pop_size; k++) {
            if (k >= n_hs) {
                std::mt19937 lrng(init_seeds[k]);
                pop[k] = make_random(prob, lrng);
            }
            weighted_local_search(pop[k], prob, ws, G, ls_moves, shortlist);
            local_calls++;
        }
        #pragma omp atomic
        ls_calls += local_calls;
    }
#else
    {
        EvalWorkspace ws; ws.reset(prob.n_types, prob.max_slot);
        for (int k = 0; k < pop_size; k++) {
            if (k >= n_hs) {
                std::mt19937 lrng(init_seeds[k]);
                pop[k] = make_random(prob, lrng);
            }
            weighted_local_search(pop[k], prob, ws, G, ls_moves, shortlist);
            ls_calls++;
        }
    }
#endif

    int inc = 0;
    for (int k = 1; k < pop_size; k++) if (G(pop[k]) < G(pop[inc])) inc = k;
    Individual incumbent = pop[inc];
    double best_g = G(incumbent);
    int stagnation = 0;

    std::vector<Individual> offspring(pop_size);
    std::vector<uint32_t> child_seeds(pop_size);

    // Fewer LS moves per child during evolution — the population compounds them.
    const int gen_moves = std::max(1, ls_moves / 2);

    for (int gen = 0; gen < n_gen; gen++) {
        for (auto& s : child_seeds) s = rng();

        offspring[0] = incumbent;   // elitism

#ifdef _OPENMP
        #pragma omp parallel
        {
            EvalWorkspace ws; ws.reset(prob.n_types, prob.max_slot);
            int local_calls = 0;
            #pragma omp for schedule(dynamic, 1)
            for (int k = 1; k < pop_size; k++) {
                std::mt19937 lrng(child_seeds[k]);
                std::uniform_int_distribution<int> ri(0, pop_size - 1);
                auto pick = [&]() {
                    const int a = ri(lrng), b = ri(lrng);
                    return (G(pop[a]) <= G(pop[b])) ? a : b;
                };
                Individual c = crossover(pop[pick()], pop[pick()], lrng, 0);
                mutate(c, prob, p_mut, lrng);
                weighted_local_search(c, prob, ws, G, gen_moves, shortlist);
                local_calls++;
                offspring[k] = std::move(c);
            }
            #pragma omp atomic
            ls_calls += local_calls;
        }
#else
        {
            EvalWorkspace ws; ws.reset(prob.n_types, prob.max_slot);
            for (int k = 1; k < pop_size; k++) {
                std::mt19937 lrng(child_seeds[k]);
                std::uniform_int_distribution<int> ri(0, pop_size - 1);
                auto pick = [&]() {
                    const int a = ri(lrng), b = ri(lrng);
                    return (G(pop[a]) <= G(pop[b])) ? a : b;
                };
                Individual c = crossover(pop[pick()], pop[pick()], lrng, 0);
                mutate(c, prob, p_mut, lrng);
                weighted_local_search(c, prob, ws, G, gen_moves, shortlist);
                ls_calls++;
                offspring[k] = std::move(c);
            }
        }
#endif

        pop.swap(offspring);

        bool advanced = false;
        for (int k = 0; k < pop_size; k++) {
            const double g = G(pop[k]);
            if (g < best_g - 1e-9) { best_g = g; incumbent = pop[k]; advanced = true; }
        }
        stagnation = advanced ? 0 : stagnation + 1;

        // ── ILS kick on stagnation ──────────────────────────────────────────
        if (restart_patience > 0 && stagnation >= restart_patience) {
            EvalWorkspace ws; ws.reset(prob.n_types, prob.max_slot);
            Individual kick = incumbent;
            const int nflips = std::max(3, prob.n_jobs / 40);
            for (int f = 0; f < nflips; f++) {
                const int j = rng() % prob.n_jobs;
                const auto& span = prob.job_type_span[j];
                kick.genes[j] = span[rng() % span.size()][1];
            }
            weighted_local_search(kick, prob, ws, G, ls_moves, shortlist);
            ls_calls++;
            pop[pop_size - 1] = kick;
            const double g = G(kick);
            if (g < best_g - 1e-9) { best_g = g; incumbent = kick; }
            stagnation = 0;
        }
    }

    // Final intensive polish on the incumbent.
    {
        EvalWorkspace ws; ws.reset(prob.n_types, prob.max_slot);
        weighted_local_search(incumbent, prob, ws, G, ls_moves * 3, shortlist);
        ls_calls++;
        evaluate(incumbent, prob, ws);
    }

    return {extract_assignment(incumbent, prob), incumbent.f1, incumbent.f2,
            G(incumbent), ls_calls};
}
