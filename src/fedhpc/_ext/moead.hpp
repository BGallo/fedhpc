/**
 * FED-HPC — MOEA/D with normalised Tchebycheff decomposition.
 *
 * Parallelism (offline-update variant):
 *   All n_weights children per generation are generated and evaluated in
 *   parallel (Phase 1).  Ideal/nadir-point reduction and neighbourhood
 *   replacement run sequentially after (Phase 2).
 *
 * Normalised scalarisation:
 *   Each objective is scaled by its observed range [ideal, nadir] so that
 *   weight vectors spread evenly across the tradeoff regardless of the
 *   absolute scale difference between f1 (slots) and f2 (dollars).
 *
 * Bounded replacement (nr, Zhang & Li 2007):
 *   A single child may overwrite at most max_replace neighbours per
 *   generation instead of the whole neighbourhood.  This limits how fast
 *   one good solution can take over a region of weight space, which is the
 *   standard fix for MOEA/D's premature-convergence / diversity-loss failure
 *   mode.  max_replace <= 0 (default) disables the cap and reproduces the
 *   original unbounded-replacement behaviour exactly, including RNG draws.
 *
 * Elitist archive (optional, archive_size > 0):
 *   Bounded external non-dominated archive updated with every feasible child
 *   from Phase 1, merged into the returned front at extraction. Recovers
 *   extreme/rare points that bounded neighbourhood replacement can overwrite
 *   and lose before they're ever read back out of `pop`. Disabled by default
 *   (archive_size <= 0), which leaves the extraction path unchanged.
 *
 * Profiling:
 *   Returns a Profile (vector of (name, ms) pairs) alongside the solutions.
 *   Phases timed per generation (summed + per-gen average reported):
 *     init_eval_ms     — heuristic seeds + random init + evaluation + initial ideal
 *     offspring_ms     — parallel offspring generation + evaluation + ideal reduction
 *     replacement_ms   — sequential neighbourhood replacement (Phase 2)
 *     extract_ms       — final extraction + Pareto filter
 */
#pragma once

#include "ga_common.hpp"

// Insert cand into a bounded non-dominated archive (no-op if cand is
// infeasible or dominated by an existing member). Newly-dominated members
// are dropped. Over capacity, evict the member with the smallest distance
// to its nearest neighbour in objective space (most crowded).
inline void archive_insert(std::vector<Individual>& archive, const Individual& cand,
                           int archive_size) {
    if (cand.cv > 0.0) return;
    for (const auto& a : archive)
        if (a.dominates(cand)) return;
    archive.erase(
        std::remove_if(archive.begin(), archive.end(),
                       [&](const Individual& a) { return cand.dominates(a); }),
        archive.end());
    archive.push_back(cand);
    if (static_cast<int>(archive.size()) <= archive_size) return;

    int worst = 0;
    double worst_nn = std::numeric_limits<double>::infinity();
    for (int i = 0; i < static_cast<int>(archive.size()); i++) {
        double nn = std::numeric_limits<double>::infinity();
        for (int j = 0; j < static_cast<int>(archive.size()); j++) {
            if (i == j) continue;
            const double d = std::hypot(archive[i].f1 - archive[j].f1,
                                        archive[i].f2 - archive[j].f2);
            nn = std::min(nn, d);
        }
        if (nn < worst_nn) { worst_nn = nn; worst = i; }
    }
    archive.erase(archive.begin() + worst);
}

inline std::pair<ResultList, Profile>
run_moead(const Problem& prob, int n_weights, int n_gen,
          int T_size, int seed, int n_threads, int max_replace = -1,
          double p_mut_start = -1.0, double p_mut_end = -1.0, int crossover_kind = 0,
          int archive_size = 0) {
    py::gil_scoped_release release;
    set_num_threads(n_threads);

    const auto t_total = Clock::now();

    std::mt19937 rng(seed);
    const auto mut_sched = make_mutation_schedule(p_mut_start, p_mut_end, 1.0 / prob.n_jobs);

    // ── Weight vectors: uniformly spaced on the 2-objective simplex ──────────
    // W_EPS floor prevents degenerate single-objective subproblems at extremes.
    constexpr double W_EPS = 1e-4;
    std::vector<std::pair<double,double>> W(n_weights);
    for (int i = 0; i < n_weights; i++) {
        const double lam = (n_weights > 1) ? (double)i / (n_weights - 1) : 0.5;
        W[i] = {lam + W_EPS, 1.0 - lam + W_EPS};
    }

    // ── T-nearest neighbourhoods ──────────────────────────────────────────────
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

    // ── Initial population ────────────────────────────────────────────────────

    const auto t_init = Clock::now();

    std::vector<uint32_t> init_seeds(n_weights);
    for (auto& s : init_seeds) s = rng();

    auto h_seeds   = make_heuristic_seeds(prob);
    const int n_hs = static_cast<int>(h_seeds.size());

    std::vector<Individual> pop(n_weights);
    for (int i = 0; i < n_hs && i < n_weights; i++)
        pop[i] = h_seeds[i];

    double z1 = std::numeric_limits<double>::infinity();
    double z2 = std::numeric_limits<double>::infinity();
    double n1 = -std::numeric_limits<double>::infinity();
    double n2 = -std::numeric_limits<double>::infinity();

#ifdef _OPENMP
    #pragma omp parallel reduction(min : z1, z2) reduction(max : n1, n2)
    {
        EvalWorkspace ws;
        ws.reset(prob.n_types, prob.max_slot);
        #pragma omp for schedule(static)
        for (int i = 0; i < n_weights; i++) {
            if (i >= n_hs) {
                std::mt19937 lrng(init_seeds[i]);
                pop[i] = make_random(prob, lrng);
            }
            evaluate(pop[i], prob, ws);
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
        EvalWorkspace ws;
        ws.reset(prob.n_types, prob.max_slot);
        for (int i = 0; i < n_weights; i++) {
            if (i >= n_hs) {
                std::mt19937 lrng(init_seeds[i]);
                pop[i] = make_random(prob, lrng);
            }
            evaluate(pop[i], prob, ws);
            if (pop[i].cv == 0.0) {
                z1 = std::min(z1, pop[i].f1);
                z2 = std::min(z2, pop[i].f2);
                n1 = std::max(n1, pop[i].f1);
                n2 = std::max(n2, pop[i].f2);
            }
        }
    }
#endif
    if (!std::isfinite(z1)) z1 = 0.0;
    if (!std::isfinite(z2)) z2 = 0.0;
    if (!std::isfinite(n1)) n1 = z1 + 1.0;
    if (!std::isfinite(n2)) n2 = z2 + 1.0;

    const double init_eval_ms = ms_since(t_init);

    // Normalised Tchebycheff: scale each objective by its observed [ideal, nadir]
    // range so that weight vectors spread evenly across the tradeoff.
    auto scalar = [&](const Individual& ind, int i) noexcept -> double {
        if (ind.cv > 0.0) return 1e18 + ind.cv * 1e6;
        const double lam1 = W[i].first, lam2 = W[i].second;
        const double r1 = std::max(n1 - z1, 1.0);
        const double r2 = std::max(n2 - z2, 1.0);
        return std::max(lam1 * (ind.f1 - z1) / r1,
                        lam2 * (ind.f2 - z2) / r2);
    };

    std::vector<Individual> children(n_weights);
    std::vector<uint32_t>   child_seeds(n_weights);

    std::vector<Individual> archive;
    if (archive_size > 0) archive.reserve(archive_size + 1);

    double offspring_ms = 0.0, replacement_ms = 0.0;

    for (int gen = 0; gen < n_gen; gen++) {
        // ── Phase 1: parallel offspring generation + ideal update ─────────────

        for (auto& s : child_seeds) s = rng();

        double dz1 = z1, dz2 = z2;
        double dn1 = n1, dn2 = n2;

        const double p_mut_gen = mut_sched.at(gen, n_gen);

        const auto t_off = Clock::now();

#ifdef _OPENMP
        #pragma omp parallel reduction(min : dz1, dz2) reduction(max : dn1, dn2)
        {
            EvalWorkspace ws;
            ws.reset(prob.n_types, prob.max_slot);
            #pragma omp for schedule(static)
            for (int i = 0; i < n_weights; i++) {
                std::mt19937 lrng(child_seeds[i]);
                std::uniform_int_distribution<int> rn(0, T - 1);
                const int a = neigh[i][rn(lrng)];
                const int b = neigh[i][rn(lrng)];
                children[i] = crossover(pop[a], pop[b], lrng, crossover_kind);
                mutate(children[i], prob, p_mut_gen, lrng);
                evaluate(children[i], prob, ws);
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
            EvalWorkspace ws;
            ws.reset(prob.n_types, prob.max_slot);
            for (int i = 0; i < n_weights; i++) {
                std::mt19937 lrng(child_seeds[i]);
                std::uniform_int_distribution<int> rn(0, T - 1);
                const int a = neigh[i][rn(lrng)];
                const int b = neigh[i][rn(lrng)];
                children[i] = crossover(pop[a], pop[b], lrng, crossover_kind);
                mutate(children[i], prob, p_mut_gen, lrng);
                evaluate(children[i], prob, ws);
                if (children[i].cv == 0.0) {
                    dz1 = std::min(dz1, children[i].f1);
                    dz2 = std::min(dz2, children[i].f2);
                    dn1 = std::max(dn1, children[i].f1);
                    dn2 = std::max(dn2, children[i].f2);
                }
            }
        }
#endif
        z1 = dz1; z2 = dz2; n1 = dn1; n2 = dn2;
        offspring_ms += ms_since(t_off);

        if (archive_size > 0)
            for (const auto& c : children) archive_insert(archive, c, archive_size);

        // ── Phase 2: sequential neighbourhood replacement ─────────────────────
        // max_replace <= 0 or >= T: unbounded, identical to the original loop
        // (same order, no RNG draws).  Otherwise each child may only take over
        // up to max_replace neighbours, chosen in a shuffled order, capping
        // how fast it can dominate a region of weight space.

        const auto t_rep = Clock::now();
        const bool capped = max_replace > 0 && max_replace < T;
        std::vector<int> order;
        if (capped) order.resize(T);
        for (int i = 0; i < n_weights; i++) {
            if (!capped) {
                for (int j : neigh[i])
                    if (scalar(children[i], j) <= scalar(pop[j], j))
                        pop[j] = children[i];
                continue;
            }
            std::copy(neigh[i].begin(), neigh[i].end(), order.begin());
            std::shuffle(order.begin(), order.end(), rng);
            int replaced = 0;
            for (int j : order) {
                if (replaced >= max_replace) break;
                if (scalar(children[i], j) <= scalar(pop[j], j)) {
                    pop[j] = children[i];
                    replaced++;
                }
            }
        }
        replacement_ms += ms_since(t_rep);
    }

    // ── Extract Pareto front ──────────────────────────────────────────────────

    const auto t_extract = Clock::now();

    ResultList results;
    for (auto& ind : pop)
        if (ind.cv == 0.0)
            results.emplace_back(extract_assignment(ind, prob), ind.f1, ind.f2);
    if (archive_size > 0)
        for (auto& ind : archive)
            results.emplace_back(extract_assignment(ind, prob), ind.f1, ind.f2);
    auto filtered = pareto_filter(std::move(results));

    const double extract_ms = ms_since(t_extract);
    const double total_ms   = ms_since(t_total);
    const double gen_d      = std::max(n_gen, 1);

    Profile profile = {
        {"n_gen",               static_cast<double>(n_gen)},
        {"total_ms",            total_ms},
        {"init_eval_ms",        init_eval_ms},
        {"offspring_total_ms",  offspring_ms},
        {"offspring_avg_ms",    offspring_ms  / gen_d},
        {"replacement_total_ms",replacement_ms},
        {"replacement_avg_ms",  replacement_ms / gen_d},
        {"extract_ms",          extract_ms},
    };

    return {std::move(filtered), std::move(profile)};
}
