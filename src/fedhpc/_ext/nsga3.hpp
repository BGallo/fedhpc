/**
 * FED-HPC — NSGA-III with structured reference-point selection (Deb & Jain 2014).
 *
 * Key differences from NSGA-II (nsga2.hpp):
 *   - Das-Dennis reference points on the unit hyperplane replace crowding distance
 *     for diversity maintenance.
 *   - Each generation: normalization → reference-point association → niching-based
 *     survival selection for the critical (last partial) front.
 *   - Tournament selection uses rank only; ties are broken randomly rather than by
 *     crowding distance.
 *   - pop_size should be set to n_divisions + 1 for best coverage; both default to
 *     100 / 99 respectively.
 *
 * Normalization (per generation, on the combined S ∪ Fl pool):
 *   1. Ideal point z*: component-wise minimum over feasible individuals (cv == 0).
 *   2. Extreme points: one per objective, found via the Achievement Scalarizing
 *      Function (ASF) using a weight vector with ε = 1e-6 on non-focus objectives.
 *   3. Intercepts a_m: line fitted through the two extreme points in translated
 *      objective space; falls back to observed maxima when the system is degenerate.
 *   4. Normalized objectives: f̄_m = (f_m − z*_m) / a_m.
 *
 * Niching (selection from the critical front Fl):
 *   - Association: each individual is mapped to its closest reference point by
 *     perpendicular distance in normalized space.
 *   - ρ_j = niching count of reference point j (individuals in S associated with j).
 *   - Iteratively: pick a random reference point j* with minimum ρ among those that
 *     still have Fl candidates; if ρ_j* == 0 select the closest Fl member, else a
 *     random one; increment ρ_j*.
 *
 * Profiling:
 *   Returns a Profile (vector of (name, ms) pairs) alongside the solutions.
 *   Phases timed per generation (summed + per-gen average reported):
 *     init_eval_ms       — heuristic seeds + random init + evaluation
 *     rank_nds_total_ms  — NDS + rank assignment on the parent population (for mating)
 *     offspring_total_ms — parallel offspring generation + evaluation
 *     combine_select_total_ms — NDS on combined pool + normalization + niching
 *     extract_ms         — final NDS + result extraction + Pareto filter
 */
#pragma once

#include "ga_common.hpp"
#include "nsga2.hpp"   // reuse fast_nds / assign_ranks

#include <array>

// ── Reference points ──────────────────────────────────────────────────────────

// Das-Dennis structured reference points for M = 2 objectives.
// n_divisions p → (p + 1) equally-spaced points on the unit simplex.
inline std::vector<std::array<double, 2>>
make_reference_points_2d(int n_divisions) {
    std::vector<std::array<double, 2>> refs;
    refs.reserve(n_divisions + 1);
    for (int i = 0; i <= n_divisions; i++) {
        const double lam = (n_divisions > 0) ? (double)i / n_divisions : 0.5;
        refs.push_back({lam, 1.0 - lam});
    }
    return refs;
}

// ── Tournament (rank-only, random tiebreak) ───────────────────────────────────

inline const Individual&
tournament3(const Individual& a, const Individual& b, std::mt19937& rng) noexcept {
    if (a.rank < b.rank) return a;
    if (b.rank < a.rank) return b;
    return (std::uniform_int_distribution<int>(0, 1)(rng) == 0) ? a : b;
}

// Generalised k-ary tournament (rank-only, random tiebreak). k=2 (default)
// call sites keep the original two-argument tournament3() call so the
// default path is an exact byte-for-byte match to the pre-existing
// implementation.
inline const Individual&
tournament3_k(const std::vector<Individual>& pop, int k, std::mt19937& rng) noexcept {
    std::uniform_int_distribution<int> ri(0, static_cast<int>(pop.size()) - 1);
    const Individual* best = &pop[ri(rng)];
    for (int i = 1; i < k; i++)
        best = &tournament3(*best, pop[ri(rng)], rng);
    return *best;
}

// ── Normalization helpers ─────────────────────────────────────────────────────

struct RefData {
    double norm_f1;
    double norm_f2;
    int    ref_idx;   // associated reference point index
    double ref_dist;  // perpendicular distance to that reference line
};

// Component-wise minimum over feasible individuals; falls back to all if none feasible.
inline std::pair<double, double>
nsga3_ideal(const std::vector<Individual>& pool, const std::vector<int>& idxs) {
    double z1 = std::numeric_limits<double>::infinity();
    double z2 = std::numeric_limits<double>::infinity();
    bool any = false;
    for (int i : idxs) {
        if (pool[i].cv == 0.0) {
            z1 = std::min(z1, pool[i].f1);
            z2 = std::min(z2, pool[i].f2);
            any = true;
        }
    }
    if (!any)
        for (int i : idxs) { z1 = std::min(z1, pool[i].f1); z2 = std::min(z2, pool[i].f2); }
    if (!std::isfinite(z1)) { z1 = 0.0; z2 = 0.0; }
    return {z1, z2};
}

// Fit hyperplane x/a1 + y/a2 = 1 through the two ASF extreme points of the pool
// in the translated objective space (translated by ideal point z1, z2).
// Returns {a1, a2}; falls back to observed maxima on degenerate input.
inline std::pair<double, double>
nsga3_intercepts(const std::vector<Individual>& pool, const std::vector<int>& idxs,
                 double z1, double z2) {
    constexpr double WEPS = 1e-6;

    // Prefer feasible individuals for the ASF scan.
    std::vector<int> src;
    src.reserve(idxs.size());
    for (int i : idxs) if (pool[i].cv == 0.0) src.push_back(i);
    if (src.empty()) src = idxs;

    int    ext1 = src[0], ext2 = src[0];
    double min1 = std::numeric_limits<double>::infinity();
    double min2 = std::numeric_limits<double>::infinity();
    for (int i : src) {
        const double fp1 = pool[i].f1 - z1;
        const double fp2 = pool[i].f2 - z2;
        const double asf1 = std::max(fp1,        fp2 / WEPS);  // focus on f1
        const double asf2 = std::max(fp1 / WEPS, fp2);         // focus on f2
        if (asf1 < min1) { min1 = asf1; ext1 = i; }
        if (asf2 < min2) { min2 = asf2; ext2 = i; }
    }

    // Translate extreme points.
    const double x1 = pool[ext1].f1 - z1, y1 = pool[ext1].f2 - z2;
    const double x2 = pool[ext2].f1 - z1, y2 = pool[ext2].f2 - z2;

    // Solve [x1 y1; x2 y2] * [1/a1; 1/a2] = [1; 1]  via Cramer's rule.
    const double det = x1 * y2 - x2 * y1;

    auto max_val = [&](int obj) -> double {
        double mv = 1e-9;
        for (int i : src)
            mv = std::max(mv, (obj == 0 ? pool[i].f1 : pool[i].f2)
                              - (obj == 0 ? z1 : z2));
        return mv;
    };

    double a1, a2;
    if (std::abs(det) < 1e-12 || ext1 == ext2) {
        a1 = max_val(0);
        a2 = max_val(1);
    } else {
        const double inv1 = (y2 - y1) / det;
        const double inv2 = (x1 - x2) / det;
        a1 = (std::abs(inv1) > 1e-12 && 1.0 / inv1 > 1e-9) ? 1.0 / inv1 : max_val(0);
        a2 = (std::abs(inv2) > 1e-12 && 1.0 / inv2 > 1e-9) ? 1.0 / inv2 : max_val(1);
    }
    return {std::max(a1, 1e-9), std::max(a2, 1e-9)};
}

// Perpendicular distance from (f1, f2) to the ray through origin with direction d.
inline double perp_dist_2d(double f1, double f2,
                            const std::array<double, 2>& d) noexcept {
    const double dlen2 = d[0] * d[0] + d[1] * d[1];
    const double proj  = (f1 * d[0] + f2 * d[1]) / dlen2;
    const double e1    = f1 - proj * d[0];
    const double e2    = f2 - proj * d[1];
    return std::sqrt(e1 * e1 + e2 * e2);
}

// Normalize objectives and associate each individual with its nearest reference point.
// idxs are indices into pool; all_data[k] corresponds to pool[idxs[k]].
inline std::vector<RefData>
nsga3_associate(const std::vector<Individual>& pool, const std::vector<int>& idxs,
                double z1, double z2, double a1, double a2,
                const std::vector<std::array<double, 2>>& refs) {
    const int N = (int)idxs.size();
    const int R = (int)refs.size();
    std::vector<RefData> data(N);

    for (int k = 0; k < N; k++) {
        const int i       = idxs[k];
        data[k].norm_f1   = (pool[i].f1 - z1) / a1;
        data[k].norm_f2   = (pool[i].f2 - z2) / a2;

        double best_dist  = std::numeric_limits<double>::infinity();
        int    best_r     = 0;
        for (int r = 0; r < R; r++) {
            const double d = perp_dist_2d(data[k].norm_f1, data[k].norm_f2, refs[r]);
            if (d < best_dist) { best_dist = d; best_r = r; }
        }
        data[k].ref_idx   = best_r;
        data[k].ref_dist  = best_dist;
    }
    return data;
}

// ── NSGA-III niching selection from the critical front ────────────────────────

// Given:
//   all_data[0..n_S-1]        → individuals already accepted into next generation (S)
//   all_data[n_S..n_S+n_Fl-1] → candidates from the critical front (Fl)
//   refs                       → reference points
//   needed                     → how many to select from Fl
// Returns indices into Fl (i.e. into all_data offset by n_S) that are selected.
inline std::vector<int>
niching_select(const std::vector<RefData>& all_data, int n_S, int n_Fl,
               int R, int needed, std::mt19937& rng) {
    // Build niching counts for S
    std::vector<int> rho(R, 0);
    for (int k = 0; k < n_S; k++)
        rho[all_data[k].ref_idx]++;

    std::vector<bool> available(n_Fl, true);
    std::vector<int>  selected;
    selected.reserve(needed);

    while ((int)selected.size() < needed) {
        // Minimum rho among reference points that still have available Fl candidates.
        int min_rho = std::numeric_limits<int>::max();
        for (int k = 0; k < n_Fl; k++) {
            if (available[k])
                min_rho = std::min(min_rho, rho[all_data[n_S + k].ref_idx]);
        }

        // Collect qualifying reference points (min rho, with available candidates).
        std::vector<int> min_refs;
        {
            std::vector<bool> seen(R, false);
            for (int k = 0; k < n_Fl; k++) {
                if (!available[k]) continue;
                const int r = all_data[n_S + k].ref_idx;
                if (rho[r] == min_rho && !seen[r]) {
                    min_refs.push_back(r);
                    seen[r] = true;
                }
            }
        }
        if (min_refs.empty()) break;

        // Random qualifying reference point.
        const int chosen_r = min_refs[
            std::uniform_int_distribution<int>(0, (int)min_refs.size() - 1)(rng)];

        // Fl candidates associated with chosen_r.
        std::vector<int> cands;
        for (int k = 0; k < n_Fl; k++)
            if (available[k] && all_data[n_S + k].ref_idx == chosen_r)
                cands.push_back(k);

        int sel_k;
        if (rho[chosen_r] == 0) {
            // No individual in S is on this reference line — pick the closest Fl member.
            sel_k = cands[0];
            for (int k : cands)
                if (all_data[n_S + k].ref_dist < all_data[n_S + sel_k].ref_dist)
                    sel_k = k;
        } else {
            // Already have at least one in S — pick a random Fl candidate.
            sel_k = cands[
                std::uniform_int_distribution<int>(0, (int)cands.size() - 1)(rng)];
        }

        selected.push_back(sel_k);
        available[sel_k] = false;
        rho[chosen_r]++;
    }
    return selected;
}

// ── NSGA-III main loop ────────────────────────────────────────────────────────

inline std::pair<ResultList, Profile>
run_nsga3(const Problem& prob, int pop_size, int n_divisions,
          int n_gen, int seed, int n_threads,
          double p_mut_start = -1.0, double p_mut_end = -1.0, int crossover_kind = 0,
          int tourn_k = 2, int local_search_interval = -1, int sched_repair = 0,
          int ablate = 0) {
    py::gil_scoped_release release;
    set_num_threads(n_threads);

    const bool abl_ls    = ablate & ABL_NO_LOCAL_SEARCH;
    const bool abl_sr     = ablate & ABL_NO_SCHED_REPAIR;
    const bool abl_seeds  = ablate & ABL_NO_HEURISTIC_SEEDS;
    const bool abl_unif   = ablate & ABL_UNIFORM_RANDOM;
    const bool abl_xover  = ablate & ABL_NO_CROSSOVER;
    const int  sr_fpb     = sched_repair >= 2 ? 1 : 0;
    auto do_ls = [&](Individual& x, const Problem& pp, EvalWorkspace& w, int mp = 3) {
        if (!abl_ls) local_search(x, pp, w, mp);
    };
    auto do_sr = [&](Individual& x, const Problem& pp, EvalWorkspace& w) {
        if (sched_repair && !abl_sr) schedule_repair(x, pp, w, 2, sr_fpb);
    };

    const auto t_total = Clock::now();

    std::mt19937 rng(seed);
    const auto mut_sched = make_mutation_schedule(p_mut_start, p_mut_end, 2.0 / prob.n_jobs);

    // See nsga2.hpp for the interval convention (<0 auto, 0 disabled, >0 explicit).
    const int ls_interval = (local_search_interval < 0)
        ? std::max(1, n_gen / 10) : local_search_interval;

    const auto refs = make_reference_points_2d(n_divisions);
    const int  R    = (int)refs.size();

    // ── Initial population ─────────────────────────────────────────────────────

    const auto t_init = Clock::now();

    std::vector<uint32_t> init_seeds(pop_size);
    for (auto& s : init_seeds) s = rng();

    auto      h_seeds = abl_seeds ? std::vector<Individual>{} : make_heuristic_seeds(prob);
    const int n_hs    = (int)h_seeds.size();

    // See nsga2.hpp for why the seeds need repairing before entering the
    // population: make_greedy() in particular is typically wildly
    // capacity-infeasible on its own.
    {
        EvalWorkspace ws_seed;
        ws_seed.reset(prob.n_types, prob.max_slot);
        for (auto& s : h_seeds) {
            do_ls(s, prob, ws_seed);
            do_sr(s, prob, ws_seed);
            evaluate(s, prob, ws_seed);
        }
    }

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
                pop[k] = make_random(prob, lrng, abl_unif);
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
                pop[k] = make_random(prob, lrng, abl_unif);
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

    double rank_nds_ms = 0.0, offspring_ms = 0.0, combine_select_ms = 0.0, local_search_ms = 0.0;

    for (int gen = 0; gen < n_gen; gen++) {
        // ── Rank parent population for tournament selection ────────────────────

        const auto t_rank = Clock::now();
        {
            auto fronts_pop = fast_nds(pop);
            assign_ranks(pop, fronts_pop);
        }
        rank_nds_ms += ms_since(t_rank);

        // ── Offspring generation — parallel ───────────────────────────────────

        for (auto& s : child_seeds) s = rng();

        const double p_mut_gen = mut_sched.at(gen, n_gen);

        const auto t_off = Clock::now();

#ifdef _OPENMP
        #pragma omp parallel
        {
            EvalWorkspace ws;
            ws.reset(prob.n_types, prob.max_slot);
            #pragma omp for schedule(static)
            for (int k = 0; k < pop_size; k++) {
                std::mt19937 lrng(child_seeds[k]);
                std::uniform_int_distribution<int> ri(0, pop_size - 1);
                const Individual& p1 = (tourn_k == 2)
                    ? tournament3(pop[ri(lrng)], pop[ri(lrng)], lrng)
                    : tournament3_k(pop, tourn_k, lrng);
                const Individual& p2 = (tourn_k == 2)
                    ? tournament3(pop[ri(lrng)], pop[ri(lrng)], lrng)
                    : tournament3_k(pop, tourn_k, lrng);
                offspring[k] = abl_xover ? p1 : crossover(p1, p2, lrng, crossover_kind);
                mutate(offspring[k], prob, p_mut_gen, lrng);
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
                const Individual& p1 = (tourn_k == 2)
                    ? tournament3(pop[ri(lrng)], pop[ri(lrng)], lrng)
                    : tournament3_k(pop, tourn_k, lrng);
                const Individual& p2 = (tourn_k == 2)
                    ? tournament3(pop[ri(lrng)], pop[ri(lrng)], lrng)
                    : tournament3_k(pop, tourn_k, lrng);
                offspring[k] = abl_xover ? p1 : crossover(p1, p2, lrng, crossover_kind);
                mutate(offspring[k], prob, p_mut_gen, lrng);
                evaluate(offspring[k], prob, ws);
            }
        }
#endif

        offspring_ms += ms_since(t_off);

        // ── Combine + NSGA-III selection ──────────────────────────────────────

        const auto t_sel = Clock::now();

        combined.clear();
        for (auto& x : pop)       combined.push_back(std::move(x));
        for (auto& x : offspring) combined.push_back(std::move(x));

        // Fast NDS on combined pool.
        auto cf = fast_nds(combined);
        assign_ranks(combined, cf);

        // Identify the critical front: first front that would overflow pop_size.
        int filled      = 0;
        int critical_fi = (int)cf.size();  // sentinel: all fronts fit
        for (int fi = 0; fi < (int)cf.size(); fi++) {
            if (filled + (int)cf[fi].size() > (size_t)pop_size) {
                critical_fi = fi;
                break;
            }
            filled += (int)cf[fi].size();
        }

        const int needed = pop_size - filled;

        pop.clear();

        if (needed == 0 || critical_fi == (int)cf.size()) {
            // All complete fronts exactly fill (or slightly underfill) pop_size.
            for (int fi = 0; fi < (int)cf.size() && (int)pop.size() < pop_size; fi++)
                for (int i : cf[fi]) {
                    if ((int)pop.size() >= pop_size) break;
                    pop.push_back(std::move(combined[i]));
                }
        } else {
            // Complete fronts 0..critical_fi-1 go directly into pop.
            // Critical front: select 'needed' via niching.

            // Build all_indices = S indices (fronts 0..critical_fi-1) ++ Fl indices.
            std::vector<int> all_indices;
            all_indices.reserve(filled + (int)cf[critical_fi].size());
            for (int fi = 0; fi < critical_fi; fi++)
                for (int i : cf[fi]) all_indices.push_back(i);
            for (int i : cf[critical_fi]) all_indices.push_back(i);

            // Normalize and associate all individuals in S ∪ Fl.
            auto [z1, z2] = nsga3_ideal(combined, all_indices);
            auto [a1, a2] = nsga3_intercepts(combined, all_indices, z1, z2);
            auto all_data = nsga3_associate(combined, all_indices, z1, z2, a1, a2, refs);

            const int n_S  = filled;
            const int n_Fl = (int)cf[critical_fi].size();

            auto sel_from_fl = niching_select(all_data, n_S, n_Fl, R, needed, rng);

            // Move accepted individuals into pop (read combined before any moves).
            for (int fi = 0; fi < critical_fi; fi++)
                for (int i : cf[fi]) pop.push_back(std::move(combined[i]));

            for (int k : sel_from_fl)
                pop.push_back(std::move(combined[cf[critical_fi][k]]));
        }

        if ((int)pop.size() > pop_size) pop.resize(pop_size);

        combine_select_ms += ms_since(t_sel);

        // ── Periodic local search ──────────────────────────────────────────────
        // See nsga2.hpp for rationale.

        if (ls_interval > 0 && (gen + 1) % ls_interval == 0) {
            const auto t_ls = Clock::now();
#ifdef _OPENMP
            #pragma omp parallel
            {
                EvalWorkspace ws;
                ws.reset(prob.n_types, prob.max_slot);
                #pragma omp for schedule(static)
                for (int k = 0; k < (int)pop.size(); k++) {
                    if (!abl_ls) local_search(pop[k], prob, ws, /*max_passes=*/1);
                    evaluate(pop[k], prob, ws);
                }
            }
#else
            {
                EvalWorkspace ws;
                ws.reset(prob.n_types, prob.max_slot);
                for (auto& ind : pop) {
                    if (!abl_ls) local_search(ind, prob, ws, 1);
                    evaluate(ind, prob, ws);
                }
            }
#endif
            local_search_ms += ms_since(t_ls);
        }
    }

    // ── Extract Pareto front ──────────────────────────────────────────────────

    const auto t_extract = Clock::now();

    // Polish the final population before ranking — see nsga2.hpp's
    // extraction step for rationale.
    {
        EvalWorkspace ws_polish;
        ws_polish.reset(prob.n_types, prob.max_slot);
        for (auto& ind : pop) {
            do_ls(ind, prob, ws_polish);
            do_sr(ind, prob, ws_polish);
            evaluate(ind, prob, ws_polish);
        }
    }

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
        {"n_gen",                     static_cast<double>(n_gen)},
        {"total_ms",                  total_ms},
        {"init_eval_ms",              init_eval_ms},
        {"rank_nds_total_ms",         rank_nds_ms},
        {"rank_nds_avg_ms",           rank_nds_ms      / gen_d},
        {"offspring_total_ms",        offspring_ms},
        {"offspring_avg_ms",          offspring_ms     / gen_d},
        {"combine_select_total_ms",   combine_select_ms},
        {"combine_select_avg_ms",     combine_select_ms / gen_d},
        {"local_search_total_ms",     local_search_ms},
        {"extract_ms",                extract_ms},
    };

    return {std::move(filtered), std::move(profile)};
}
