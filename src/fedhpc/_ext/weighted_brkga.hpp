/**
 * FED-HPC — weighted-sum solver, random-key / serial-SGS encoding (options B+C).
 *
 * An alternative to the memetic solver in weighted.hpp. Instead of searching a
 * per-job type-assignment vector and decoding start times with a fixed SPT
 * rule, the genotype is a vector of random keys (Bean 1994; Gonçalves &
 * Resende BRKGA) in three blocks, and the decoder is a serial schedule
 * generation scheme (SGS) driven by an evolved priority order:
 *
 *   key[0     .. n)   priority   — job order for the serial SGS (higher first)
 *   key[n     .. 2n)  delay      — per-job start floor, arrival + f(key)*H
 *                                  (option C: parameterized-active schedules,
 *                                   Mendes, Gonçalves & Resende 2005)
 *   key[2n    .. 3n)  type       — index into the job's feasible types, sorted
 *                                  by (cost asc, p_occ asc)
 *
 * The SGS places every job at its earliest capacity-feasible slot on its
 * decoded type, at or after its delay floor, jobs taken in priority order.
 * Serial SGS generates exactly the active schedules, and for a regular
 * objective (sum of turnarounds) an optimal schedule is active — so the fixed
 * SPT order of weighted.hpp's decoder is no longer a structural ceiling.
 *
 * BRKGA population management: elite / mutant / biased-uniform-crossover.
 * Every crossover child is a valid key vector, so no repair is needed on the
 * genotype. The decoded schedule is made feasible by the SGS (capacity) plus
 * the same soft penalty as weighted.hpp for the budget / Pareto-cap.
 *
 * Determinism: bit-for-bit stable for a fixed (seed, n_threads). Per-thread
 * RNGs seeded from the master RNG before every parallel region.
 */
#pragma once

#include "ga_common.hpp"
#include "weighted.hpp"   // WeightedScalar, WeightedResult

// A decoded individual: the key vector plus its decoded schedule + fitness.
struct RKIndividual {
    std::vector<double> keys;   // 3 * n_jobs
    Individual sched;           // decoded genes + f1/f2/cv
    double g = 0.0;
};

// Per-job feasible-type list, sorted by (cost asc, p_occ asc): the order the
// type key indexes into. Built once per run.
inline std::vector<std::vector<std::array<int, 3>>>
brkga_type_options(const Problem& prob) {
    // [j] = list of {type_id, block_begin, block_end}, cost-sorted
    std::vector<std::vector<std::array<int, 3>>> opt(prob.n_jobs);
    for (int j = 0; j < prob.n_jobs; j++) {
        opt[j] = prob.job_type_span[j];
        std::sort(opt[j].begin(), opt[j].end(),
                  [&](const std::array<int, 3>& a, const std::array<int, 3>& b) {
                      const SlotInfo& sa = prob.job_slots[j][a[1]];
                      const SlotInfo& sb = prob.job_slots[j][b[1]];
                      if (sa.cost != sb.cost) return sa.cost < sb.cost;
                      return sa.p_occ < sb.p_occ;
                  });
    }
    return opt;
}

// Index of type_id `tid` in job j's cost-sorted feasible-type list `topt[j]`.
inline int brkga_type_slot(const std::vector<std::array<int, 3>>& os, int tid) {
    for (int i = 0; i < (int)os.size(); i++)
        if (os[i][0] == tid) return i;
    return 0;
}

// Decode a key vector into a schedule via the serial SGS, then evaluate.
//
// polish: 0 = none; 1 = re-optimise start times on the key-chosen types only
// (schedule_repair + forward-backward improvement); 2 = also run the scalar
// type-flip local search (memetic BRKGA) and write the improved type vector
// back into the type keys (Lamarckian) so good type choices propagate through
// crossover. delay_mult == 0 disables option C.
inline void brkga_decode(RKIndividual& rk, const Problem& prob, EvalWorkspace& ws,
                         const std::vector<std::vector<std::array<int, 3>>>& topt,
                         const WeightedScalar& G, double delay_mult,
                         int polish, int shortlist, int ablate) {
    const int n = prob.n_jobs;
    const int stride = prob.max_slot + 1;
    Individual& ind = rk.sched;
    ind.genes.assign(n, 0);

    // ── 1. decode type + per-job delay floor ──────────────────────────────
    std::vector<int> floor_t(n, 0), blk_begin(n), blk_end(n);
    for (int j = 0; j < n; j++) {
        const auto& os = topt[j];
        const int K = (int)os.size();
        int ti = (int)(rk.keys[2 * n + j] * K);
        if (ti >= K) ti = K - 1;
        if (ti < 0) ti = 0;
        blk_begin[j] = os[ti][1];
        blk_end[j]   = os[ti][2];
        ind.genes[j] = os[ti][1];
        const SlotInfo& first = prob.job_slots[j][os[ti][1]];
        // paper-style: floor = earliest + key^2 * delay_mult * this job's p_occ
        const double d = rk.keys[n + j];
        floor_t[j] = first.start + (int)(d * d * delay_mult * first.p_occ);
    }

    // ── 2. priority order (higher key first) ──────────────────────────────
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (rk.keys[a] != rk.keys[b]) return rk.keys[a] > rk.keys[b];
        return a < b;
    });

    // ── 3. serial SGS: earliest feasible slot >= floor, in priority order ──
    for (int idx : ws.dirty) ws.occ[idx] = 0;
    ws.dirty.clear();

    for (int j : order) {
        const int b = blk_begin[j], e = blk_end[j];
        const int m = prob.job_slots[j][b].type_id;
        const int mcap = prob.type_cap[m];
        const int mbase = m * stride;

        int chosen = b;
        bool placed = false;
        for (int k = b; k < e; k++) {
            const SlotInfo& s = prob.job_slots[j][k];
            if (s.start < floor_t[j]) continue;
            bool fits = true;
            if (mcap >= 0) {
                for (int t = s.start; t < s.start + s.p_occ; t++)
                    if (ws.occ[mbase + t] + prob.init_occ_flat[mbase + t] >= mcap) {
                        fits = false; break;
                    }
            }
            if (fits) { chosen = k; placed = true; break; }
        }
        if (!placed) chosen = e - 1;   // floor past every feasible slot
        ind.genes[j] = chosen;

        if (mcap >= 0) {
            const SlotInfo& ap = prob.job_slots[j][chosen];
            for (int t = ap.start; t < ap.start + ap.p_occ; t++) {
                const int idx = mbase + t;
                if (ws.occ[idx] == 0) ws.dirty.push_back(idx);
                ws.occ[idx]++;
            }
        }
    }

    if (polish >= 1) {
        schedule_repair(ind, prob, ws, 2, /*free_pool_balance=*/0);
        evaluate(ind, prob, ws);
        forward_backward_improve(ind, prob, ws, 2, /*free_pool_balance=*/0);
    }
    if (polish >= 2) {
        // Scalar type-flip local search (memetic BRKGA), then write the
        // improved type vector back into the type keys.
        weighted_local_search(ind, prob, ws, G, 4, shortlist, ablate);
        for (int j = 0; j < n; j++) {
            const int ti = brkga_type_slot(topt[j],
                               prob.job_slots[j][ind.genes[j]].type_id);
            rk.keys[2 * n + j] = (ti + 0.5) / (double)topt[j].size();
        }
        forward_backward_improve(ind, prob, ws, 2, /*free_pool_balance=*/0);
    }
    evaluate(ind, prob, ws);
    rk.g = G(ind);
}

inline WeightedResult
run_weighted_brkga(const Problem& prob, double w1, double w2, double f1_cap,
                   int pop_size, int n_gen, int seed, int n_threads,
                   int ablate, const std::vector<std::vector<int>>& extra_seeds,
                   int use_delay, int local_polish) {
    py::gil_scoped_release release;
    set_num_threads(n_threads);

    const int n = prob.n_jobs;
    const int L = 3 * n;
    const int fpb = (ablate & ABL_NO_FREE_POOL) ? 0 : 1;
    const double delay_mult = use_delay ? 3.0 : 0.0;   // floor = est + key^2 * 3 * p_occ
    const int polish = local_polish;
    const int shortlist = 24;

    WeightedScalar G{w1, w2, f1_cap,
                     (w1 > 0.0 ? w1 : (w2 > 0.0 ? w2 : 1.0)) * 1e3};
    auto topt = brkga_type_options(prob);

    // BRKGA parameters (Gonçalves & Resende defaults).
    const int n_elite  = std::max(1, (int)std::lround(0.20 * pop_size));
    const int n_mutant = std::max(1, (int)std::lround(0.15 * pop_size));
    const double rhoe  = 0.70;   // prob. of inheriting the elite parent's key

    std::mt19937 rng(seed);
    int decodes = 0;

    // ── initial population: random keys + biased keys from extra_seeds ─────
    std::vector<RKIndividual> pop(pop_size);
    std::vector<uint32_t> iseed(pop_size);
    for (auto& s : iseed) s = rng();

    // extra_seeds are per-job type-assignment genomes (slot indices). Encode
    // the implied type choice into the type-key block; priority/delay random.
    const int n_seed = std::min<int>((int)extra_seeds.size(), pop_size / 2);

#ifdef _OPENMP
    #pragma omp parallel
    {
        EvalWorkspace ws; ws.reset(prob.n_types, prob.max_slot);
        int local = 0;
        #pragma omp for schedule(dynamic, 1)
        for (int k = 0; k < pop_size; k++) {
            std::mt19937 lr(iseed[k]);
            std::uniform_real_distribution<double> u(0.0, 1.0);
            pop[k].keys.resize(L);
            for (int i = 0; i < L; i++) pop[k].keys[i] = u(lr);
            if (k < n_seed) {
                const auto& g = extra_seeds[k];
                for (int j = 0; j < n && j < (int)g.size(); j++) {
                    const int tid = prob.job_slots[j][std::clamp(g[j], 0,
                                        (int)prob.job_slots[j].size() - 1)].type_id;
                    const auto& os = topt[j];
                    for (int ti = 0; ti < (int)os.size(); ti++)
                        if (os[ti][0] == tid) {
                            pop[k].keys[2 * n + j] =
                                (ti + 0.5) / (double)os.size();
                            break;
                        }
                }
            }
            brkga_decode(pop[k], prob, ws, topt, G, delay_mult, polish, shortlist, ablate);
            local++;
        }
        #pragma omp atomic
        decodes += local;
    }
#else
    {
        EvalWorkspace ws; ws.reset(prob.n_types, prob.max_slot);
        for (int k = 0; k < pop_size; k++) {
            std::mt19937 lr(iseed[k]);
            std::uniform_real_distribution<double> u(0.0, 1.0);
            pop[k].keys.resize(L);
            for (int i = 0; i < L; i++) pop[k].keys[i] = u(lr);
            brkga_decode(pop[k], prob, ws, topt, G, delay_mult, polish, shortlist, ablate);
            decodes++;
        }
    }
#endif

    auto by_g = [](const RKIndividual& a, const RKIndividual& b) { return a.g < b.g; };
    std::sort(pop.begin(), pop.end(), by_g);

    std::vector<RKIndividual> next(pop_size);
    std::vector<uint32_t> cseed(pop_size);

    for (int gen = 0; gen < n_gen; gen++) {
        for (auto& s : cseed) s = rng();

        // elite carried over unchanged
        for (int k = 0; k < n_elite; k++) next[k] = pop[k];

#ifdef _OPENMP
        #pragma omp parallel
        {
            EvalWorkspace ws; ws.reset(prob.n_types, prob.max_slot);
            int local = 0;
            #pragma omp for schedule(dynamic, 1)
            for (int k = n_elite; k < pop_size; k++) {
                std::mt19937 lr(cseed[k]);
                std::uniform_real_distribution<double> u(0.0, 1.0);
                next[k].keys.resize(L);
                if (k >= pop_size - n_mutant) {
                    for (int i = 0; i < L; i++) next[k].keys[i] = u(lr);
                } else {
                    const int ea = std::uniform_int_distribution<int>(0, n_elite - 1)(lr);
                    const int nb = std::uniform_int_distribution<int>(
                                       n_elite, pop_size - 1)(lr);
                    for (int i = 0; i < L; i++)
                        next[k].keys[i] = (u(lr) < rhoe) ? pop[ea].keys[i]
                                                         : pop[nb].keys[i];
                }
                brkga_decode(next[k], prob, ws, topt, G, delay_mult, polish, shortlist, ablate);
                local++;
            }
            #pragma omp atomic
            decodes += local;
        }
#else
        {
            EvalWorkspace ws; ws.reset(prob.n_types, prob.max_slot);
            for (int k = n_elite; k < pop_size; k++) {
                std::mt19937 lr(cseed[k]);
                std::uniform_real_distribution<double> u(0.0, 1.0);
                next[k].keys.resize(L);
                if (k >= pop_size - n_mutant) {
                    for (int i = 0; i < L; i++) next[k].keys[i] = u(lr);
                } else {
                    const int ea = std::uniform_int_distribution<int>(0, n_elite - 1)(lr);
                    const int nb = std::uniform_int_distribution<int>(
                                       n_elite, pop_size - 1)(lr);
                    for (int i = 0; i < L; i++)
                        next[k].keys[i] = (u(lr) < rhoe) ? pop[ea].keys[i]
                                                         : pop[nb].keys[i];
                }
                brkga_decode(next[k], prob, ws, topt, G, delay_mult, polish, shortlist, ablate);
                decodes++;
            }
        }
#endif
        pop.swap(next);
        std::sort(pop.begin(), pop.end(), by_g);
    }

    // Final polish on the incumbent — start-time only (the type vector is the
    // chromosome's to own): schedule_repair + Extract-from-Preempt re-decode +
    // forward-backward improvement, all on the key-chosen types.
    (void)fpb;
    {
        EvalWorkspace ws; ws.reset(prob.n_types, prob.max_slot);
        Individual& best = pop[0].sched;
        schedule_repair(best, prob, ws, 3, /*free_pool_balance=*/0);
        evaluate(best, prob, ws);
        finish_decode(best, prob, ws, /*decode_order=*/1, /*fbi_passes=*/6,
                      /*free_pool_balance=*/0);
        pop[0].g = G(best);
    }

    const Individual& best = pop[0].sched;
    return {extract_assignment(best, prob), best.f1, best.f2, pop[0].g, decodes};
}
