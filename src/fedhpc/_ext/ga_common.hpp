/**
 * FED-HPC multi-objective scheduling — shared GA data structures and operators.
 *
 * Performance notes
 * -----------------
 * evaluate():
 *   Flat int[] occupancy grid (type_id × slot) + per-thread dirty list so only
 *   touched cells are zeroed between calls; no heap allocation in the hot loop.
 *
 * Offspring generation:
 *   All children per generation are created and evaluated in parallel.  Each
 *   OpenMP thread owns a seeded mt19937 + EvalWorkspace (no shared mutable state).
 *   Seeds are drawn from the master RNG *before* the parallel region, giving
 *   deterministic results for any fixed (seed, n_threads) pair.
 *
 * GIL:
 *   Both run_nsga2 and run_moead release the Python GIL for their entire
 *   duration so other Python threads are never blocked.
 */
#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef _OPENMP
#  include <omp.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace py = pybind11;

// ── Profiling ────────────────────────────────────────────────────────────────
//
// Profile is a flat list of (name, milliseconds) pairs.  Each algorithm appends
// named phase timings so callers can build a table without knowing the schema
// in advance.  n_gen is always the first entry so the Python layer can compute
// per-generation averages without a separate parameter.

using Clock     = std::chrono::steady_clock;
using TimePoint = Clock::time_point;
using Profile   = std::vector<std::pair<std::string, double>>;

inline double ms_since(TimePoint t0) noexcept {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// ── Threading ─────────────────────────────────────────────────────────────────

static inline void set_num_threads(int n) {
#ifdef _OPENMP
    if (n > 0) omp_set_num_threads(n);
#else
    (void)n;
#endif
}

// ── Data types ────────────────────────────────────────────────────────────────

struct SlotInfo {
    int    type_id;
    int    start;
    int    p_occ;
    double f1_contrib;  // t + p_occ - arrival_j  (precomputed)
    double cost;        // c[j,m]
};

struct Problem {
    int    n_jobs;
    int    n_types;   // number of distinct type ids
    int    max_slot;  // max(start + p_occ) over all job_slots — exclusive upper bound
    double budget;

    std::vector<std::vector<SlotInfo>> job_slots;  // [j] = feasible slots for job j
    std::vector<int> type_cap;        // type_cap[m]: capacity, -1 = unlimited
    std::vector<int> init_occ_flat;   // flat [type_id * (max_slot+1) + t] occupancy
                                      //   from running jobs; 0 for unlimited types

    std::vector<double> type_risk;    // type_risk[m]: revocation-risk probability, 0 if unset
    std::vector<std::vector<double>> slot_cum_weight;
    // [j] = cumulative sampling weights over job_slots[j], for make_random().
    // Precomputed once here (not per draw) since job_slots is fixed for the
    // whole run.
};

// Per-thread occupancy workspace — no heap allocation inside evaluate().
struct EvalWorkspace {
    std::vector<int> occ;    // same layout as Problem::init_occ_flat
    std::vector<int> dirty;  // flat indices of cells with occ != 0

    void reset(int n_types, int max_slot) {
        occ.assign(static_cast<size_t>(n_types) * (max_slot + 1), 0);
        dirty.clear();
        dirty.reserve(4096);
    }
};

struct Individual {
    std::vector<int> genes;
    double f1 = 0.0, f2 = 0.0, cv = 0.0;
    int    rank     = 0;
    double crowding = 0.0;

    // Constrained dominance (Deb 2002)
    bool dominates(const Individual& o) const noexcept {
        if (cv > 0 && o.cv > 0) return cv < o.cv;
        if (cv == 0 && o.cv > 0) return true;
        if (cv > 0 && o.cv == 0) return false;
        return (f1 <= o.f1 && f2 <= o.f2) && (f1 < o.f1 || f2 < o.f2);
    }
};

// ── Evaluation ────────────────────────────────────────────────────────────────

// Flat-grid occupancy: zero only dirtied cells, no heap allocation in hot path.
inline void evaluate(Individual& ind, const Problem& prob, EvalWorkspace& ws) {
    for (int idx : ws.dirty) ws.occ[idx] = 0;
    ws.dirty.clear();

    ind.f1 = 0.0;
    ind.f2 = 0.0;
    ind.cv = 0.0;

    const int stride = prob.max_slot + 1;

    for (int j = 0; j < prob.n_jobs; j++) {
        const SlotInfo& s = prob.job_slots[j][ind.genes[j]];
        ind.f1 += s.f1_contrib;
        ind.f2 += s.cost;

        const int cap = (s.type_id < (int)prob.type_cap.size())
                        ? prob.type_cap[s.type_id] : -1;
        if (cap < 0) continue;

        const int base = s.type_id * stride;
        for (int t = s.start; t < s.start + s.p_occ; t++) {
            const int idx = base + t;
            if (ws.occ[idx] == 0) ws.dirty.push_back(idx);
            ws.occ[idx]++;
        }
    }

    for (int idx : ws.dirty) {
        const int cap   = prob.type_cap[idx / stride];
        const int total = ws.occ[idx] + prob.init_occ_flat[idx];
        const int exc   = total - cap;
        if (exc > 0) ind.cv += exc;
    }
    if (ind.f2 > prob.budget) ind.cv += ind.f2 - prob.budget;
}

// ── Genetic operators ─────────────────────────────────────────────────────────

// Cost-and-revocation-risk-biased sampling weights for a job's slot list.
// Cheaper / lower-risk slots get a higher weight, but the ratio is bounded
// (worst slot keeps 20% of the best slot's weight) so random initialization
// still explores costlier options instead of collapsing onto the cheapest.
//
// Falls back to uniform weights when the job's slots show no effective-cost
// variation at all — this is not just a numerical safeguard, it is the
// common case whenever every feasible slot is a free (cost=0), zero-risk
// on-prem type: with no signal to weight on, uniform sampling is correct.
inline void compute_slot_weights(const std::vector<SlotInfo>& slots,
                                  const std::vector<double>& type_risk,
                                  std::vector<double>& cum_weight) {
    const int n = static_cast<int>(slots.size());
    cum_weight.resize(n);

    std::vector<double> eff_cost(n);
    double min_c = std::numeric_limits<double>::infinity();
    double max_c = -std::numeric_limits<double>::infinity();
    for (int k = 0; k < n; k++) {
        const int tid = slots[k].type_id;
        const double risk = (tid < (int)type_risk.size()) ? type_risk[tid] : 0.0;
        eff_cost[k] = slots[k].cost * (1.0 + risk);
        min_c = std::min(min_c, eff_cost[k]);
        max_c = std::max(max_c, eff_cost[k]);
    }

    const double range = max_c - min_c;
    double running = 0.0;
    for (int k = 0; k < n; k++) {
        const double w = (range < 1e-12)
            ? 1.0
            : (1.0 - 0.8 * (eff_cost[k] - min_c) / range);
        running += w;
        cum_weight[k] = running;
    }
}

// Weighted-random genome: each gene is drawn from its job's precomputed
// cost/risk-biased distribution (see compute_slot_weights) instead of
// uniformly, so random-initialized individuals start out already leaning
// toward cheaper, lower-revocation-risk assignments.
inline Individual make_random(const Problem& prob, std::mt19937& rng) {
    Individual ind;
    ind.genes.resize(prob.n_jobs);
    for (int j = 0; j < prob.n_jobs; j++) {
        const auto& cum = prob.slot_cum_weight[j];
        std::uniform_real_distribution<double> u(0.0, cum.back());
        const double r = u(rng);
        const int idx = static_cast<int>(std::upper_bound(cum.begin(), cum.end(), r) - cum.begin());
        ind.genes[j] = std::min(idx, static_cast<int>(cum.size()) - 1);
    }
    return ind;
}

// 2-point crossover: takes a contiguous segment from p2, rest from p1.
inline Individual crossover_two_point(const Individual& p1, const Individual& p2,
                                      std::mt19937& rng) {
    Individual child;
    const int n = static_cast<int>(p1.genes.size());
    child.genes.resize(n);
    if (n <= 2) {
        std::uniform_int_distribution<int> coin(0, 1);
        for (int j = 0; j < n; j++)
            child.genes[j] = coin(rng) ? p1.genes[j] : p2.genes[j];
        return child;
    }
    std::uniform_int_distribution<int> pt(1, n - 1);
    int c1 = pt(rng), c2 = pt(rng);
    if (c1 > c2) std::swap(c1, c2);
    for (int j = 0; j < n; j++)
        child.genes[j] = (j >= c1 && j < c2) ? p2.genes[j] : p1.genes[j];
    return child;
}

// Uniform crossover: each gene independently drawn from p1 or p2 (coin flip).
// Higher allelic mixing than two-point; can disrupt building blocks more but
// explores combinations two-point structurally cannot reach in one step.
inline Individual crossover_uniform(const Individual& p1, const Individual& p2,
                                    std::mt19937& rng) {
    Individual child;
    const int n = static_cast<int>(p1.genes.size());
    child.genes.resize(n);
    std::uniform_int_distribution<int> coin(0, 1);
    for (int j = 0; j < n; j++)
        child.genes[j] = coin(rng) ? p1.genes[j] : p2.genes[j];
    return child;
}

// CrossoverKind: 0 = two-point (default, original behaviour), 1 = uniform.
inline Individual crossover(const Individual& p1, const Individual& p2,
                             std::mt19937& rng, int kind = 0) {
    return (kind == 1) ? crossover_uniform(p1, p2, rng)
                        : crossover_two_point(p1, p2, rng);
}

inline Individual make_greedy(const Problem& prob, bool minimize_cost) {
    Individual ind;
    ind.genes.resize(prob.n_jobs);
    for (int j = 0; j < prob.n_jobs; j++) {
        int best = 0;
        double best_val = minimize_cost ? prob.job_slots[j][0].cost
                                        : prob.job_slots[j][0].f1_contrib;
        for (int k = 1; k < (int)prob.job_slots[j].size(); k++) {
            const double val = minimize_cost ? prob.job_slots[j][k].cost
                                             : prob.job_slots[j][k].f1_contrib;
            if (val < best_val) { best_val = val; best = k; }
        }
        ind.genes[j] = best;
    }
    return ind;
}

// Earliest possible start on any type — no waiting past arrival.
inline Individual make_no_wait(const Problem& prob) {
    Individual ind;
    ind.genes.resize(prob.n_jobs);
    for (int j = 0; j < prob.n_jobs; j++) {
        int best = 0, best_start = prob.job_slots[j][0].start;
        for (int k = 1; k < (int)prob.job_slots[j].size(); k++)
            if (prob.job_slots[j][k].start < best_start) {
                best_start = prob.job_slots[j][k].start; best = k;
            }
        ind.genes[j] = best;
    }
    return ind;
}

// Earliest start on finite-capacity (on-prem) types; fall back to any type.
inline Individual make_no_burst(const Problem& prob) {
    Individual ind;
    ind.genes.resize(prob.n_jobs);
    for (int j = 0; j < prob.n_jobs; j++) {
        int best = -1, best_start = std::numeric_limits<int>::max();
        for (int k = 0; k < (int)prob.job_slots[j].size(); k++) {
            const SlotInfo& s = prob.job_slots[j][k];
            const int cap = (s.type_id < (int)prob.type_cap.size())
                            ? prob.type_cap[s.type_id] : -1;
            if (cap >= 0 && s.start < best_start) {
                best_start = s.start; best = k;
            }
        }
        if (best < 0) {
            best = 0; best_start = prob.job_slots[j][0].start;
            for (int k = 1; k < (int)prob.job_slots[j].size(); k++)
                if (prob.job_slots[j][k].start < best_start) {
                    best_start = prob.job_slots[j][k].start; best = k;
                }
        }
        ind.genes[j] = best;
    }
    return ind;
}

// Earliest start on unlimited-capacity (cloud) types; fall back to any type.
inline Individual make_full_burst(const Problem& prob) {
    Individual ind;
    ind.genes.resize(prob.n_jobs);
    for (int j = 0; j < prob.n_jobs; j++) {
        int best = -1, best_start = std::numeric_limits<int>::max();
        for (int k = 0; k < (int)prob.job_slots[j].size(); k++) {
            const SlotInfo& s = prob.job_slots[j][k];
            const int cap = (s.type_id < (int)prob.type_cap.size())
                            ? prob.type_cap[s.type_id] : -1;
            if (cap < 0 && s.start < best_start) {
                best_start = s.start; best = k;
            }
        }
        if (best < 0) {
            best = 0; best_start = prob.job_slots[j][0].start;
            for (int k = 1; k < (int)prob.job_slots[j].size(); k++)
                if (prob.job_slots[j][k].start < best_start) {
                    best_start = prob.job_slots[j][k].start; best = k;
                }
        }
        ind.genes[j] = best;
    }
    return ind;
}

// Each job starts at its earliest slot + round(frac * max_slot) time units.
inline Individual make_fixed_wait(const Problem& prob, double frac) {
    Individual ind;
    ind.genes.resize(prob.n_jobs);
    const int delay = static_cast<int>(std::round(frac * prob.max_slot));
    for (int j = 0; j < prob.n_jobs; j++) {
        int min_start = prob.job_slots[j][0].start;
        for (int k = 1; k < (int)prob.job_slots[j].size(); k++)
            min_start = std::min(min_start, prob.job_slots[j][k].start);
        const int target = min_start + delay;
        int best = 0, best_dist = std::numeric_limits<int>::max();
        for (int k = 0; k < (int)prob.job_slots[j].size(); k++) {
            const int d = std::abs(prob.job_slots[j][k].start - target);
            if (d < best_dist) { best_dist = d; best = k; }
        }
        ind.genes[j] = best;
    }
    return ind;
}

// Stagger jobs across the horizon by index: job j targets min_start_j + j*(max_slot/(n_jobs-1)).
inline Individual make_star_wait(const Problem& prob) {
    Individual ind;
    ind.genes.resize(prob.n_jobs);
    for (int j = 0; j < prob.n_jobs; j++) {
        int min_start = prob.job_slots[j][0].start;
        for (int k = 1; k < (int)prob.job_slots[j].size(); k++)
            min_start = std::min(min_start, prob.job_slots[j][k].start);
        const int step = (prob.n_jobs > 1)
            ? static_cast<int>(std::round((double)j * prob.max_slot / (prob.n_jobs - 1)))
            : 0;
        const int target = min_start + step;
        int best = 0, best_dist = std::numeric_limits<int>::max();
        for (int k = 0; k < (int)prob.job_slots[j].size(); k++) {
            const int d = std::abs(prob.job_slots[j][k].start - target);
            if (d < best_dist) { best_dist = d; best = k; }
        }
        ind.genes[j] = best;
    }
    return ind;
}

// All deterministic seed individuals:
// greedy-time, greedy-cost, no-wait, no-burst, full-burst,
// fixed-wait-25%, fixed-wait-50%, star-wait.
inline std::vector<Individual> make_heuristic_seeds(const Problem& prob) {
    return {
        make_greedy(prob, false),     // 0: min turnaround (f1)
        make_greedy(prob, true),      // 1: min cost (f2)
        make_no_wait(prob),           // 2: ASAP any type
        make_no_burst(prob),          // 3: ASAP on-prem, fallback cloud
        make_full_burst(prob),        // 4: ASAP cloud, fallback on-prem
        make_fixed_wait(prob, 0.25),  // 5: 25% horizon delay
        make_fixed_wait(prob, 0.50),  // 6: 50% horizon delay
        make_star_wait(prob),         // 7: staggered by job index
    };
}

inline void mutate(Individual& ind, const Problem& prob, double p_mut,
                   std::mt19937& rng) {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (int j = 0; j < prob.n_jobs; j++) {
        if (u(rng) < p_mut) {
            const int n = static_cast<int>(prob.job_slots[j].size());
            ind.genes[j] = std::uniform_int_distribution<int>(0, n - 1)(rng);
        }
    }
}

// ── Mutation-rate annealing ───────────────────────────────────────────────────
//
// p_mut_start < 0  ⇒ resolve to the caller's formula default (e.g. 2/n_jobs).
// p_mut_end   < 0  ⇒ hold constant at the resolved start rate (no annealing).
// Otherwise the rate is linearly interpolated start→end across generations.
// Passing both as -1 (the default in every binding) reproduces the original
// fixed-rate behaviour exactly, generation for generation.
struct MutationSchedule {
    double lo, hi;
    double at(int gen, int n_gen) const noexcept {
        if (n_gen <= 1) return lo;
        const double frac = static_cast<double>(gen) / (n_gen - 1);
        return lo + frac * (hi - lo);
    }
};

inline MutationSchedule make_mutation_schedule(double p_mut_start, double p_mut_end,
                                               double default_rate) noexcept {
    const double lo = (p_mut_start >= 0.0) ? p_mut_start : default_rate;
    const double hi = (p_mut_end   >= 0.0) ? p_mut_end   : lo;
    return {lo, hi};
}

// ── Pareto utilities ──────────────────────────────────────────────────────────

using ResultList = std::vector<std::tuple<std::vector<std::tuple<int,int>>, double, double>>;

inline ResultList pareto_filter(ResultList results) {
    const double eps = 1e-9;
    const int N = static_cast<int>(results.size());
    std::vector<bool> keep(N, true);

    for (int i = 0; i < N; i++) {
        if (!keep[i]) continue;
        const double f1i = std::get<1>(results[i]);
        const double f2i = std::get<2>(results[i]);
        for (int j = i + 1; j < N; j++) {
            if (!keep[j]) continue;
            const double f1j = std::get<1>(results[j]);
            const double f2j = std::get<2>(results[j]);
            if (std::abs(f1i - f1j) < eps && std::abs(f2i - f2j) < eps) {
                keep[j] = false; continue;
            }
            if (f1i <= f1j && f2i <= f2j && (f1i < f1j || f2i < f2j)) {
                keep[j] = false; continue;
            }
            if (f1j <= f1i && f2j <= f2i && (f1j < f1i || f2j < f2i)) {
                keep[i] = false; break;
            }
        }
    }
    ResultList out;
    for (int i = 0; i < N; i++)
        if (keep[i]) out.push_back(std::move(results[i]));
    return out;
}

inline std::vector<std::tuple<int,int>> extract_assignment(const Individual& ind,
                                                             const Problem& prob) {
    std::vector<std::tuple<int,int>> asgn;
    asgn.reserve(prob.n_jobs);
    for (int j = 0; j < prob.n_jobs; j++) {
        const SlotInfo& s = prob.job_slots[j][ind.genes[j]];
        asgn.emplace_back(s.type_id, s.start);
    }
    return asgn;
}

// ── Problem construction ──────────────────────────────────────────────────────

inline Problem build_problem(
    int    n_jobs,
    double budget,
    const std::vector<std::vector<std::tuple<int,int,int,double,double>>>& raw_slots,
    const std::vector<int>& type_cap,
    const std::vector<double>& type_risk,
    const std::vector<std::tuple<int,int,int>>& raw_init_occ
) {
    Problem prob;
    prob.n_jobs    = n_jobs;
    prob.budget    = budget;
    prob.type_cap  = type_cap;
    prob.type_risk = type_risk;
    prob.n_types   = static_cast<int>(type_cap.size());

    prob.job_slots.resize(n_jobs);
    prob.slot_cum_weight.resize(n_jobs);
    int max_slot = 0;
    for (int j = 0; j < n_jobs; j++) {
        prob.job_slots[j].reserve(raw_slots[j].size());
        for (auto& [tid, start, pocc, f1c, cost] : raw_slots[j]) {
            prob.job_slots[j].push_back({tid, start, pocc, f1c, cost});
            max_slot = std::max(max_slot, start + pocc);
        }
        compute_slot_weights(prob.job_slots[j], prob.type_risk, prob.slot_cum_weight[j]);
    }
    prob.max_slot = max_slot;

    const int stride = max_slot + 1;
    prob.init_occ_flat.assign(static_cast<size_t>(prob.n_types) * stride, 0);
    for (auto& [m, t, cnt] : raw_init_occ) {
        if (m < prob.n_types && prob.type_cap[m] >= 0 && t < stride)
            prob.init_occ_flat[m * stride + t] = cnt;
    }

    return prob;
}
