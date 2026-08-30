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
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <unordered_map>
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

// ── Ablation flags ───────────────────────────────────────────────────────────
//
// Bitmask passed to run_nsga2/3, run_moead and run_weighted for the ablation
// study (scripts/ablation_real.py). Default 0 = the full algorithm; each bit
// removes one component so its contribution can be measured in isolation.
// Not part of any tuned default — purely diagnostic, like `profile`.
enum AblateFlag {
    ABL_NO_HEURISTIC_SEEDS = 1 << 0,  // random-init only (skip make_heuristic_seeds)
    ABL_UNIFORM_RANDOM     = 1 << 1,  // make_random uniform, not cost/risk-weighted
    ABL_NO_LOCAL_SEARCH    = 1 << 2,  // skip local_search() everywhere
    ABL_NO_SCHED_REPAIR    = 1 << 3,  // skip schedule_repair() (overrides sched_repair)
    ABL_NO_CANDIDATE_GEOM  = 1 << 4,  // job_candidates: earliest-K only, no geometric reach
    ABL_NO_CROSSOVER       = 1 << 5,  // offspring from one parent (mutation only)
    ABL_NO_ILS_KICK        = 1 << 6,  // weighted: no perturbation restart on stagnation
    ABL_NO_ELITISM         = 1 << 7,  // weighted: do not carry the incumbent forward
    ABL_NO_EST_SHORTLIST   = 1 << 8,  // weighted_local_search: unranked candidate scan
    ABL_NO_FREE_POOL       = 1 << 9,  // weighted: schedule_repair without free_pool_balance
};

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

    std::vector<std::vector<int>> job_candidates;
    // [j] = shortlist of indices into job_slots[j] for local_search(): the
    // best few (by f1_contrib, then cost) slots *per feasible type*. Per-job
    // slot counts run into the thousands (one per feasible start time), but
    // the (f1_contrib, cost) Pareto frontier collapses to a single point per
    // job in practice — every job's unconstrained-best slot is unambiguous,
    // which is exactly why capacity congestion happens (many jobs want the
    // same one slot). A shortlist ranked by (f1_contrib, cost) alone would
    // therefore only ever surface *one* type (whichever is cheapest) at many
    // nearby times, never a fallback type — so it's built per-type instead,
    // capped at PER_TYPE_K slots each, to keep every feasible type reachable
    // for congestion-relief moves while staying cheap enough to scan.

    std::vector<std::vector<std::array<int, 3>>> job_type_span;
    // [j] = list of {type_id, begin, end} — one entry per contiguous block of
    // job_slots[j] that shares a type_id. job_slots[j] is built type-by-type
    // (see moea._job_slots), each block ascending in start time, so a block is
    // exactly the feasible (start-sorted) slot range for one (job, type) pair.
    // Used by schedule_repair() to walk a job's slots on a single type in
    // earliest-start order without rescanning / re-grouping.
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
inline Individual make_random(const Problem& prob, std::mt19937& rng,
                              bool uniform = false) {
    Individual ind;
    ind.genes.resize(prob.n_jobs);
    for (int j = 0; j < prob.n_jobs; j++) {
        const int n = static_cast<int>(prob.job_slots[j].size());
        if (uniform) {
            ind.genes[j] = std::uniform_int_distribution<int>(0, n - 1)(rng);
            continue;
        }
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

// Greedy single-objective seed, tie-broken on the *other* objective.
// Picking the primary-minimal slot alone leaves ties (very common — many
// slots often share the same earliest start or same zero cost) resolved by
// arbitrary job_slots iteration order, which silently anchors the seed (and
// the population it draws from) on whichever type happens to come first
// rather than the true co-optimal slot. Comparing the secondary objective
// within an epsilon band closes that gap at zero extra asymptotic cost.
inline Individual make_greedy(const Problem& prob, bool minimize_cost) {
    Individual ind;
    ind.genes.resize(prob.n_jobs);
    constexpr double eps = 1e-9;
    for (int j = 0; j < prob.n_jobs; j++) {
        int best = 0;
        const auto primary_of   = [&](int k) { return minimize_cost ? prob.job_slots[j][k].cost
                                                                     : prob.job_slots[j][k].f1_contrib; };
        const auto secondary_of = [&](int k) { return minimize_cost ? prob.job_slots[j][k].f1_contrib
                                                                     : prob.job_slots[j][k].cost; };
        double best_primary = primary_of(0), best_secondary = secondary_of(0);
        for (int k = 1; k < (int)prob.job_slots[j].size(); k++) {
            const double primary = primary_of(k), secondary = secondary_of(k);
            const bool strictly_better = primary < best_primary - eps;
            const bool tied_but_cheaper_secondary =
                primary < best_primary + eps && secondary < best_secondary;
            if (strictly_better || tied_but_cheaper_secondary) {
                best_primary = primary; best_secondary = secondary; best = k;
            }
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

// Capacity-aware list-scheduling seed: minimises total completion time
// *under* each type's capacity constraint, instead of ignoring it like the
// other greedy seeds.
//
// Rationale: reaching the cost-minimal extreme forces (almost) every job
// onto the same one or two free/cheapest types, whose capacity is far
// smaller than the job count — the true optimum there requires spreading
// jobs across the horizon in a specific load-balanced way. That's a
// textbook parallel-machine, capacity-constrained, minimise-total-
// completion-time scheduling problem: process jobs in order of their own
// earliest feasible start on their preferred type, and greedily place each
// at the earliest still-has-room window (first-fit list scheduling). This
// is the same shape of result as SPT/list-scheduling optimality for that
// problem — local_search()'s bounded candidate shortlist can polish
// starting from here, but can't discover this global a rearrangement from
// scratch (see its own comment for why).
inline Individual make_list_schedule(const Problem& prob) {
    Individual ind;
    ind.genes.resize(prob.n_jobs);

    // Each job's preferred type: same lexicographic (cost, then f1_contrib)
    // minimisation make_greedy(prob, /*minimize_cost=*/true) uses, so the
    // "preferred type" here matches what that seed would pick per job.
    std::vector<int> pref_type(prob.n_jobs);
    for (int j = 0; j < prob.n_jobs; j++) {
        const auto& slots = prob.job_slots[j];
        int best = 0;
        for (int k = 1; k < (int)slots.size(); k++) {
            const bool cheaper = slots[k].cost < slots[best].cost - 1e-9;
            const bool tied_faster = slots[k].cost < slots[best].cost + 1e-9
                                    && slots[k].f1_contrib < slots[best].f1_contrib;
            if (cheaper || tied_faster) best = k;
        }
        pref_type[j] = slots[best].type_id;
    }

    std::vector<std::vector<int>> groups(prob.n_types);
    for (int j = 0; j < prob.n_jobs; j++) groups[pref_type[j]].push_back(j);

    const int stride = prob.max_slot + 1;
    std::vector<int> occ = prob.init_occ_flat;  // seed from running-job occupancy

    for (int m = 0; m < prob.n_types; m++) {
        auto& jobs_m = groups[m];
        if (jobs_m.empty()) continue;
        const int cap = prob.type_cap[m];

        if (cap < 0) {
            // Unlimited-capacity type: no congestion possible, just take
            // each job's own preferred (cheapest, then fastest) slot on m.
            for (int j : jobs_m) {
                const auto& slots = prob.job_slots[j];
                int best = -1;
                for (int k = 0; k < (int)slots.size(); k++) {
                    if (slots[k].type_id != m) continue;
                    if (best < 0 || slots[k].cost < slots[best].cost - 1e-9
                        || (slots[k].cost < slots[best].cost + 1e-9
                            && slots[k].f1_contrib < slots[best].f1_contrib))
                        best = k;
                }
                ind.genes[j] = best;
            }
            continue;
        }

        // Finite-capacity type: process jobs first-come-first-served by
        // their own earliest feasible start on this type, greedily placing
        // each at the earliest window with room across its whole [start,
        // start+p_occ) span.
        std::vector<std::pair<int,int>> order;  // (earliest_start_on_m, job_id)
        order.reserve(jobs_m.size());
        for (int j : jobs_m) {
            int earliest = std::numeric_limits<int>::max();
            for (const SlotInfo& s : prob.job_slots[j])
                if (s.type_id == m) earliest = std::min(earliest, s.start);
            order.emplace_back(earliest, j);
        }
        std::sort(order.begin(), order.end());

        for (auto& [_, j] : order) {
            const auto& slots = prob.job_slots[j];
            std::vector<int> cand_idx;  // indices with type_id == m, start ascending
            for (int k = 0; k < (int)slots.size(); k++)
                if (slots[k].type_id == m) cand_idx.push_back(k);
            std::sort(cand_idx.begin(), cand_idx.end(), [&](int a, int b) {
                return slots[a].start < slots[b].start;
            });

            int chosen = -1;
            for (int k : cand_idx) {
                const SlotInfo& s = slots[k];
                bool fits = true;
                const int base = m * stride;
                for (int t = s.start; t < s.start + s.p_occ; t++)
                    if (occ[base + t] >= cap) { fits = false; break; }
                if (fits) { chosen = k; break; }
            }
            if (chosen < 0) {
                // Type m is saturated across its whole horizon for this job
                // (rare) — fall back to the job's cheapest slot on any type;
                // evaluate() will penalise it normally like any other
                // individual if that reintroduces a violation.
                chosen = 0;
                for (int k = 1; k < (int)slots.size(); k++)
                    if (slots[k].cost < slots[chosen].cost - 1e-9
                        || (slots[k].cost < slots[chosen].cost + 1e-9
                            && slots[k].f1_contrib < slots[chosen].f1_contrib))
                        chosen = k;
            } else {
                const SlotInfo& s = slots[chosen];
                const int base = m * stride;
                for (int t = s.start; t < s.start + s.p_occ; t++) occ[base + t]++;
            }
            ind.genes[j] = chosen;
        }
    }
    return ind;
}

// All deterministic seed individuals:
// greedy-time, greedy-cost, no-wait, full-burst,
// fixed-wait-25%, fixed-wait-50%, star-wait, list-schedule.
//
// make_no_burst() (earliest start on finite-capacity types, cloud fallback)
// was dropped from this list: an ablation across all 10 bundled instances
// (see moea.time_seeds()) showed it converges to *exactly* the same
// post-repair (f1, f2) as make_no_wait() on every single one — the two are
// provably redundant once local_search() repairs them, so keeping both only
// added construction cost with no diversity benefit. The other seeds don't
// have this property: greedy_time/greedy_cost/full_burst/fixed_wait_25/
// fixed_wait_50 each converge to points identical to their neighbours on
// roughly half the tested instances and to genuinely distinct points on the
// other half (particularly on the smaller, less capacity-congested
// instances) — removing any of *those* would lose real basin coverage on a
// majority of cases, even though they look redundant on the two largest,
// most heavily congested instances alone.
inline std::vector<Individual> make_heuristic_seeds(const Problem& prob) {
    return {
        make_greedy(prob, false),     // 0: min turnaround (f1)
        make_greedy(prob, true),      // 1: min cost (f2)
        make_no_wait(prob),           // 2: ASAP any type
        make_full_burst(prob),        // 3: ASAP cloud, fallback on-prem
        make_fixed_wait(prob, 0.25),  // 4: 25% horizon delay
        make_fixed_wait(prob, 0.50),  // 5: 50% horizon delay
        make_star_wait(prob),         // 6: staggered by job index
        make_list_schedule(prob),     // 7: capacity-aware list scheduling
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

// ── Local search (congestion repair + iso-objective cost/time descent) ────────
//
// Coordinate descent over job assignments, restricted to each job's
// per-type candidate shortlist (Problem::job_candidates — see its comment
// for why the full slot list isn't the right search space here). For each
// job, in turn:
//   - if the current slot contributes to a capacity violation, any
//     candidate that strictly reduces this job's marginal violation is
//     preferred, regardless of cost/f1_contrib (repair always wins);
//   - among candidates tied on marginal violation, only a *weakly
//     dominating* one (same-or-better f1_contrib and cost, strictly better
//     in at least one) is accepted (safe polish — never regresses either
//     objective without an offsetting violation reduction).
//
// This targets exactly the failure mode plain mutation/crossover can't:
// resolving many jobs contending for the same over-subscribed slot needs a
// *directed* move to a specific alternative, and escaping a flat iso-f1
// plateau toward the cheapest co-optimal combination needs a move that
// dominance-based selection alone has no gradient to find.
//
// Mutates ind.genes in place; does NOT refresh ind.f1/f2/cv — call
// evaluate() again afterwards. Safe to call on an already-feasible,
// already-locally-optimal individual (returns false, does no useful work
// beyond one O(n_jobs) scan).
inline bool local_search(Individual& ind, const Problem& prob, EvalWorkspace& ws,
                          int max_passes = 3) {
    const int stride = prob.max_slot + 1;
    constexpr double eps = 1e-9;
    bool any_improved = false;

    for (int pass = 0; pass < max_passes; pass++) {
        // Rebuild occupancy from the individual's current genes.
        for (int idx : ws.dirty) ws.occ[idx] = 0;
        ws.dirty.clear();
        for (int j = 0; j < prob.n_jobs; j++) {
            const SlotInfo& s = prob.job_slots[j][ind.genes[j]];
            const int cap = prob.type_cap[s.type_id];
            if (cap < 0) continue;
            const int base = s.type_id * stride;
            for (int t = s.start; t < s.start + s.p_occ; t++) {
                const int idx = base + t;
                if (ws.occ[idx] == 0) ws.dirty.push_back(idx);
                ws.occ[idx]++;
            }
        }

        const auto marginal_violation = [&](const SlotInfo& s, int cap) -> int {
            if (cap < 0) return 0;
            const int base = s.type_id * stride;
            int v = 0;
            for (int t = s.start; t < s.start + s.p_occ; t++) {
                const int idx = base + t;
                const int exc = ws.occ[idx] + prob.init_occ_flat[idx] + 1 - cap;
                if (exc > 0) v += exc;
            }
            return v;
        };

        bool pass_improved = false;

        for (int j = 0; j < prob.n_jobs; j++) {
            const int cur_k = ind.genes[j];
            const SlotInfo& cur = prob.job_slots[j][cur_k];
            const int cur_cap = prob.type_cap[cur.type_id];

            // Remove this job's own contribution so violation below is
            // computed relative to "everyone else".
            if (cur_cap >= 0) {
                const int base = cur.type_id * stride;
                for (int t = cur.start; t < cur.start + cur.p_occ; t++)
                    ws.occ[base + t]--;
            }

            int best_k = cur_k;
            int best_viol = marginal_violation(cur, cur_cap);
            double best_f1c = cur.f1_contrib, best_cost = cur.cost;

            for (int k : prob.job_candidates[j]) {
                if (k == cur_k) continue;
                const SlotInfo& cand = prob.job_slots[j][k];
                const int cand_cap = prob.type_cap[cand.type_id];
                const int v = marginal_violation(cand, cand_cap);
                if (v < best_viol) {
                    best_viol = v; best_k = k;
                    best_f1c = cand.f1_contrib; best_cost = cand.cost;
                } else if (v == best_viol) {
                    const bool weakly_better =
                        cand.f1_contrib <= best_f1c + eps && cand.cost <= best_cost + eps;
                    const bool strictly_better =
                        cand.f1_contrib < best_f1c - eps || cand.cost < best_cost - eps;
                    if (weakly_better && strictly_better) {
                        best_k = k; best_f1c = cand.f1_contrib; best_cost = cand.cost;
                    }
                }
            }

            if (best_k != cur_k) { ind.genes[j] = best_k; pass_improved = true; }

            const SlotInfo& applied = prob.job_slots[j][ind.genes[j]];
            const int applied_cap = prob.type_cap[applied.type_id];
            if (applied_cap >= 0) {
                const int base = applied.type_id * stride;
                for (int t = applied.start; t < applied.start + applied.p_occ; t++) {
                    const int idx = base + t;
                    if (ws.occ[idx] == 0) ws.dirty.push_back(idx);
                    ws.occ[idx]++;
                }
            }
        }

        any_improved = any_improved || pass_improved;
        if (!pass_improved) break;
    }
    return any_improved;
}

// ── Earliest-feasible list-scheduling repair ─────────────────────────────────
//
// Problem-specific decoder. Cost (f2) depends *only* on which type each job
// runs on — it is completely independent of the start time. So for any fixed
// type assignment, f1 (total turnaround) is minimised by scheduling every job
// as early as capacity allows: a classic capacity-constrained list-scheduling
// problem the generic operators have no gradient toward.
//
// schedule_repair() rewrites each gene's *start* (and, with free_pool_balance,
// possibly its type — but only to a weakly-dominating one): jobs are processed
// shortest-processing-time-first (SPT — the classic rule for minimising total
// completion time on parallel machines) and each is moved to the earliest
// capacity-feasible slot. Because job_slots[j] is start-sorted within a type
// (job_type_span), the first feasible slot found is the earliest one, so f1 is
// non-increasing per job and capacity violations only ever shrink. With the
// default single-type scan, cost also never changes.
//
// This is where local_search() (bounded per-type shortlist, coordinate
// descent) structurally can't reach: relocating the whole population of jobs
// congesting a cheap type's early window out across the horizon in one
// coherent sweep — exactly the move the cost-minimal corner of the front
// needs. Mutates ind.genes in place; does NOT refresh f1/f2/cv (call
// evaluate() afterwards). Returns true if any gene moved.
//
// free_pool_balance != 0: each job's earliest-slot scan spans *every* type
// that weakly dominates its current one — cost ≤ current AND p_occ ≤ current —
// not just the current type. On instances where several types share a price
// (e.g. multiple free on-prem pools) this lets the decoder spread load across
// all of them instead of queueing every job on the one pool its gene names,
// which is otherwise the dominant source of avoidable turnaround at the
// cost-minimal end. Still a weakly-dominating move: cost never rises, p_occ
// never rises, start never rises. Default 0 keeps the single-type scan.
inline bool schedule_repair(Individual& ind, const Problem& prob, EvalWorkspace& ws,
                            int max_passes = 2, int free_pool_balance = 0) {
    const int stride = prob.max_slot + 1;
    bool any_improved = false;

    // Per-job candidate type-spans that weakly dominate the current gene
    // (rebuilt each pass since the current gene can change). Reused scratch.
    std::vector<std::array<int, 3>> dom_spans;

    for (int pass = 0; pass < max_passes; pass++) {
        // Rebuild finite-type occupancy from the individual's current genes.
        for (int idx : ws.dirty) ws.occ[idx] = 0;
        ws.dirty.clear();
        for (int j = 0; j < prob.n_jobs; j++) {
            const SlotInfo& s = prob.job_slots[j][ind.genes[j]];
            if (prob.type_cap[s.type_id] < 0) continue;
            const int base = s.type_id * stride;
            for (int t = s.start; t < s.start + s.p_occ; t++) {
                const int idx = base + t;
                if (ws.occ[idx] == 0) ws.dirty.push_back(idx);
                ws.occ[idx]++;
            }
        }

        // Shortest-processing-time-first (SPT). Placing jobs in ascending
        // p_occ order and giving each the earliest feasible slot is the
        // classic list-scheduling rule that minimises total completion time
        // (hence f1 = total turnaround) on parallel identical machines — an
        // A/B on the 10min instance confirmed SPT beats both current-start
        // order and release-date order at the cost-minimal corner (f1 gap to
        // the proven optimum roughly halved). Ties broken by current start.
        std::vector<int> order(prob.n_jobs);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            const SlotInfo& sa = prob.job_slots[a][ind.genes[a]];
            const SlotInfo& sb = prob.job_slots[b][ind.genes[b]];
            if (sa.p_occ != sb.p_occ) return sa.p_occ < sb.p_occ;
            return sa.start < sb.start;
        });

        bool pass_improved = false;

        for (int j : order) {
            const int cur_k = ind.genes[j];
            const SlotInfo& cur = prob.job_slots[j][cur_k];
            const int tid = cur.type_id;
            const int cap = prob.type_cap[tid];

            // Locate this job's slot block for its current type.
            int begin = -1, end = -1;
            for (const auto& sp : prob.job_type_span[j])
                if (sp[0] == tid) { begin = sp[1]; end = sp[2]; break; }
            if (begin < 0) continue;

            // Candidate type-spans to scan for j's earliest feasible slot.
            dom_spans.clear();
            if (free_pool_balance) {
                for (const auto& sp : prob.job_type_span[j]) {
                    const SlotInfo& first = prob.job_slots[j][sp[1]];
                    if (first.cost <= cur.cost + 1e-9 && first.p_occ <= cur.p_occ)
                        dom_spans.push_back(sp);
                }
            } else {
                dom_spans.push_back({tid, begin, end});
            }

            // Drop j's own contribution (finite type only) so the scan sees
            // "everyone else".
            if (cap >= 0) {
                const int base = tid * stride;
                for (int t = cur.start; t < cur.start + cur.p_occ; t++)
                    ws.occ[base + t]--;
            }

            int chosen = cur_k, chosen_start = cur.start;
            for (const auto& sp : dom_spans) {
                const int m = sp[0], b = sp[1], e = sp[2];
                const int mcap = prob.type_cap[m];
                if (mcap < 0) {
                    // Unlimited: earliest slot is the block start.
                    const int st = prob.job_slots[j][b].start;
                    if (st < chosen_start) { chosen = b; chosen_start = st; }
                    continue;
                }
                const int mbase = m * stride;
                for (int k = b; k < e; k++) {
                    const SlotInfo& s = prob.job_slots[j][k];
                    if (s.start >= chosen_start) break;   // no earlier slot ahead
                    bool fits = true;
                    for (int t = s.start; t < s.start + s.p_occ; t++)
                        if (ws.occ[mbase + t] + prob.init_occ_flat[mbase + t] >= mcap) {
                            fits = false; break;
                        }
                    if (fits) { chosen = k; chosen_start = s.start; break; }
                }
            }

            if (chosen != cur_k) { ind.genes[j] = chosen; pass_improved = true; }

            const SlotInfo& applied = prob.job_slots[j][ind.genes[j]];
            if (prob.type_cap[applied.type_id] >= 0) {
                const int abase = applied.type_id * stride;
                for (int t = applied.start; t < applied.start + applied.p_occ; t++) {
                    const int idx = abase + t;
                    if (ws.occ[idx] == 0) ws.dirty.push_back(idx);
                    ws.occ[idx]++;
                }
            }
        }

        any_improved = any_improved || pass_improved;
        if (!pass_improved) break;
    }
    return any_improved;
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
    const std::vector<std::tuple<int,int,int>>& raw_init_occ,
    int ablate = 0
) {
    Problem prob;
    prob.n_jobs    = n_jobs;
    prob.budget    = budget;
    prob.type_cap  = type_cap;
    prob.type_risk = type_risk;
    prob.n_types   = static_cast<int>(type_cap.size());

    prob.job_slots.resize(n_jobs);
    prob.slot_cum_weight.resize(n_jobs);
    prob.job_candidates.resize(n_jobs);
    prob.job_type_span.resize(n_jobs);
    constexpr int PER_TYPE_K = 4;
    int max_slot = 0;
    for (int j = 0; j < n_jobs; j++) {
        prob.job_slots[j].reserve(raw_slots[j].size());
        for (auto& [tid, start, pocc, f1c, cost] : raw_slots[j]) {
            prob.job_slots[j].push_back({tid, start, pocc, f1c, cost});
            max_slot = std::max(max_slot, start + pocc);
        }
        compute_slot_weights(prob.job_slots[j], prob.type_risk, prob.slot_cum_weight[j]);

        // Contiguous same-type blocks of job_slots[j] (see job_type_span doc).
        {
            const auto& sl = prob.job_slots[j];
            int b = 0;
            const int m = static_cast<int>(sl.size());
            while (b < m) {
                int e = b + 1;
                while (e < m && sl[e].type_id == sl[b].type_id) e++;
                prob.job_type_span[j].push_back({sl[b].type_id, b, e});
                b = e;
            }
        }

        // Build the per-type candidate shortlist for local_search(): group
        // this job's slot indices by type, sort each group by (f1_contrib,
        // cost), keep the first PER_TYPE_K.
        std::unordered_map<int, std::vector<int>> by_type;
        for (int k = 0; k < (int)prob.job_slots[j].size(); k++)
            by_type[prob.job_slots[j][k].type_id].push_back(k);
        auto& cand = prob.job_candidates[j];
        for (auto& [tid, idxs] : by_type) {
            std::sort(idxs.begin(), idxs.end(), [&](int a, int b) {
                const SlotInfo& sa = prob.job_slots[j][a];
                const SlotInfo& sb = prob.job_slots[j][b];
                if (sa.f1_contrib != sb.f1_contrib) return sa.f1_contrib < sb.f1_contrib;
                return sa.cost < sb.cost;
            });
            // f1_contrib is a strictly increasing function of start for a
            // fixed (job, type) pair (p_occ is constant across start times
            // there), so this sort already orders idxs by start time
            // ascending — i.e. earliest-first.
            const int n = (int)idxs.size();
            const int n_keep = std::min(n, PER_TYPE_K);
            cand.insert(cand.end(), idxs.begin(), idxs.begin() + n_keep);

            // The earliest-PER_TYPE_K block alone means local_search() can
            // only ever resolve *local* congestion (a job's own next couple
            // of slots) — it has no visibility into a free slot far later in
            // the horizon, which is exactly what's needed once many jobs are
            // all congesting a type's early slots simultaneously. Add
            // geometrically-spaced positions further out (cheap: O(log n)
            // extra candidates) plus the very last slot as a guaranteed
            // last-resort, so local_search can reach across the whole
            // horizon without scanning it.
            if (!(ablate & ABL_NO_CANDIDATE_GEOM)) {
                for (int pos = PER_TYPE_K * 2; pos < n; pos *= 2)
                    cand.push_back(idxs[pos]);
                if (n > n_keep) cand.push_back(idxs[n - 1]);
            }
        }
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
