/**
 * NSGA-II and MOEA/D for FED-HPC multi-objective scheduling — optimised build.
 *
 * Performance changes vs the original:
 *
 *  evaluate():
 *    Replaced std::map<pair<int,int>,int> with a flat int[] occupancy grid
 *    (type_id × slot) plus a per-thread dirty list so only touched cells are
 *    zeroed between calls.  Eliminates all heap allocation inside the hot loop.
 *
 *  fast_nds():
 *    The O(N²) dominance-computation phase is now a parallel-for over rows p.
 *    Each row writes only to dominated_by[p] / dom_count[p] — no data races.
 *    The front-propagation phase remains sequential (dependency chain).
 *
 *  Offspring generation (NSGA-II & MOEA/D):
 *    All children per generation are created and evaluated in parallel.
 *    Each OpenMP thread keeps its own seeded mt19937 + EvalWorkspace so there
 *    is no shared mutable state in the hot path.
 *    Seeds are drawn from the master RNG *before* the parallel region, giving
 *    deterministic results for any fixed (seed, n_threads) pair.
 *
 *  MOEA/D parallelism:
 *    All n_weights children per generation are generated in parallel
 *    (parallel MOEA/D variant: offline update).  Ideal-point reduction and
 *    neighbourhood replacement run sequentially after.
 *
 *  GIL:
 *    Both run_nsga2 and run_moead release the Python GIL for their entire
 *    duration so other Python threads are never blocked.
 *
 *  n_threads:
 *    Both entry points accept an explicit thread count (0 = let OpenMP pick).
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef _OPENMP
#  include <omp.h>
#endif

#include <algorithm>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <tuple>
#include <vector>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Inline helpers
// ---------------------------------------------------------------------------

static inline void set_num_threads(int n) {
#ifdef _OPENMP
    if (n > 0) omp_set_num_threads(n);
#else
    (void)n;
#endif
}

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

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
    std::vector<int> type_cap;          // type_cap[m]: capacity, -1 = unlimited
    std::vector<int> init_occ_flat;     // flat [type_id * (max_slot+1) + t] occupancy
                                        //   from running jobs; 0 for unlimited types
};

// ---------------------------------------------------------------------------
// Per-thread occupancy workspace — no heap allocation inside evaluate()
// ---------------------------------------------------------------------------

struct EvalWorkspace {
    std::vector<int> occ;    // same layout as Problem::init_occ_flat
    std::vector<int> dirty;  // flat indices of cells with occ != 0

    void reset(int n_types, int max_slot) {
        occ.assign(static_cast<size_t>(n_types) * (max_slot + 1), 0);
        dirty.clear();
        dirty.reserve(4096);
    }
};

// ---------------------------------------------------------------------------
// Individual
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// evaluate() — flat-grid occupancy, zero heap allocation in hot path
// ---------------------------------------------------------------------------

void evaluate(Individual& ind, const Problem& prob, EvalWorkspace& ws) {
    // Zero only cells dirtied by the previous call on this workspace.
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
        if (cap < 0) continue;  // unlimited — skip capacity tracking

        const int base = s.type_id * stride;
        for (int t = s.start; t < s.start + s.p_occ; t++) {
            const int idx = base + t;
            if (ws.occ[idx] == 0) ws.dirty.push_back(idx);
            ws.occ[idx]++;
        }
    }

    // Accumulate violations from dirty cells.
    for (int idx : ws.dirty) {
        const int cap   = prob.type_cap[idx / stride];
        const int total = ws.occ[idx] + prob.init_occ_flat[idx];
        const int exc   = total - cap;
        if (exc > 0) ind.cv += exc;
    }
    if (ind.f2 > prob.budget) ind.cv += ind.f2 - prob.budget;
}

// ---------------------------------------------------------------------------
// Genetic operators
// ---------------------------------------------------------------------------

Individual make_random(const Problem& prob, std::mt19937& rng) {
    Individual ind;
    ind.genes.resize(prob.n_jobs);
    for (int j = 0; j < prob.n_jobs; j++) {
        const int n = static_cast<int>(prob.job_slots[j].size());
        ind.genes[j] = std::uniform_int_distribution<int>(0, n - 1)(rng);
    }
    return ind;
}

Individual crossover(const Individual& p1, const Individual& p2, std::mt19937& rng) {
    Individual child;
    const int n = static_cast<int>(p1.genes.size());
    child.genes.resize(n);
    if (n <= 2) {
        std::uniform_int_distribution<int> coin(0, 1);
        for (int j = 0; j < n; j++)
            child.genes[j] = coin(rng) ? p1.genes[j] : p2.genes[j];
        return child;
    }
    // 2-point crossover: takes a contiguous segment from p2, rest from p1.
    // Produces structurally different offspring vs uniform when parents are similar.
    std::uniform_int_distribution<int> pt(1, n - 1);
    int c1 = pt(rng), c2 = pt(rng);
    if (c1 > c2) std::swap(c1, c2);
    for (int j = 0; j < n; j++)
        child.genes[j] = (j >= c1 && j < c2) ? p2.genes[j] : p1.genes[j];
    return child;
}

Individual make_greedy(const Problem& prob, bool minimize_cost) {
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

void mutate(Individual& ind, const Problem& prob, double p_mut, std::mt19937& rng) {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (int j = 0; j < prob.n_jobs; j++) {
        if (u(rng) < p_mut) {
            const int n = static_cast<int>(prob.job_slots[j].size());
            ind.genes[j] = std::uniform_int_distribution<int>(0, n - 1)(rng);
        }
    }
}

// ---------------------------------------------------------------------------
// fast_nds — O(N²) with parallelised dominance-computation phase
// ---------------------------------------------------------------------------

std::vector<std::vector<int>> fast_nds(const std::vector<Individual>& pop) {
    const int N = static_cast<int>(pop.size());
    std::vector<int>              dom_count(N, 0);
    std::vector<std::vector<int>> dominated_by(N);

    // Each row p writes only to dominated_by[p] and dom_count[p] — no races.
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

    // Front propagation — sequential (each front depends on the previous).
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

void assign_ranks(std::vector<Individual>& pop,
                  const std::vector<std::vector<int>>& fronts) {
    for (int r = 0; r < (int)fronts.size(); r++)
        for (int i : fronts[r]) pop[i].rank = r;
}

void crowding_distance(std::vector<Individual>& pop, const std::vector<int>& front) {
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

const Individual& tournament(const Individual& a, const Individual& b) noexcept {
    if (a.rank < b.rank) return a;
    if (b.rank < a.rank) return b;
    return (a.crowding >= b.crowding) ? a : b;
}

// ---------------------------------------------------------------------------
// Shared: Pareto filter
// ---------------------------------------------------------------------------

using ResultList = std::vector<std::tuple<std::vector<std::tuple<int,int>>, double, double>>;

ResultList pareto_filter(ResultList results) {
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

std::vector<std::tuple<int,int>> extract_assignment(const Individual& ind,
                                                      const Problem& prob) {
    std::vector<std::tuple<int,int>> asgn;
    asgn.reserve(prob.n_jobs);
    for (int j = 0; j < prob.n_jobs; j++) {
        const SlotInfo& s = prob.job_slots[j][ind.genes[j]];
        asgn.emplace_back(s.type_id, s.start);
    }
    return asgn;
}

// ---------------------------------------------------------------------------
// NSGA-II
// ---------------------------------------------------------------------------

ResultList run_nsga2(const Problem& prob, int pop_size, int n_gen, int seed,
                     int n_threads) {
    py::gil_scoped_release release;
    set_num_threads(n_threads);

    std::mt19937 rng(seed);
    const double p_mut = 2.0 / prob.n_jobs;

    // ── Initial population ────────────────────────────────────────────────

    // Pre-draw seeds for each individual so the parallel region is deterministic.
    std::vector<uint32_t> init_seeds(pop_size);
    for (auto& s : init_seeds) s = rng();

    std::vector<Individual> pop(pop_size);
    // Anchor both ends of the Pareto front with extremal greedy seeds.
    if (pop_size >= 1) pop[0] = make_greedy(prob, false); // minimize time (f1)
    if (pop_size >= 2) pop[1] = make_greedy(prob, true);  // minimize cost (f2)

#ifdef _OPENMP
    #pragma omp parallel
    {
        EvalWorkspace ws;
        ws.reset(prob.n_types, prob.max_slot);
        #pragma omp for schedule(static)
        for (int k = 0; k < pop_size; k++) {
            if (k >= 2) {
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
            if (k >= 2) {
                std::mt19937 lrng(init_seeds[k]);
                pop[k] = make_random(prob, lrng);
            }
            evaluate(pop[k], prob, ws);
        }
    }
#endif

    // Pre-allocate buffers reused across generations.
    std::vector<Individual> offspring(pop_size);
    std::vector<Individual> combined;
    combined.reserve(2 * pop_size);
    std::vector<uint32_t> child_seeds(pop_size);

    for (int gen = 0; gen < n_gen; gen++) {
        // ── Non-dominated sort + crowding ────────────────────────────────

        auto fronts = fast_nds(pop);
        assign_ranks(pop, fronts);
        for (auto& f : fronts) crowding_distance(pop, f);

        // ── Offspring generation — parallel ──────────────────────────────

        for (auto& s : child_seeds) s = rng();

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

        // ── Combine + select ─────────────────────────────────────────────

        combined.clear();
        for (auto& x : pop)      combined.push_back(std::move(x));
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
    }

    // ── Extract Pareto front ─────────────────────────────────────────────

    auto fronts = fast_nds(pop);
    assign_ranks(pop, fronts);
    ResultList results;
    for (auto& ind : pop)
        if (ind.rank == 0 && ind.cv == 0.0)
            results.emplace_back(extract_assignment(ind, prob), ind.f1, ind.f2);
    return pareto_filter(std::move(results));
}

// ---------------------------------------------------------------------------
// MOEA/D — parallel offspring generation (offline-update variant)
// ---------------------------------------------------------------------------

ResultList run_moead(const Problem& prob, int n_weights, int n_gen,
                     int T_size, int seed, int n_threads) {
    py::gil_scoped_release release;
    set_num_threads(n_threads);

    std::mt19937 rng(seed);
    const double p_mut = 1.0 / prob.n_jobs;

    // Weight vectors: uniformly spaced on the 2-objective simplex.
    // W_EPS floor prevents degenerate single-objective subproblems at the extremes.
    constexpr double W_EPS = 1e-4;
    std::vector<std::pair<double,double>> W(n_weights);
    for (int i = 0; i < n_weights; i++) {
        const double lam = (n_weights > 1) ? (double)i / (n_weights - 1) : 0.5;
        W[i] = {lam + W_EPS, 1.0 - lam + W_EPS};
    }

    // Precompute T-nearest neighbourhoods.
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

    // ── Initial population ────────────────────────────────────────────────

    std::vector<uint32_t> init_seeds(n_weights);
    for (auto& s : init_seeds) s = rng();

    std::vector<Individual> pop(n_weights);
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
            std::mt19937 lrng(init_seeds[i]);
            pop[i] = make_random(prob, lrng);
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
            std::mt19937 lrng(init_seeds[i]);
            pop[i] = make_random(prob, lrng);
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

    // Normalized Tchebycheff scalarization: each objective is scaled by its observed
    // range [ideal, nadir] so that weight vectors evenly spread across the tradeoff
    // regardless of the absolute scale difference between f1 and f2.
    auto scalar = [&](const Individual& ind, int i) noexcept -> double {
        if (ind.cv > 0.0) return 1e18 + ind.cv * 1e6;
        const double lam1 = W[i].first, lam2 = W[i].second;
        const double r1 = std::max(n1 - z1, 1.0);
        const double r2 = std::max(n2 - z2, 1.0);
        return std::max(lam1 * (ind.f1 - z1) / r1,
                        lam2 * (ind.f2 - z2) / r2);
    };

    // Pre-allocate children buffer.
    std::vector<Individual> children(n_weights);
    std::vector<uint32_t>   child_seeds(n_weights);

    for (int gen = 0; gen < n_gen; gen++) {
        // ── Phase 1: parallel offspring generation ───────────────────────

        for (auto& s : child_seeds) s = rng();

        double dz1 = z1, dz2 = z2;
        double dn1 = n1, dn2 = n2;

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
                children[i] = crossover(pop[a], pop[b], lrng);
                mutate(children[i], prob, p_mut, lrng);
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
                children[i] = crossover(pop[a], pop[b], lrng);
                mutate(children[i], prob, p_mut, lrng);
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

        // ── Phase 2: sequential neighbourhood update ─────────────────────

        for (int i = 0; i < n_weights; i++) {
            for (int j : neigh[i]) {
                if (scalar(children[i], j) <= scalar(pop[j], j))
                    pop[j] = children[i];
            }
        }
    }

    ResultList results;
    for (auto& ind : pop)
        if (ind.cv == 0.0)
            results.emplace_back(extract_assignment(ind, prob), ind.f1, ind.f2);
    return pareto_filter(std::move(results));
}

// ---------------------------------------------------------------------------
// Build Problem from Python data — precompute flat init_occ
// ---------------------------------------------------------------------------

Problem build_problem(
    int    n_jobs,
    double budget,
    const std::vector<std::vector<std::tuple<int,int,int,double,double>>>& raw_slots,
    const std::vector<int>& type_cap,
    const std::vector<std::tuple<int,int,int>>& raw_init_occ
) {
    Problem prob;
    prob.n_jobs   = n_jobs;
    prob.budget   = budget;
    prob.type_cap = type_cap;
    prob.n_types  = static_cast<int>(type_cap.size());

    prob.job_slots.resize(n_jobs);
    int max_slot = 0;
    for (int j = 0; j < n_jobs; j++) {
        prob.job_slots[j].reserve(raw_slots[j].size());
        for (auto& [tid, start, pocc, f1c, cost] : raw_slots[j]) {
            prob.job_slots[j].push_back({tid, start, pocc, f1c, cost});
            max_slot = std::max(max_slot, start + pocc);
        }
    }
    prob.max_slot = max_slot;

    // Flat init_occ: only for types with finite capacity (type_cap[m] >= 0).
    const int stride = max_slot + 1;
    prob.init_occ_flat.assign(static_cast<size_t>(prob.n_types) * stride, 0);
    for (auto& [m, t, cnt] : raw_init_occ) {
        if (m < prob.n_types && prob.type_cap[m] >= 0 && t < stride)
            prob.init_occ_flat[m * stride + t] = cnt;
    }

    return prob;
}

// ---------------------------------------------------------------------------
// Pybind11 module
// ---------------------------------------------------------------------------

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
            return run_nsga2(
                build_problem(n_jobs, budget, job_slots, type_cap, init_occ),
                pop_size, n_gen, seed, n_threads);
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
        "NSGA-II Pareto frontier approximation (parallel).\n"
        "n_threads=0 lets OpenMP choose (all available cores)."
    );

    m.def("moead",
        [](int n_jobs, double budget,
           const std::vector<std::vector<std::tuple<int,int,int,double,double>>>& job_slots,
           const std::vector<int>& type_cap,
           const std::vector<std::tuple<int,int,int>>& init_occ,
           int n_weights, int n_gen, int neighborhood_size,
           int seed, int n_threads) {
            return run_moead(
                build_problem(n_jobs, budget, job_slots, type_cap, init_occ),
                n_weights, n_gen, neighborhood_size, seed, n_threads);
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
        "MOEA/D Pareto frontier approximation (parallel, Tchebycheff decomposition).\n"
        "n_threads=0 lets OpenMP choose (all available cores)."
    );
}
