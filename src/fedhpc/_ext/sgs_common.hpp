/**
 * FED-HPC — priority-permutation + non-delay Schedule Generation Scheme (SGS).
 *
 * This is the "standard literature" chromosome/decoder alternative to
 * ga_common.hpp's job_slots-index representation, ported from the Python
 * prototype (scripts/priority_sgs_problem.py) to C++ so it can run at a
 * matched (pop_size, n_gen) eval budget with permutation-aware operators.
 *
 * Chromosome: Individual::genes has length 2*n_jobs (reuses the existing
 * Individual struct from ga_common.hpp — f1/f2/cv/rank/crowding are
 * representation-agnostic, only the genes layout and decode differ):
 *   genes[0 .. n_jobs)        — a permutation of job ids: processing order.
 *   genes[n_jobs + j]         — type-choice index into cand[j] (that job's
 *                               feasible types, sorted by cost ascending),
 *                               for job id j (NOT permutation position).
 *
 * Decode (non-delay SGS, matching decode() in priority_sgs_problem.py
 * exactly): walk jobs in permutation order; for each job, try its
 * type-choice-selected candidate first, then wrap through the remaining
 * cost-sorted candidates, taking the first that admits a capacity-feasible
 * start in [a_min, t_max_excl). If none do (every feasible type saturated
 * for this job's whole window), force the type-choice candidate at a_min and
 * charge the capacity excess to cv — the same fallback convention the
 * Python prototype used, so results stay comparable.
 *
 * Deliberately NOT implemented for this representation: local_search(),
 * schedule_repair(), and MOEA/D's scalar_ls_interval polish. Those are all
 * built around job_slots indexing and don't have an SGS analogue here — this
 * keeps the comparison to the job_slots scheme honest (same class of
 * "vanilla GA loop" the seeded pymoo prototype used, just at C++ speed with
 * permutation-aware operators).
 */
#pragma once

#include "ga_common.hpp"

#include <numeric>

// ── Data types ────────────────────────────────────────────────────────────────

struct SgsCandidate {
    int    type_id;
    int    p_occ;
    int    a_min;       // earliest feasible start (Instance.T[j,m].start)
    int    t_max_excl;  // exclusive upper bound on start (Instance.T[j,m].stop)
    int    cap;         // -1 = unlimited capacity
    double cost;
};

struct SgsProblem {
    int    n_jobs;
    int    n_types;
    int    horizon;   // occupancy grid length per type (== Instance.horizon)
    double budget;

    std::vector<double> arrival;                     // [j]
    std::vector<std::vector<SgsCandidate>> cand;      // [j], sorted by cost ascending
    std::vector<int> init_occ_flat;                   // [type_id*horizon + t]
};

// Per-thread decode workspace — mirrors EvalWorkspace's dirty-list convention
// (occ tracks only genes-added occupancy; init_occ_flat is added at check
// time), so only touched cells need clearing between decodes.
struct SgsWorkspace {
    std::vector<int> occ;
    std::vector<int> dirty;

    void reset(const SgsProblem& prob) {
        occ.assign(static_cast<size_t>(prob.n_types) * prob.horizon, 0);
        dirty.clear();
        dirty.reserve(4096);
    }
};

// ── Decode / evaluate ─────────────────────────────────────────────────────────

// Earliest t in [c.a_min, c.t_max_excl) such that [t, t+p_occ) is entirely
// below capacity, using the same jump-ahead scan as the Python prototype:
// on a blocked window, jump straight past the first blocking slot found
// rather than re-scanning one slot at a time.
inline bool sgs_scan_earliest(const SgsWorkspace& ws, const SgsProblem& prob,
                              const SgsCandidate& c, int& start_out) noexcept {
    const int base = c.type_id * prob.horizon;
    int t = c.a_min;
    while (t < c.t_max_excl) {
        int blocking = -1;
        const int end = t + c.p_occ;
        for (int u = t; u < end; u++) {
            if (ws.occ[base + u] + prob.init_occ_flat[base + u] >= c.cap) {
                blocking = u;
                break;
            }
        }
        if (blocking < 0) { start_out = t; return true; }
        t = blocking + 1;
    }
    return false;
}

// Decodes `ind` and fills f1/f2/cv. If `out_assignment` is non-null, also
// records (type_id, start) per job id (for extract_assignment_sgs).
inline void evaluate_sgs(Individual& ind, const SgsProblem& prob, SgsWorkspace& ws,
                         std::vector<std::pair<int,int>>* out_assignment = nullptr) {
    for (int idx : ws.dirty) ws.occ[idx] = 0;
    ws.dirty.clear();

    ind.f1 = 0.0;
    ind.f2 = 0.0;
    ind.cv = 0.0;

    const int n = prob.n_jobs;
    if (out_assignment) out_assignment->assign(n, {0, 0});

    for (int pos = 0; pos < n; pos++) {
        const int j = ind.genes[pos];
        const auto& cands = prob.cand[j];
        const int ncand = static_cast<int>(cands.size());
        int k0 = ind.genes[n + j];
        if (k0 < 0) k0 = 0;
        if (k0 >= ncand) k0 = ncand - 1;

        bool placed = false;
        int chosen = k0, start = 0;

        for (int off = 0; off < ncand; off++) {
            const int idx = (k0 + off) % ncand;
            const SgsCandidate& c = cands[idx];
            if (c.cap < 0) {
                start = c.a_min; chosen = idx; placed = true; break;
            }
            int s;
            if (sgs_scan_earliest(ws, prob, c, s)) {
                start = s; chosen = idx; placed = true; break;
            }
        }

        if (!placed) {
            // Every feasible type saturated for this job's whole window:
            // force the type-choice candidate at its earliest start and
            // charge the resulting overcrowding to cv (does not touch occ,
            // matching the Python prototype's fallback exactly).
            chosen = k0;
            const SgsCandidate& c = cands[chosen];
            start = c.a_min;
            const int base = c.type_id * prob.horizon;
            long long excess = 0;
            for (int u = start; u < start + c.p_occ; u++) {
                const int total = ws.occ[base + u] + prob.init_occ_flat[base + u];
                const int exc = total - c.cap + 1;
                if (exc > 0) excess += exc;
            }
            ind.cv += static_cast<double>(excess);
        } else {
            const SgsCandidate& c = cands[chosen];
            if (c.cap >= 0) {
                const int base = c.type_id * prob.horizon;
                for (int u = start; u < start + c.p_occ; u++) {
                    const int idx = base + u;
                    if (ws.occ[idx] == 0) ws.dirty.push_back(idx);
                    ws.occ[idx]++;
                }
            }
        }

        const SgsCandidate& applied = cands[chosen];
        ind.f1 += (start + applied.p_occ - prob.arrival[j]);
        ind.f2 += applied.cost;
        if (out_assignment) (*out_assignment)[j] = {applied.type_id, start};
    }

    if (ind.f2 > prob.budget) ind.cv += (ind.f2 - prob.budget);
}

inline std::vector<std::tuple<int,int>> extract_assignment_sgs(
    const Individual& ind, const SgsProblem& prob, SgsWorkspace& ws) {
    std::vector<std::pair<int,int>> asgn;
    Individual tmp = ind;  // evaluate_sgs mutates nothing in ind beyond f1/f2/cv
    evaluate_sgs(tmp, prob, ws, &asgn);
    std::vector<std::tuple<int,int>> out;
    out.reserve(asgn.size());
    for (auto& [tid, start] : asgn) out.emplace_back(tid, start);
    return out;
}

// ── Genetic operators ─────────────────────────────────────────────────────────

inline Individual make_random_sgs(const SgsProblem& prob, std::mt19937& rng) {
    Individual ind;
    const int n = prob.n_jobs;
    ind.genes.resize(2 * n);
    for (int i = 0; i < n; i++) ind.genes[i] = i;
    std::shuffle(ind.genes.begin(), ind.genes.begin() + n, rng);
    for (int j = 0; j < n; j++) {
        const int ncand = static_cast<int>(prob.cand[j].size());
        ind.genes[n + j] = std::uniform_int_distribution<int>(0, ncand - 1)(rng);
    }
    return ind;
}

// Order Crossover (OX, Davis 1985) on a permutation of [0, n).
inline void ox_crossover(const int* p1, const int* p2, std::vector<int>& child,
                         int n, std::mt19937& rng) {
    child.assign(n, -1);
    std::uniform_int_distribution<int> pt(0, n - 1);
    int a = pt(rng), b = pt(rng);
    if (a > b) std::swap(a, b);

    std::vector<bool> used(n, false);
    for (int i = a; i <= b; i++) { child[i] = p1[i]; used[p1[i]] = true; }

    int idx = (b + 1) % n;
    for (int k = 0; k < n; k++) {
        const int src = (b + 1 + k) % n;
        const int val = p2[src];
        if (!used[val]) {
            child[idx] = val; used[val] = true;
            idx = (idx + 1) % n;
        }
    }
}

// OX on the permutation half + two-point/uniform on the type-choice half
// (that half has independent per-job domains, so the ordinary index
// crossover from ga_common.hpp applies unchanged in spirit).
inline Individual crossover_sgs(const Individual& p1, const Individual& p2,
                                std::mt19937& rng, int kind, int n_jobs) {
    Individual child;
    child.genes.resize(2 * n_jobs);

    std::vector<int> child_perm;
    ox_crossover(p1.genes.data(), p2.genes.data(), child_perm, n_jobs, rng);
    for (int i = 0; i < n_jobs; i++) child.genes[i] = child_perm[i];

    if (kind == 1) {  // uniform
        std::uniform_int_distribution<int> coin(0, 1);
        for (int j = 0; j < n_jobs; j++)
            child.genes[n_jobs + j] = coin(rng) ? p1.genes[n_jobs + j] : p2.genes[n_jobs + j];
    } else if (n_jobs <= 2) {
        std::uniform_int_distribution<int> coin(0, 1);
        for (int j = 0; j < n_jobs; j++)
            child.genes[n_jobs + j] = coin(rng) ? p1.genes[n_jobs + j] : p2.genes[n_jobs + j];
    } else {  // two-point (default)
        std::uniform_int_distribution<int> pt(1, n_jobs - 1);
        int c1 = pt(rng), c2 = pt(rng);
        if (c1 > c2) std::swap(c1, c2);
        for (int j = 0; j < n_jobs; j++)
            child.genes[n_jobs + j] =
                (j >= c1 && j < c2) ? p2.genes[n_jobs + j] : p1.genes[n_jobs + j];
    }
    return child;
}

// Swap mutation on the permutation half (standard for permutation
// representations — a per-gene resample would produce duplicate/missing job
// ids); per-gene resample on the type-choice half (independent domains, same
// style as ga_common.hpp's mutate()).
inline void mutate_sgs(Individual& ind, const SgsProblem& prob, double p_mut,
                       std::mt19937& rng) {
    const int n = prob.n_jobs;
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::uniform_int_distribution<int> ri(0, n - 1);
    for (int i = 0; i < n; i++)
        if (u(rng) < p_mut) std::swap(ind.genes[i], ind.genes[ri(rng)]);
    for (int j = 0; j < n; j++)
        if (u(rng) < p_mut) {
            const int ncand = static_cast<int>(prob.cand[j].size());
            ind.genes[n + j] = std::uniform_int_distribution<int>(0, ncand - 1)(rng);
        }
}

// ── Heuristic seeds ────────────────────────────────────────────────────────────
//
// Direct port of the 5 seed genomes validated in
// scripts/compare_priority_sgs_seeded_vs_known.py's build_heuristic_seeds():
// the other heuristics degenerate to identical genomes on this dataset (see
// that function's docstring), so repeating them here would add construction
// cost for zero diversity — the same reasoning ga_common.hpp's own
// make_heuristic_seeds() comment gives for dropping make_no_burst().

inline Individual make_seed_arrival_order(const SgsProblem& prob, int type_choice_rule) {
    // type_choice_rule: 0 = cheapest (index 0), 1 = cloud-first (unlimited
    // type if any, else mid), 2 = mid-cost, 3 = priciest (last index).
    Individual ind;
    const int n = prob.n_jobs;
    ind.genes.resize(2 * n);

    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b) { return prob.arrival[a] < prob.arrival[b]; });
    for (int i = 0; i < n; i++) ind.genes[i] = order[i];

    for (int j = 0; j < n; j++) {
        const auto& c = prob.cand[j];
        const int ncand = static_cast<int>(c.size());
        int choice = 0;
        switch (type_choice_rule) {
            case 0: choice = 0; break;
            case 1: {
                int cloud = -1;
                for (int k = 0; k < ncand; k++) if (c[k].cap < 0) { cloud = k; break; }
                choice = (cloud >= 0) ? cloud : ncand / 2;
                break;
            }
            case 2: choice = ncand / 2; break;
            case 3: choice = ncand - 1; break;
        }
        ind.genes[n + j] = choice;
    }
    return ind;
}

inline std::vector<Individual> make_heuristic_seeds_sgs(const SgsProblem& prob) {
    Individual cheapest_arrival = make_seed_arrival_order(prob, 0);
    Individual cheapest_reverse = cheapest_arrival;
    std::reverse(cheapest_reverse.genes.begin(),
                cheapest_reverse.genes.begin() + prob.n_jobs);
    return {
        cheapest_arrival,                       // cheapest-type, arrival-order
        cheapest_reverse,                       // cheapest-type, reverse-arrival-order
        make_seed_arrival_order(prob, 1),       // cloud-first, arrival-order
        make_seed_arrival_order(prob, 2),       // mid-cost type, arrival-order
        make_seed_arrival_order(prob, 3),       // priciest-type, arrival-order
    };
}

// ── Problem construction ──────────────────────────────────────────────────────

inline SgsProblem build_sgs_problem(
    int n_jobs, double budget, int horizon,
    const std::vector<double>& arrival,
    // raw_cand[j] = (type_id, p_occ, a_min, t_max_excl, cap, cost) tuples,
    // pre-sorted by cost ascending (done in Python, mirroring job_candidates
    // in priority_sgs_problem.py).
    const std::vector<std::vector<std::tuple<int,int,int,int,int,double>>>& raw_cand,
    const std::vector<std::tuple<int,int,int>>& raw_init_occ  // (type_id, t, count)
) {
    SgsProblem prob;
    prob.n_jobs  = n_jobs;
    prob.budget  = budget;
    prob.horizon = horizon;
    prob.arrival = arrival;
    prob.cand.resize(n_jobs);

    int max_type = 0;
    for (int j = 0; j < n_jobs; j++) {
        prob.cand[j].reserve(raw_cand[j].size());
        for (auto& [tid, pocc, amin, tmax, cap, cost] : raw_cand[j]) {
            prob.cand[j].push_back({tid, pocc, amin, tmax, cap, cost});
            max_type = std::max(max_type, tid);
        }
    }
    prob.n_types = max_type + 1;
    prob.init_occ_flat.assign(static_cast<size_t>(prob.n_types) * horizon, 0);
    for (auto& [tid, t, cnt] : raw_init_occ)
        if (tid < prob.n_types && t < horizon)
            prob.init_occ_flat[static_cast<size_t>(tid) * horizon + t] = cnt;

    return prob;
}
