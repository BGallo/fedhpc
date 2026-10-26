/**
 * FED-HPC — weighted-sum solver on a dispatch decoder (on-prem/cloud split search).
 *
 * Solves the same scalar as weighted.hpp,  min w1·f1 + w2·f2  (soft cap
 * f1 ≤ f1_cap), but with a different representation:
 *
 *   genome  — per job: 0 = auto, 1 = cloud, 2 = on-prem (never offloaded).
 *   decoder — time-driven non-delay list scheduling of the on-prem jobs: at
 *             every t, waiting jobs in WSPT order (p_min / w, then release)
 *             start on any capacity-feasible on-prem type (the decoder picks
 *             the type — equivalent pools need no gene). A cloud job runs on
 *             its best cloud type for this scalar (cloud capacity never
 *             binds, so it starts at release and its score is fixed).
 *             `auto` jobs are offloaded to cloud once waiting longer would
 *             score worse than the cloud option (deadline rule, the decoder
 *             analogue of the R3 rejection deadlines in formulations.py).
 *   search  — best-improvement local search over single genome flips,
 *             candidates ranked by their own Δg estimate and decoded exactly
 *             in parallel, plus ILS kicks on stagnation.
 *
 * Why: on the 10-min instances every on-prem type is equivalent per job
 * (same p_occ, cost 0), so the slot-level genome of weighted.hpp spends
 * almost all of its search on pool↔pool plateaus, and its left-shift repair
 * cannot build a feasible from-scratch schedule. Time-driven SPT dispatch
 * reproduces the exact optimum's on-prem schedule to within ~0.2 % of f1.
 *
 * Deterministic for a fixed seed, independent of n_threads: candidates are
 * decoded in parallel but reduced in index order.
 */
#pragma once

#include "ga_common.hpp"

struct DOpt { int type, p; double cost; int earliest, latest; };

struct DJob {
    double arrival, weight;
    std::vector<DOpt> onprem;   // finite-capacity options
    std::vector<DOpt> cloud;    // unlimited options
};

struct DProblem {
    int n = 0, H = 0;
    std::vector<DJob> jobs;
    std::vector<int> pool_type;              // pool index -> type id
    std::vector<std::vector<int>> free0;     // [pool][t] free capacity, t in [0, H)
    std::vector<int> order;                  // static WSPT priority order
    std::vector<int> release;                // earliest on-prem start (or INT_MAX)
    std::vector<int> pmin;                   // min on-prem p (or 0)
    std::vector<std::vector<int>> opt_pool;  // [j][k] pool index of onprem[k]
    std::vector<int> rank;                   // [j] position in `order`
};

inline DProblem build_dproblem(
    int H,
    const std::vector<std::tuple<double, double,
        std::vector<std::tuple<int, int, double, int, int>>>>& jobs_in,
    const std::vector<int>& type_cap,
    const std::vector<std::tuple<int, int, int>>& init_occ) {
    DProblem P;
    P.n = (int)jobs_in.size();
    P.H = H;
    std::vector<int> type_pool(type_cap.size(), -1);
    for (int m = 0; m < (int)type_cap.size(); m++)
        if (type_cap[m] >= 0) {
            type_pool[m] = (int)P.pool_type.size();
            P.pool_type.push_back(m);
            P.free0.emplace_back(H, type_cap[m]);
        }
    for (auto& [m, t, c] : init_occ)
        if (m < (int)type_pool.size() && type_pool[m] >= 0 && t >= 0 && t < H)
            P.free0[type_pool[m]][t] -= c;
    P.jobs.resize(P.n);
    P.release.assign(P.n, INT_MAX);
    P.pmin.assign(P.n, 0);
    P.opt_pool.resize(P.n);
    for (int j = 0; j < P.n; j++) {
        auto& [a, w, opts] = jobs_in[j];
        DJob& J = P.jobs[j];
        J.arrival = a; J.weight = w;
        for (auto& [m, p, c, e, l] : opts) {
            DOpt o{m, p, c, e, l};
            if (m < (int)type_cap.size() && type_cap[m] >= 0) {
                J.onprem.push_back(o);
                P.opt_pool[j].push_back(type_pool[m]);
            } else {
                J.cloud.push_back(o);
            }
        }
        int pm = INT_MAX;
        for (auto& o : J.onprem) { P.release[j] = std::min(P.release[j], o.earliest); pm = std::min(pm, o.p); }
        P.pmin[j] = J.onprem.empty() ? 0 : pm;
    }
    P.order.resize(P.n);
    std::iota(P.order.begin(), P.order.end(), 0);
    std::stable_sort(P.order.begin(), P.order.end(), [&](int a, int b) {
        const double ka = P.pmin[a] / std::max(P.jobs[a].weight, 1e-12);
        const double kb = P.pmin[b] / std::max(P.jobs[b].weight, 1e-12);
        if (ka != kb) return ka < kb;
        return P.release[a] < P.release[b];
    });
    P.rank.resize(P.n);
    for (int i = 0; i < P.n; i++) P.rank[P.order[i]] = i;
    return P;
}

struct DScalar {
    double w1, w2, f1_cap, cap_pen;
    std::vector<double> cloud_score;   // [j] best cloud score (inf if none)
    std::vector<int> cloud_opt;        // [j] index into jobs[j].cloud (-1 if none)
    double mu = 0;                     // priority mix (0 = WSPT)
    std::vector<int> order, rank;      // dispatch priority order for this scalar
};

// Dispatch priority for this scalar: index v_j = (w1·w_j + mu·regret_j) / p_j,
// highest first. regret_j = what sending j to cloud costs over running it
// on-prem at release (capped when j has no cloud option). mu = 0 is WSPT —
// optimal-order for sum w_j C_j — and ignores the cloud alternative; larger mu
// hands contended capacity to the jobs whose cloud option is dearest.
inline void set_priority(const DProblem& P, DScalar& S, double mu) {
    S.mu = mu;
    std::vector<double> v(P.n, 0.0);
    for (int j = 0; j < P.n; j++) {
        const DJob& J = P.jobs[j];
        if (J.onprem.empty()) continue;
        double s_on = std::numeric_limits<double>::infinity();
        for (const DOpt& o : J.onprem)
            s_on = std::min(s_on, S.w1 * J.weight * (o.earliest + o.p - J.arrival) + S.w2 * o.cost);
        double regret = S.cloud_score[j] - s_on;
        if (!std::isfinite(regret)) regret = 1e6 * (S.w1 + S.w2);
        regret = std::max(regret, 0.0);
        v[j] = (S.w1 * J.weight + mu * regret) / std::max(P.pmin[j], 1);
    }
    S.order.resize(P.n);
    std::iota(S.order.begin(), S.order.end(), 0);
    std::stable_sort(S.order.begin(), S.order.end(), [&](int a, int b) {
        if (v[a] != v[b]) return v[a] > v[b];
        return P.release[a] < P.release[b];
    });
    S.rank.resize(P.n);
    for (int i = 0; i < P.n; i++) S.rank[S.order[i]] = i;
}

inline DScalar make_scalar(const DProblem& P, double w1, double w2, double f1_cap) {
    DScalar S{w1, w2, f1_cap, (w1 > 0.0 ? w1 : (w2 > 0.0 ? w2 : 1.0)) * 1e3, {}, {}};
    S.cloud_score.assign(P.n, std::numeric_limits<double>::infinity());
    S.cloud_opt.assign(P.n, -1);
    for (int j = 0; j < P.n; j++) {
        const DJob& J = P.jobs[j];
        for (int k = 0; k < (int)J.cloud.size(); k++) {
            const DOpt& o = J.cloud[k];
            const double s = w1 * J.weight * (o.earliest + o.p - J.arrival) + w2 * o.cost;
            if (s < S.cloud_score[j]) { S.cloud_score[j] = s; S.cloud_opt[j] = k; }
        }
    }
    set_priority(P, S, 0.0);
    return S;
}

enum DFlag {
    DF_NO_DEADLINE   = 1 << 0,   // `auto` = on-prem (no deadline offload)
    DF_TIE_MOST_FREE = 1 << 1,   // pool tie-break: most free capacity at t
    DF_TIE_BEST_FIT  = 1 << 2,   // pool tie-break: least free capacity at t
    DF_RANDOM_RANK   = 1 << 3,   // LS: random candidates instead of est-ranked
    DF_NO_RECALL     = 1 << 4,   // LS: only offload moves (no cloud→on-prem)
    DF_NO_KICK       = 1 << 5,   // no ILS kicks
    DF_MIX_RANK      = 1 << 6,   // LS: half est-ranked, half random candidates
    DF_OLD_EST       = 1 << 7,   // LS: weighted.hpp's type-flip estimate
                                 //     (w1·Δp + w2·Δcost, improving only), ranked
    DF_SERIAL        = 1 << 8,   // decoder: serial SGS instead of dispatch
    DF_BEST_OF       = 1 << 9,   // decoder: both, keep the lower pen_g
};

struct DSched {
    double f1 = 0, f2 = 0, g = 0, pen_g = 0;
    int cv = 0, n_cloud = 0, n_auto_off = 0;
    std::vector<int> type, start;
    std::vector<char> on;              // 1 = placed on-prem
};

struct DWork {
    std::vector<std::vector<int>> free;
    std::vector<int> alt_type, alt_start;       // serial-decoder scratch (best-of)
    std::vector<char> alt_on;
    std::vector<int> maxrun, waiting, merged;
    std::vector<std::vector<int>> rel_bucket;   // jobs (priority order) released at t
};

inline double pen(const DScalar& S, double f1, double f2, int cv) {
    double g = S.w1 * f1 + S.w2 * f2;
    if (f1 > S.f1_cap) g += S.cap_pen * (f1 - S.f1_cap);
    if (cv > 0) g += 1e9 + 1e6 * cv;
    return g;
}

inline void dispatch(const DProblem& P, const DScalar& S, const std::vector<char>& gen,
                     int flags, DWork& W, DSched& out) {
    const int H = P.H, np = (int)P.pool_type.size();
    const bool deadline = !(flags & DF_NO_DEADLINE);
    W.free = P.free0;
    W.maxrun.assign(np, 0);
    out.type.assign(P.n, -1); out.start.assign(P.n, -1); out.on.assign(P.n, 0);
    out.f1 = out.f2 = 0; out.cv = out.n_cloud = out.n_auto_off = 0;

    auto to_cloud = [&](int j) -> bool {
        const int k = S.cloud_opt[j];
        if (k < 0) return false;
        const DOpt& o = P.jobs[j].cloud[k];
        out.type[j] = o.type; out.start[j] = o.earliest;
        out.f1 += P.jobs[j].weight * (o.earliest + o.p - P.jobs[j].arrival);
        out.f2 += o.cost;
        out.n_cloud++;
        return true;
    };

    if ((int)W.rel_bucket.size() != H + 1) W.rel_bucket.assign(H + 1, {});
    for (auto& b : W.rel_bucket) b.clear();
    for (int j : S.order) {
        if (gen[j] == 1 || P.jobs[j].onprem.empty()) {
            if (!to_cloud(j)) {
                if (P.jobs[j].onprem.empty()) { out.cv++; continue; }
            } else continue;
        }
        W.rel_bucket[std::min(P.release[j], H)].push_back(j);
    }
    W.waiting.clear();
    // buckets are filled in priority order; merge them into `waiting` by rank
    auto by_rank = [&](int a, int b) { return S.rank[a] < S.rank[b]; };

    for (int t = 0; t < H; t++) {
        if (!W.rel_bucket[t].empty()) {
            W.merged.resize(W.waiting.size() + W.rel_bucket[t].size());
            std::merge(W.waiting.begin(), W.waiting.end(), W.rel_bucket[t].begin(),
                       W.rel_bucket[t].end(), W.merged.begin(), by_rank);
            W.waiting.swap(W.merged);
        }
        if (W.waiting.empty()) continue;
        int any_room = 0;
        for (int q = 0; q < np; q++) {
            int r = 0;
            const auto& f = W.free[q];
            while (t + r < H && f[t + r] > 0) r++;
            W.maxrun[q] = r;
            any_room |= (r > 0);
        }
        size_t keep = 0;
        for (size_t i = 0; i < W.waiting.size(); i++) {
            const int j = W.waiting[i];
            const DJob& J = P.jobs[j];
            int best_k = -1; double best_s = 0; int best_free = 0;
            if (any_room) {
                for (int k = 0; k < (int)J.onprem.size(); k++) {
                    const DOpt& o = J.onprem[k];
                    const int q = P.opt_pool[j][k];
                    if (t < o.earliest || t > o.latest || o.p > W.maxrun[q]) continue;
                    const double s = S.w1 * J.weight * (t + o.p - J.arrival) + S.w2 * o.cost;
                    const int fr = W.free[q][t];
                    bool better = best_k < 0 || s < best_s - 1e-12;
                    if (!better && std::abs(s - best_s) <= 1e-12) {
                        if (flags & DF_TIE_MOST_FREE) better = fr > best_free;
                        else if (flags & DF_TIE_BEST_FIT) better = fr < best_free;
                    }
                    if (better) { best_k = k; best_s = s; best_free = fr; }
                }
            }
            if (best_k >= 0) {
                if (deadline && gen[j] == 0 && best_s >= S.cloud_score[j] && to_cloud(j)) {
                    out.n_auto_off++;
                    continue;
                }
                const DOpt& o = J.onprem[best_k];
                const int q = P.opt_pool[j][best_k];
                auto& f = W.free[q];
                int newrun = W.maxrun[q];
                for (int u = t; u < t + o.p; u++) {
                    if (--f[u] == 0 && u - t < newrun) newrun = u - t;
                }
                W.maxrun[q] = newrun;
                if (newrun == 0) {
                    any_room = 0;
                    for (int qq = 0; qq < np; qq++) any_room |= (W.maxrun[qq] > 0);
                }
                out.type[j] = o.type; out.start[j] = t; out.on[j] = 1;
                out.f1 += J.weight * (t + o.p - J.arrival);
                out.f2 += o.cost;
                continue;
            }
            // not placed at t: offload if waiting past t already scores worse
            int last = -1;
            double cmin = std::numeric_limits<double>::infinity();
            for (const DOpt& o : J.onprem) { last = std::max(last, o.latest); cmin = std::min(cmin, o.cost); }
            if (deadline && gen[j] == 0 &&
                S.w1 * J.weight * (t + 1 + P.pmin[j] - J.arrival) + S.w2 * cmin >= S.cloud_score[j] &&
                to_cloud(j)) {
                out.n_auto_off++;
                continue;
            }
            if (t + 1 > last) {             // horizon: no on-prem start left
                if (!to_cloud(j)) out.cv++;
                continue;
            }
            W.waiting[keep++] = j;
        }
        W.waiting.resize(keep);
    }
    for (int j : W.waiting) if (!to_cloud(j)) out.cv++;
    out.g = S.w1 * out.f1 + S.w2 * out.f2;
    out.pen_g = pen(S, out.f1, out.f2, out.cv);
}

// Serial SGS decoder (the rule of weighted.hpp's schedule_repair, from
// scratch): on-prem jobs in this scalar's priority order, each at its earliest
// capacity-feasible start on any on-prem type. An `auto` job whose earliest
// on-prem score is no better than its cloud score goes to cloud (the deadline
// rule); a job with no on-prem start left before its latest start overflows
// to cloud. On lightly loaded instances this reproduces the exact on-prem
// schedule where non-delay dispatch is ~0.2-0.8 % of the f1 range off; under
// congestion it fragments capacity and is far worse — hence DF_BEST_OF.
inline void serial_sgs(const DProblem& P, const DScalar& S, const std::vector<char>& gen,
                       int flags, DWork& W, DSched& out) {
    const bool deadline = !(flags & DF_NO_DEADLINE);
    W.free = P.free0;
    out.type.assign(P.n, -1); out.start.assign(P.n, -1); out.on.assign(P.n, 0);
    out.f1 = out.f2 = 0; out.cv = out.n_cloud = out.n_auto_off = 0;
    auto to_cloud = [&](int j) -> bool {
        const int k = S.cloud_opt[j];
        if (k < 0) return false;
        const DOpt& o = P.jobs[j].cloud[k];
        out.type[j] = o.type; out.start[j] = o.earliest;
        out.f1 += P.jobs[j].weight * (o.earliest + o.p - P.jobs[j].arrival);
        out.f2 += o.cost;
        out.n_cloud++;
        return true;
    };
    for (int j : S.order) {
        const DJob& J = P.jobs[j];
        if (gen[j] == 1 || J.onprem.empty()) {
            if (!to_cloud(j)) out.cv++;
            continue;
        }
        int best_k = -1, best_t = 0; double best_s = 0;
        for (int k = 0; k < (int)J.onprem.size(); k++) {
            const DOpt& o = J.onprem[k];
            const auto& f = W.free[P.opt_pool[j][k]];
            int t = std::max(o.earliest, 0), found = -1;
            while (t <= o.latest) {          // earliest window of o.p free slots
                int u = t;
                const int e = t + o.p;
                while (u < e && f[u] > 0) u++;
                if (u == e) { found = t; break; }
                t = u + 1;
            }
            if (found < 0) continue;
            const double sc = S.w1 * J.weight * (found + o.p - J.arrival) + S.w2 * o.cost;
            if (best_k < 0 || sc < best_s - 1e-12) { best_k = k; best_s = sc; best_t = found; }
        }
        if (best_k < 0) { if (!to_cloud(j)) out.cv++; continue; }
        if (deadline && gen[j] == 0 && best_s >= S.cloud_score[j] && to_cloud(j)) {
            out.n_auto_off++;
            continue;
        }
        const DOpt& o = J.onprem[best_k];
        auto& f = W.free[P.opt_pool[j][best_k]];
        for (int u = best_t; u < best_t + o.p; u++) f[u]--;
        out.type[j] = o.type; out.start[j] = best_t; out.on[j] = 1;
        out.f1 += J.weight * (best_t + o.p - J.arrival);
        out.f2 += o.cost;
    }
    out.g = S.w1 * out.f1 + S.w2 * out.f2;
    out.pen_g = pen(S, out.f1, out.f2, out.cv);
}

// Decoder selected by flags: dispatch (default), serial SGS, or the better of both.
inline void decode(const DProblem& P, const DScalar& S, const std::vector<char>& gen,
                   int flags, DWork& W, DSched& out) {
    if (flags & DF_SERIAL) { serial_sgs(P, S, gen, flags, W, out); return; }
    dispatch(P, S, gen, flags, W, out);
    if (!(flags & DF_BEST_OF)) return;
    DSched alt;
    alt.type.swap(W.alt_type); alt.start.swap(W.alt_start); alt.on.swap(W.alt_on);
    serial_sgs(P, S, gen, flags, W, alt);
    if (alt.pen_g < out.pen_g - 1e-12) std::swap(out, alt);
    W.alt_type.swap(alt.type); W.alt_start.swap(alt.start); W.alt_on.swap(alt.on);
}

struct DStats {
    long long decodes = 0;
    int ls_moves = 0, kicks = 0, iters = 0;
    double mu = 0;
    double decode_ms = 0;
    std::vector<std::pair<double, double>> trace;   // (elapsed ms, best pen_g)
};

// Best-improvement LS over single genome flips; leaves the result in gen/cur.
inline void dispatch_ls(const DProblem& P, const DScalar& S, std::vector<char>& gen,
                        DSched& cur, int flags, int K, int max_iters,
                        std::mt19937& rng, std::vector<DWork>& works, DStats& st,
                        TimePoint t0) {
    struct Cand { double est; int j; char to; };
    std::vector<Cand> cand;
    std::vector<DSched> res;
    for (int it = 0; it < max_iters; it++) {
        st.iters++;
        cand.clear();
        for (int j = 0; j < P.n; j++) {
            const DJob& J = P.jobs[j];
            if (flags & DF_OLD_EST) {
                // weighted.hpp's shortlist: unweighted Δp_occ, improving estimates only
                const int kc = S.cloud_opt[j];
                if (J.onprem.empty() || kc < 0) continue;
                const DOpt& c = J.cloud[kc];
                const DOpt* o = &J.onprem[0];
                for (const DOpt& x : J.onprem) if (x.type == cur.type[j]) { o = &x; break; }
                const double d = S.w1 * (c.p - o->p) + S.w2 * (c.cost - o->cost);
                if (cur.on[j]) { if (d < -1e-12) cand.push_back({d, j, 1}); }
                else if (!(flags & DF_NO_RECALL) && -d < -1e-12)
                    cand.push_back({-d, j, gen[j] == 1 ? (char)0 : (char)2});
                continue;
            }
            if (cur.on[j]) {
                if (S.cloud_opt[j] < 0) continue;
                // own on-prem score (completion from the schedule)
                double s_on = 0;
                for (const DOpt& o : J.onprem) if (o.type == cur.type[j]) {
                    s_on = S.w1 * J.weight * (cur.start[j] + o.p - J.arrival) + S.w2 * o.cost; break; }
                cand.push_back({S.cloud_score[j] - s_on, j, 1});
            } else if (!(flags & DF_NO_RECALL) && !J.onprem.empty()) {
                // optimistic no-wait on-prem score
                double s_on = std::numeric_limits<double>::infinity();
                for (const DOpt& o : J.onprem)
                    s_on = std::min(s_on, S.w1 * J.weight * (o.earliest + o.p - J.arrival) + S.w2 * o.cost);
                cand.push_back({s_on - S.cloud_score[j], j, gen[j] == 1 ? (char)0 : (char)2});
            }
        }
        if (cand.empty()) break;
        const int k = std::min<int>(K, (int)cand.size());
        if ((flags & DF_MIX_RANK) && !(flags & DF_RANDOM_RANK)) {
            const int h = k / 2;
            std::partial_sort(cand.begin(), cand.begin() + h, cand.end(),
                              [](const Cand& a, const Cand& b) { return a.est < b.est; });
            for (int i = h; i < k; i++) {
                std::uniform_int_distribution<int> d(i, (int)cand.size() - 1);
                std::swap(cand[i], cand[d(rng)]);
            }
        } else if (flags & DF_RANDOM_RANK) {
            for (int i = 0; i < k; i++) {
                std::uniform_int_distribution<int> d(i, (int)cand.size() - 1);
                std::swap(cand[i], cand[d(rng)]);
            }
        } else {
            std::partial_sort(cand.begin(), cand.begin() + k, cand.end(),
                              [](const Cand& a, const Cand& b) { return a.est < b.est; });
        }
        res.resize(k);
        const TimePoint td = Clock::now();
#ifdef _OPENMP
        // Already inside a parallel region (one memetic child per thread):
        // decode serially on this thread's workspace.
        #pragma omp parallel if (!omp_in_parallel())
        {
            DWork& W = works[omp_get_thread_num()];   // 0 in a nested (inactive) team
            std::vector<char> g2;
            #pragma omp for schedule(dynamic, 1)
            for (int i = 0; i < k; i++) {
                g2 = gen;
                g2[cand[i].j] = cand[i].to;
                decode(P, S, g2, flags, W, res[i]);
            }
        }
#else
        for (int i = 0; i < k; i++) {
            std::vector<char> g2 = gen;
            g2[cand[i].j] = cand[i].to;
            decode(P, S, g2, flags, works[0], res[i]);
        }
#endif
        st.decode_ms += ms_since(td);
        st.decodes += k;
        int bi = -1;
        double bg = cur.pen_g - 1e-12;
        for (int i = 0; i < k; i++) if (res[i].pen_g < bg) { bg = res[i].pen_g; bi = i; }
        if (bi < 0) break;
        gen[cand[bi].j] = cand[bi].to;
        cur = std::move(res[bi]);
        st.ls_moves++;
        st.trace.emplace_back(ms_since(t0), cur.pen_g);
    }
}

struct DispatchResult {
    std::vector<std::tuple<int,int>> asgn;
    DSched best;
    DStats st;
    double total_ms = 0;
};

inline DispatchResult run_weighted_dispatch(
    const DProblem& P, double w1, double w2, double f1_cap, int seed, int n_threads,
    int K, int max_iters, int n_kicks, int kick_size, int flags, int ls, double mu,
    int n_mu) {
    py::gil_scoped_release release;
    set_num_threads(n_threads);
    const TimePoint t0 = Clock::now();
    DScalar S = make_scalar(P, w1, w2, f1_cap);
    int nt = 1;
#ifdef _OPENMP
    nt = omp_get_max_threads();
#endif
    std::vector<DWork> works(nt);
    std::mt19937 rng(seed);
    DStats st;

    // Starting priority mixes: the given mu, or the n_mu best of a grid by
    // the all-auto decode.
    static const double grid[] = {0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0};
    std::vector<std::pair<double, double>> starts;   // (decode pen_g, mu)
    if (mu >= 0) {
        starts.push_back({0.0, mu});
    } else {
        std::vector<char> g0(P.n, 0);
        for (double m : grid) {
            set_priority(P, S, m);
            DSched c;
            decode(P, S, g0, flags, works[0], c);
            st.decodes++;
            starts.push_back({c.pen_g, m});
        }
        std::stable_sort(starts.begin(), starts.end());
        starts.resize(std::min<int>(std::max(n_mu, 1), (int)starts.size()));
    }

    std::vector<char> best_gen;
    DSched best;
    double best_mu = starts[0].second;
    const int kicks_per = n_kicks / (int)starts.size();
    for (auto& [_, m] : starts) {
        set_priority(P, S, m);
        std::vector<char> gen(P.n, 0);
        DSched cur;
        decode(P, S, gen, flags, works[0], cur);
        st.decodes++;
        if (ls) dispatch_ls(P, S, gen, cur, flags, K, max_iters, rng, works, st, t0);
        std::vector<char> lbest_gen = gen;
        DSched lbest = cur;
        if (ls && !(flags & DF_NO_KICK)) {
            for (int kk = 0; kk < kicks_per; kk++) {
                st.kicks++;
                std::vector<char> g2 = lbest_gen;
                // flip kick_size jobs: offload random on-prem jobs / recall random cloud jobs
                for (int f = 0; f < kick_size; f++) {
                    const int j = rng() % P.n;
                    if (lbest.on[j] && S.cloud_opt[j] >= 0) g2[j] = 1;
                    else if (!lbest.on[j] && !P.jobs[j].onprem.empty()) g2[j] = (g2[j] == 1) ? 0 : 2;
                }
                DSched c;
                decode(P, S, g2, flags, works[0], c);
                st.decodes++;
                dispatch_ls(P, S, g2, c, flags, K, max_iters, rng, works, st, t0);
                if (c.pen_g < lbest.pen_g - 1e-12) {
                    lbest = std::move(c); lbest_gen = std::move(g2);
                }
            }
        }
        if (best.type.empty() || lbest.pen_g < best.pen_g - 1e-12) {
            best = std::move(lbest); best_gen = std::move(lbest_gen); best_mu = m;
            st.trace.emplace_back(ms_since(t0), best.pen_g);
        }
    }
    st.mu = best_mu;

    DispatchResult R;
    R.asgn.resize(P.n);
    for (int j = 0; j < P.n; j++) R.asgn[j] = {best.type[j], best.start[j]};
    R.best = std::move(best);
    R.st = std::move(st);
    R.total_ms = ms_since(t0);
    return R;
}
