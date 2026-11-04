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

#include <climits>
#include <cstdint>
#include <memory>
#include <type_traits>

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
//
// edf: instead, earliest indifference deadline first — the last start at which
// on-prem still scores better than cloud, release + regret_j / (w1·w_j),
// capped at the job's latest on-prem start (and that cap alone when it has no
// cloud option or w1 = 0). It keeps every job that can beat cloud ahead of its
// own deadline, where WSPT lets long jobs wait past theirs and get offloaded.
inline void set_priority(const DProblem& P, DScalar& S, double mu, bool edf = false) {
    S.mu = mu;
    std::vector<double> v(P.n, 0.0);
    for (int j = 0; j < P.n; j++) {
        const DJob& J = P.jobs[j];
        if (J.onprem.empty()) continue;
        double s_on = std::numeric_limits<double>::infinity();
        int last = 0;
        for (const DOpt& o : J.onprem) {
            s_on = std::min(s_on, S.w1 * J.weight * (o.earliest + o.p - J.arrival) + S.w2 * o.cost);
            last = std::max(last, o.latest);
        }
        double regret = S.cloud_score[j] - s_on;
        if (edf) {
            double dl = last;
            if (std::isfinite(regret) && S.w1 * J.weight > 0)
                dl = std::min<double>(last, P.release[j] + std::max(regret, 0.0) / (S.w1 * J.weight));
            v[j] = -dl;
            continue;
        }
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
    DF_WHY           = 1 << 10,  // diagnostics: record why each job went to cloud
    DF_EDF           = 1 << 11,  // priority: indifference-deadline EDF only (no mu grid)
    DF_EDF_ADD       = 1 << 12,  // priority: EDF as an extra start beside the mu starts
    DF_EDF_NODL      = 1 << 13,  // the EDF start decodes with DF_NO_DEADLINE (mu starts keep it)
    DF_REF_DECODE    = 1 << 14,  // LS: reference decoder (no DEngine)
    DF_NO_RESTART    = 1 << 15,  // LS: DEngine full decode per candidate
    DF_PRUNE         = 1 << 16,  // LS: stop a candidate once its lower bound exceeds the incumbent
    DF_FOCUS_KICKS   = 1 << 17,  // all kicks to the best start after LS (default: split over the starts)
    DF_FULL_RECORD   = 1 << 18,  // LS: re-record the incumbent by a full decode (default: from the move's restart)
    DF_PRUNE_LATE    = 1 << 19,  // LS: DF_PRUNE only in a step's batches after the first (no improving move yet)
    DF_NO_FIT        = 1 << 20,  // DEngine: scan every waiting job in a slot with room (no fit filter)
    DF_ABS_EST       = 1 << 21,  // LS: rank candidates by |est| (closest to indifference) instead of est
};

// DSched::why codes (dispatch decoder, DF_WHY only)
enum DWhy : char { DW_ONPREM = 0, DW_GENE = 1, DW_OFF_PLACE = 2, DW_OFF_WAIT = 3,
                   DW_HORIZON = 4, DW_LEFTOVER = 5 };

struct DSched {
    double f1 = 0, f2 = 0, g = 0, pen_g = 0;
    int cv = 0, n_cloud = 0, n_auto_off = 0;
    std::vector<int> type, start;
    std::vector<char> on;              // 1 = placed on-prem
    std::vector<char> why;             // DWhy per job (DF_WHY only)
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
    const bool why = flags & DF_WHY;
    W.free = P.free0;
    W.maxrun.assign(np, 0);
    out.type.assign(P.n, -1); out.start.assign(P.n, -1); out.on.assign(P.n, 0);
    out.f1 = out.f2 = 0; out.cv = out.n_cloud = out.n_auto_off = 0;
    if (why) out.why.assign(P.n, DW_ONPREM);

    auto to_cloud = [&](int j, char reason) -> bool {
        const int k = S.cloud_opt[j];
        if (k < 0) return false;
        if (why) out.why[j] = reason;
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
            if (!to_cloud(j, DW_GENE)) {
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
                if (deadline && gen[j] == 0 && best_s >= S.cloud_score[j] && to_cloud(j, DW_OFF_PLACE)) {
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
                to_cloud(j, DW_OFF_WAIT)) {
                out.n_auto_off++;
                continue;
            }
            if (t + 1 > last) {             // horizon: no on-prem start left
                if (!to_cloud(j, DW_HORIZON)) out.cv++;
                continue;
            }
            W.waiting[keep++] = j;
        }
        W.waiting.resize(keep);
    }
    for (int j : W.waiting) if (!to_cloud(j, DW_LEFTOVER)) out.cv++;
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

// ── Fast dispatch engine (DF_REF_DECODE off) ─────────────────────────────────
//
// Same decisions as dispatch(), cheaper per decode (2026-11-03,
// results/generated_bench/dispatch_study):
//
//   queue    waiting jobs are a bitset over priority rank, so merging a slot's
//            releases and dropping a job are O(1) and a scan visits only set
//            bits; a job's deadline/horizon removal slot is precomputed, so a
//            slot with no free room only touches the jobs with an event there.
//   fit      (unless DF_NO_FIT) in a slot with room, a pool whose free run is
//            shorter than its longest job masks the scan to the jobs with an
//            option that fits (per pool, length-bucket rank bitsets); the
//            skipped jobs' events are handled after the scan. The scan reads
//            its job data from rank-ordered arrays (DFast::rjob / ropt).
//   restart  an LS candidate differs from the incumbent in one gene, so it
//            decodes identically up to the first slot t* that gene can change.
//            The incumbent is recorded (state every DF_CKPT slots, outcome and
//            decision slot per job); a candidate resumes from the checkpoint at
//            or before t* and stops once its state equals the incumbent's again
//            (every job decided differently is decided in both and its on-prem
//            footprint has ended), taking the rest from the incumbent. An
//            accepted move re-records the incumbent the same way
//            (record_flip; DF_FULL_RECORD: a full record instead).
//   prune   (DF_PRUNE) while a candidate runs, decided scores + each waiting job's
//            min(cloud, start next slot) + each unreleased job's
//            min(cloud, start at release) bound its final score from below;
//            once that exceeds the incumbent's pen_g it cannot be accepted.
//
// f1/f2 of a recorded schedule are summed in job-index order (independent of
// the decode path); a candidate's are the incumbent's plus the change over the
// jobs it re-decided — equal to a full decode up to rounding (~1e-14 relative).

constexpr int DF_CKPT = 4;

// Per (scalar, priority, deadline mode) precomputation, jobs indexed by rank.
struct DFast {
    int nw = 0;
    std::vector<int> last;               // [j] latest on-prem start
    std::vector<double> cmin;            // [j] cheapest on-prem cost
    std::vector<int> evt0, evt2;         // [j] removal slot when waiting (gene 0 / otherwise), INT_MAX if none
    // score lower bound of a job waiting at the end of slot t: min(cs, al + be·(t+1)),
    // equal to cs from slot xc on; of an unreleased job: lbrel
    std::vector<double> al, be, cs, lbrel, lbrel_suf;   // lbrel_suf[t]: jobs released at or after t
    std::vector<int> xc;
    struct Word { int w; uint64_t rel, e0, e2, xc; };
    std::vector<std::vector<Word>> sp;   // [t] nonzero words of the per-slot rank bitsets
    // fit filter: fit[(fofs[q] + b) * nw + w], the jobs with an option on pool q
    // of p <= fthr[q][b] (fthr ascending, at most DF_FIT_B per pool)
    std::vector<std::vector<int>> fthr;
    std::vector<int> fofs;
    std::vector<uint64_t> fit;
    // the room scan's job data by rank: options ropt[roff[r], roff[r + 1])
    struct ROpt { int p, e, l, q, type; double cost; };
    struct RJob { double w, a, cs; int evt0, evt2, j; };
    std::vector<int> roff;
    std::vector<ROpt> ropt;
    std::vector<RJob> rjob;
};

constexpr int DF_FIT_B = 16;

inline DFast build_dfast(const DProblem& P, const DScalar& S, bool deadline) {
    DFast F;
    const int H = P.H, nw = (P.n + 63) / 64;
    F.nw = nw;
    std::vector<uint64_t> rel((size_t)(H + 1) * nw, 0), e0((size_t)H * nw, 0), e2((size_t)H * nw, 0), xb((size_t)H * nw, 0);
    auto set = [&](std::vector<uint64_t>& b, int t, int r) { b[(size_t)t * nw + (r >> 6)] |= 1ULL << (r & 63); };
    F.last.assign(P.n, -1); F.cmin.assign(P.n, std::numeric_limits<double>::infinity());
    F.evt0.assign(P.n, INT_MAX); F.evt2.assign(P.n, INT_MAX);
    F.al.assign(P.n, 0); F.be.assign(P.n, 0); F.cs.assign(P.n, 0); F.lbrel.assign(P.n, 0);
    F.xc.assign(P.n, INT_MAX); F.lbrel_suf.assign(H + 2, 0);
    for (int j = 0; j < P.n; j++) {
        const DJob& J = P.jobs[j];
        if (J.onprem.empty()) continue;
        double a = std::numeric_limits<double>::infinity(), lr = S.cloud_score[j];
        for (const DOpt& o : J.onprem) {
            F.last[j] = std::max(F.last[j], o.latest); F.cmin[j] = std::min(F.cmin[j], o.cost);
            a = std::min(a, S.w1 * J.weight * (o.p - J.arrival) + S.w2 * o.cost);
            lr = std::min(lr, S.w1 * J.weight * (o.earliest + o.p - J.arrival) + S.w2 * o.cost);
        }
        const int r = S.rank[j], t_rel = std::min(P.release[j], H);
        set(rel, t_rel, r);
        F.al[j] = a; F.be[j] = S.w1 * J.weight; F.cs[j] = S.cloud_score[j]; F.lbrel[j] = lr;
        F.lbrel_suf[t_rel] += lr;
        int x = INT_MAX;                 // first t with al + be·(t+1) >= cs
        if (std::isfinite(F.cs[j])) {
            if (F.be[j] <= 0) x = (a >= F.cs[j]) ? 0 : INT_MAX;
            else {
                x = std::max(0, (int)std::ceil((F.cs[j] - a) / F.be[j] - 1.0));
                while (x > 0 && a + F.be[j] * x >= F.cs[j]) x--;
                while (a + F.be[j] * (x + 1.0) < F.cs[j]) x++;
            }
        }
        F.xc[j] = x;
        if (x < H) set(xb, x, r);
        if (t_rel >= H) continue;
        // the first slot >= release at which a waiting, unplaced job leaves the queue
        const int tl = std::max(t_rel, F.last[j]);           // horizon: t + 1 > last
        if (tl < H) { set(e2, tl, r); F.evt2[j] = tl; }
        int td = INT_MAX;                                     // deadline (gene 0)
        if (deadline)
            for (int t = t_rel; t <= std::min(tl, H - 1); t++)
                if (S.w1 * J.weight * (t + 1 + P.pmin[j] - J.arrival) + S.w2 * F.cmin[j] >= S.cloud_score[j]) { td = t; break; }
        const int t0 = std::min(td, tl);
        if (t0 < H) { set(e0, t0, r); F.evt0[j] = t0; }
    }
    for (int t = H - 1; t >= 0; t--) F.lbrel_suf[t] += F.lbrel_suf[t + 1];
    const int np = (int)P.pool_type.size();
    F.fthr.assign(np, {}); F.fofs.assign(np + 1, 0);
    for (int j = 0; j < P.n; j++)
        for (int k = 0; k < (int)P.jobs[j].onprem.size(); k++) F.fthr[P.opt_pool[j][k]].push_back(P.jobs[j].onprem[k].p);
    for (int q = 0; q < np; q++) {
        auto& v = F.fthr[q];
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
        if ((int)v.size() > DF_FIT_B) {                       // quantiles, keeping the largest
            std::vector<int> u;
            for (int b = 1; b <= DF_FIT_B; b++) u.push_back(v[(size_t)b * v.size() / DF_FIT_B - 1]);
            u.erase(std::unique(u.begin(), u.end()), u.end());
            v = u;
        }
        F.fofs[q + 1] = F.fofs[q] + (int)v.size();
    }
    F.fit.assign((size_t)F.fofs[np] * nw, 0);
    for (int j = 0; j < P.n; j++)
        for (int k = 0; k < (int)P.jobs[j].onprem.size(); k++) {
            const int q = P.opt_pool[j][k], r = S.rank[j];
            const auto& v = F.fthr[q];
            for (int b = (int)(std::lower_bound(v.begin(), v.end(), P.jobs[j].onprem[k].p) - v.begin()); b < (int)v.size(); b++)
                F.fit[(size_t)(F.fofs[q] + b) * nw + (r >> 6)] |= 1ULL << (r & 63);
        }
    F.roff.assign(P.n + 1, 0); F.rjob.resize(P.n);
    for (int r = 0; r < P.n; r++) {
        const int j = S.order[r];
        const DJob& J = P.jobs[j];
        for (int k = 0; k < (int)J.onprem.size(); k++) {
            const DOpt& o = J.onprem[k];
            F.ropt.push_back({o.p, o.earliest, o.latest, P.opt_pool[j][k], o.type, o.cost});
        }
        F.roff[r + 1] = (int)F.ropt.size();
        F.rjob[r] = {J.weight, J.arrival, S.cloud_score[j], F.evt0[j], F.evt2[j], j};
    }
    F.sp.assign(H + 1, {});
    for (int t = 0; t <= H; t++)
        for (int w = 0; w < nw; w++) {
            const size_t i = (size_t)t * nw + w;
            const uint64_t a = rel[i], b = t < H ? e0[i] : 0, c = t < H ? e2[i] : 0, d = t < H ? xb[i] : 0;
            if (a | b | c | d) F.sp[t].push_back({w, a, b, c, d});
        }
    return F;
}

// Waiting jobs' bound: sums over the linear-phase jobs (al, be) and capped ones (cs).
struct DBound { double A = 0, B = 0, C = 0; };

// A recorded incumbent.
struct DInc {
    std::vector<char> gen;
    std::vector<uint64_t> g0, g1;        // rank bitsets: gene 0 / effective gene-cloud
    std::vector<size_t> foff;            // [t / DF_CKPT] offset of the u >= t free rows
    std::vector<int> free_snap;          // checkpointed free capacity, rows u >= t
    std::vector<uint64_t> wait_snap;     // checkpointed queue
    std::vector<DBound> bnd;             // checkpointed waiting bound
    std::vector<int> nwait, cv;          // [t] at the start of slot t, t in [0, H]
    std::vector<double> dsc;             // [t] score of the decisions before slot t
    std::vector<double> gc_suf;          // [t] lbrel of the gene-cloud jobs released at or after t
    std::vector<int> dslot, type, start, fend;   // per job; dslot -1 = init, H = leftover; fend 0 = cloud
    std::vector<signed char> kind;               // per job: run()'s decision kind
    std::vector<double> c1, c2;
    std::vector<int> dec_ptr, dec_job;   // CSR: jobs decided at slot t
    DSched s;                            // the schedule, f1/f2 in job-index order
};

struct DFlipWork {
    std::vector<std::vector<int>> free;
    std::vector<uint64_t> wait, g0, g1;
    std::vector<int> maxrun;
    std::vector<char> gen;
    std::vector<int> stamp, pstamp, sstamp, ctype, cstart, cfend, cslot;   // == id: decided / tracked / settled
    std::vector<signed char> ckind;
    std::vector<double> c1, c2, gc_suf;
    std::vector<int> decided, slot_dec, touched;
    std::vector<std::vector<int>> settle_at;    // [t] tracked jobs that settle at the end of t
    int id = 0;
    long long slots = 0, pruned = 0, converged = 0;
};

struct DEngine {
    const DProblem& P; const DScalar& S; const DFast& F; const int flags; const bool deadline;
    DEngine(const DProblem& P_, const DScalar& S_, const DFast& F_, int fl)
        : P(P_), S(S_), F(F_), flags(fl), deadline(!(fl & DF_NO_DEADLINE)) {}

    // Slots [t0, H) and the leftovers from the given state, deciding as
    // dispatch() does. decide(j, type, start, fend, c1, c2, slot, kind) reports
    // each decision (kind 0 on-prem, 1 cloud, 2 cv, 3 deadline offload);
    // end_slot(t) returns true to stop after slot t. lb, when given, is kept.
    template <class Decide, class EndSlot>
    bool run(int t0, std::vector<std::vector<int>>& free, std::vector<uint64_t>& wait, int& n_wait,
             const std::vector<char>& gen, const std::vector<uint64_t>& g0, const std::vector<uint64_t>& g1,
             std::vector<int>& maxrun, DBound* lb, Decide&& decide, EndSlot&& end_slot) const {
        const int H = P.H, np = (int)P.pool_type.size(), nw = F.nw;
        const bool fit = !(flags & DF_NO_FIT);
        if (fit && (int)maxrun.size() < 2 * np) maxrun.resize(2 * np);   // [np, 2np): fit buckets
        int tcur = t0;
        auto lb_add = [&](int j, int t, double sgn) {
            if (F.xc[j] < t) lb->C += sgn * F.cs[j]; else { lb->A += sgn * F.al[j]; lb->B += sgn * F.be[j]; }
        };
        auto drop = [&](int r) {
            wait[r >> 6] &= ~(1ULL << (r & 63)); n_wait--;
            if (lb) lb_add(S.order[r], tcur, -1.0);
        };
        auto cloud = [&](int j, int t, int kind) -> bool {
            const int k = S.cloud_opt[j];
            if (k < 0) return false;
            const DOpt& o = P.jobs[j].cloud[k];
            decide(j, o.type, o.earliest, 0, P.jobs[j].weight * (o.earliest + o.p - P.jobs[j].arrival), o.cost, t, kind);
            return true;
        };
        auto event = [&](int r, int t) {      // waiting, not placed, removal slot t
            const int j = S.order[r];
            const DJob& J = P.jobs[j];
            if (deadline && gen[j] == 0 &&
                S.w1 * J.weight * (t + 1 + P.pmin[j] - J.arrival) + S.w2 * F.cmin[j] >= S.cloud_score[j] &&
                cloud(j, t, 3)) { drop(r); return; }
            if (t + 1 > F.last[j]) { if (!cloud(j, t, 1)) decide(j, -1, -1, 0, 0.0, 0.0, t, 2); drop(r); }
        };
        for (int t = t0; t < H; t++) {
            tcur = t;
            const auto& sp = F.sp[t];
            for (const auto& e : sp) {
                const uint64_t add = e.rel & ~g1[e.w];
                if (!add) continue;
                wait[e.w] |= add; n_wait += __builtin_popcountll(add);
                if (lb) for (uint64_t y = add; y; y &= y - 1) lb_add(S.order[(e.w << 6) | __builtin_ctzll(y)], t, 1.0);
            }
            if (n_wait > 0) {
                // removals at t among the waiting jobs ranked at or after (cw, cbits)
                auto events_from = [&](int cw, uint64_t cbits) {
                    for (const auto& e : sp) {
                        if (e.w < cw) continue;
                        uint64_t y = (deadline ? ((e.e0 & g0[e.w]) | (e.e2 & ~g0[e.w])) : e.e2) & wait[e.w];
                        if (e.w == cw) y &= cbits;
                        for (; y; y &= y - 1) event((e.w << 6) | __builtin_ctzll(y), t);
                    }
                };
                int any_room = 0;
                for (int q = 0; q < np; q++) {
                    int r = 0;
                    const auto& f = free[q];
                    while (t + r < H && f[t + r] > 0) r++;
                    maxrun[q] = r; any_room |= (r > 0);
                }
                // fit filter: per pool, the bucket covering its free run (-1: no room);
                // filt once some pool's run is shorter than its longest job (else
                // the mask keeps every job), used once any word was filtered
                int* bq = fit ? maxrun.data() + np : nullptr;
                bool filt = false, used = false;
                auto bucket = [&](int q) {
                    const auto& v = F.fthr[q];
                    if (v.empty()) { bq[q] = -1; return; }
                    bq[q] = maxrun[q] <= 0 ? -1
                          : F.fofs[q] + std::min<int>((int)(std::lower_bound(v.begin(), v.end(), maxrun[q]) - v.begin()), (int)v.size() - 1);
                    filt |= bq[q] < F.fofs[q + 1] - 1;
                };
                if (fit) for (int q = 0; q < np; q++) bucket(q);
                auto fits = [&](int w) {
                    uint64_t m = 0;
                    for (int q = 0; q < np; q++) if (bq[q] >= 0) m |= F.fit[(size_t)bq[q] * nw + w];
                    return m;
                };
                int w = 0;
                uint64_t x = 0;
                if (any_room) {
                    for (; w < nw && any_room; w++) {
                        x = wait[w];
                        if (filt && x) { x &= fits(w); used = true; }
                        while (x && any_room) {
                            const int b = __builtin_ctzll(x); x &= x - 1;
                            const int r = (w << 6) | b;
                            const DFast::RJob& J = F.rjob[r];
                            const bool g0r = deadline && (g0[w] >> b & 1);   // gene 0 (auto) under the deadline rule
                            const DFast::ROpt* best = nullptr; double best_s = 0; int best_free = 0;
                            for (int k = F.roff[r]; k < F.roff[r + 1]; k++) {
                                const DFast::ROpt& o = F.ropt[k];
                                const int q = o.q;
                                if (t < o.e || t > o.l || o.p > maxrun[q]) continue;
                                const double s = S.w1 * J.w * (t + o.p - J.a) + S.w2 * o.cost;
                                const int fr = free[q][t];
                                bool better = !best || s < best_s - 1e-12;
                                if (!better && std::abs(s - best_s) <= 1e-12) {
                                    if (flags & DF_TIE_MOST_FREE) better = fr > best_free;
                                    else if (flags & DF_TIE_BEST_FIT) better = fr < best_free;
                                }
                                if (better) { best = &o; best_s = s; best_free = fr; }
                            }
                            const int j = J.j;
                            if (best) {
                                if (g0r && best_s >= J.cs && cloud(j, t, 3)) {
                                    drop(r); continue;
                                }
                                const DFast::ROpt& o = *best;
                                const int q = o.q;
                                auto& f = free[q];
                                int newrun = maxrun[q];
                                for (int u = t; u < t + o.p; u++)
                                    if (--f[u] == 0 && u - t < newrun) newrun = u - t;
                                if (fit && newrun != maxrun[q]) {   // narrow the rest of this word too
                                    maxrun[q] = newrun; bucket(q);
                                    if (filt) { x &= fits(w); used = true; }
                                }
                                maxrun[q] = newrun;
                                if (newrun == 0) {
                                    any_room = 0;
                                    for (int qq = 0; qq < np; qq++) any_room |= (maxrun[qq] > 0);
                                }
                                decide(j, o.type, t, t + o.p, J.w * (t + o.p - J.a), o.cost, t, 0);
                                drop(r);
                                continue;
                            }
                            if ((g0r ? J.evt0 : J.evt2) == t) event(r, t);
                        }
                    }
                    if (used) events_from(0, ~0ULL);         // the skipped (non-fitting) jobs' events
                    else if (!any_room) events_from(w - 1, x);   // the rest of word w-1 and later words
                } else {
                    events_from(0, ~0ULL);
                }
            }
            if (lb)                           // waiting jobs whose bound caps at the end of t
                for (const auto& e : sp)
                    for (uint64_t y = e.xc & wait[e.w]; y; y &= y - 1) {
                        const int j = S.order[(e.w << 6) | __builtin_ctzll(y)];
                        lb->A -= F.al[j]; lb->B -= F.be[j]; lb->C += F.cs[j];
                    }
            if (end_slot(t)) return true;
        }
        for (int w = 0; w < nw; w++)
            for (uint64_t y = wait[w]; y; y &= y - 1) {
                const int j = S.order[(w << 6) | __builtin_ctzll(y)];
                if (!cloud(j, H, 1)) decide(j, -1, -1, 0, 0.0, 0.0, H, 2);
            }
        return false;
    }

    // Init decisions (gene cloud, no on-prem option) in priority order and the gene bitsets.
    template <class Decide>
    void init(const std::vector<char>& gen, std::vector<uint64_t>& g0, std::vector<uint64_t>& g1,
              Decide&& decide) const {
        g0.assign(F.nw, 0); g1.assign(F.nw, 0);
        for (int r = 0; r < P.n; r++) {
            const int j = S.order[r];
            if (gen[j] == 1 || P.jobs[j].onprem.empty()) {
                const int k = S.cloud_opt[j];
                if (k >= 0) {
                    const DOpt& o = P.jobs[j].cloud[k];
                    decide(j, o.type, o.earliest, 0, P.jobs[j].weight * (o.earliest + o.p - P.jobs[j].arrival), o.cost, -1, 1);
                    g1[r >> 6] |= 1ULL << (r & 63);
                    continue;
                }
                if (P.jobs[j].onprem.empty()) { decide(j, -1, -1, 0, 0.0, 0.0, -1, 2); continue; }
            }
            if (gen[j] == 0) g0[r >> 6] |= 1ULL << (r & 63);
        }
    }

    // Full decode of gen, recorded for restarts (R.s holds the schedule).
    void record(const std::vector<char>& gen, DInc& R, DFlipWork& W) const {
        const int H = P.H, np = (int)P.pool_type.size(), nw = F.nw, nck = H / DF_CKPT + 1;
        R.gen = gen;
        R.dslot.assign(P.n, -1); R.type.assign(P.n, -1); R.start.assign(P.n, -1); R.fend.assign(P.n, 0);
        R.kind.assign(P.n, 0);
        R.c1.assign(P.n, 0); R.c2.assign(P.n, 0);
        R.foff.assign(nck + 1, 0);
        for (int c = 0; c < nck; c++) R.foff[c + 1] = R.foff[c] + (size_t)np * (H - c * DF_CKPT);
        R.free_snap.resize(R.foff[nck]); R.wait_snap.resize((size_t)nck * nw); R.bnd.resize(nck);
        R.nwait.resize(H + 1); R.cv.resize(H + 1); R.dsc.assign(H + 1, 0); R.gc_suf.assign(H + 2, 0);
        R.dec_ptr.assign(H + 2, 0);
        std::vector<int> cnt(H + 1, 0);
        double dsum = 0;
        int cv = 0, ncl = 0, noff = 0;
        auto decide = [&](int j, int ty, int st, int fe, double a, double b, int slot, int kind) {
            R.dslot[j] = slot; R.type[j] = ty; R.start[j] = st; R.fend[j] = fe; R.c1[j] = a; R.c2[j] = b;
            R.kind[j] = (signed char)kind;
            dsum += S.w1 * a + S.w2 * b;
            if (slot >= 0) cnt[slot]++;
            if (kind == 2) cv++; else if (kind != 0) ncl++;
            if (kind == 3) noff++;
        };
        init(gen, R.g0, R.g1, decide);
        for (int r = 0; r < P.n; r++)
            if (R.g1[r >> 6] >> (r & 63) & 1) {
                const int j = S.order[r];
                if (!P.jobs[j].onprem.empty()) R.gc_suf[std::min(P.release[j], H)] += F.lbrel[j];
            }
        for (int t = H - 1; t >= 0; t--) R.gc_suf[t] += R.gc_suf[t + 1];
        W.free = P.free0; W.wait.assign(nw, 0); W.maxrun.assign(np, 0);
        int n_wait = 0;
        DBound lb;
        auto snap = [&](int t) {
            R.nwait[t] = n_wait; R.cv[t] = cv; R.dsc[t] = dsum;
            if (t % DF_CKPT) return;
            const int c = t / DF_CKPT;
            R.bnd[c] = lb;
            for (int q = 0; q < np; q++)
                std::copy(W.free[q].begin() + t, W.free[q].end(), R.free_snap.begin() + R.foff[c] + (size_t)q * (H - t));
            std::copy(W.wait.begin(), W.wait.end(), R.wait_snap.begin() + (size_t)c * nw);
        };
        snap(0);
        run(0, W.free, W.wait, n_wait, gen, R.g0, R.g1, W.maxrun, &lb, decide,
            [&](int t) { snap(t + 1); return false; });
        // decisions by slot (CSR)
        for (int t = 0; t <= H; t++) R.dec_ptr[t + 1] = R.dec_ptr[t] + cnt[t];
        R.dec_job.resize(R.dec_ptr[H + 1]);
        for (int j = 0; j < P.n; j++)
            if (R.dslot[j] >= 0) R.dec_job[R.dec_ptr[R.dslot[j]] + --cnt[R.dslot[j]]] = j;
        DSched& s = R.s;
        s.type = R.type; s.start = R.start;
        s.on.assign(P.n, 0);
        for (int j = 0; j < P.n; j++) s.on[j] = R.fend[j] > 0;
        s.f1 = s.f2 = 0;
        for (int j = 0; j < P.n; j++) { s.f1 += R.c1[j]; s.f2 += R.c2[j]; }
        s.cv = cv; s.n_cloud = ncl; s.n_auto_off = noff;
        s.g = S.w1 * s.f1 + S.w2 * s.f2;
        s.pen_g = pen(S, s.f1, s.f2, s.cv);
    }

    // pen_g of a full decode of gen (no recording); with a finite thr, +inf
    // once the lower bound exceeds thr.
    double full_pen(const std::vector<char>& gen, DFlipWork& W,
                    double thr = std::numeric_limits<double>::infinity()) const {
        const int H = P.H, np = (int)P.pool_type.size();
        const bool prune = std::isfinite(thr);
        if ((int)W.c1.size() != P.n) { W.c1.assign(P.n, 0); W.c2.assign(P.n, 0); }
        std::fill(W.c1.begin(), W.c1.end(), 0.0); std::fill(W.c2.begin(), W.c2.end(), 0.0);
        int cv = 0;
        double dsum = 0;
        auto decide = [&](int j, int, int, int, double a, double b, int, int kind) {
            W.c1[j] = a; W.c2[j] = b; cv += kind == 2; dsum += S.w1 * a + S.w2 * b;
        };
        init(gen, W.g0, W.g1, decide);
        if (prune) {
            W.gc_suf.assign(H + 2, 0.0);
            for (int j = 0; j < P.n; j++)
                if (gen[j] == 1 && S.cloud_opt[j] >= 0 && !P.jobs[j].onprem.empty())
                    W.gc_suf[std::min(P.release[j], H)] += F.lbrel[j];
            for (int t = H - 1; t >= 0; t--) W.gc_suf[t] += W.gc_suf[t + 1];
        }
        W.free = P.free0; W.wait.assign(F.nw, 0); W.maxrun.assign(np, 0);
        int n_wait = 0;
        DBound lb;
        const bool cut = run(0, W.free, W.wait, n_wait, gen, W.g0, W.g1, W.maxrun, prune ? &lb : nullptr, decide,
            [&](int t) {
                W.slots++;
                return prune && dsum + lb.A + lb.B * (t + 1) + lb.C + F.lbrel_suf[t + 1] - W.gc_suf[t + 1] > thr;
            });
        if (cut) { W.pruned++; return std::numeric_limits<double>::infinity(); }
        double f1 = 0, f2 = 0;
        for (int j = 0; j < P.n; j++) { f1 += W.c1[j]; f2 += W.c2[j]; }
        return pen(S, f1, f2, cv);
    }

    // pen_g of the incumbent with job j's gene set to `to` (+inf when pruned).
    double flip(const DInc& R, int j, char to, DFlipWork& W, bool prune) const {
        return flip_impl<false>(R, j, to, W, prune);
    }

    // Re-record R with job j's gene set to `to`, equal to record() of the new
    // genome (float sums up to rounding): the restart decode re-snapshots
    // slots (t*, convergence]; the earlier slots differ only by job j (queue
    // bit, its init score) and the later ones by the score / cv shift.
    void record_flip(DInc& R, int j, char to, DFlipWork& W) const {
        flip_impl<true>(R, j, to, W, false);
    }

private:
    template <bool REC>
    double flip_impl(std::conditional_t<REC, DInc&, const DInc&> R, int j, char to, DFlipWork& W, bool prune) const {
        const int H = P.H, np = (int)P.pool_type.size(), nw = F.nw;
        // the schedule does not change: only the gene (and its g0 bit)
        auto same = [&]() {
            if constexpr (REC) {
                R.gen[j] = to;
                if (!P.jobs[j].onprem.empty()) {
                    const int r = S.rank[j];
                    if (to == 0) R.g0[r >> 6] |= 1ULL << (r & 63); else R.g0[r >> 6] &= ~(1ULL << (r & 63));
                }
            }
            return R.s.pen_g;
        };
        if (P.jobs[j].onprem.empty()) return same();
        // effective gene: 1 cloud (has a cloud option), 0 auto (deadline rule on), 2 on-prem
        auto eff = [&](char x) { return (x == 1 && S.cloud_opt[j] >= 0) ? 1 : (x == 0 && deadline ? 0 : 2); };
        const int eg = eff(R.gen[j]), et = eff(to);
        if (eg == et) return same();
        // t*: the first slot the gene can change a decision
        const int d = R.dslot[j];
        int ts;
        if (et == 1) ts = d;                                  // leaves the queue: from its decision
        else if (eg == 1) ts = std::min(P.release[j], H);     // joins the queue at release
        else if (eg == 0) {                                   // auto -> on-prem
            if (R.fend[j] > 0) return same();                 //   placed: same either way
            ts = d;
        } else ts = std::min(d, F.evt0[j] == INT_MAX ? H : F.evt0[j]);   // on-prem -> auto
        ts = std::max(ts, 0) / DF_CKPT * DF_CKPT;
        if ((int)W.stamp.size() != P.n) {
            W.stamp.assign(P.n, 0); W.pstamp.assign(P.n, 0); W.sstamp.assign(P.n, 0); W.ctype.assign(P.n, 0);
            W.cstart.assign(P.n, 0); W.cfend.assign(P.n, 0); W.cslot.assign(P.n, 0); W.ckind.assign(P.n, 0);
            W.c1.assign(P.n, 0); W.c2.assign(P.n, 0);
        }
        if ((int)W.settle_at.size() != H + 1) W.settle_at.assign(H + 1, {});
        for (int t : W.touched) W.settle_at[t].clear();
        W.touched.clear();
        const int id = ++W.id, c = ts / DF_CKPT;
        W.free.resize(np);
        for (int q = 0; q < np; q++) {
            W.free[q].resize(H);
            const int* src = &R.free_snap[R.foff[c] + (size_t)q * (H - ts)];
            std::copy(src, src + (H - ts), W.free[q].begin() + ts);
        }
        W.wait.assign(R.wait_snap.begin() + (size_t)c * nw, R.wait_snap.begin() + (size_t)(c + 1) * nw);
        W.maxrun.resize(np);
        int n_wait = R.nwait[ts];
        DBound lb = R.bnd[c];
        double csum = 0;                                      // score of the candidate's decisions
        const double sj_old = S.w1 * R.c1[j] + S.w2 * R.c2[j];
        const double dbase = R.dsc[ts] - (eg == 1 ? sj_old : 0.0);
        const int relj = std::min(P.release[j], H), r = S.rank[j];
        const uint64_t bit = 1ULL << (r & 63);
        W.g0 = R.g0; W.g1 = R.g1;
        W.g0[r >> 6] &= ~bit; W.g1[r >> 6] &= ~bit;
        if (to == 0) W.g0[r >> 6] |= bit;                    // as init(); read only with the deadline rule
        W.gen = R.gen; W.gen[j] = to;
        W.decided.clear(); W.slot_dec.clear();
        int dcv = 0, n_open = 0;              // tracked jobs not settled yet
        // a tracked job settles once decided in both runs, identically or with
        // both on-prem footprints over: at the end of slot max(fend) - 1
        auto track = [&](int k) { if (W.pstamp[k] != id) { W.pstamp[k] = id; n_open++; } };
        auto check = [&](int k, int t) {
            if (W.sstamp[k] == id || W.stamp[k] != id || R.dslot[k] > t) return;
            const int at = (W.ctype[k] == R.type[k] && W.cstart[k] == R.start[k])
                               ? t : std::max({t, W.cfend[k] - 1, R.fend[k] - 1});
            W.sstamp[k] = id;
            if (at <= t) { n_open--; return; }
            if (W.settle_at[at].empty()) W.touched.push_back(at);
            W.settle_at[at].push_back(k);
        };
        auto decide = [&](int k, int ty, int st, int fe, double a, double b, int slot, int kind) {
            W.stamp[k] = id; W.ctype[k] = ty; W.cstart[k] = st; W.cfend[k] = fe; W.c1[k] = a; W.c2[k] = b;
            W.cslot[k] = slot; W.ckind[k] = (signed char)kind;
            csum += S.w1 * a + S.w2 * b;
            dcv += kind == 2;
            W.decided.push_back(k);
            W.slot_dec.push_back(k);
            track(k);
        };
        if (et == 1) {                                        // decided at init instead
            if (W.wait[r >> 6] & bit) {
                W.wait[r >> 6] &= ~bit; n_wait--;
                if (F.xc[j] < ts) lb.C -= F.cs[j]; else { lb.A -= F.al[j]; lb.B -= F.be[j]; }
            }
            W.g1[r >> 6] |= bit;
            const DOpt& o = P.jobs[j].cloud[S.cloud_opt[j]];
            decide(j, o.type, o.earliest, 0, P.jobs[j].weight * (o.earliest + o.p - P.jobs[j].arrival), o.cost, -1, 1);
            W.slot_dec.clear();
        } else track(j);
        if constexpr (REC) {
            // slots up to t*: job j leaves the queue / the init decisions
            const double dpre = (et == 1 ? csum : 0.0) - (eg == 1 ? sj_old : 0.0);
            for (int t = 0; t <= ts; t++) R.dsc[t] += dpre;
            if (et == 1)                                      // j waited over (relj, d]
                for (int t = relj + 1; t <= ts; t++) {
                    R.nwait[t]--;
                    if (t % DF_CKPT) continue;
                    const int cc = t / DF_CKPT;
                    R.wait_snap[(size_t)cc * nw + (r >> 6)] &= ~bit;
                    DBound& B = R.bnd[cc];
                    if (F.xc[j] < t) B.C -= F.cs[j]; else { B.A -= F.al[j]; B.B -= F.be[j]; }
                }
        }
        int conv_t = -1;
        bool pruned = false;
        const double thr = R.s.pen_g + 1e-9 * std::max(1.0, std::abs(R.s.pen_g));
        auto end_slot = [&](int t) {
            W.slots++;
            for (int i = R.dec_ptr[t]; i < R.dec_ptr[t + 1]; i++) { const int k = R.dec_job[i]; track(k); check(k, t); }
            for (int k : W.slot_dec) check(k, t);
            W.slot_dec.clear();
            n_open -= (int)W.settle_at[t].size();
            if (n_open == 0 && t + 1 < H) { conv_t = t; return true; }
            if constexpr (REC) {                              // snapshot the start of t + 1
                const int u = t + 1;
                R.nwait[u] = n_wait; R.cv[u] = R.cv[ts] + dcv; R.dsc[u] = dbase + csum;
                if (u % DF_CKPT == 0) {
                    const int cc = u / DF_CKPT;
                    R.bnd[cc] = lb;
                    for (int q = 0; q < np; q++)
                        std::copy(W.free[q].begin() + u, W.free[q].end(), R.free_snap.begin() + R.foff[cc] + (size_t)q * (H - u));
                    std::copy(W.wait.begin(), W.wait.end(), R.wait_snap.begin() + (size_t)cc * nw);
                }
            }
            if (prune) {
                double unrel = F.lbrel_suf[t + 1] - R.gc_suf[t + 1];
                if (relj > t) { if (et == 1) unrel -= F.lbrel[j]; if (eg == 1) unrel += F.lbrel[j]; }
                if (dbase + csum + lb.A + lb.B * (t + 1) + lb.C + unrel > thr) { pruned = true; return true; }
            }
            return false;
        };
        run(ts, W.free, W.wait, n_wait, W.gen, W.g0, W.g1, W.maxrun, (prune || REC) ? &lb : nullptr, decide, end_slot);
        if (pruned) { W.pruned++; return std::numeric_limits<double>::infinity(); }
        double f1 = R.s.f1, f2 = R.s.f2;
        for (int k : W.decided) { f1 += W.c1[k] - R.c1[k]; f2 += W.c2[k] - R.c2[k]; }
        int cv = R.cv[ts] + dcv;
        if (conv_t >= 0) { W.converged++; cv += R.s.cv - R.cv[conv_t + 1]; }
        if constexpr (REC) commit(R, j, ts, conv_t, dbase + csum, R.cv[ts] + dcv, W);
        return pen(S, f1, f2, cv);
    }

    // record_flip's tail: slots after the convergence shift by the change in
    // score / cv, the re-decided jobs replace theirs, gc_suf / the per-slot
    // index / the schedule follow.
    void commit(DInc& R, int j, int ts, int conv_t, double dsc_conv, int cv_conv, DFlipWork& W) const {
        const int H = P.H, relj = std::min(P.release[j], H), r = S.rank[j];
        if (conv_t >= 0) {
            const int u = conv_t + 1;
            const double dd = dsc_conv - R.dsc[u];
            const int dc = cv_conv - R.cv[u];
            for (int t = u; t <= H; t++) { R.dsc[t] += dd; R.cv[t] += dc; }
        }
        (void)ts;
        const bool was_gc = R.g1[r >> 6] >> (r & 63) & 1, is_gc = W.g1[r >> 6] >> (r & 63) & 1;
        if (was_gc != is_gc)
            for (int t = 0; t <= relj; t++) R.gc_suf[t] += is_gc ? F.lbrel[j] : -F.lbrel[j];
        R.gen.swap(W.gen); R.g0.swap(W.g0); R.g1.swap(W.g1);
        DSched& s = R.s;
        for (int k : W.decided) {
            const int ok = R.kind[k], nk = W.ckind[k];
            s.cv += (nk == 2) - (ok == 2); s.n_cloud += (nk == 1 || nk == 3) - (ok == 1 || ok == 3);
            s.n_auto_off += (nk == 3) - (ok == 3);
            R.dslot[k] = W.cslot[k]; R.type[k] = W.ctype[k]; R.start[k] = W.cstart[k]; R.fend[k] = W.cfend[k];
            R.c1[k] = W.c1[k]; R.c2[k] = W.c2[k]; R.kind[k] = (signed char)nk;
            s.type[k] = W.ctype[k]; s.start[k] = W.cstart[k]; s.on[k] = W.cfend[k] > 0;
        }
        std::fill(R.dec_ptr.begin(), R.dec_ptr.end(), 0);
        for (int k = 0; k < P.n; k++) if (R.dslot[k] >= 0) R.dec_ptr[R.dslot[k] + 1]++;
        for (int t = 0; t <= H; t++) R.dec_ptr[t + 1] += R.dec_ptr[t];
        R.dec_job.resize(R.dec_ptr[H + 1]);
        {
            std::vector<int>& at = W.touched;    // reuse: fill pointers
            for (int t : at) W.settle_at[t].clear();
            at.assign(R.dec_ptr.begin(), R.dec_ptr.end() - 1);
            for (int k = 0; k < P.n; k++) if (R.dslot[k] >= 0) R.dec_job[at[R.dslot[k]]++] = k;
            at.clear();
        }
        s.f1 = s.f2 = 0;
        for (int k = 0; k < P.n; k++) { s.f1 += R.c1[k]; s.f2 += R.c2[k]; }
        s.g = S.w1 * s.f1 + S.w2 * s.f2;
        s.pen_g = pen(S, s.f1, s.f2, s.cv);
    }
public:
};

struct DStats {
    long long decodes = 0;
    long long flip_slots = 0, pruned = 0, converged = 0, records = 0;   // DEngine LS
    int ls_moves = 0, kicks = 0, iters = 0;
    double mu = 0;
    double decode_ms = 0;
    std::vector<std::pair<double, double>> trace;   // (elapsed ms, best pen_g)
};

// Index of the lowest value improving on cur (by more than 1e-12), or -1. A
// later candidate must beat the best so far by a relative 1e-12: equal-valued
// candidates (interchangeable jobs) then resolve to the first index whatever
// order the decoder summed f1/f2 in, so all decode engines take the same path.
template <class Val>
inline int best_improving(size_t k, Val&& val, double cur) {
    int bi = -1;
    double bg = cur - 1e-12;
    for (int i = 0; i < (int)k; i++) {
        const double v = val(i);
        if (v < bg - (bi < 0 ? 0.0 : 1e-12 * std::max(1.0, std::abs(bg)))) { bg = v; bi = i; }
    }
    return bi;
}

// Best-improvement LS over single genome flips; leaves the result in gen/cur.
// Each step decodes candidates in batches of K until one improves, up to
// K_max (<= K: one batch). With F (and neither DF_REF_DECODE nor the serial decoders) candidates are
// decoded by DEngine: restarted from the recorded incumbent unless
// DF_NO_RESTART, pruned by the lower bound with DF_PRUNE; cur is re-recorded.
inline void dispatch_ls(const DProblem& P, const DScalar& S, std::vector<char>& gen,
                        DSched& cur, int flags, int K, int max_iters,
                        std::mt19937& rng, std::vector<DWork>& works, DStats& st,
                        TimePoint t0, const DFast* F = nullptr, int K_max = 0) {
    struct Cand { double est; int j; char to; };
    std::vector<Cand> cand;
    std::vector<DSched> res;
    const bool fast = F && !(flags & (DF_REF_DECODE | DF_SERIAL | DF_BEST_OF | DF_WHY));
    std::unique_ptr<DEngine> E;
    DInc inc;
    std::vector<double> pens;
    std::vector<DFlipWork> fws(works.size());
    if (fast) {
        E = std::make_unique<DEngine>(P, S, *F, flags);
        E->record(gen, inc, fws[0]);
        st.records++;
        cur = inc.s;
    }
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
        // K_max candidates in batches of K: a batch is decoded only when the
        // previous ones had no improving move, so the LS stops after K_max
        // non-improving candidates but an improving step costs only K
        const int k = std::min<int>(std::max(K, K_max), (int)cand.size());
        if (flags & DF_ABS_EST) for (Cand& c : cand) c.est = std::abs(c.est);
        std::vector<int> pos;              // candidate indices in batch order
        if ((flags & DF_MIX_RANK) && !(flags & DF_RANDOM_RANK)) {
            const int h = k / 2;
            std::partial_sort(cand.begin(), cand.begin() + h, cand.end(),
                              [](const Cand& a, const Cand& b) { return a.est < b.est; });
            for (int i = h; i < k; i++) {
                std::uniform_int_distribution<int> d(i, (int)cand.size() - 1);
                std::swap(cand[i], cand[d(rng)]);
            }
            // each batch: the next K/2 by estimate, then the next random ones
            const int he = std::max(K / 2, 1);
            for (int e = 0, r = h; e < h || r < k;) {
                for (int c = 0; c < he && e < h; c++) pos.push_back(e++);
                for (int c = 0; c < K - he && r < k; c++) pos.push_back(r++);
            }
        } else {
            if (flags & DF_RANDOM_RANK) {
                for (int i = 0; i < k; i++) {
                    std::uniform_int_distribution<int> d(i, (int)cand.size() - 1);
                    std::swap(cand[i], cand[d(rng)]);
                }
            } else {
                std::partial_sort(cand.begin(), cand.begin() + k, cand.end(),
                                  [](const Cand& a, const Cand& b) { return a.est < b.est; });
            }
            pos.resize(k);
            std::iota(pos.begin(), pos.end(), 0);
        }
        bool moved = false;
        for (int b0 = 0; b0 < k && !moved; b0 += K) {
            const int m = std::min(K, k - b0);
            const int* ix = pos.data() + b0;
            const TimePoint td = Clock::now();
            if (fast) {
                pens.assign(m, 0.0);
                const bool restart = !(flags & DF_NO_RESTART),
                           prune = (flags & DF_PRUNE) || ((flags & DF_PRUNE_LATE) && b0 > 0);
                const double thr = prune ? cur.pen_g + 1e-9 * std::max(1.0, std::abs(cur.pen_g))
                                         : std::numeric_limits<double>::infinity();
#ifdef _OPENMP
                #pragma omp parallel if (!omp_in_parallel())
                {
                    DFlipWork& W = fws[omp_get_thread_num()];
                    std::vector<char> g2;
                    #pragma omp for schedule(dynamic, 1)
                    for (int i = 0; i < m; i++) {
                        const Cand& c = cand[ix[i]];
                        if (restart) pens[i] = E->flip(inc, c.j, c.to, W, prune);
                        else { g2 = gen; g2[c.j] = c.to; pens[i] = E->full_pen(g2, W, thr); }
                    }
                }
#else
                for (int i = 0; i < m; i++) {
                    const Cand& c = cand[ix[i]];
                    if (restart) pens[i] = E->flip(inc, c.j, c.to, fws[0], prune);
                    else { std::vector<char> g2 = gen; g2[c.j] = c.to; pens[i] = E->full_pen(g2, fws[0], thr); }
                }
#endif
                st.decode_ms += ms_since(td);
                st.decodes += m;
                const int bi = best_improving(m, [&](int i) { return pens[i]; }, cur.pen_g);
                if (bi < 0) continue;
                gen[cand[ix[bi]].j] = cand[ix[bi]].to;
                if (restart && !(flags & DF_FULL_RECORD)) E->record_flip(inc, cand[ix[bi]].j, cand[ix[bi]].to, fws[0]);
                else E->record(gen, inc, fws[0]);
                st.records++;
                cur = inc.s;
            } else {
                res.resize(m);
#ifdef _OPENMP
                // Already inside a parallel region (one memetic child per thread):
                // decode serially on this thread's workspace.
                #pragma omp parallel if (!omp_in_parallel())
                {
                    DWork& W = works[omp_get_thread_num()];   // 0 in a nested (inactive) team
                    std::vector<char> g2;
                    #pragma omp for schedule(dynamic, 1)
                    for (int i = 0; i < m; i++) {
                        g2 = gen;
                        g2[cand[ix[i]].j] = cand[ix[i]].to;
                        decode(P, S, g2, flags, W, res[i]);
                    }
                }
#else
                for (int i = 0; i < m; i++) {
                    std::vector<char> g2 = gen;
                    g2[cand[ix[i]].j] = cand[ix[i]].to;
                    decode(P, S, g2, flags, works[0], res[i]);
                }
#endif
                st.decode_ms += ms_since(td);
                st.decodes += m;
                const int bi = best_improving(m, [&](int i) { return res[i].pen_g; }, cur.pen_g);
                if (bi < 0) continue;
                gen[cand[ix[bi]].j] = cand[ix[bi]].to;
                cur = std::move(res[bi]);
            }
            moved = true;
            st.ls_moves++;
            st.trace.emplace_back(ms_since(t0), cur.pen_g);
        }
        if (!moved) break;
    }
    for (const DFlipWork& W : fws) { st.flip_slots += W.slots; st.pruned += W.pruned; st.converged += W.converged; }
}

struct DispatchResult {
    std::vector<std::tuple<int,int>> asgn;
    DSched best;
    std::vector<char> gen;             // genome of `best`
    DStats st;
    double total_ms = 0;
};

inline DispatchResult run_weighted_dispatch(
    const DProblem& P, double w1, double w2, double f1_cap, int seed, int n_threads,
    int K, int max_iters, int n_kicks, int kick_size, int flags_in, int ls, double mu,
    int n_mu, int kick_patience = 0, int K_max = 0) {
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
    // mu = -2 marks the EDF start (set_priority's edf order)
    std::vector<std::pair<double, double>> starts;   // (decode pen_g, mu)
    if (flags_in & DF_EDF) {
        starts.push_back({0.0, -2.0});
    } else if (mu >= 0) {
        starts.push_back({0.0, mu});
    } else {
        std::vector<char> g0(P.n, 0);
        for (double m : grid) {
            set_priority(P, S, m);
            DSched c;
            decode(P, S, g0, flags_in, works[0], c);
            st.decodes++;
            starts.push_back({c.pen_g, m});
        }
        std::stable_sort(starts.begin(), starts.end());
        starts.resize(std::min<int>(std::max(n_mu, 1), (int)starts.size()));
    }
    if ((flags_in & DF_EDF_ADD) && !(flags_in & DF_EDF)) starts.push_back({0.0, -2.0});

    std::vector<char> best_gen;
    DSched best;
    double best_mu = starts[0].second;
    // per-start priority, flags and fast-decoder tables; the EDF order serves
    // jobs before their indifference deadline, so its start may drop the
    // deadline offload (DF_EDF_NODL)
    auto start_flags = [&](double m) {
        return (m == -2.0 && (flags_in & DF_EDF_NODL)) ? (flags_in | DF_NO_DEADLINE) : flags_in;
    };
    // ILS kicks from (gen, cur) of start m; kick_patience > 0 stops after that
    // many kicks in a row without improving
    auto kicks = [&](double m, int flags, const DFast& F, int nk, std::vector<char>& lbest_gen, DSched& lbest) {
        if (!ls || (flags & DF_NO_KICK)) return;
        for (int kk = 0, stale = 0; kk < nk; kk++) {
            if (kick_patience > 0 && stale >= kick_patience) break;
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
            dispatch_ls(P, S, g2, c, flags, K, max_iters, rng, works, st, t0, &F, K_max);
            if (c.pen_g < lbest.pen_g - 1e-12) {
                lbest = std::move(c); lbest_gen = std::move(g2); stale = 0;
            } else stale++;
        }
        (void)m;
    };
    auto keep = [&](double m, std::vector<char>& g, DSched& c) {
        if (best.type.empty() || c.pen_g < best.pen_g - 1e-12) {
            best = c; best_gen = g; best_mu = m;
            st.trace.emplace_back(ms_since(t0), best.pen_g);
        }
    };
    // Each start: local search, then its share of the kicks (DF_FOCUS_KICKS:
    // local search from every start, then all kicks to the best start).
    const bool split = !(flags_in & DF_FOCUS_KICKS);
    const int kicks_per = n_kicks / (int)starts.size();
    for (auto& [_, m0] : starts) {
        const double m = m0;
        set_priority(P, S, m, m == -2.0);
        const int flags = start_flags(m);
        const DFast F = build_dfast(P, S, !(flags & DF_NO_DEADLINE));
        std::vector<char> gen(P.n, 0);
        DSched cur;
        decode(P, S, gen, flags, works[0], cur);
        st.decodes++;
        if (ls) dispatch_ls(P, S, gen, cur, flags, K, max_iters, rng, works, st, t0, &F, K_max);
        if (split) kicks(m, flags, F, kicks_per, gen, cur);
        keep(m, gen, cur);
    }
    if (!split) {
        const double m = best_mu;
        set_priority(P, S, m, m == -2.0);
        const int flags = start_flags(m);
        const DFast F = build_dfast(P, S, !(flags & DF_NO_DEADLINE));
        std::vector<char> gen = best_gen;
        DSched cur = best;
        kicks(m, flags, F, n_kicks, gen, cur);
        keep(m, gen, cur);
    }
    st.mu = best_mu;

    DispatchResult R;
    R.asgn.resize(P.n);
    for (int j = 0; j < P.n; j++) R.asgn[j] = {best.type[j], best.start[j]};
    R.best = std::move(best);
    R.gen = std::move(best_gen);
    R.st = std::move(st);
    R.total_ms = ms_since(t0);
    return R;
}
