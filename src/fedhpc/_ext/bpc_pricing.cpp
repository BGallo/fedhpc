// Label-setting pricing for fedhpc's branch-and-price master (bpc.py).
//
// One Pricer per (instance type, B&B node). A column is one instance's
// timeline: a sequence of (job, start) arcs in the type's space-time DAG over
// slots 0..H, starting from a release-group source (r, cost -rho_r); waiting
// is free. An arc of local job i started at t has reduced cost A_i + B_i * t
// (bpc.py folds every dual into A and B, so B_i >= 0 is not assumed).
//
// Memory (ng-routes, Baldacci et al. 2011). After visiting job i a label
// remembers  (mask & retain_i) | {i}  and may not visit a remembered job
// again. retain_i = all jobs is the elementary DP; a smaller neighbourhood
// N(i) allows some repeat visits, so columns may cover a job twice (the
// master gives them multiplicity coefficients). Jobs of a Ryan-Foster
// differ/same pair are always remembered, so those rules are exact.
//
// Labels are never copied along wait arcs: a label created at slot t is
// "active" from t on, and every active label tries every arc starting at the
// current slot. Dominance (cost <=, mask subset, same same-pair parity) is
// checked within a slot's arrivals and against the active set.
//
// Limited-memory subset-row cuts (Pecin et al. 2017), at most 64. Cut c has
// a job set S_c and a memory M_c >= S_c; a label carries one state bit per
// cut. Visiting i: the state drops to 0 if i is outside M_c; if i is in S_c
// a set bit pays cost_c (= -dual >= 0) and clears, a clear bit sets. A label
// a dominates b only if cost_a + sum of cost_c over cuts set in a but not in
// b is <= cost_b: a pays each such cut at most once more than b would.
// Cut costs are >= 0, so g() and fixable() below stay valid lower bounds.
//
// Completion bound. g(t) = min cost from slot t to H over all paths, repeats
// allowed (one backward pass); g is nondecreasing in t. A label with
// cost + g(t) >= thr (thr = -tol) can never yield an improving column, so it
// is retired; a newly built label below that line is kept. min_lb, the
// minimum over the best complete consistent label and every retired bound,
// is a valid lower bound on the true minimum reduced cost (exact when the
// minimum is below thr) — the Lagrangian bound in bpc.py uses it.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include <immintrin.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using u64 = std::uint64_t;

namespace {

using Path = std::vector<std::pair<int, int>>;                 // (local job, start)
using ColOut = std::tuple<double, int, Path>;                   // (rc, release, path)
using PriceOut = std::tuple<double, bool, long long, std::vector<ColOut>>;
// price_cols: (rc, release, path in global job ids, cost1, cost2, rc at the eval duals)
using RichCol = std::tuple<double, int, Path, double, double, double>;
using RichOut = std::tuple<double, bool, long long, std::vector<RichCol>>;

constexpr double EPS = 1e-9;

// ---------------------------------------------------------------------------
// Dominance scans for one-word masks without cuts or same pairs (the SoA
// path, Pricer::run_soa). Entry k dominates x when ce[k] <= c + EPS and
// mask[k] is a subset of x's; x dominates entry k when c <= cost[k] + EPS
// and x's mask is a subset of mask[k]. The AVX2 versions do the scalar
// arithmetic four entries at a time, so they agree bit for bit.
// ---------------------------------------------------------------------------
inline bool any_dom_scalar(const double* ce, const u64* m, size_t n, double c, u64 mx) {
    const double cv = c + EPS;
    for (size_t k = 0; k < n; ++k)
        if (ce[k] <= cv && !(m[k] & ~mx)) return true;
    return false;
}
inline bool any_domd_scalar(const double* co, const u64* m, size_t n, double c, u64 mx) {
    for (size_t k = 0; k < n; ++k)
        if (c <= co[k] + EPS && !(mx & ~m[k])) return true;
    return false;
}
__attribute__((target("avx2"))) inline
bool any_dom_avx2(const double* ce, const u64* m, size_t n, double c, u64 mx) {
    const __m256d cv = _mm256_set1_pd(c + EPS);
    const __m256i xv = _mm256_set1_epi64x((long long)mx);
    const __m256i z = _mm256_setzero_si256();
    size_t k = 0;
    for (; k + 4 <= n; k += 4) {
        __m256d le = _mm256_cmp_pd(_mm256_loadu_pd(ce + k), cv, _CMP_LE_OQ);
        __m256i bad = _mm256_andnot_si256(xv, _mm256_loadu_si256((const __m256i*)(m + k)));
        __m256d ok = _mm256_castsi256_pd(_mm256_cmpeq_epi64(bad, z));
        if (_mm256_movemask_pd(_mm256_and_pd(le, ok))) return true;
    }
    return any_dom_scalar(ce + k, m + k, n - k, c, mx);
}
__attribute__((target("avx2"))) inline
bool any_domd_avx2(const double* co, const u64* m, size_t n, double c, u64 mx) {
    const __m256d cv = _mm256_set1_pd(c), ev = _mm256_set1_pd(EPS);
    const __m256i xv = _mm256_set1_epi64x((long long)mx);
    const __m256i z = _mm256_setzero_si256();
    size_t k = 0;
    for (; k + 4 <= n; k += 4) {
        __m256d le = _mm256_cmp_pd(cv, _mm256_add_pd(_mm256_loadu_pd(co + k), ev), _CMP_LE_OQ);
        __m256i bad = _mm256_andnot_si256(_mm256_loadu_si256((const __m256i*)(m + k)), xv);
        __m256d ok = _mm256_castsi256_pd(_mm256_cmpeq_epi64(bad, z));
        if (_mm256_movemask_pd(_mm256_and_pd(le, ok))) return true;
    }
    return any_domd_scalar(co + k, m + k, n - k, c, mx);
}
const bool HAS_AVX2 = __builtin_cpu_supports("avx2");

// Structure-of-arrays label entries for the SoA path: cost, effective cost
// against a non-empty label (cost + first for an empty one), mask, arena id,
// empty flag.
struct Soa {
    std::vector<double> cost, ceff;
    std::vector<u64> mask;
    std::vector<int> id;
    std::vector<char> src;
    size_t n() const { return cost.size(); }
    void push(double c, double ce, u64 m, int i, char s) {
        cost.push_back(c), ceff.push_back(ce), mask.push_back(m), id.push_back(i), src.push_back(s);
    }
    void insert(size_t pos, double c, double ce, u64 m, int i, char s) {
        cost.insert(cost.begin() + pos, c), ceff.insert(ceff.begin() + pos, ce);
        mask.insert(mask.begin() + pos, m), id.insert(id.begin() + pos, i);
        src.insert(src.begin() + pos, s);
    }
    void move(size_t w, size_t k) {
        cost[w] = cost[k], ceff[w] = ceff[k], mask[w] = mask[k], id[w] = id[k], src[w] = src[k];
    }
    void resize(size_t n) {
        cost.resize(n), ceff.resize(n), mask.resize(n), id.resize(n), src.resize(n);
    }
    void release() { Soa().swap(*this); }
    void swap(Soa& o) {
        cost.swap(o.cost), ceff.swap(o.ceff), mask.swap(o.mask), id.swap(o.id), src.swap(o.src);
    }
};

class Pricer {
public:
    Pricer(int H, std::vector<int> p, const std::vector<std::vector<int>>& starts,
           const std::vector<std::pair<int, int>>& differ,
           const std::vector<std::pair<int, int>>& same)
        : H_(H), n_((int)p.size()), p_(std::move(p)) {
        if ((int)starts.size() != n_) throw std::invalid_argument("starts length != n");
        if (n_ > 64 * 32) throw std::invalid_argument("more than 2048 jobs on one type");
        for (int q : p_)
            if (q < 1) throw std::invalid_argument("processing times must be >= 1");
        nw_ = 1;
        while (64 * nw_ < n_) nw_ *= 2;
        at_.assign(H_ + 1, {});
        for (int i = 0; i < n_; ++i)
            for (int t : starts[i])
                if (t >= 0 && t < H_ && t + p_[i] <= H_) at_[t].push_back(i);
        keep_.assign(nw_, 0);
        differ_.assign((size_t)n_ * nw_, 0);
        for (auto [a, b] : differ) {
            check(a), check(b);
            set_bit(&differ_[(size_t)a * nw_], b);
            set_bit(&differ_[(size_t)b * nw_], a);
            set_bit(keep_.data(), a), set_bit(keep_.data(), b);
        }
        if (same.size() > 64) throw std::invalid_argument("more than 64 same pairs");
        same_empty_ = same.empty();
        toggle_.assign(n_, 0);
        for (size_t s = 0; s < same.size(); ++s) {
            auto [a, b] = same[s];
            check(a), check(b);
            toggle_[a] ^= u64(1) << s;
            toggle_[b] ^= u64(1) << s;
            set_bit(keep_.data(), a), set_bit(keep_.data(), b);
        }
        set_memory(std::nullopt);
        inS_.assign(n_, 0);
        inM_.assign(n_, 0);
    }

    // cuts[c] = (S_c, M_c) in local job indices; M_c is extended by S_c.
    void set_cuts(const std::vector<std::pair<std::vector<int>, std::vector<int>>>& cuts) {
        if (cuts.size() > 64) throw std::invalid_argument("more than 64 subset-row cuts");
        nc_ = (int)cuts.size();
        inS_.assign(n_, 0);
        inM_.assign(n_, 0);
        for (int c = 0; c < nc_; ++c) {
            const u64 bit = u64(1) << c;
            for (int i : cuts[c].first) check(i), inS_[i] |= bit, inM_[i] |= bit;
            for (int i : cuts[c].second) check(i), inM_[i] |= bit;
        }
    }

    // ng[i] = local jobs remembered after visiting i (besides i itself);
    // None = elementary (remember everything).
    void set_memory(const std::optional<std::vector<std::vector<int>>>& ng) {
        retain_.assign((size_t)n_ * nw_, 0);
        for (int i = 0; i < n_; ++i) {
            u64* r = &retain_[(size_t)i * nw_];
            if (!ng) {
                for (int j = 0; j < n_; ++j) set_bit(r, j);
            } else {
                if ((int)ng->size() != n_) throw std::invalid_argument("ng length != n");
                for (int j : (*ng)[i]) check(j), set_bit(r, j);
                for (int w = 0; w < nw_; ++w) r[w] |= keep_[w];
                set_bit(r, i);
            }
        }
    }

    // first_cost: added once to every non-empty path (on its first arc);
    // must be >= 0 so g() stays a valid completion bound.
    PriceOut price(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<std::pair<int, double>>& sources, int max_cols,
                   double tol, long long max_labels, double first_cost,
                   const std::vector<double>& cut_cost) {
        if ((int)A.size() != n_ || (int)B.size() != n_)
            throw std::invalid_argument("dual vector length != n");
        if (first_cost < 0) throw std::invalid_argument("first_cost must be >= 0");
        if ((int)cut_cost.size() != nc_) throw std::invalid_argument("cut_cost length != cuts");
        for (double c : cut_cost)
            if (c < 0) throw std::invalid_argument("cut costs must be >= 0");
        first_ = first_cost;
        ccost_ = cut_cost;
        switch (nw_) {
            case 1: return run<1>(A, B, sources, max_cols, tol, max_labels);
            case 2: return run<2>(A, B, sources, max_cols, tol, max_labels);
            case 4: return run<4>(A, B, sources, max_cols, tol, max_labels);
            case 8: return run<8>(A, B, sources, max_cols, tol, max_labels);
            case 16: return run<16>(A, B, sources, max_cols, tol, max_labels);
            default: return run<32>(A, B, sources, max_cols, tol, max_labels);
        }
    }

    int n_jobs() const { return n_; }

    // DP kernel: 0 auto (the SoA path with AVX2 scans when it applies and
    // the CPU has AVX2), 1 the AoS path, 2 SoA with scalar scans, 3 SoA with
    // AVX2 scans. The SoA path needs one mask word (<= 64 jobs) and no cuts
    // or same pairs; elsewhere every kernel runs the AoS path. Results do
    // not depend on it.
    void set_kernel(int k) {
        if (k < 0 || k > 3) throw std::invalid_argument("kernel must be 0..3");
        if (k == 3 && !HAS_AVX2) throw std::invalid_argument("this CPU has no AVX2");
        kernel_ = k;
    }

    // Per local job: its global id and cost terms (an arc of job i started
    // at t adds c1a[i] + c1b[i] * t to f1 and c2[i] to f2), so price_cols
    // can hand back finished columns.
    void set_info(std::vector<int> job_ids, std::vector<double> c1a, std::vector<double> c1b,
                  std::vector<double> c2) {
        if ((int)job_ids.size() != n_ || (int)c1a.size() != n_ || (int)c1b.size() != n_
            || (int)c2.size() != n_)
            throw std::invalid_argument("info length != n");
        gid_ = std::move(job_ids), c1a_ = std::move(c1a), c1b_ = std::move(c1b), c2_ = std::move(c2);
    }

    // price(), then each column with its global path, cost1, cost2 and its
    // reduced cost at a second dual vector (eA, eB, e_sources, e_first;
    // NaN when eA is empty) — what column generation filters smoothed
    // columns by.
    RichOut price_cols(const std::vector<double>& A, const std::vector<double>& B,
                       const std::vector<std::pair<int, double>>& sources, int max_cols,
                       double tol, long long max_labels, double first_cost,
                       const std::vector<double>& cut_cost, const std::vector<double>& eA,
                       const std::vector<double>& eB,
                       const std::vector<std::pair<int, double>>& e_sources, double e_first) {
        if ((int)gid_.size() != n_) throw std::invalid_argument("set_info not called");
        const bool ev = !eA.empty();
        if (ev && ((int)eA.size() != n_ || (int)eB.size() != n_))
            throw std::invalid_argument("eval dual vector length != n");
        auto [min_lb, exact, n_labels, raw] =
            price(A, B, sources, max_cols, tol, max_labels, first_cost, cut_cost);
        std::vector<RichCol> out;
        out.reserve(raw.size());
        for (auto& [rc, release, path] : raw) {
            double c1 = 0.0, c2 = 0.0, er = 0.0;
            for (auto& [i, t] : path) {
                c1 += c1a_[i] + c1b_[i] * t;
                c2 += c2_[i];
                if (ev) er += eA[i] + eB[i] * t;
                i = gid_[i];
            }
            if (ev) {
                er += e_first;
                double src = std::numeric_limits<double>::infinity();
                for (auto [r, c] : e_sources)
                    if (r == release) src = c;
                er += src;
            } else {
                er = std::numeric_limits<double>::quiet_NaN();
            }
            out.emplace_back(rc, release, std::move(path), c1, c2, er);
        }
        return {min_lb, exact, n_labels, std::move(out)};
    }

    // Reduced-cost fixing: every arc (i, t) whose cheapest path through it —
    // repeats allowed, first_cost ignored, so a lower bound on any column
    // using it — has reduced cost >= threshold.
    //   through(i, t) = F(t) + A_i + B_i t + g(t + p_i)
    // F(t): cheapest prefix from a source to slot t; g: cheapest completion.
    std::vector<std::pair<int, int>> fixable(
            const std::vector<double>& A, const std::vector<double>& B,
            const std::vector<std::pair<int, double>>& sources, double threshold) const {
        if ((int)A.size() != n_ || (int)B.size() != n_)
            throw std::invalid_argument("dual vector length != n");
        const double INF = std::numeric_limits<double>::infinity();
        std::vector<double> g(H_ + 1, 0.0), F(H_ + 1, INF);
        for (int t = H_ - 1; t >= 0; --t) {
            double v = g[t + 1];
            for (int i : at_[t]) v = std::min(v, A[i] + B[i] * t + g[t + p_[i]]);
            g[t] = v;
        }
        for (auto [r, c] : sources) {
            int s = std::min(std::max(r, 0), H_);
            F[s] = std::min(F[s], c);
        }
        for (int t = 0; t < H_; ++t) {
            if (F[t] == INF) continue;
            F[t + 1] = std::min(F[t + 1], F[t]);
            for (int i : at_[t]) {
                int t2 = t + p_[i];
                F[t2] = std::min(F[t2], F[t] + A[i] + B[i] * t);
            }
        }
        std::vector<std::pair<int, int>> out;
        for (int t = 0; t < H_; ++t)
            for (int i : at_[t]) {
                double through = F[t] + A[i] + B[i] * t + g[t + p_[i]];
                if (through >= threshold) out.emplace_back(i, t);
            }
        return out;
    }

private:
    int H_, n_, nw_, nc_ = 0;
    bool same_empty_ = true;
    long long label_hint_ = 0;
    int kernel_ = 0;
    std::vector<int> gid_;
    std::vector<double> c1a_, c1b_, c2_;   // labels of the previous call: the arena's reserve
    double first_ = 0.0;
    std::vector<u64> inS_, inM_;   // per job: cuts whose S / M contain it
    std::vector<double> ccost_;
    std::vector<int> p_;
    std::vector<std::vector<int>> at_;   // at_[t]: local jobs startable at t
    std::vector<u64> keep_, differ_, retain_, toggle_;

    void check(int i) const {
        if (i < 0 || i >= n_) throw std::invalid_argument("local job index out of range");
    }
    static void set_bit(u64* m, int i) { m[i >> 6] |= u64(1) << (i & 63); }
    static bool has_bit(const u64* m, int i) { return (m[i >> 6] >> (i & 63)) & 1; }
    template <int NW>
    static bool only_bit(const u64* m, int i) {
        for (int w = 0; w < NW; ++w)
            if (m[w] != (w == (i >> 6) ? u64(1) << (i & 63) : 0)) return false;
        return true;
    }

    // Label storage for path reconstruction and column output. Dominance
    // works on compact Ent copies (below), kept contiguous in the active set
    // and the buckets, so the hot scans read memory in order.
    struct Arena {
        std::vector<double> cost;
        std::vector<int> pred, job, start;
        std::vector<char> alive;
        int add(double c, int pr, int j, int s) {
            cost.push_back(c), pred.push_back(pr), job.push_back(j), start.push_back(s);
            alive.push_back(1);
            return (int)cost.size() - 1;
        }
        void reserve(size_t n) {
            cost.reserve(n), pred.reserve(n), job.reserve(n), start.reserve(n), alive.reserve(n);
        }
    };

    // One label's dominance state: cost, memory mask, whether it is an
    // empty (source) label, pending same-pair parity and cut state bits,
    // and its arena id.
    template <int NW>
    struct Ent {
        double cost;
        int id;
        int src;
        u64 pend, cs;
        u64 mask[NW];
    };

    // a dominates b: cost <= (plus the cuts a may still pay and b not),
    // mask(a) subset of mask(b), same pending parity. An empty label still
    // owes `first` on its first arc, so it competes at cost + first against
    // non-empty labels. PLAIN (no cuts, no same pairs): pend and cs are 0.
    template <int NW, bool PLAIN>
    bool dom(const Ent<NW>& a, const Ent<NW>& b) const {
        double ca = a.cost + (a.src && !b.src ? first_ : 0.0);
        if (ca > b.cost + EPS) return false;
        if constexpr (!PLAIN) {
            if (a.pend != b.pend) return false;
            for (u64 x = a.cs & ~b.cs; x; x &= x - 1) {
                ca += ccost_[__builtin_ctzll(x)];
                if (ca > b.cost + EPS) return false;
            }
        }
        u64 bad = 0;
        for (int w = 0; w < NW; ++w) bad |= a.mask[w] & ~b.mask[w];
        return bad == 0;
    }

    template <int NW>
    PriceOut run(const std::vector<double>& A, const std::vector<double>& B,
                 const std::vector<std::pair<int, double>>& sources, int max_cols, double tol,
                 long long max_labels) {
        // Without cuts or same pairs every label's pending and cut state is 0.
        if constexpr (NW == 1)
            if (nc_ == 0 && same_empty_ && kernel_ != 1)
                return run_soa(A, B, sources, max_cols, tol, max_labels,
                               kernel_ == 3 || (kernel_ == 0 && HAS_AVX2));
        if (nc_ == 0 && same_empty_) return run_impl<NW, true>(A, B, sources, max_cols, tol, max_labels);
        return run_impl<NW, false>(A, B, sources, max_cols, tol, max_labels);
    }

    template <int NW, bool PLAIN>
    PriceOut run_impl(const std::vector<double>& A, const std::vector<double>& B,
                      const std::vector<std::pair<int, double>>& sources, int max_cols,
                      double tol, long long max_labels) {
        using E = Ent<NW>;
        const double INF = std::numeric_limits<double>::infinity();
        const double thr = -tol;
        // completion bound g(t), repeats allowed
        std::vector<double> g(H_ + 1, 0.0);
        for (int t = H_ - 1; t >= 0; --t) {
            double v = g[t + 1];
            for (int i : at_[t]) v = std::min(v, A[i] + B[i] * t + g[t + p_[i]]);
            g[t] = v;
        }

        Arena L;
        L.reserve(std::min<long long>(label_hint_ + label_hint_ / 4 + 64, max_labels));
        std::vector<std::vector<E>> bucket(H_ + 1);
        std::vector<E> active, local;
        std::vector<int> cand;
        double pruned_min = INF;     // lower bound over everything not enumerated
        double best = INF;           // best complete consistent label
        bool exact = true;

        // Bucket insertion (dominance within the arrivals of slot t2). A
        // candidate is first checked against the earlier survivors of the
        // same (slot, job) — scanned cheapest first, with masks drawn from
        // that job's small memory, they dominate nearly every candidate —
        // and takes an arena slot only if it survives.
        auto insert = [&](int t2, E& x, int pr, int j, int s, bool ext) -> bool {
            if (ext)
                for (const E& y : local)
                    if (L.alive[y.id] && dom<NW, PLAIN>(y, x)) return false;
            auto& bk = bucket[t2];
            for (size_t k = 0; k < bk.size();) {
                if (dom<NW, PLAIN>(bk[k], x)) return false;
                if (dom<NW, PLAIN>(x, bk[k])) {
                    L.alive[bk[k].id] = 0;
                    bk[k] = bk.back();
                    bk.pop_back();
                } else {
                    ++k;
                }
            }
            x.id = L.add(x.cost, pr, j, s);
            bk.push_back(x);
            if (ext) local.push_back(x);
            return true;
        };

        for (auto [r, c] : sources) {
            E x{};
            x.cost = c, x.src = 1;
            insert(std::min(std::max(r, 0), H_), x, -1, -1, r, false);
        }

        auto by_cost = [](const E& x, double v) { return x.cost < v; };
        for (int t = 0; t <= H_; ++t) {
            // Merge this slot's arrivals. `active` stays sorted by cost: a
            // label that dominates x costs at most x.cost + EPS, one x
            // dominates at least x.cost - EPS (an empty label pays `first`
            // on top, which only makes it costlier).
            for (E& x : bucket[t]) {
                const double c = x.cost;
                size_t hi = std::upper_bound(active.begin(), active.end(), c + EPS,
                                             [](double v, const E& y) { return v < y.cost; })
                            - active.begin();
                bool dominated = false;
                for (size_t k = 0; k < hi; ++k)
                    if (dom<NW, PLAIN>(active[k], x)) { dominated = true; break; }
                if (dominated) { L.alive[x.id] = 0; continue; }
                size_t lo = std::lower_bound(active.begin(), active.end(), c - EPS, by_cost)
                            - active.begin();
                size_t w = lo;
                for (size_t k = lo; k < active.size(); ++k) {
                    if (dom<NW, PLAIN>(x, active[k])) L.alive[active[k].id] = 0;
                    else active[w++] = active[k];
                }
                active.resize(w);
                active.insert(std::lower_bound(active.begin() + lo, active.end(), c, by_cost), x);
                if (x.pend == 0) {
                    best = std::min(best, c);
                    if (c < thr) cand.push_back(x.id);
                }
            }
            std::vector<E>().swap(bucket[t]);
            // retire labels that can no longer improve
            {
                size_t w = 0;
                for (size_t k = 0; k < active.size(); ++k) {
                    double lb = active[k].cost + g[t];
                    if (lb >= thr || t == H_) {
                        if (lb >= thr) pruned_min = std::min(pruned_min, lb);
                    } else {
                        active[w++] = active[k];
                    }
                }
                active.resize(w);
            }
            if (t == H_ || !exact) break;
            if (at_[t].empty()) continue;
            // Cheapest first (active is sorted): an extension costs at least
            // cost + arc + g(t2) (first_cost and cut costs are >= 0), so the
            // first label whose optimistic bound fails ends the scan for this
            // arc — that bound also lower-bounds every later label's, keeping
            // min_lb valid.
            for (int i : at_[t]) {
                const int t2 = t + p_[i];
                const double arc = A[i] + B[i] * t;
                const double rest = arc + g[t2];
                const u64* ret = &retain_[(size_t)i * NW];
                const u64* dif = &differ_[(size_t)i * NW];
                const u64 cS = inS_[i], cM = inM_[i];
                const int iw = i >> 6;
                const u64 ib = u64(1) << (i & 63);
                local.clear();
                for (size_t k = 0; k < active.size(); ++k) {
                    const E& a = active[k];
                    if (a.cost + rest >= thr) {
                        pruned_min = std::min(pruned_min, a.cost + rest);
                        break;
                    }
                    if (a.mask[iw] & ib) continue;
                    u64 clash = 0;
                    for (int w = 0; w < NW; ++w) clash |= a.mask[w] & dif[w];
                    if (clash) continue;
                    E x;
                    x.cost = a.cost + arc + (a.src ? first_ : 0.0);
                    x.src = 0;
                    if constexpr (PLAIN) {
                        x.pend = 0, x.cs = 0;
                    } else {
                        u64 st = a.cs & cM;
                        for (u64 y = st & cS; y; y &= y - 1) x.cost += ccost_[__builtin_ctzll(y)];
                        x.cs = st ^ cS;
                        x.pend = a.pend ^ toggle_[i];
                    }
                    double lb = x.cost + g[t2];
                    if (lb >= thr) {
                        pruned_min = std::min(pruned_min, lb);
                        continue;
                    }
                    if ((long long)L.cost.size() >= max_labels) { exact = false; break; }
                    for (int w = 0; w < NW; ++w) x.mask[w] = a.mask[w] & ret[w];
                    x.mask[iw] |= ib;
                    const bool src = a.src;
                    if (insert(t2, x, a.id, i, t, true) && PLAIN && !src && only_bit<NW>(x.mask, i))
                        break;   // a stored {i}-memory extension dominates every later one
                }
                if (!exact) break;
            }
        }

        return finish(L, cand, exact ? std::min(best, pruned_min) : -INF, exact, max_cols);
    }

    // The DP of run_impl<1, true> on structure-of-arrays storage, with the
    // dominance scans done by any_dom/any_domd (AVX2 or scalar). Buckets
    // are antichains, so "is the candidate dominated" and "whom does it
    // dominate" can be two passes. The per-(slot, job) survivor list of
    // run_impl is not needed: its entries are in the bucket being scanned.
    PriceOut run_soa(const std::vector<double>& A, const std::vector<double>& B,
                     const std::vector<std::pair<int, double>>& sources, int max_cols,
                     double tol, long long max_labels, bool avx2) {
        return avx2 ? run_soa_avx2(A, B, sources, max_cols, tol, max_labels)
                    : run_soa_body<false>(A, B, sources, max_cols, tol, max_labels);
    }
    // The whole DP compiled for AVX2, so the scans inline.
    __attribute__((target("avx2")))
    PriceOut run_soa_avx2(const std::vector<double>& A, const std::vector<double>& B,
                          const std::vector<std::pair<int, double>>& sources, int max_cols,
                          double tol, long long max_labels) {
        return run_soa_body<true>(A, B, sources, max_cols, tol, max_labels);
    }
    template <bool AVX>
    __attribute__((always_inline)) inline
    PriceOut run_soa_body(const std::vector<double>& A, const std::vector<double>& B,
                          const std::vector<std::pair<int, double>>& sources, int max_cols,
                          double tol, long long max_labels) {
        auto any_dom = [](const double* ce, const u64* m, size_t n, double c, u64 mx) {
            if constexpr (AVX) return any_dom_avx2(ce, m, n, c, mx);
            else return any_dom_scalar(ce, m, n, c, mx);
        };
        auto any_domd = [](const double* co, const u64* m, size_t n, double c, u64 mx) {
            if constexpr (AVX) return any_domd_avx2(co, m, n, c, mx);
            else return any_domd_scalar(co, m, n, c, mx);
        };
        const double INF = std::numeric_limits<double>::infinity();
        const double thr = -tol;
        std::vector<double> g(H_ + 1, 0.0);
        for (int t = H_ - 1; t >= 0; --t) {
            double v = g[t + 1];
            for (int i : at_[t]) v = std::min(v, A[i] + B[i] * t + g[t + p_[i]]);
            g[t] = v;
        }
        Arena L;
        L.reserve(std::min<long long>(label_hint_ + label_hint_ / 4 + 64, max_labels));
        std::vector<Soa> bucket(H_ + 1);
        Soa act;
        std::vector<int> cand;
        double pruned_min = INF, best = INF;
        bool exact = true;

        // an empty (source) label x: a dominates it iff a.cost <= c + EPS
        // and a is empty too; it dominates b iff c (+ first if b is not
        // empty) <= b.cost + EPS
        for (auto [r, c] : sources) {
            Soa& bk = bucket[std::min(std::max(r, 0), H_)];
            bool dominated = false;
            for (size_t k = 0; k < bk.n(); ++k)
                if (bk.cost[k] <= c + EPS && bk.mask[k] == 0) { dominated = true; break; }
            if (dominated) continue;
            size_t w = 0;
            for (size_t k = 0; k < bk.n(); ++k) {
                if (c + (bk.src[k] ? 0.0 : first_) <= bk.cost[k] + EPS) L.alive[bk.id[k]] = 0;
                else bk.move(w++, k);
            }
            bk.resize(w);
            bk.push(c, c + first_, 0, L.add(c, -1, -1, r), 1);
        }

        for (int t = 0; t <= H_; ++t) {
            Soa& bk = bucket[t];
            for (size_t e = 0; e < bk.n(); ++e) {
                const double c = bk.cost[e];
                const u64 m = bk.mask[e];
                const char s = bk.src[e];
                const size_t na = act.n();
                size_t hi = std::upper_bound(act.cost.begin(), act.cost.end(), c + EPS) - act.cost.begin();
                if (any_dom(s ? act.cost.data() : act.ceff.data(), act.mask.data(), hi, c, m)) {
                    L.alive[bk.id[e]] = 0;
                    continue;
                }
                size_t lo = std::lower_bound(act.cost.begin(), act.cost.end(), c - EPS) - act.cost.begin();
                if (s || any_domd(act.cost.data() + lo, act.mask.data() + lo, na - lo, c, m)) {
                    size_t w = lo;
                    for (size_t k = lo; k < na; ++k) {
                        double cx = c + (s && !act.src[k] ? first_ : 0.0);
                        if (cx <= act.cost[k] + EPS && !(m & ~act.mask[k])) L.alive[act.id[k]] = 0;
                        else act.move(w++, k);
                    }
                    act.resize(w);
                }
                size_t pos = std::lower_bound(act.cost.begin() + lo, act.cost.end(), c) - act.cost.begin();
                act.insert(pos, c, bk.ceff[e], m, bk.id[e], s);
                best = std::min(best, c);
                if (c < thr) cand.push_back(bk.id[e]);
            }
            bk.release();
            // retire: act is sorted by cost, so the labels with cost + g(t)
            // >= thr are a suffix, and its first entry has the least bound
            {
                const double gt = g[t];
                size_t cut = std::partition_point(act.cost.begin(), act.cost.end(),
                                                  [&](double c) { return c + gt < thr; })
                             - act.cost.begin();
                if (cut < act.n()) pruned_min = std::min(pruned_min, act.cost[cut] + gt);
                act.resize(t == H_ ? 0 : cut);
            }
            if (t == H_ || !exact) break;
            for (int i : at_[t]) {
                const int t2 = t + p_[i];
                const double arc = A[i] + B[i] * t;
                const double rest = arc + g[t2];
                const u64 ret = retain_[i], dif = differ_[i], ib = u64(1) << i;
                Soa& out = bucket[t2];
                for (size_t k = 0; k < act.n(); ++k) {
                    const double ac = act.cost[k];
                    if (ac + rest >= thr) {
                        pruned_min = std::min(pruned_min, ac + rest);
                        break;
                    }
                    const u64 am = act.mask[k];
                    if ((am & ib) || (am & dif)) continue;
                    const double c = ac + arc + (act.src[k] ? first_ : 0.0);
                    const double lb = c + g[t2];
                    if (lb >= thr) {
                        pruned_min = std::min(pruned_min, lb);
                        continue;
                    }
                    if ((long long)L.cost.size() >= max_labels) { exact = false; break; }
                    const u64 nm = (am & ret) | ib;
                    const size_t no = out.n();
                    if (any_dom(out.ceff.data(), out.mask.data(), no, c, nm)) continue;
                    if (any_domd(out.cost.data(), out.mask.data(), no, c, nm)) {
                        size_t w = 0;
                        for (size_t q = 0; q < no; ++q) {
                            if (c <= out.cost[q] + EPS && !(nm & ~out.mask[q])) L.alive[out.id[q]] = 0;
                            else out.move(w++, q);
                        }
                        out.resize(w);
                    }
                    out.push(c, c, nm, L.add(c, act.id[k], i, t), 0);
                    if (!act.src[k] && nm == ib) break;   // dominates every later candidate
                }
                if (!exact) break;
            }
        }
        return finish(L, cand, exact ? std::min(best, pruned_min) : -INF, exact, max_cols);
    }

    // The max_cols cheapest distinct candidate columns: a heap, not a full sort.
    PriceOut finish(Arena& L, std::vector<int>& cand, double min_lb, bool exact, int max_cols) {
        label_hint_ = (long long)L.cost.size();
        auto costlier = [&](int a, int b) { return L.cost[a] > L.cost[b]; };
        std::make_heap(cand.begin(), cand.end(), costlier);
        std::vector<ColOut> cols;
        std::unordered_set<std::size_t> seen;
        while (!cand.empty() && (int)cols.size() < max_cols) {
            std::pop_heap(cand.begin(), cand.end(), costlier);
            const int id = cand.back();
            cand.pop_back();
            Path path;
            int x = id;
            while (L.job[x] >= 0) {
                path.emplace_back(L.job[x], L.start[x]);
                x = L.pred[x];
            }
            if (path.empty()) continue;
            std::reverse(path.begin(), path.end());
            std::size_t h = std::hash<int>()(L.start[x]);
            for (auto [j, s] : path) h = h * 1000003u ^ (std::size_t)(j * 65537 + s);
            if (!seen.insert(h).second) continue;
            cols.emplace_back(L.cost[id], L.start[x], std::move(path));
        }
        return {min_lb, exact, (long long)L.cost.size(), std::move(cols)};
    }
};

}  // namespace

PYBIND11_MODULE(_bpcprice, mod) {
    mod.doc() = "Label-setting pricing (elementary or ng-route) for fedhpc's BPC master";
    py::class_<Pricer>(mod, "Pricer")
        .def(py::init<int, std::vector<int>, const std::vector<std::vector<int>>&,
                      const std::vector<std::pair<int, int>>&,
                      const std::vector<std::pair<int, int>>&>(),
             py::arg("horizon"), py::arg("p"), py::arg("starts"),
             py::arg("differ") = std::vector<std::pair<int, int>>{},
             py::arg("same") = std::vector<std::pair<int, int>>{})
        .def("set_memory", &Pricer::set_memory, py::arg("ng"))
        .def("set_kernel", &Pricer::set_kernel, py::arg("kernel"))
        .def("set_info", &Pricer::set_info, py::arg("job_ids"), py::arg("c1a"), py::arg("c1b"),
             py::arg("c2"))
        .def("price_cols", &Pricer::price_cols, py::arg("A"), py::arg("B"), py::arg("sources"),
             py::arg("max_cols"), py::arg("tol"), py::arg("max_labels"), py::arg("first_cost"),
             py::arg("cut_cost"), py::arg("eA"), py::arg("eB"), py::arg("e_sources"),
             py::arg("e_first"), py::call_guard<py::gil_scoped_release>())
        .def("set_cuts", &Pricer::set_cuts, py::arg("cuts"))
        .def("price", &Pricer::price, py::arg("A"), py::arg("B"), py::arg("sources"),
             py::arg("max_cols") = 10, py::arg("tol") = 1e-6,
             py::arg("max_labels") = 2000000LL, py::arg("first_cost") = 0.0,
             py::arg("cut_cost") = std::vector<double>{},
             py::call_guard<py::gil_scoped_release>())
        .def("fixable", &Pricer::fixable, py::arg("A"), py::arg("B"), py::arg("sources"),
             py::arg("threshold"), py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("n_jobs", &Pricer::n_jobs);
}
