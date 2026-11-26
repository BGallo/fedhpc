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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using u64 = std::uint64_t;

namespace {

using Path = std::vector<std::pair<int, int>>;                 // (local job, start)
using ColOut = std::tuple<double, int, Path>;                   // (rc, release, path)
using PriceOut = std::tuple<double, bool, long long, std::vector<ColOut>>;

constexpr double EPS = 1e-9;

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
    struct Arena {
        std::vector<double> cost;
        std::vector<int> pred, job, start;
        std::vector<u64> pend, mask, cs;
        std::vector<char> alive;
        double first = 0.0;   // paid by an empty (source) label's first arc
        const double* ccost = nullptr;
        int add(double c, int pr, int j, int s, u64 pd, u64 cst, const u64* m) {
            cost.push_back(c), pred.push_back(pr), job.push_back(j), start.push_back(s);
            pend.push_back(pd), cs.push_back(cst), alive.push_back(1);
            mask.insert(mask.end(), m, m + NW);
            return (int)cost.size() - 1;
        }
        const u64* m(int id) const { return &mask[(size_t)id * NW]; }
        void pop() {
            cost.pop_back(), pred.pop_back(), job.pop_back(), start.pop_back();
            pend.pop_back(), cs.pop_back(), alive.pop_back();
            mask.resize(mask.size() - NW);
        }
        // a dominates b: cost <= (plus the cuts a may still pay and b not),
        // mask(a) subset of mask(b), same pending parity. An empty label
        // still owes `first` on its first arc, so it competes at cost +
        // first against non-empty labels.
        bool dom(int a, int b) const {
            double ca = cost[a] + (job[a] < 0 && job[b] >= 0 ? first : 0.0);
            if (ca > cost[b] + EPS || pend[a] != pend[b]) return false;
            for (u64 x = cs[a] & ~cs[b]; x; x &= x - 1) {
                ca += ccost[__builtin_ctzll(x)];
                if (ca > cost[b] + EPS) return false;
            }
            const u64 *ma = m(a), *mb = m(b);
            for (int w = 0; w < NW; ++w)
                if (ma[w] & ~mb[w]) return false;
            return true;
        }
    };

    template <int NW>
    PriceOut run(const std::vector<double>& A, const std::vector<double>& B,
                 const std::vector<std::pair<int, double>>& sources, int max_cols, double tol,
                 long long max_labels) {
        const double INF = std::numeric_limits<double>::infinity();
        const double thr = -tol;
        // completion bound g(t), repeats allowed
        std::vector<double> g(H_ + 1, 0.0);
        for (int t = H_ - 1; t >= 0; --t) {
            double v = g[t + 1];
            for (int i : at_[t]) v = std::min(v, A[i] + B[i] * t + g[t + p_[i]]);
            g[t] = v;
        }

        Arena<NW> L;
        L.first = first_;
        L.ccost = ccost_.data();
        std::vector<std::vector<int>> bucket(H_ + 1);
        std::vector<int> active, cand;
        double pruned_min = INF;     // lower bound over everything not enumerated
        double best = INF;           // best complete consistent label
        bool exact = true;
        u64 zero[NW] = {0};
        u64 nm[NW];

        auto insert = [&](int t2, double c, int pr, int j, int s, u64 pd, u64 cst, const u64* m) {
            // dominance within the arrivals of slot t2
            auto& bk = bucket[t2];
            int id = L.add(c, pr, j, s, pd, cst, m);
            for (size_t k = 0; k < bk.size();) {
                int o = bk[k];
                if (L.dom(o, id)) {
                    L.pop();
                    return;
                }
                if (L.dom(id, o)) {
                    L.alive[o] = 0;
                    bk[k] = bk.back();
                    bk.pop_back();
                } else {
                    ++k;
                }
            }
            bk.push_back(id);
        };

        for (auto [r, c] : sources) insert(std::min(std::max(r, 0), H_), c, -1, -1, r, 0, 0, zero);

        for (int t = 0; t <= H_; ++t) {
            // merge this slot's arrivals into the active set
            for (int id : bucket[t]) {
                if (!L.alive[id]) continue;
                bool dominated = false;
                for (int a : active)
                    if (L.dom(a, id)) { dominated = true; break; }
                if (dominated) { L.alive[id] = 0; continue; }
                for (size_t k = 0; k < active.size();) {
                    if (L.dom(id, active[k])) {
                        L.alive[active[k]] = 0;
                        active[k] = active.back();
                        active.pop_back();
                    } else {
                        ++k;
                    }
                }
                active.push_back(id);
                if (L.pend[id] == 0) {
                    best = std::min(best, L.cost[id]);
                    if (L.cost[id] < thr) cand.push_back(id);
                }
            }
            bucket[t].clear();
            bucket[t].shrink_to_fit();
            // retire labels that can no longer improve
            for (size_t k = 0; k < active.size();) {
                int a = active[k];
                double lb = L.cost[a] + g[t];
                if (lb >= thr || t == H_) {
                    if (lb >= thr) pruned_min = std::min(pruned_min, lb);
                    active[k] = active.back();
                    active.pop_back();
                } else {
                    ++k;
                }
            }
            if (t == H_ || !exact) break;
            if (at_[t].empty()) continue;
            // Cheapest first: an extension costs at least cost + arc + g(t2)
            // (first_cost and cut costs are >= 0), so the first label whose
            // optimistic bound fails ends the scan for this arc — that bound
            // also lower-bounds every later label's, keeping min_lb valid.
            std::sort(active.begin(), active.end(),
                      [&](int x, int y) { return L.cost[x] < L.cost[y]; });
            for (int i : at_[t]) {
                const int t2 = t + p_[i];
                const double arc = A[i] + B[i] * t;
                const double rest = arc + g[t2];
                const u64* ret = &retain_[(size_t)i * NW];
                const u64* dif = &differ_[(size_t)i * NW];
                const u64 cS = inS_[i], cM = inM_[i];
                for (size_t k = 0; k < active.size(); ++k) {
                    int a = active[k];
                    if (L.cost[a] + rest >= thr) {
                        pruned_min = std::min(pruned_min, L.cost[a] + rest);
                        break;
                    }
                    const u64* ma = L.m(a);
                    if (has_bit(ma, i)) continue;
                    bool clash = false;
                    for (int w = 0; w < NW; ++w)
                        if (ma[w] & dif[w]) { clash = true; break; }
                    if (clash) continue;
                    double c = L.cost[a] + arc + (L.job[a] < 0 ? first_ : 0.0);
                    u64 st = L.cs[a] & cM;
                    for (u64 x = st & cS; x; x &= x - 1) c += ccost_[__builtin_ctzll(x)];
                    st ^= cS;
                    double lb = c + g[t2];
                    if (lb >= thr) {
                        pruned_min = std::min(pruned_min, lb);
                        continue;
                    }
                    if ((long long)L.cost.size() >= max_labels) { exact = false; break; }
                    for (int w = 0; w < NW; ++w) nm[w] = ma[w] & ret[w];
                    set_bit(nm, i);
                    insert(t2, c, a, i, t, L.pend[a] ^ toggle_[i], st, nm);
                }
                if (!exact) break;
            }
        }

        double min_lb = exact ? std::min(best, pruned_min) : -INF;

        std::sort(cand.begin(), cand.end(),
                  [&](int a, int b) { return L.cost[a] < L.cost[b]; });
        std::vector<ColOut> cols;
        std::unordered_set<std::size_t> seen;
        for (int id : cand) {
            if ((int)cols.size() >= max_cols) break;
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
