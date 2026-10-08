// Column-generation pricing for Rm || sum w_j C_j + TMUC (Chen et al., EJOR 2024).
//
// A column is (machine k, job subset S) run in WSPT order; its cost is
//   c(k,S) = U_k + sum_{j in S} (w_j C_j + L_k p_jk).
// Given job duals beta, the per-machine subproblem
//   min_S  c(k,S) - beta(S)
// is the paper's pseudo-polynomial DP (Eq. 19) over jobs in WSPT order and the
// machine's load t (= completion time of the last job, schedules idle-free):
//   F_j(t) = min(F_{j-1}(t), F_{j-1}(t - p_jk) + w_j t + L_k p_jk - beta_j),
//   F_0(0) = U_k.
// Loads are capped at a horizon H_k (valid for every solution of cost <= UB).
// Jobs with beta_j <= (w_j + L_k) p_jk can never lower the value and are
// skipped.  price() returns min_S per machine (for the Lagrangian bound) and,
// for each machine, up to `per_machine` columns with the most negative
// value - alpha_k, one per distinct load t.
//
// enumerate() lists *every* subset S of machine k with load <= H and
// c(k,S) - beta(S) <= threshold (a DFS over jobs in WSPT order, pruned by the
// backward DP B_i(t) = best completion from job i on at load t).  With
// threshold = v_k + (UB - 1 - L(beta)) this is every column that can appear in
// a solution cheaper than UB, so a set-partitioning MIP over the enumerated
// columns decides optimality exactly (reduced-cost column enumeration).
#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using i64 = long long;

namespace {

class Pricer {
public:
    Pricer(const std::vector<std::vector<int>>& p, const std::vector<int>& w,
           const std::vector<int>& U, const std::vector<int>& L)
        : n_((int)w.size()), m_((int)U.size()), w_(w), U_(U), L_(L) {
        if ((int)p.size() != n_ || (int)L.size() != m_)
            throw std::invalid_argument("inconsistent instance dimensions");
        p_.resize((size_t)n_ * m_);
        for (int j = 0; j < n_; ++j) {
            if ((int)p[j].size() != m_) throw std::invalid_argument("p row length != m");
            for (int k = 0; k < m_; ++k) p_[(size_t)j * m_ + k] = p[j][k];
        }
        order_.resize(m_);
        for (int k = 0; k < m_; ++k) {
            auto& o = order_[k];
            o.resize(n_);
            std::iota(o.begin(), o.end(), 0);
            std::sort(o.begin(), o.end(), [&](int a, int b) {
                i64 l = (i64)P(a, k) * w_[b], r = (i64)P(b, k) * w_[a];
                return l != r ? l < r : a < b;
            });
        }
    }

    // Returns (min_S value per machine, columns); column =
    // (machine, jobs, exact integer cost, value - alpha_k).
    std::pair<std::vector<double>, std::vector<std::tuple<int, std::vector<int>, i64, double>>>
    price(const std::vector<double>& beta, const std::vector<double>& alpha,
          const std::vector<int>& horizon, int per_machine, double tol) const {
        if ((int)beta.size() != n_ || (int)alpha.size() != m_ || (int)horizon.size() != m_)
            throw std::invalid_argument("dual / horizon length mismatch");
        std::vector<double> best(m_, 0.0);
        std::vector<std::tuple<int, std::vector<int>, i64, double>> cols;
        const double INF = std::numeric_limits<double>::infinity();

        std::vector<int> items;
        std::vector<double> F;
        std::vector<char> take;
        for (int k = 0; k < m_; ++k) {
            const int H = std::max(0, horizon[k]);
            items.clear();
            for (int j : order_[k]) {
                int pj = P(j, k);
                if (pj <= H && beta[j] > (double)(w_[j] + L_[k]) * pj + 1e-9) items.push_back(j);
            }
            const int W = H + 1;
            F.assign(W, INF);
            F[0] = (double)U_[k];
            take.assign(items.size() * (size_t)W, 0);
            int hi = 0;
            for (size_t i = 0; i < items.size(); ++i) {
                int j = items[i], pj = P(j, k);
                double lin = (double)L_[k] * pj - beta[j];
                char* tk = &take[i * (size_t)W];
                int top = std::min(hi + pj, H);
                for (int t = top; t >= pj; --t) {
                    double f = F[t - pj];
                    if (f == INF) continue;
                    double cand = f + (double)w_[j] * t + lin;
                    if (cand < F[t]) {
                        F[t] = cand;
                        tk[t] = 1;
                    }
                }
                hi = top;
            }
            // Loads t >= 1 with the most negative reduced cost.
            std::vector<int> ts;
            for (int t = 1; t <= hi; ++t)
                if (F[t] < INF) {
                    best[k] = std::min(best[k], F[t]);
                    if (F[t] - alpha[k] < -tol) ts.push_back(t);
                }
            if ((int)ts.size() > per_machine) {
                std::partial_sort(ts.begin(), ts.begin() + per_machine, ts.end(),
                                  [&](int a, int b) { return F[a] < F[b]; });
                ts.resize(per_machine);
            }
            for (int t0 : ts) {
                std::vector<int> jobs;
                int t = t0;
                for (size_t i = items.size(); i-- > 0 && t > 0;) {
                    if (take[i * (size_t)W + t]) {
                        jobs.push_back(items[i]);
                        t -= P(items[i], k);
                    }
                }
                if (t != 0) throw std::logic_error("pricing backtrack failed");
                std::sort(jobs.begin(), jobs.end());
                cols.emplace_back(k, jobs, cost(k, jobs), F[t0] - alpha[k]);
            }
        }
        return {best, cols};
    }

    // Returns (columns (jobs, exact cost, reduced cost), complete); complete is
    // false when max_cols was hit and the list is truncated.
    std::pair<std::vector<std::tuple<std::vector<int>, i64, double>>, bool>
    enumerate(int k, const std::vector<double>& beta, int horizon, double threshold,
              i64 max_cols) const {
        if (k < 0 || k >= m_ || (int)beta.size() != n_)
            throw std::invalid_argument("bad machine / dual length");
        const int H = std::max(0, horizon);
        const int W = H + 1;
        std::vector<int> items;
        for (int j : order_[k])
            if (P(j, k) <= H) items.push_back(j);
        const int ni = (int)items.size();
        // B[i*W + t]: min over subsets of items i.. of their reduced-cost
        // contribution when the machine's load before item i is t (empty = 0).
        std::vector<double> B((size_t)(ni + 1) * W, 0.0);
        for (int i = ni - 1; i >= 0; --i) {
            int j = items[i], pj = P(j, k);
            double lin = (double)L_[k] * pj - beta[j];
            const double* nx = &B[(size_t)(i + 1) * W];
            double* cu = &B[(size_t)i * W];
            for (int t = 0; t <= H; ++t) {
                double v = nx[t];
                if (t + pj <= H) v = std::min(v, (double)w_[j] * (t + pj) + lin + nx[t + pj]);
                cu[t] = v;
            }
        }
        std::vector<std::tuple<std::vector<int>, i64, double>> out;
        std::vector<int> cur;
        bool complete = true;
        const double thr = threshold + 1e-7 * (1.0 + std::abs(threshold));
        // iterative DFS: frame = (item index, load, accumulated value, state)
        struct Frame { int i, t; double acc; int state; };
        std::vector<Frame> st;
        st.push_back({0, 0, (double)U_[k], 0});
        while (!st.empty()) {
            Frame& f = st.back();
            if (f.i == ni) {
                if (!cur.empty()) {
                    if ((i64)out.size() >= max_cols) { complete = false; break; }
                    std::vector<int> js(cur);
                    std::sort(js.begin(), js.end());
                    out.emplace_back(js, cost(k, js), f.acc);
                }
                st.pop_back();
                if (!st.empty() && st.back().state == 2) cur.pop_back();
                continue;
            }
            int j = items[f.i], pj = P(j, k);
            if (f.state == 0) {                // try taking item i
                f.state = 1;
                int t2 = f.t + pj;
                if (t2 <= H) {
                    double a2 = f.acc + (double)w_[j] * t2 + (double)L_[k] * pj - beta[j];
                    if (a2 + B[(size_t)(f.i + 1) * W + t2] <= thr) {
                        f.state = 2;
                        cur.push_back(j);
                        st.push_back({f.i + 1, t2, a2, 0});
                        continue;
                    }
                }
            }
            if (f.state == 2) {                // returned from the take branch
                f.state = 1;                   // (cur already popped by child)
            }
            if (f.state == 1) {                // try skipping item i
                f.state = 3;
                if (f.acc + B[(size_t)(f.i + 1) * W + f.t] <= thr) {
                    st.push_back({f.i + 1, f.t, f.acc, 0});
                    continue;
                }
            }
            st.pop_back();                      // both branches done
            if (!st.empty() && st.back().state == 2) cur.pop_back();
        }
        return {out, complete};
    }

    i64 cost(int k, std::vector<int> jobs) const {
        if (jobs.empty()) return 0;
        const auto& o = order_[k];
        std::vector<int> pos(n_);
        for (int r = 0; r < n_; ++r) pos[o[r]] = r;
        std::sort(jobs.begin(), jobs.end(), [&](int a, int b) { return pos[a] < pos[b]; });
        i64 t = 0, c = U_[k];
        for (int j : jobs) {
            t += P(j, k);
            c += (i64)w_[j] * t;
        }
        return c + (i64)L_[k] * t;
    }

private:
    int n_, m_;
    std::vector<int> p_, w_, U_, L_;
    std::vector<std::vector<int>> order_;

    int P(int j, int k) const { return p_[(size_t)j * m_ + k]; }
};

}  // namespace

PYBIND11_MODULE(_rmprice, mod) {
    mod.doc() = "Column-generation pricing DP for Rm||sum wC + TMUC";
    py::class_<Pricer>(mod, "Pricer")
        .def(py::init<const std::vector<std::vector<int>>&, const std::vector<int>&,
                      const std::vector<int>&, const std::vector<int>&>(),
             py::arg("p"), py::arg("w"), py::arg("U"), py::arg("L"))
        .def("price", &Pricer::price, py::arg("beta"), py::arg("alpha"), py::arg("horizon"),
             py::arg("per_machine") = 5, py::arg("tol") = 1e-6,
             py::call_guard<py::gil_scoped_release>())
        .def("enumerate", &Pricer::enumerate, py::arg("machine"), py::arg("beta"),
             py::arg("horizon"), py::arg("threshold"), py::arg("max_cols") = 50000000,
             py::call_guard<py::gil_scoped_release>())
        .def("cost", &Pricer::cost, py::arg("machine"), py::arg("jobs"));
}
