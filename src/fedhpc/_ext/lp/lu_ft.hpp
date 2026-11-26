// fedlp: sparse LU basis factor with Forrest-Tomlin updates.
//
// M B = U, M = R_t ... R_1 L_k ... L_1:
//   - L etas: column etas from the factorization, v[i] -= l_i v[p];
//   - R etas: row etas from the Forrest-Tomlin updates, v[p] -= sum mu_j v[p_j];
//   - U: upper triangular in the pivot ("rank") order `ord_`. Its columns are
//     the basis positions; position c pivots on row prow_[c] with diagonal
//     diag_[c]. Off-diagonal entries are kept both by column (ucol_) and by
//     row (urow_).
//
// Factorization is left-looking, over the basis columns sparsest first.
// L^-1 a_c gives the column's entries at rows already pivoted (its U column)
// and the remaining rows. Among those it pivots by threshold partial pivoting
// (at least 0.1 of the largest), the sparsest row first.
//
// A Forrest-Tomlin update replaces position c's U column by the "spike"
// M a_q, kept from the entering column's FTRAN, and moves c to the end of
// the order. The rows pivoted after c are then used to eliminate c's pivot
// row (now last), and the multipliers become one row eta R. Factors stay
// compact: one short row eta per pivot, where the product form adds a full
// column eta. Refactorise after `interval` updates, or when a new diagonal
// is tiny.
//
// FTRAN skips etas and U columns whose pivot entry is zero, and BTRAN works
// in rank order, so both cost what the nonzeros they touch cost.
#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <queue>
#include <vector>

#include "factor.hpp"
#include "model.hpp"

namespace fedlp {

class LuFT final : public BasisFactor {
public:
    void reset(int m) override {
        m_ = m;
        clear();
        spike_.assign(m, 0.0);
        tmp_.assign(m, 0.0);
        rowbuf_.assign(m, 0.0);
        w_.assign(m, 0.0);
    }

    bool factorize(const Model& M, std::vector<int>& head) override {
        const int m = m_;
        clear();
        diag_.assign(m, 0.0);
        prow_.assign(m, -1);
        ucol_.assign(m, {});
        urow_.assign(m, {});
        std::vector<int> rcount(m, 0);
        for (int c = 0; c < m; ++c) {
            const int j = head[c];
            const int* ix = M.col_idx(j);
            for (int k = 0, n = M.col_nnz(j); k < n; ++k) ++rcount[ix[k]];
        }
        std::vector<int> order(m);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(),
                         [&](int a, int b) { return M.col_nnz(head[a]) < M.col_nnz(head[b]); });
        std::vector<char> taken(m, 0);
        std::vector<double>& v = tmp_;
        std::fill(v.begin(), v.end(), 0.0);
        eta_of_row_.assign(m, -1);
        std::vector<int> nz;
        std::vector<char> seen(m, 0);
        for (int c : order) {
            const int j = head[c];
            nz.clear();
            const int* ix = M.col_idx(j);
            const double* val = M.col_val(j);
            for (int k = 0, n = M.col_nnz(j); k < n; ++k) {
                v[ix[k]] = val[k];
                if (!seen[ix[k]]) seen[ix[k]] = 1, nz.push_back(ix[k]);
            }
            apply_L_sparse(v.data(), nz, seen);
            double mx = 0.0;
            for (int i : nz)
                if (!taken[i]) mx = std::max(mx, std::fabs(v[i]));
            if (mx < 1e-11) {
                for (int i : nz) v[i] = 0.0, seen[i] = 0;
                return false;
            }
            int p = -1;
            for (int i : nz) {
                if (taken[i] || std::fabs(v[i]) < 0.1 * mx) continue;
                if (p < 0 || rcount[i] < rcount[p] ||
                    (rcount[i] == rcount[p] && std::fabs(v[i]) > std::fabs(v[p])))
                    p = i;
            }
            const double inv = 1.0 / v[p];
            diag_[c] = v[p];
            prow_[c] = p;
            lpiv_.push_back(p);
            eta_of_row_[p] = (int)lpiv_.size() - 1;
            for (int i : nz) {
                const double x = v[i];
                if (std::fabs(x) > 1e-14) {
                    if (taken[i]) ucol_[c].push_back({i, x});          // U column
                    else if (i != p) lidx_.push_back(i), lval_.push_back(x * inv);   // L eta
                }
                v[i] = 0.0, seen[i] = 0;
            }
            lstart_.push_back((int)lidx_.size());
            taken[p] = 1;
            ord_.push_back(c);
        }
        for (int c = 0; c < m; ++c)
            for (auto [p, x] : ucol_[c]) urow_[p].push_back({c, x});
        updates_ = 0;
        return true;
    }

    void ftran(const Model& M, int j, double* out) override {
        std::vector<double>& v = spike_;
        std::fill(v.begin(), v.end(), 0.0);
        const int* ix = M.col_idx(j);
        const double* val = M.col_val(j);
        nzf_.clear();
        if ((int)seenf_.size() != m_) seenf_.assign(m_, 0);
        for (int k = 0, n = M.col_nnz(j); k < n; ++k) {
            v[ix[k]] = val[k];
            if (!seenf_[ix[k]]) seenf_[ix[k]] = 1, nzf_.push_back(ix[k]);
        }
        apply_L_sparse(v.data(), nzf_, seenf_);
        for (int i : nzf_) seenf_[i] = 0;
        apply_R(v.data());
        // keep the spike (M a_j) for update(); solve U into out
        std::copy(v.begin(), v.end(), tmp_.begin());
        out_nz_.clear();
        solve_U(tmp_.data(), out, &out_nz_);
        have_spike_ = true;
    }
    void ftran_inplace(double* x) override {
        apply_L(x);
        apply_R(x);
        std::copy(x, x + m_, tmp_.begin());
        solve_U(tmp_.data(), x);
    }
    void btran_inplace(double* y) override {
        // y is indexed by position: z' U = y', then z' M
        // row-oriented: once z at c's pivot row is known, take its share off
        // the later columns (urow_), skipping zeros
        std::vector<double>& z = tmp_;
        std::fill(z.begin(), z.end(), 0.0);
        for (int c : ord_) {
            const double zc = y[c] / diag_[c];
            z[prow_[c]] = zc;
            if (zc == 0.0) continue;
            for (auto [c2, x] : urow_[prow_[c]]) y[c2] -= x * zc;
        }
        apply_Rt(z.data());
        apply_Lt(z.data());
        std::copy(z.begin(), z.end(), y);
    }
    const double* row(int r) override {
        std::fill(rowbuf_.begin(), rowbuf_.end(), 0.0);
        rowbuf_[r] = 1.0;
        btran_inplace(rowbuf_.data());
        return rowbuf_.data();
    }

    // Forrest-Tomlin: position r takes the column whose spike the last
    // ftran(M, j, out) kept. Returns false when the new diagonal is tiny
    // (the caller refactorises).
    bool update(int r, const double* alpha) override {
        if (!have_spike_) return false;
        have_spike_ = false;
        const int c = r, pk = prow_[c];
        // det(B_new) = alpha_r det(B): the new diagonal must be alpha_r times
        // the old one, which checks the update's accuracy
        const double expected = alpha[r] * diag_[c];
        // 1. drop c's old U column
        for (auto [p, x] : ucol_[c]) erase_from(urow_[p], c);
        ucol_[c].clear();
        // 2. gather pivot row pk (now last) into w, by position
        std::vector<int> wnz;
        for (auto [c2, x] : urow_[pk]) {
            w_[c2] = x;
            wnz.push_back(c2);
            erase_from(ucol_[c2], pk);
        }
        urow_[pk].clear();
        // 3. the spike becomes column c: its entry at pk joins w, the rest U
        for (int p = 0; p < m_; ++p) {
            const double x = spike_[p];
            if (std::fabs(x) <= 1e-14) continue;
            if (p == pk) {
                w_[c] += x;
                wnz.push_back(c);
            } else {
                ucol_[c].push_back({p, x});
                urow_[p].push_back({c, x});
            }
        }
        // 4. move c to the end of the order
        const int k = (int)(std::find(ord_.begin(), ord_.end(), c) - ord_.begin());
        ord_.erase(ord_.begin() + k);
        ord_.push_back(c);
        // 5. eliminate row pk's entries left of the new diagonal
        rtarget_.push_back(pk);
        double mu_max = 0.0;
        for (int i = k, last = (int)ord_.size() - 1; i < last; ++i) {
            const int cj = ord_[i];
            const double x = w_[cj];
            if (x == 0.0) continue;
            w_[cj] = 0.0;
            const int pj = prow_[cj];
            const double mu = x / diag_[cj];
            for (auto [c2, y] : urow_[pj]) {
                if (w_[c2] == 0.0) wnz.push_back(c2);
                w_[c2] -= mu * y;
            }
            ridx_.push_back(pj), rval_.push_back(mu);
            mu_max = std::max(mu_max, std::fabs(mu));
        }
        rstart_.push_back((int)ridx_.size());
        const double d = w_[c];
        for (int c2 : wnz) w_[c2] = 0.0;
        w_[c] = 0.0;
        ++updates_;
        diag_[c] = d;
        if (!(std::fabs(d) > 1e-11)) return false;
        // no pivoting is possible in a Forrest-Tomlin elimination: large
        // multipliers mean growth, so refactorise instead of carrying it
        if (mu_max > kMaxGrowth) return false;
        return std::fabs(d - expected) <= 1e-9 * (1.0 + std::fabs(expected));
    }
    const char* name() const override { return "lu"; }
    const std::vector<int>* last_nz() const override { return &out_nz_; }
    int updates() const { return updates_; }

private:
    static constexpr double kMaxGrowth = 1e4;
    int m_ = 0, updates_ = 0;
    bool have_spike_ = false;
    std::vector<int> lpiv_, lstart_, lidx_;
    std::vector<double> lval_;
    std::vector<int> rtarget_, rstart_, ridx_;
    std::vector<double> rval_;
    std::vector<double> diag_;
    std::vector<int> prow_, ord_;
    std::vector<std::vector<std::pair<int, double>>> ucol_, urow_;
    std::vector<double> spike_, tmp_, rowbuf_, w_;
    std::vector<int> eta_of_row_, nzf_;   // L eta pivoting on each row; ftran's pattern
    std::vector<int> out_nz_;             // last ftran(M, j, out)'s nonzero positions
    std::vector<char> seenf_;

    // L applied to a sparse v whose pattern is nz (seen marks it): only the
    // etas of the touched pivot rows, in eta order (a fill row is always
    // pivoted later than the eta that fills it, so a min-heap keeps order).
    void apply_L_sparse(double* v, std::vector<int>& nz, std::vector<char>& seen) const {
        std::priority_queue<int, std::vector<int>, std::greater<int>> heap;
        for (int i : nz)
            if (eta_of_row_[i] >= 0) heap.push(eta_of_row_[i]);
        int last = -1;
        while (!heap.empty()) {
            const int k = heap.top();
            heap.pop();
            if (k == last) continue;
            last = k;
            const double vp = v[lpiv_[k]];
            if (vp == 0.0) continue;
            for (int e = lstart_[k], f = lstart_[k + 1]; e < f; ++e) {
                const int i = lidx_[e];
                v[i] -= lval_[e] * vp;
                if (!seen[i]) {
                    seen[i] = 1, nz.push_back(i);
                    if (eta_of_row_[i] > k) heap.push(eta_of_row_[i]);
                }
            }
        }
    }

    void clear() {
        lpiv_.clear(), lidx_.clear(), lval_.clear(), lstart_.assign(1, 0);
        rtarget_.clear(), ridx_.clear(), rval_.clear(), rstart_.assign(1, 0);
        ord_.clear();
        have_spike_ = false;
    }
    static void erase_from(std::vector<std::pair<int, double>>& v, int key) {
        for (size_t i = 0; i < v.size(); ++i)
            if (v[i].first == key) {
                v[i] = v.back();
                v.pop_back();
                return;
            }
    }

    void apply_L(double* v) const {
        for (size_t k = 0; k < lpiv_.size(); ++k) {
            const double vp = v[lpiv_[k]];
            if (vp == 0.0) continue;
            for (int e = lstart_[k], f = lstart_[k + 1]; e < f; ++e) v[lidx_[e]] -= lval_[e] * vp;
        }
    }
    void apply_Lt(double* z) const {
        for (int k = (int)lpiv_.size() - 1; k >= 0; --k) {
            double s = 0.0;
            for (int e = lstart_[k], f = lstart_[k + 1]; e < f; ++e) s += lval_[e] * z[lidx_[e]];
            z[lpiv_[k]] -= s;
        }
    }
    void apply_R(double* v) const {
        for (size_t k = 0; k < rtarget_.size(); ++k) {
            double s = 0.0;
            for (int e = rstart_[k], f = rstart_[k + 1]; e < f; ++e) s += rval_[e] * v[ridx_[e]];
            v[rtarget_[k]] -= s;
        }
    }
    void apply_Rt(double* z) const {
        for (int k = (int)rtarget_.size() - 1; k >= 0; --k) {
            const double zt = z[rtarget_[k]];
            if (zt == 0.0) continue;
            for (int e = rstart_[k], f = rstart_[k + 1]; e < f; ++e) z[ridx_[e]] -= rval_[e] * zt;
        }
    }
    // U x = v (v by rows, consumed), x by position; nz collects x's pattern
    void solve_U(double* v, double* x, std::vector<int>* nz = nullptr) const {
        for (int i = (int)ord_.size() - 1; i >= 0; --i) {
            const int c = ord_[i];
            const double xc = v[prow_[c]] / diag_[c];
            x[c] = xc;
            if (xc == 0.0) continue;
            if (nz) nz->push_back(c);
            for (auto [p, u] : ucol_[c]) v[p] -= u * xc;
        }
    }
};

}  // namespace fedlp
