// fedlp: basis factor interface, and the sparse product-form factor.
//
// The simplex needs, from a factor of the basis B (position i of B holds
// column head[i]):
//   ftran   x <- B^-1 x          (entering column, primal values)
//   btran   y' <- y' B^-1        (duals, pivot rows)
//   row(r)  e_r' B^-1            (the pivot row's BTRAN)
//   update  after column alpha = B^-1 a_q replaces position r
// factorize() may reorder `head` (a sparse factor pivots each basis column
// on a row of its choosing; position i is then the row its column pivots
// on), so the simplex rebuilds its position maps after every factorization.
//
// SparsePFI is a product-form inverse, B^-1 = E_k ... E_1: each eta E_k is
// the identity except column p_k. It is built by left-looking Gauss-Jordan
// over the basis columns, sparsest first, so the many singleton columns of a
// master (slacks, artificials, null columns) give trivial etas and only a
// small bump fills in. Pivot rows are chosen by threshold partial pivoting
// (|x_i| >= 0.1 max), the sparsest row among those. FTRAN skips every eta
// whose pivot entry is zero (hyper-sparsity), so its cost follows the
// nonzeros actually touched. A pivot appends one eta, O(nnz(alpha)), where
// the explicit inverse's update is O(m^2): the factor for masters past a few
// hundred rows.
#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "dense_inverse.hpp"
#include "model.hpp"

namespace fedlp {

class BasisFactor {
public:
    virtual ~BasisFactor() = default;
    virtual void reset(int m) = 0;
    // false when B is numerically singular
    virtual bool factorize(const Model& M, std::vector<int>& head) = 0;
    virtual void ftran(const Model& M, int j, double* out) = 0;   // out = B^-1 a_j
    virtual void ftran_inplace(double* x) = 0;
    virtual void btran_inplace(double* y) = 0;
    virtual const double* row(int r) = 0;                          // e_r' B^-1
    virtual void update(int r, const double* alpha) = 0;
    virtual const char* name() const = 0;
};

// The explicit dense inverse behind the interface.
class DenseFactor final : public BasisFactor {
public:
    void reset(int m) override { inv_.reset(m), m_ = m, tmp_.assign(m, 0.0); }
    bool factorize(const Model& M, std::vector<int>& head) override { return inv_.factorize(M, head); }
    void ftran(const Model& M, int j, double* out) override { inv_.ftran(M, j, out); }
    void ftran_inplace(double* x) override {
        for (int i = 0; i < m_; ++i) {
            const double* bi = inv_.row(i);
            double s = 0.0;
            for (int k = 0; k < m_; ++k) s += bi[k] * x[k];
            tmp_[i] = s;
        }
        std::copy(tmp_.begin(), tmp_.end(), x);
    }
    void btran_inplace(double* y) override {
        std::fill(tmp_.begin(), tmp_.end(), 0.0);
        for (int i = 0; i < m_; ++i) {
            const double c = y[i];
            if (c == 0.0) continue;
            const double* bi = inv_.row(i);
            for (int k = 0; k < m_; ++k) tmp_[k] += c * bi[k];
        }
        std::copy(tmp_.begin(), tmp_.end(), y);
    }
    const double* row(int r) override { return inv_.row(r); }
    void update(int r, const double* alpha) override { inv_.update(r, alpha); }
    const char* name() const override { return "dense"; }

private:
    DenseInverse inv_;
    int m_ = 0;
    std::vector<double> tmp_;
};

class SparsePFI final : public BasisFactor {
public:
    void reset(int m) override {
        m_ = m;
        clear();
        work_.assign(m, 0.0);
        rowbuf_.assign(m, 0.0);
    }

    bool factorize(const Model& M, std::vector<int>& head) override {
        const int m = m_;
        clear();
        // static row counts of B, to prefer sparse pivot rows
        std::vector<int> rcount(m, 0);
        for (int i = 0; i < m; ++i) {
            const int j = head[i];
            const int* ix = M.col_idx(j);
            for (int k = 0, n = M.col_nnz(j); k < n; ++k) ++rcount[ix[k]];
        }
        std::vector<int> order(m);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(),
                         [&](int a, int b) { return M.col_nnz(head[a]) < M.col_nnz(head[b]); });
        std::vector<char> taken(m, 0);
        std::vector<int> newhead(m, -1);
        std::vector<double>& x = work_;
        for (int oi : order) {
            const int j = head[oi];
            std::fill(x.begin(), x.end(), 0.0);
            const int* ix = M.col_idx(j);
            const double* v = M.col_val(j);
            for (int k = 0, n = M.col_nnz(j); k < n; ++k) x[ix[k]] = v[k];
            apply_ftran(x.data());
            double mx = 0.0;
            for (int i = 0; i < m; ++i)
                if (!taken[i]) mx = std::max(mx, std::fabs(x[i]));
            if (mx < 1e-11) return false;
            int p = -1;
            for (int i = 0; i < m; ++i) {
                if (taken[i] || std::fabs(x[i]) < 0.1 * mx) continue;
                if (p < 0 || rcount[i] < rcount[p] ||
                    (rcount[i] == rcount[p] && std::fabs(x[i]) > std::fabs(x[p])))
                    p = i;
            }
            push_eta(p, x.data());
            taken[p] = 1;
            newhead[p] = j;
        }
        head = newhead;
        n_factor_ = (int)piv_.size();
        return true;
    }

    void ftran(const Model& M, int j, double* out) override {
        std::fill(out, out + m_, 0.0);
        const int* ix = M.col_idx(j);
        const double* v = M.col_val(j);
        for (int k = 0, n = M.col_nnz(j); k < n; ++k) out[ix[k]] = v[k];
        apply_ftran(out);
    }
    void ftran_inplace(double* x) override { apply_ftran(x); }
    void btran_inplace(double* y) override { apply_btran(y); }
    const double* row(int r) override {
        std::fill(rowbuf_.begin(), rowbuf_.end(), 0.0);
        rowbuf_[r] = 1.0;
        apply_btran(rowbuf_.data());
        return rowbuf_.data();
    }
    void update(int r, const double* alpha) override { push_eta(r, alpha); }
    const char* name() const override { return "sparse"; }
    size_t etas() const { return piv_.size(); }
    size_t eta_nnz() const { return idx_.size(); }

private:
    int m_ = 0, n_factor_ = 0;
    // etas: pivot row, 1/alpha_p, off-pivot entries -alpha_i/alpha_p
    std::vector<int> piv_, start_, idx_;
    std::vector<double> diag_, val_;
    std::vector<double> work_, rowbuf_;

    void clear() {
        piv_.clear(), diag_.clear(), idx_.clear(), val_.clear();
        start_.assign(1, 0);
    }

    void push_eta(int p, const double* x) {
        const double inv = 1.0 / x[p];
        piv_.push_back(p);
        diag_.push_back(inv);
        for (int i = 0; i < m_; ++i) {
            if (i == p) continue;
            const double a = x[i];
            if (std::fabs(a) > 1e-14) idx_.push_back(i), val_.push_back(-a * inv);
        }
        start_.push_back((int)idx_.size());
    }

    void apply_ftran(double* x) const {
        const int ne = (int)piv_.size();
        for (int k = 0; k < ne; ++k) {
            const int p = piv_[k];
            const double xp = x[p];
            if (xp == 0.0) continue;   // hyper-sparse skip
            x[p] = xp * diag_[k];
            for (int e = start_[k], f = start_[k + 1]; e < f; ++e) x[idx_[e]] += val_[e] * xp;
        }
    }

    void apply_btran(double* y) const {
        for (int k = (int)piv_.size() - 1; k >= 0; --k) {
            const int p = piv_[k];
            double s = y[p] * diag_[k];
            for (int e = start_[k], f = start_[k + 1]; e < f; ++e) s += y[idx_[e]] * val_[e];
            y[p] = s;
        }
    }
};

}  // namespace fedlp
