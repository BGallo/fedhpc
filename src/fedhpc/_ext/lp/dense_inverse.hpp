// fedlp: explicit dense basis inverse.
//
// Column generation masters have few rows (tens to a few hundred) and many
// columns, so an explicit B^-1 is affordable and gives what the primal
// simplex needs cheaply:
//   - FTRAN of a sparse column: alpha = sum_k a_k B^-1 e_k, O(m nnz(a));
//   - row r of B^-1 (the pivot row's BTRAN) for free;
//   - the update after a pivot is one rank-1 operation, O(m^2), on
//     contiguous rows (vectorised by the compiler).
// Rounding accumulates in the updates, so the simplex refactorises
// periodically and whenever the pivot element computed by row and by
// column disagree.
//
// The interface (factorize / ftran / row / update) is what a sparse LU with
// Forrest-Tomlin updates would provide, for larger masters.
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "model.hpp"

namespace fedlp {

class DenseInverse {
public:
    void reset(int m) {
        m_ = m;
        inv_.assign((size_t)m * m, 0.0);
    }
    int size() const { return m_; }

    // B^-1 of the basis whose position i holds column head[i]. Gauss-Jordan
    // on [B | I] with partial pivoting; zero multipliers are skipped, so the
    // near-identity bases of a master factor in about O(m^2 nnz). Returns
    // false when B is numerically singular.
    bool factorize(const Model& M, const std::vector<int>& head) {
        const int m = m_;
        std::vector<double> B((size_t)m * m, 0.0);
        for (int i = 0; i < m; ++i) {
            const int j = head[i];
            const int* ix = M.col_idx(j);
            const double* v = M.col_val(j);
            for (int k = 0, n = M.col_nnz(j); k < n; ++k) B[(size_t)ix[k] * m + i] = v[k];
        }
        std::fill(inv_.begin(), inv_.end(), 0.0);
        for (int i = 0; i < m; ++i) inv_[(size_t)i * m + i] = 1.0;
        std::vector<int> nz;
        nz.reserve(m);
        for (int c = 0; c < m; ++c) {
            int p = c;
            double best = std::fabs(B[(size_t)c * m + c]);
            for (int r = c + 1; r < m; ++r) {
                const double a = std::fabs(B[(size_t)r * m + c]);
                if (a > best) best = a, p = r;
            }
            if (best < 1e-11) return false;
            if (p != c) {
                std::swap_ranges(B.begin() + (size_t)p * m, B.begin() + (size_t)(p + 1) * m,
                                 B.begin() + (size_t)c * m);
                std::swap_ranges(inv_.begin() + (size_t)p * m, inv_.begin() + (size_t)(p + 1) * m,
                                 inv_.begin() + (size_t)c * m);
            }
            double* bc = &B[(size_t)c * m];
            double* ic = &inv_[(size_t)c * m];
            const double inv = 1.0 / bc[c];
            // the pivot row's nonzeros, so the eliminations below touch only them
            nz.clear();
            for (int k = 0; k < m; ++k) {
                bc[k] *= inv, ic[k] *= inv;
                if (bc[k] != 0.0 || ic[k] != 0.0) nz.push_back(k);
            }
            for (int r = 0; r < m; ++r) {
                if (r == c) continue;
                const double f = B[(size_t)r * m + c];
                if (f == 0.0) continue;
                double* br = &B[(size_t)r * m];
                double* ir = &inv_[(size_t)r * m];
                for (int k : nz) br[k] -= f * bc[k], ir[k] -= f * ic[k];
            }
        }
        return true;
    }

    // out = B^-1 a_j
    void ftran(const Model& M, int j, double* out) const {
        std::fill(out, out + m_, 0.0);
        const int* ix = M.col_idx(j);
        const double* v = M.col_val(j);
        for (int k = 0, n = M.col_nnz(j); k < n; ++k) {
            const int c = ix[k];
            const double a = v[k];
            const double* col = inv_.data() + c;
            for (int i = 0; i < m_; ++i) out[i] += col[(size_t)i * m_] * a;
        }
    }

    // row r of B^-1 (e_r' B^-1)
    const double* row(int r) const { return inv_.data() + (size_t)r * m_; }

    // After column alpha = B^-1 a_q replaces basis position r.
    void update(int r, const double* alpha) {
        const int m = m_;
        double* br = &inv_[(size_t)r * m];
        const double inv = 1.0 / alpha[r];
        for (int k = 0; k < m; ++k) br[k] *= inv;
        for (int i = 0; i < m; ++i) {
            const double f = alpha[i];
            if (i == r || f == 0.0) continue;
            double* bi = &inv_[(size_t)i * m];
            for (int k = 0; k < m; ++k) bi[k] -= f * br[k];
        }
    }

private:
    int m_ = 0;
    std::vector<double> inv_;   // row-major
};

}  // namespace fedlp
