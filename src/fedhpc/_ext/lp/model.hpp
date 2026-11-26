// fedlp: a simplex LP solver for column generation masters.
//
// Model: the LP  min c'x  s.t.  A x (<=, =, >=) b,  l <= x <= u,  stored by
// columns, with a row-wise copy for pivot rows. Columns 0..m-1 are the rows'
// logicals: column r is sigma_r e_r, sigma_r = +1 for <= and = rows (bounds
// [0, inf) and [0, 0]) and -1 for >= rows ([0, inf)), so every row reads
// A x + sigma s = b and the simplex treats logicals and structurals alike.
//
// Columns are appended (column generation) and removed (tombstoned: a dead
// column keeps its storage but is never priced or made basic). The row-wise
// copy is append-only and skips dead columns; it is compacted when more
// than half its entries are dead.
#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace fedlp {

constexpr double kInf = std::numeric_limits<double>::infinity();

class Model {
public:
    struct RowEntry {
        int col;
        double val;
    };

    Model(const std::vector<char>& sense, const std::vector<double>& rhs) : m_((int)rhs.size()), rhs_(rhs) {
        if ((int)sense.size() != m_) throw std::invalid_argument("sense and rhs lengths differ");
        rows_.resize(m_);
        start_.push_back(0);
        for (int r = 0; r < m_; ++r) {
            const char s = sense[r];
            if (s != 'L' && s != 'E' && s != 'G') throw std::invalid_argument("row sense must be L, E or G");
            const double sigma = s == 'G' ? -1.0 : 1.0;
            append(0.0, 0.0, s == 'E' ? 0.0 : kInf, &r, &sigma, 1);
        }
        sense_ = sense;
    }

    int rows() const { return m_; }
    int cols() const { return (int)obj_.size(); }                // logicals included
    int structurals() const { return cols() - m_; }
    char sense(int r) const { return sense_[r]; }
    const std::vector<double>& rhs() const { return rhs_; }

    // Append a structural column; returns its model index (>= rows()).
    int add_column(double c, double lb, double ub, const int* idx, const double* val, int nnz) {
        if (!(lb > -kInf) && !(ub < kInf)) throw std::invalid_argument("free columns are not supported");
        if (lb > ub) throw std::invalid_argument("lower bound above upper bound");
        for (int k = 0; k < nnz; ++k)
            if (idx[k] < 0 || idx[k] >= m_) throw std::invalid_argument("row index out of range");
        return append(c, lb, ub, idx, val, nnz);
    }

    void remove_column(int j) {
        check_structural(j);
        if (!alive_[j]) return;
        alive_[j] = 0;
        dead_entries_ += col_nnz(j);
        live_entries_ -= col_nnz(j);
        if (dead_entries_ > live_entries_ && dead_entries_ > 4096) compact_rows();
    }

    void set_bounds(int j, double lb, double ub) {
        check_structural(j);
        if (lb > ub) throw std::invalid_argument("lower bound above upper bound");
        if (!(lb > -kInf) && !(ub < kInf)) throw std::invalid_argument("free columns are not supported");
        lb_[j] = lb, ub_[j] = ub;
    }

    // nonzeros of the live columns
    long long nnz() const { return live_entries_; }
    double obj(int j) const { return obj_[j]; }
    void set_obj(int j, double c) {
        check_structural(j);
        obj_[j] = c;
    }
    double lb(int j) const { return lb_[j]; }
    double ub(int j) const { return ub_[j]; }
    bool alive(int j) const { return alive_[j]; }
    int col_nnz(int j) const { return start_[j + 1] - start_[j]; }
    const int* col_idx(int j) const { return idx_.data() + start_[j]; }
    const double* col_val(int j) const { return val_.data() + start_[j]; }
    const std::vector<RowEntry>& row(int r) const { return rows_[r]; }

    // y' a_j
    double dot(const double* y, int j) const {
        const int* ix = col_idx(j);
        const double* v = col_val(j);
        double s = 0.0;
        for (int k = 0, n = col_nnz(j); k < n; ++k) s += y[ix[k]] * v[k];
        return s;
    }

private:
    int m_;
    std::vector<char> sense_;
    std::vector<double> rhs_;
    std::vector<int> start_, idx_;
    std::vector<double> val_, obj_, lb_, ub_;
    std::vector<char> alive_;
    std::vector<std::vector<RowEntry>> rows_;
    long long live_entries_ = 0, dead_entries_ = 0;

    void check_structural(int j) const {
        if (j < m_ || j >= cols()) throw std::invalid_argument("not a structural column");
    }

    int append(double c, double lb, double ub, const int* idx, const double* val, int nnz) {
        const int j = cols();
        for (int k = 0; k < nnz; ++k) {
            if (val[k] == 0.0) continue;
            idx_.push_back(idx[k]);
            val_.push_back(val[k]);
            rows_[idx[k]].push_back({j, val[k]});
            ++live_entries_;
        }
        start_.push_back((int)idx_.size());
        obj_.push_back(c), lb_.push_back(lb), ub_.push_back(ub), alive_.push_back(1);
        return j;
    }

    void compact_rows() {
        for (auto& row : rows_) {
            size_t w = 0;
            for (size_t k = 0; k < row.size(); ++k)
                if (alive_[row[k].col]) row[w++] = row[k];
            row.resize(w);
        }
        dead_entries_ = 0;
    }
};

}  // namespace fedlp
