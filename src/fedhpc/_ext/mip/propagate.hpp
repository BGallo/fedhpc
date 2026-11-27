// Activity-based domain propagation (feasibility-based bound tightening),
// event driven: a row is examined only when one of its columns' bounds has
// changed since it was last examined (SCIP's and Gurobi's scheme; Achterberg
// 2007, ch. 7), where fedhpc.mip's vectorised Python version re-derived
// every row each round.
//
// Row i, lo_i <= sum_j a_ij x_j <= hi_i. Examining it computes its activity
// bounds exactly from the current domain (finite parts and counts of
// infinite contributions), so nothing drifts and no activity is stored. With
// the entry's own contribution removed (Brearley, Mitra & Williams 1975):
//
//     a > 0:  x_j <= (hi - minact^{-j}) / a,   x_j >= (lo - maxact^{-j}) / a
//     a < 0:  x_j >= (hi - minact^{-j}) / a,   x_j <= (lo - maxact^{-j}) / a
//
// A row whose slack hi - minact (lo side: maxact - lo) is at least its
// largest |a_j| (u_j - l_j) cannot tighten anything from that side and is
// skipped after the first pass. Integer bounds round inwards with int_tol;
// a continuous bound moves only by more than min_rel_change of its width.
// Updates apply at once (Gauss-Seidel): later entries of the same row use
// the activity computed at the scan's start, which holds the looser bounds,
// so every bound is still implied. Sound, incomplete, as the Python one.
//
// The domain lives here with a trail: change() tightens and propagates,
// undo(level) restores every bound changed since level(). Dives and
// backtracking fix-and-propagate push and pop without copying the domain.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace fedmip {

constexpr double kInf = std::numeric_limits<double>::infinity();

struct PropOptions {
    double int_tol = 1e-6;
    double feas_tol = 1e-6;
    double min_rel_change = 1e-3;
    double max_abs_bound = 1e9;
    long long max_work = 0;   // entry visits per call before stopping (0: 200 nnz + 1e6)
};

class Propagator {
public:
    PropOptions opt;

    Propagator(int m, int n, std::vector<int> rstart, std::vector<int> col, std::vector<double> val,
               std::vector<double> row_lo, std::vector<double> row_hi, std::vector<char> integer)
        : m_(m), n_(n), rstart_(std::move(rstart)), col_(std::move(col)), val_(std::move(val)),
          lo_(std::move(row_lo)), hi_(std::move(row_hi)), int_(std::move(integer)) {
        // column-wise copy: rows to wake when a bound changes
        cstart_.assign(n_ + 1, 0);
        for (int e = 0, ne = (int)col_.size(); e < ne; ++e) ++cstart_[col_[e] + 1];
        for (int j = 0; j < n_; ++j) cstart_[j + 1] += cstart_[j];
        crow_.resize(col_.size());
        std::vector<int> fill(cstart_.begin(), cstart_.end() - 1);
        for (int i = 0; i < m_; ++i)
            for (int e = rstart_[i]; e < rstart_[i + 1]; ++e) crow_[fill[col_[e]]++] = i;
        lb_.assign(n_, 0.0), ub_.assign(n_, 0.0);
        inq_.assign(m_, 0);
    }

    int rows() const { return m_; }
    int cols() const { return n_; }
    const std::vector<double>& lb() const { return lb_; }
    const std::vector<double>& ub() const { return ub_; }
    int infeasible_row() const { return bad_row_; }
    long long tightened() const { return tightened_; }
    int rounds() const { return rounds_; }
    long long work() const { return work_; }

    // a fresh domain (clears the trail)
    void reset(const double* lb, const double* ub) {
        lb_.assign(lb, lb + n_), ub_.assign(ub, ub + n_);
        trail_.clear();
    }
    size_t level() const { return trail_.size(); }
    void undo(size_t level) {
        while (trail_.size() > level) {
            const Change& c = trail_.back();
            (c.upper ? ub_ : lb_)[c.j] = c.old;
            trail_.pop_back();
        }
    }

    // Tighten the given bounds (never loosens) and propagate from their
    // rows; false when the domain becomes empty (the caller undoes).
    bool change(const int* js, const double* nl, const double* nu, int k, int max_rounds) {
        clear_queue();
        for (int t = 0; t < k; ++t) {
            const int j = js[t];
            bool moved = false;
            if (nl[t] > lb_[j]) set_bound(j, false, nl[t]), moved = true;
            if (nu[t] < ub_[j]) set_bound(j, true, nu[t]), moved = true;
            if (!moved) continue;
            if (lb_[j] > ub_[j] + opt.feas_tol * (1.0 + std::fabs(ub_[j]))) {
                bad_row_ = -1;
                return false;
            }
            wake(j);
        }
        return run(max_rounds);
    }

    // Propagate the whole current domain (every row, or the given ones).
    bool propagate_all(int max_rounds, const int* rows = nullptr, int nrows = -1) {
        clear_queue();
        if (rows) {
            for (int t = 0; t < nrows; ++t) push(rows[t]);
        } else {
            for (int i = 0; i < m_; ++i) push(i);
        }
        return run(max_rounds);
    }

    // activity bounds of every row under the current domain
    void activity_bounds(double* mn, double* mx) const {
        for (int i = 0; i < m_; ++i) {
            Act a = activity(i);
            mn[i] = a.nmin ? -kInf : a.fmin;
            mx[i] = a.nmax ? kInf : a.fmax;
        }
    }

private:
    struct Change {
        int j;
        bool upper;
        double old;
    };
    struct Act {
        double fmin = 0.0, fmax = 0.0, span = 0.0;   // span: max |a| (u - l) over finite ranges
        int nmin = 0, nmax = 0;
        bool inf_span = false;
    };

    int m_, n_;
    std::vector<int> rstart_, col_, cstart_, crow_;
    std::vector<double> val_, lo_, hi_;
    std::vector<char> int_;
    std::vector<double> lb_, ub_;
    std::vector<Change> trail_;
    std::vector<int> queue_, next_;
    std::vector<char> inq_;
    int bad_row_ = -1, rounds_ = 0;
    long long tightened_ = 0, work_ = 0;

    void set_bound(int j, bool upper, double v) {
        double& b = upper ? ub_[j] : lb_[j];
        trail_.push_back({j, upper, b});
        b = v;
        ++tightened_;
    }
    void push(int i) {
        if (!inq_[i]) inq_[i] = 1, next_.push_back(i);
    }
    void wake(int j) {
        for (int e = cstart_[j]; e < cstart_[j + 1]; ++e) push(crow_[e]);
    }
    void clear_queue() {
        for (int i : next_) inq_[i] = 0;
        for (int i : queue_) inq_[i] = 0;
        next_.clear(), queue_.clear();
        bad_row_ = -1, tightened_ = 0, rounds_ = 0, work_ = 0;
    }

    Act activity(int i) const {
        Act a;
        for (int e = rstart_[i]; e < rstart_[i + 1]; ++e) {
            const int j = col_[e];
            const double v = val_[e], l = lb_[j], u = ub_[j];
            const double cmin = v > 0 ? v * l : v * u, cmax = v > 0 ? v * u : v * l;
            if (std::isfinite(cmin)) a.fmin += cmin; else ++a.nmin;
            if (std::isfinite(cmax)) a.fmax += cmax; else ++a.nmax;
            const double w = std::fabs(v) * (u - l);
            if (std::isfinite(w)) a.span = std::max(a.span, w); else a.inf_span = true;
        }
        return a;
    }

    bool run(int max_rounds) {
        const long long cap = opt.max_work > 0 ? opt.max_work : 200LL * (long long)col_.size() + 1000000;
        while (!next_.empty() && rounds_ < max_rounds) {
            ++rounds_;
            queue_.swap(next_);
            next_.clear();
            for (size_t t = 0; t < queue_.size(); ++t) {
                const int i = queue_[t];
                inq_[i] = 0;
                if (!examine(i)) {
                    for (size_t u = t + 1; u < queue_.size(); ++u) inq_[queue_[u]] = 0;
                    queue_.clear();
                    return false;
                }
                if (work_ > cap) {
                    for (size_t u = t + 1; u < queue_.size(); ++u) inq_[queue_[u]] = 0;
                    queue_.clear();
                    return true;
                }
            }
            queue_.clear();
        }
        return true;
    }

    // examine row i: infeasibility, then the bounds it implies
    bool examine(int i) {
        const int s = rstart_[i], f = rstart_[i + 1];
        work_ += f - s;
        const Act a = activity(i);
        const double hi = hi_[i], lo = lo_[i];
        const double minact = a.nmin ? -kInf : a.fmin, maxact = a.nmax ? kInf : a.fmax;
        if (minact > hi + opt.feas_tol * (1.0 + std::fabs(hi)) || maxact < lo - opt.feas_tol * (1.0 + std::fabs(lo))) {
            bad_row_ = i;
            return false;
        }
        // which sides can tighten at all
        const bool use_hi = std::isfinite(hi) && a.nmin <= 1 && (a.inf_span || a.nmin == 1 || hi - a.fmin < a.span);
        const bool use_lo = std::isfinite(lo) && a.nmax <= 1 && (a.inf_span || a.nmax == 1 || a.fmax - lo < a.span);
        if (!use_hi && !use_lo) return true;
        work_ += f - s;
        for (int e = s; e < f; ++e) {
            const int j = col_[e];
            const double v = val_[e], l = lb_[j], u = ub_[j];
            double new_lb = -kInf, new_ub = kInf;
            if (use_hi) {
                const double cmin = v > 0 ? v * l : v * u;
                const bool own_inf = !std::isfinite(cmin);
                if (a.nmin - (int)own_inf == 0) {
                    const double b = (hi - (own_inf ? a.fmin : a.fmin - cmin)) / v;
                    if (v > 0) new_ub = b; else new_lb = b;
                }
            }
            if (use_lo) {
                const double cmax = v > 0 ? v * u : v * l;
                const bool own_inf = !std::isfinite(cmax);
                if (a.nmax - (int)own_inf == 0) {
                    const double b = (lo - (own_inf ? a.fmax : a.fmax - cmax)) / v;
                    if (v > 0) new_lb = std::max(new_lb, b); else new_ub = std::min(new_ub, b);
                }
            }
            if (!tighten(j, new_lb, new_ub)) {
                bad_row_ = i;
                return false;
            }
        }
        return true;
    }

    // clean and apply candidate bounds for column j; false when empty
    bool tighten(int j, double nlb, double nub) {
        const double l = lb_[j], u = ub_[j];
        const bool isint = int_[j] != 0;
        if (std::isfinite(nub) && isint) nub = std::floor(nub + opt.int_tol);
        if (std::isfinite(nlb) && isint) nlb = std::ceil(nlb - opt.int_tol);
        if (std::fabs(nub) > opt.max_abs_bound || nub != nub) nub = kInf;
        if (std::fabs(nlb) > opt.max_abs_bound || nlb != nlb) nlb = -kInf;
        double width = u - l;
        if (!std::isfinite(width)) width = std::max(1.0, std::fabs(std::isfinite(u) ? u : std::isfinite(l) ? l : 1.0));
        const double thr = isint ? 0.5 : opt.min_rel_change * std::max(width, 1e-6);
        bool moved = false;
        if (nub < u - thr) set_bound(j, true, nub), moved = true;
        if (nlb > l + thr) set_bound(j, false, nlb), moved = true;
        if (!moved) return true;
        double& L = lb_[j];
        double& U = ub_[j];
        if (L > U + opt.feas_tol * (1.0 + std::fabs(U))) return false;
        if (L > U) {   // crossing within tolerance: snap
            const double mid = isint ? std::round(L) : 0.5 * (L + U);
            set_bound(j, false, mid), set_bound(j, true, mid);
        }
        wake(j);
        return true;
    }
};

}  // namespace fedmip
