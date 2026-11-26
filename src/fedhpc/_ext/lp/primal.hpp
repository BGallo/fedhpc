// fedlp: bounded primal simplex for column generation masters.
//
// Column generation re-solves one LP after appending columns, which keeps
// the last basis primal feasible: the primal simplex simply continues from
// it, with the basis factor, duals, reduced costs and pricing weights all
// carried over. The techniques follow the simplex literature for the
// highly degenerate set-partitioning masters of branch and price:
//
// - Devex pricing (Harris 1973; Forrest & Goldfarb 1992). Reduced costs
//   and reference weights are maintained through the pivot row, computed
//   row-wise from the nonzeros of e_r' B^-1 (hyper-sparse pivot rows,
//   Hall & McKinnon), so a pivot does not scan every column.
// - Harris two-pass ratio test with bound flips and bound shifting: a
//   leaving variable slightly past its bound gets its bound shifted to its
//   value instead of a backward step (Koberstein 2005; SoPlex, CLP).
// - Bound perturbation against stalling: after a streak of degenerate
//   pivots, the bounds that degenerate basic variables sit on are relaxed
//   by small random amounts. At optimality the true bounds are restored
//   and a phase 1 / phase 2 clean-up removes any residual infeasibility.
// - A sum-of-infeasibilities phase 1 for crashed or warm-started bases that
//   are not primal feasible (bound changes, removed basic columns).
// - Stability control: the pivot element is computed both from the column
//   (FTRAN) and from the row; a disagreement triggers refactorisation, as
//   does a fixed pivot interval.
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <vector>

#include "dense_inverse.hpp"
#include "model.hpp"

namespace fedlp {

enum class Status : int { Optimal = 0, Infeasible = 1, Unbounded = 2, IterationLimit = 3, Numerical = 5 };

struct Options {
    double primal_tol = 1e-9;    // primal feasibility
    double dual_tol = 1e-9;      // optimality, scaled by 1 + |c_j|
    double pivot_tol = 1e-7;     // smallest acceptable pivot element
    double perturb = 5e-7;       // relative size of anti-degeneracy perturbation
    int degenerate_trigger = 10; // degenerate pivots in a row before perturbing
    int refactor_interval = 200; // pivots between refactorisations
    long long max_iter = 10'000'000;
    // Sifting: pivots work on the basic columns plus a working set of
    // candidate nonbasic columns; the rest stay frozen at their bounds until
    // a pricing pass with the current duals admits them. The working set is
    // capped at max(work_min, work_per_row * rows) nonbasic structurals.
    bool sifting = true;
    int work_min = 500;
    double work_per_row = 3.0;
};

struct Stats {
    long long iterations = 0, phase1_iterations = 0, flips = 0, degenerate = 0;
    long long refactors = 0, stability_refactors = 0, perturbations = 0, cleanups = 0;
    long long row_pivots = 0, col_pivots = 0;   // how pivot rows were computed
    long long sifts = 0, admitted = 0, evicted = 0;
    double seconds = 0.0;
};

class PrimalSimplex {
public:
    enum : char { BASIC = 0, AT_LB = 1, AT_UB = 2, DEAD = 3 };

    explicit PrimalSimplex(Model& M) : M_(M) { sync_columns(); }

    Options opt;

    // --- model change notifications (the façade calls these after it
    //     changes the Model) --------------------------------------------
    void sync_columns() {
        for (int j = (int)st_.size(); j < M_.cols(); ++j) add_var(j);
    }
    void column_removed(int j) {
        if (st_[j] == DEAD) return;
        if (st_[j] == BASIC) {
            basis_valid_ = false;   // re-crash on the next solve
        } else if (basis_valid_ && x_[j] != 0.0) {
            shift_basics(j, -x_[j]);
        }
        st_[j] = DEAD;
        x_[j] = 0.0;
    }
    void bounds_changed(int j) {
        if (st_[j] == DEAD || st_[j] == BASIC) return;   // a basic one is fixed by phase 1
        const double nx = rest_value(j, st_[j]);
        st_[j] = std::isfinite(M_.lb(j)) && nx == M_.lb(j) ? AT_LB : AT_UB;
        if (basis_valid_ && nx != x_[j]) shift_basics(j, nx - x_[j]);
        x_[j] = nx;
    }

    Status solve() {
        auto t0 = std::chrono::steady_clock::now();
        sync_columns();
        next_valid_ = false;
        degenerate_ = 0;
        Status s = solve_impl();
        stats_.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return s;
    }

    // --- results ----------------------------------------------------------
    double value(int j) const { return st_[j] == BASIC ? xb_[pos_[j]] : x_[j]; }
    double objective() const {
        double v = 0.0;
        for (int j = 0; j < M_.cols(); ++j)
            if (st_[j] != DEAD) v += M_.obj(j) * value(j);
        return v;
    }
    const std::vector<double>& duals() const { return y_; }
    double reduced_cost(int j) const { return st_[j] == BASIC || st_[j] == DEAD ? 0.0 : d_[j]; }
    char var_status(int j) const { return st_[j]; }
    const Stats& stats() const { return stats_; }
    void reset_stats() { stats_ = Stats(); }
    void invalidate_basis() { basis_valid_ = false; }

private:
    Model& M_;
    DenseInverse binv_;
    bool basis_valid_ = false;
    // per variable (logicals first)
    std::vector<char> st_;
    std::vector<int> pos_;
    std::vector<double> x_, d_, w_, pl_, pu_;
    // per basis position / row
    std::vector<int> head_;
    std::vector<double> xb_, y_;
    // work buffers
    std::vector<double> alpha_, rho_, acc_;
    std::vector<int> touched_;
    std::vector<char> in_touched_;
    // sifting working set: columns pivots and pricing look at (logicals,
    // new and recently attractive columns); in_work_ marks membership
    std::vector<int> work_;
    std::vector<char> in_work_;
    bool perturbed_ = false;
    int since_refactor_ = 0;
    int degenerate_ = 0;        // degenerate pivots in a row, this solve
    int next_q_ = -1;           // entering column chosen by the last fused pivot row
    bool next_valid_ = false;
    unsigned long long rng_ = 0x2545F4914F6CDD1Dull;
    Stats stats_;

    // ------------------------------------------------------------------ util
    double urand() {
        rng_ ^= rng_ << 13, rng_ ^= rng_ >> 7, rng_ ^= rng_ << 17;
        return (double)(rng_ >> 11) * (1.0 / 9007199254740992.0);
    }
    double lbw(int j) const { return M_.lb(j) - pl_[j]; }
    double ubw(int j) const { return M_.ub(j) + pu_[j]; }
    // Optimality tolerance of column j: relative to its cost, plus the
    // rounding of d_j = c_j - y'a_j itself, which with big-M artificials in
    // the master is a difference of large duals (~ eps |y|_inf nnz(a_j)).
    double dtol(int j) const {
        return opt.dual_tol * (1.0 + std::fabs(M_.obj(j))) + 1e-14 * ymax_ * (1.0 + M_.col_nnz(j));
    }
    double ymax_ = 0.0;   // |y|_inf, kept current with y
    void update_ymax() {
        double v = 0.0;
        for (double x : y_) v = std::max(v, std::fabs(x));
        ymax_ = v;
    }

    // value a nonbasic variable rests at, for a status
    double rest_value(int j, char s) const {
        const double l = M_.lb(j), u = M_.ub(j);
        if (s == AT_UB && std::isfinite(u)) return u;
        return std::isfinite(l) ? l : u;
    }

    void add_var(int j) {
        st_.push_back(DEAD), pos_.push_back(-1);
        x_.push_back(0.0), d_.push_back(0.0), w_.push_back(1.0), pl_.push_back(0.0), pu_.push_back(0.0);
        acc_.push_back(0.0), in_touched_.push_back(0), in_work_.push_back(0);
        if (!M_.alive(j)) return;
        admit(j);   // a new column is a candidate
        const char s = std::isfinite(M_.lb(j)) ? AT_LB : AT_UB;
        st_[j] = s;
        x_[j] = rest_value(j, s);
        if (basis_valid_) {
            if (x_[j] != 0.0) shift_basics(j, x_[j]);
            d_[j] = M_.obj(j) - M_.dot(y_.data(), j);
        }
    }

    void admit(int j) {
        if (!in_work_[j]) in_work_[j] = 1, work_.push_back(j);
    }

    // nonbasic j moved by delta: x_B -= delta B^-1 a_j
    void shift_basics(int j, double delta) {
        const int m = M_.rows();
        alpha_.resize(m);
        binv_.ftran(M_, j, alpha_.data());
        for (int i = 0; i < m; ++i) xb_[i] -= delta * alpha_[i];
    }

    // ------------------------------------------------------------ basis
    // Logical basis, except that an equality row takes a structural
    // singleton column with a positive coefficient when there is one (the
    // master's artificials and null columns) — usually a feasible start.
    void crash() {
        const int m = M_.rows(), n = M_.cols();
        for (int j = 0; j < n; ++j) {
            if (st_[j] == DEAD) continue;
            const char s = std::isfinite(M_.lb(j)) ? AT_LB : AT_UB;
            st_[j] = s, x_[j] = rest_value(j, s), pos_[j] = -1, w_[j] = 1.0;
            pl_[j] = pu_[j] = 0.0;
        }
        head_.assign(m, -1);
        for (int r = 0; r < m; ++r)
            if (M_.sense(r) != 'E') head_[r] = r;
        for (int j = m; j < n; ++j) {
            if (st_[j] == DEAD || M_.col_nnz(j) != 1) continue;
            const int r = M_.col_idx(j)[0];
            const double a = M_.col_val(j)[0];
            if (head_[r] >= 0 || a <= 0.0) continue;
            const double v = M_.rhs()[r] / a;
            if (v >= M_.lb(j) - 1e-12 && v <= M_.ub(j) + 1e-12) head_[r] = j;
        }
        for (int r = 0; r < m; ++r)
            if (head_[r] < 0) head_[r] = r;
        for (int i = 0; i < m; ++i) st_[head_[i]] = BASIC, pos_[head_[i]] = i;
        for (int j = 0; j < n; ++j)
            if (st_[j] != DEAD) admit(j);
        perturbed_ = false;
        binv_.reset(m);
        basis_valid_ = refactor();
    }

    bool refactor() {
        ++stats_.refactors;
        since_refactor_ = 0;
        if (!binv_.factorize(M_, head_)) return false;
        compute_xb();
        compute_y();
        compute_d();
        return true;
    }

    void compute_xb() {
        const int m = M_.rows();
        std::vector<double> r(M_.rhs());
        for (int j = 0; j < M_.cols(); ++j) {
            if (st_[j] == BASIC || st_[j] == DEAD || x_[j] == 0.0) continue;
            const int* ix = M_.col_idx(j);
            const double* v = M_.col_val(j);
            for (int k = 0, nz = M_.col_nnz(j); k < nz; ++k) r[ix[k]] -= v[k] * x_[j];
        }
        xb_.assign(m, 0.0);
        for (int i = 0; i < m; ++i) {
            const double* bi = binv_.row(i);
            double s = 0.0;
            for (int k = 0; k < m; ++k) s += bi[k] * r[k];
            xb_[i] = s;
        }
    }
    void compute_y() {
        const int m = M_.rows();
        y_.assign(m, 0.0);
        for (int i = 0; i < m; ++i) {
            const double c = M_.obj(head_[i]);
            if (c == 0.0) continue;
            const double* bi = binv_.row(i);
            for (int k = 0; k < m; ++k) y_[k] += c * bi[k];
        }
        update_ymax();
    }
    void compute_d() {
        for (int j = 0; j < M_.cols(); ++j)
            d_[j] = st_[j] == BASIC || st_[j] == DEAD ? 0.0 : M_.obj(j) - M_.dot(y_.data(), j);
    }

    // largest bound violation of a basic variable, against working bounds
    double infeasibility() const {
        double v = 0.0;
        for (int i = 0; i < M_.rows(); ++i) {
            const int j = head_[i];
            v = std::max(v, lbw(j) - xb_[i]);
            v = std::max(v, xb_[i] - ubw(j));
        }
        return v;
    }

    // ------------------------------------------------------------ pricing
    bool eligible(int j) const {
        const char s = st_[j];
        if (s == BASIC || s == DEAD || lbw(j) == ubw(j)) return false;
        const double d = d_[j], t = dtol(j);
        return (s == AT_LB && d < -t) || (s == AT_UB && d > t);
    }
    int select_devex() const {
        int q = -1;
        double best = 0.0;
        for (int j : work_) {
            if (!eligible(j)) continue;
            const double sc = d_[j] * d_[j] / w_[j];
            if (sc > best) best = sc, q = j;
        }
        return q;
    }

    // ------------------------------------------------------------ ratio test
    struct Step {
        int r = -1;          // leaving position, -1: bound flip
        double theta = 0.0;
        bool unbounded = false;
        bool hit_lo = false; // the leaving variable reached lo(r) (else hi(r))
    };

    // Harris two-pass test for entering q moving in direction dir (+1 up).
    // lo/hi(i): bounds the basic variable at position i must respect.
    template <class Lo, class Hi>
    Step ratio_test(int q, double dir, Lo lo, Hi hi) {
        const int m = M_.rows();
        const double tol = opt.primal_tol, piv = opt.pivot_tol;
        const double range = ubw(q) - lbw(q);
        double tmax = range;
        for (int i = 0; i < m; ++i) {
            const double a = dir * alpha_[i];
            if (a > piv) {
                const double l = lo(i);
                if (l > -kInf) tmax = std::min(tmax, (xb_[i] - l + tol) / a);
            } else if (a < -piv) {
                const double u = hi(i);
                if (u < kInf) tmax = std::min(tmax, (u - xb_[i] + tol) / -a);
            }
        }
        Step s;
        if (!(tmax < kInf)) {
            s.unbounded = true;
            return s;
        }
        if (range <= tmax) {   // bound flip of the entering variable
            s.theta = range;
            return s;
        }
        double bestpiv = 0.0;
        for (int i = 0; i < m; ++i) {
            const double a = dir * alpha_[i];
            double t;
            if (a > piv) {
                const double l = lo(i);
                if (!(l > -kInf)) continue;
                t = (xb_[i] - l) / a;
            } else if (a < -piv) {
                const double u = hi(i);
                if (!(u < kInf)) continue;
                t = (u - xb_[i]) / -a;
            } else {
                continue;
            }
            if (t <= tmax && std::fabs(a) > bestpiv) bestpiv = std::fabs(a), s.r = i, s.theta = t, s.hit_lo = a > 0;
        }
        if (s.theta < 0.0) s.theta = 0.0;   // the leaving bound gets shifted (see pivot)
        return s;
    }

    // ------------------------------------------------------------ pivoting
    // Pivot q in at position r after a step theta in direction dir. With
    // `maintain`, updates the reduced costs, Devex weights and duals
    // through the row-wise pivot row (phase 2); otherwise they are
    // recomputed later (phase 1). Returns false on numerical failure.
    bool pivot(int q, int r, double dir, double theta, bool maintain, bool to_upper) {
        const int m = M_.rows();
        const int l = head_[r];
        // primal update
        for (int i = 0; i < m; ++i) xb_[i] -= dir * theta * alpha_[i];
        const double xq = x_[q] + dir * theta;
        // the leaving variable rests at the bound it reached; when Harris
        // let it overshoot, that bound is shifted to its value
        const double xl = xb_[r];
        if (to_upper) {
            if (xl > ubw(l)) pu_[l] += xl - ubw(l), perturbed_ = true;
            x_[l] = ubw(l), st_[l] = AT_UB;
        } else {
            if (xl < lbw(l)) pl_[l] += lbw(l) - xl, perturbed_ = true;
            x_[l] = lbw(l), st_[l] = AT_LB;
        }
        pos_[l] = -1;
        admit(l);
        bool stable = true;
        if (maintain) {
            const double arq = alpha_[r];
            rho_.assign(binv_.row(r), binv_.row(r) + m);
            const double td = d_[q] / arq;
            const double wq = w_[q];
            const double inv2 = 1.0 / (arq * arq);
            double arq_row = 0.0, wmax = 0.0;
            // The pivot row alpha_r = e_r' B^-1 A over the nonbasic columns:
            // row-wise through the nonzeros of e_r' B^-1 when that is
            // sparse, else column by column (sequential memory), fused with
            // the reduced-cost and Devex updates and the choice of the next
            // entering column (hybrid, as in HiGHS).
            long long rowcost = 0;
            for (int k = 0; k < m; ++k)
                if (std::fabs(rho_[k]) >= 1e-14) rowcost += (long long)M_.row(k).size();
            if (3 * rowcost < M_.nnz()) {
                ++stats_.row_pivots;
                for (int k = 0; k < m; ++k) {
                    const double rk = rho_[k];
                    if (std::fabs(rk) < 1e-14) continue;
                    for (const auto& e : M_.row(k)) {
                        const int j = e.col;
                        if (st_[j] == BASIC || st_[j] == DEAD || !in_work_[j]) continue;
                        if (!in_touched_[j]) in_touched_[j] = 1, touched_.push_back(j), acc_[j] = 0.0;
                        acc_[j] += rk * e.val;
                    }
                }
                arq_row = in_touched_[q] ? acc_[q] : 0.0;
                for (int j : touched_) {
                    const double a = acc_[j];
                    if (j != q && j != l) {
                        d_[j] -= td * a;
                        w_[j] = std::max(w_[j], a * a * inv2 * wq);
                        wmax = std::max(wmax, w_[j]);
                    }
                    in_touched_[j] = 0, acc_[j] = 0.0;
                }
                touched_.clear();
                d_[l] = -td;
                w_[l] = std::max(wq * inv2, 1.0);
            } else {
                ++stats_.col_pivots;
                d_[l] = -td;
                w_[l] = std::max(wq * inv2, 1.0);
                double best = 0.0;
                int bq = -1;
                if (eligible(l)) best = d_[l] * d_[l] / w_[l], bq = l;
                arq_row = M_.dot(rho_.data(), q);
                for (int j : work_) {
                    const char sj = st_[j];
                    if (sj == BASIC || sj == DEAD || j == l || j == q) continue;
                    const double a = M_.dot(rho_.data(), j);
                    if (a != 0.0) {
                        d_[j] -= td * a;
                        w_[j] = std::max(w_[j], a * a * inv2 * wq);
                        wmax = std::max(wmax, w_[j]);
                    }
                    if (eligible(j)) {
                        const double sc = d_[j] * d_[j] / w_[j];
                        if (sc > best) best = sc, bq = j;
                    }
                }
                next_q_ = bq;
                next_valid_ = true;
            }
            stable = std::fabs(arq_row - arq) <= 1e-9 * (1.0 + std::fabs(arq));
            for (int k = 0; k < m; ++k) y_[k] += td * rho_[k];
            update_ymax();
            if (wmax > 1e8) {   // reset the reference framework
                std::fill(w_.begin(), w_.end(), 1.0);
                next_valid_ = false;
            }
        }
        binv_.update(r, alpha_.data());
        head_[r] = q, pos_[q] = r, st_[q] = BASIC, xb_[r] = xq, d_[q] = 0.0;
        if (!stable) ++stats_.stability_refactors;
        if (!stable || ++since_refactor_ >= opt.refactor_interval) {
            next_valid_ = false;   // the reduced costs are recomputed
            if (!refactor()) return false;
        }
        return true;
    }

    // Optimal on the working set: price the frozen columns with the
    // current duals and admit the eligible ones; evict the least attractive
    // working columns past the cap. Returns whether any column was admitted.
    bool sift() {
        ++stats_.sifts;
        const int m = M_.rows(), n = M_.cols();
        const size_t cap = (size_t)std::max<double>(opt.work_min, opt.work_per_row * m);
        int added = 0;
        for (int j = m; j < n; ++j) {
            if (in_work_[j] || st_[j] == BASIC || st_[j] == DEAD) continue;
            d_[j] = M_.obj(j) - M_.dot(y_.data(), j);
            if (eligible(j)) {
                w_[j] = 1.0;
                admit(j);
                ++added;
            }
        }
        stats_.admitted += added;
        // compact the list; past the cap, evict the nonbasic structurals
        // whose reduced costs keep them furthest from entering
        std::vector<std::pair<double, int>> cand;
        size_t w = 0;
        for (int j : work_) {
            if (st_[j] == DEAD) { in_work_[j] = 0; continue; }
            if (j >= m && st_[j] != BASIC && !eligible(j))
                cand.push_back({st_[j] == AT_LB ? d_[j] : -d_[j], j});
            work_[w++] = j;
        }
        work_.resize(w);
        if (cand.size() > cap) {
            const size_t drop = cand.size() - cap / 2;
            std::nth_element(cand.begin(), cand.begin() + drop, cand.end(),
                             [](const auto& a, const auto& b) { return a.first > b.first; });
            for (size_t k = 0; k < drop; ++k) in_work_[cand[k].second] = 0;
            stats_.evicted += drop;
            w = 0;
            for (int j : work_)
                if (in_work_[j]) work_[w++] = j;
            work_.resize(w);
        }
        return added > 0;
    }

    // after a fresh compute_d: admit every eligible column; any?
    bool admit_eligible_all() {
        bool any = false;
        for (int j = 0, n = M_.cols(); j < n; ++j)
            if (eligible(j)) admit(j), any = true;
        return any;
    }

    // relax the bounds degenerate basic variables sit on
    void perturb() {
        ++stats_.perturbations;
        perturbed_ = true;
        for (int i = 0; i < M_.rows(); ++i) {
            const int j = head_[i];
            const double l = lbw(j), u = ubw(j);
            const double tl = opt.primal_tol * (1.0 + std::fabs(l));
            if (std::isfinite(l) && std::fabs(xb_[i] - l) <= tl)
                pl_[j] += (1.0 + urand()) * opt.perturb * (1.0 + std::fabs(M_.lb(j)));
            const double tu = opt.primal_tol * (1.0 + std::fabs(u));
            if (std::isfinite(u) && std::fabs(xb_[i] - u) <= tu)
                pu_[j] += (1.0 + urand()) * opt.perturb * (1.0 + std::fabs(M_.ub(j)));
        }
    }
    void unperturb() {
        perturbed_ = false;
        for (int j = 0; j < M_.cols(); ++j) {
            pl_[j] = pu_[j] = 0.0;
            if (st_[j] == AT_LB || st_[j] == AT_UB) x_[j] = rest_value(j, st_[j]);
        }
        compute_xb();
    }

    // ------------------------------------------------------------ phases
    // Phase 2 from a primal feasible basis. Returns Optimal when no
    // column prices out (against the maintained reduced costs).
    Status phase2(bool allow_perturb) {
        int& degenerate = degenerate_;   // the streak spans the rounds of a solve
        next_valid_ = false;
        while (true) {
            if (stats_.iterations >= opt.max_iter) return Status::IterationLimit;
            int q = next_valid_ ? next_q_ : select_devex();
            next_valid_ = false;
            if (q < 0) {
                if (!(opt.sifting && sift())) return Status::Optimal;
                q = select_devex();
                if (q < 0) return Status::Optimal;
            }
            const double dir = d_[q] < 0 ? 1.0 : -1.0;
            alpha_.resize(M_.rows());
            binv_.ftran(M_, q, alpha_.data());
            Step s = ratio_test(q, dir, [&](int i) { return lbw(head_[i]); },
                                [&](int i) { return ubw(head_[i]); });
            if (s.unbounded) return Status::Unbounded;
            ++stats_.iterations;
            if (s.r < 0) {
                for (int i = 0; i < M_.rows(); ++i) xb_[i] -= dir * s.theta * alpha_[i];
                x_[q] = dir > 0 ? ubw(q) : lbw(q);
                st_[q] = dir > 0 ? AT_UB : AT_LB;
                ++stats_.flips;
                continue;
            }
            const bool to_upper = !s.hit_lo;
            const bool degen = s.theta * std::fabs(alpha_[s.r]) <= opt.primal_tol;
            if (degen) ++stats_.degenerate;
            degenerate = degen ? degenerate + 1 : 0;
            if (!pivot(q, s.r, dir, s.theta, true, to_upper)) return Status::Numerical;
            if (allow_perturb && degenerate >= opt.degenerate_trigger) {
                perturb();
                degenerate = 0;
            }
        }
    }

    // Sum-of-infeasibilities phase 1: an infeasible basic variable's
    // violated bound is relaxed to infinity and it gets cost -1 (below its
    // lower bound) or +1 (above its upper bound).
    Status phase1() {
        const int m = M_.rows();
        std::vector<double> c1(m), y1(m);
        while (true) {
            if (stats_.iterations >= opt.max_iter) return Status::IterationLimit;
            bool any = false;
            for (int i = 0; i < m; ++i) {
                const int j = head_[i];
                const double tl = opt.primal_tol * (1.0 + std::fabs(lbw(j)));
                const double tu = opt.primal_tol * (1.0 + std::fabs(ubw(j)));
                c1[i] = xb_[i] < lbw(j) - tl ? -1.0 : xb_[i] > ubw(j) + tu ? 1.0 : 0.0;
                any |= c1[i] != 0.0;
            }
            if (!any) return Status::Optimal;
            std::fill(y1.begin(), y1.end(), 0.0);
            for (int i = 0; i < m; ++i) {
                if (c1[i] == 0.0) continue;
                const double* bi = binv_.row(i);
                for (int k = 0; k < m; ++k) y1[k] += c1[i] * bi[k];
            }
            int q = -1;
            double best = 0.0, dq = 0.0;
            for (int j = 0, n = M_.cols(); j < n; ++j) {
                const char s = st_[j];
                if (s == BASIC || s == DEAD || lbw(j) == ubw(j)) continue;
                const double d = -M_.dot(y1.data(), j);
                if ((s == AT_LB && d < -1e-12) || (s == AT_UB && d > 1e-12)) {
                    const double sc = d * d / w_[j];
                    if (sc > best) best = sc, q = j, dq = d;
                }
            }
            if (q < 0) return Status::Infeasible;
            const double dir = dq < 0 ? 1.0 : -1.0;
            alpha_.resize(m);
            binv_.ftran(M_, q, alpha_.data());
            // an infeasible variable may move up to its violated bound (and
            // leaves the basis there), not past it
            Step s = ratio_test(
                q, dir, [&](int i) { return c1[i] < 0 ? -kInf : c1[i] > 0 ? ubw(head_[i]) : lbw(head_[i]); },
                [&](int i) { return c1[i] > 0 ? kInf : c1[i] < 0 ? lbw(head_[i]) : ubw(head_[i]); });
            if (s.unbounded) return Status::Infeasible;   // cannot reduce the infeasibility further
            ++stats_.iterations, ++stats_.phase1_iterations;
            if (s.r < 0) {
                for (int i = 0; i < m; ++i) xb_[i] -= dir * s.theta * alpha_[i];
                x_[q] = dir > 0 ? ubw(q) : lbw(q);
                st_[q] = dir > 0 ? AT_UB : AT_LB;
                continue;
            }
            // below its lower bound, it stops at it (AT_LB); above its upper, AT_UB
            const bool to_upper = c1[s.r] < 0 ? false : c1[s.r] > 0 ? true : !s.hit_lo;
            if (!pivot(q, s.r, dir, s.theta, false, to_upper)) return Status::Numerical;
        }
    }

    Status solve_impl() {
        if (!basis_valid_) {
            crash();
            if (!basis_valid_) return Status::Numerical;
        }
        bool allow_perturb = true;
        int cleanups_here = 0;
        for (int round = 0; round < 50; ++round) {
            if (infeasibility() > opt.primal_tol) {
                Status s = phase1();
                compute_y(), compute_d();
                if (s != Status::Optimal) return s;
            }
            Status s = phase2(allow_perturb);
            if (s != Status::Optimal) return s;
            if (perturbed_) {
                ++stats_.cleanups;
                unperturb();
                // after repeated clean-ups, finish on the true bounds
                allow_perturb = ++cleanups_here < 3;
                if (infeasibility() > opt.primal_tol) continue;
            }
            // fresh duals and reduced costs over every column before
            // declaring optimality
            compute_y(), compute_d();
            if (!admit_eligible_all()) return Status::Optimal;
        }
        return Status::Numerical;
    }
};

}  // namespace fedlp
