// Feasibility Jump (Luteberget & Sartor, "Feasibility Jump: an LP-free
// Lagrangian MIP heuristic", Math. Prog. Comp. 2023; HiGHS's "J" source):
// weighted local search over the original rows, no LP.
//
// Each row i has a weight w_i (start 1) and a violation
// viol_i(x) = max(0, lo_i - a_i'x, a_i'x - hi_i). A variable's *jump value*
// is the value in its domain minimising sum_i w_i viol_i over its rows (a
// convex piecewise-linear function of x_j: its minimum is at a breakpoint
// where a row's activity reaches one of its bounds, or at a domain bound),
// plus w_obj c_j x_j. A step samples variables of violated rows, moves the
// one whose jump lowers the weighted violation most; when none does (a
// local minimum), the weights of the violated rows grow by one and a random
// variable of a random violated row jumps. A feasible point is kept as the
// incumbent; with obj_weight > 0 the search continues for better objective
// values within the budget (FJ's later phase).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace fedmip {

struct FJResult {
    bool found = false;
    std::vector<double> x;
    double objective = 0.0;
    long long steps = 0, work = 0;
};

class FeasibilityJump {
public:
    FeasibilityJump(int m, int n, const int* rstart, const int* col, const double* val, const double* lo,
                    const double* hi, const double* lb, const double* ub, const char* integer, const double* c)
        : m_(m), n_(n), rs_(rstart, rstart + m + 1), col_(col, col + rstart[m]), val_(val, val + rstart[m]),
          lo_(lo, lo + m), hi_(hi, hi + m), lb_(lb, lb + n), ub_(ub, ub + n), int_(integer, integer + n),
          c_(c, c + n) {
        cs_.assign(n_ + 1, 0);
        for (int e = 0; e < (int)col_.size(); ++e) ++cs_[col_[e] + 1];
        for (int j = 0; j < n_; ++j) cs_[j + 1] += cs_[j];
        crow_.resize(col_.size()), cval_.resize(col_.size());
        std::vector<int> fill(cs_.begin(), cs_.end() - 1);
        for (int i = 0; i < m_; ++i)
            for (int e = rs_[i]; e < rs_[i + 1]; ++e) {
                const int p = fill[col_[e]]++;
                crow_[p] = i, cval_[p] = val_[e];
            }
    }

    FJResult run(const double* x0, long long max_work, unsigned long long seed, double obj_weight, int sample = 25) {
        rng_ = seed * 0x9E3779B97F4A7C15ull + 1;
        x_.assign(n_, 0.0);
        for (int j = 0; j < n_; ++j) {
            double v = x0 ? x0[j] : 0.0;
            v = std::min(std::max(v, lb_[j]), ub_[j]);
            if (!std::isfinite(v)) v = std::isfinite(lb_[j]) ? lb_[j] : std::isfinite(ub_[j]) ? ub_[j] : 0.0;
            if (int_[j]) v = std::round(v);
            x_[j] = std::min(std::max(v, lb_[j]), ub_[j]);
        }
        act_.assign(m_, 0.0);
        for (int i = 0; i < m_; ++i)
            for (int e = rs_[i]; e < rs_[i + 1]; ++e) act_[i] += val_[e] * x_[col_[e]];
        w_.assign(m_, 1.0);
        vpos_.assign(m_, -1), viol_.clear();
        for (int i = 0; i < m_; ++i)
            if (violation(i) > 0) add_viol(i);
        FJResult res;
        double best_obj = std::numeric_limits<double>::infinity();
        long long work = 0, steps = 0, since_best = 0;
        while (work < max_work) {
            if (viol_.empty()) {
                const double obj = objective();
                if (obj < best_obj - 1e-9) {
                    best_obj = obj, res.found = true, res.x = x_, res.objective = obj, since_best = 0;
                }
                if (obj_weight <= 0.0 || ++since_best > 2000) break;
            }
            ++steps;
            // candidates: variables of sampled violated rows (or, feasible,
            // with an objective to improve: random columns with a cost)
            int bestj = -1;
            double bestv = 0.0, bestgain = 1e-9;
            const int tries = viol_.empty() ? sample : std::min<int>(sample, 4 * (int)viol_.size() + 1);
            for (int t = 0; t < tries; ++t) {
                int j;
                if (viol_.empty()) {
                    j = (int)(next() % (unsigned long long)n_);
                } else {
                    const int i = viol_[next() % viol_.size()];
                    const int len = rs_[i + 1] - rs_[i];
                    if (len == 0) continue;
                    j = col_[rs_[i] + (int)(next() % (unsigned long long)len)];
                }
                if (lb_[j] == ub_[j]) continue;
                double v, gain;
                work += jump(j, obj_weight, v, gain);
                if (gain > bestgain) bestgain = gain, bestj = j, bestv = v;
            }
            if (bestj < 0) {
                if (viol_.empty()) break;
                // local minimum: heavier violated rows, then a forced move
                for (int i : viol_) w_[i] += 1.0;
                const int i = viol_[next() % viol_.size()];
                const int len = rs_[i + 1] - rs_[i];
                if (len == 0) break;
                const int j = col_[rs_[i] + (int)(next() % (unsigned long long)len)];
                double v, gain;
                work += jump(j, 0.0, v, gain);
                if (v == x_[j]) continue;
                move(j, v);
                work += cs_[j + 1] - cs_[j];
                continue;
            }
            move(bestj, bestv);
            work += cs_[bestj + 1] - cs_[bestj];
        }
        res.steps = steps, res.work = work;
        return res;
    }

private:
    int m_, n_;
    std::vector<int> rs_, col_, cs_, crow_;
    std::vector<double> val_, cval_, lo_, hi_, lb_, ub_, c_;
    std::vector<char> int_;
    std::vector<double> x_, act_, w_;
    std::vector<int> viol_, vpos_;
    std::vector<std::pair<double, double>> bp_;   // (value, slope change) breakpoints
    unsigned long long rng_ = 1;

    unsigned long long next() {
        rng_ ^= rng_ << 13, rng_ ^= rng_ >> 7, rng_ ^= rng_ << 17;
        return rng_;
    }
    double tol(double b) const { return 1e-6 * (1.0 + std::fabs(b)); }
    double violation(int i) const {
        const double a = act_[i];
        if (a < lo_[i] - tol(lo_[i])) return lo_[i] - a;
        if (a > hi_[i] + tol(hi_[i])) return a - hi_[i];
        return 0.0;
    }
    void add_viol(int i) { vpos_[i] = (int)viol_.size(), viol_.push_back(i); }
    void del_viol(int i) {
        const int p = vpos_[i], last = viol_.back();
        viol_[p] = last, vpos_[last] = p, viol_.pop_back(), vpos_[i] = -1;
    }
    double objective() const {
        double s = 0.0;
        for (int j = 0; j < n_; ++j) s += c_[j] * x_[j];
        return s;
    }
    void move(int j, double v) {
        const double d = v - x_[j];
        x_[j] = v;
        for (int p = cs_[j]; p < cs_[j + 1]; ++p) {
            const int i = crow_[p];
            act_[i] += cval_[p] * d;
            const bool bad = violation(i) > 0;
            if (bad && vpos_[i] < 0) add_viol(i);
            else if (!bad && vpos_[i] >= 0) del_viol(i);
        }
    }

    // The jump value of j: argmin over its domain of
    //   f(v) = sum_i w_i viol_i(act_i + a_ij (v - x_j)) + ow c_j v,
    // convex piecewise linear: scan its breakpoints in order, tracking the
    // slope. Returns the work done; v and the gain f(x_j) - f(v).
    long long jump(int j, double ow, double& v, double& gain) {
        const double xj = x_[j], l = lb_[j], u = ub_[j];
        bp_.clear();
        double slope = ow * c_[j];        // slope of f at v -> -inf
        double f0 = ow * c_[j] * xj;
        for (int p = cs_[j]; p < cs_[j + 1]; ++p) {
            const int i = crow_[p];
            const double a = cval_[p], w = w_[i], r = act_[i] - a * xj;   // activity without j
            f0 += w * violation(i);
            // the row term as a function of v: w * max(0, lo - r - a v, r + a v - hi)
            if (std::isfinite(lo_[i])) {
                const double t = (lo_[i] - r) / a;          // a v = lo - r
                // below lo: slope -w a for a > 0 (v < t), +w|a| ... as a > 0:
                // term w (lo - r - a v) for v < t, so slope -w a there
                if (a > 0) slope -= w * a, bp_.push_back({t, w * a});
                else bp_.push_back({t, w * (-a)});          // a < 0: term grows for v > t
            }
            if (std::isfinite(hi_[i])) {
                const double t = (hi_[i] - r) / a;
                if (a > 0) bp_.push_back({t, w * a});
                else slope -= w * (-a), bp_.push_back({t, w * (-a)});
            }
        }
        std::sort(bp_.begin(), bp_.end());
        // the minimiser: the first point where the slope turns >= 0
        double best = std::isfinite(l) ? l : (std::isfinite(u) ? u : xj);
        if (slope < 0) {
            best = std::isfinite(u) ? u : xj;
            for (const auto& [t, ds] : bp_) {
                slope += ds;
                if (slope >= 0) {
                    best = t;
                    break;
                }
            }
        }
        best = std::min(std::max(best, l), u);
        if (int_[j]) {
            // the better of the two integers around it (convexity)
            const double fl = std::max(l, std::floor(best)), ce = std::min(u, std::ceil(best));
            best = value_at(j, fl, ow) <= value_at(j, ce, ow) ? fl : ce;
        }
        v = best;
        gain = f0 - value_at(j, best, ow);
        return 2 * (cs_[j + 1] - cs_[j]) + (long long)bp_.size();
    }
    double value_at(int j, double v, double ow) const {
        const double d = v - x_[j];
        double f = ow * c_[j] * v;
        for (int p = cs_[j]; p < cs_[j + 1]; ++p) {
            const int i = crow_[p];
            const double a = act_[i] + cval_[p] * d;
            double vi = 0.0;
            if (a < lo_[i] - tol(lo_[i])) vi = lo_[i] - a;
            else if (a > hi_[i] + tol(hi_[i])) vi = a - hi_[i];
            f += w_[i] * vi;
        }
        return f;
    }
};

}  // namespace fedmip
