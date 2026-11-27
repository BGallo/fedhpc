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
#include <cstdio>
#include <cstdint>
#include <vector>

#include <memory>
#include <string>

#include "factor.hpp"
#include "lu_ft.hpp"

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define FEDLP_X86 1
#endif

namespace fedlp {
namespace kern {
// a[j] = v0[j] rho[i0[j]] + v1[j] rho[i1[j]] + v2[j] rho[i2[j]] over a padded
// 3-wide ELLPACK copy of the columns (padding: index m, where rho is 0). The
// dual's column-wise PRICE ran at 3.7 instructions a cycle, ~45 a column:
// instruction-bound, so 4 columns a step with AVX2 gathers and FMAs.
inline void ell3_scalar(const int* i0, const int* i1, const int* i2, const double* v0, const double* v1,
                        const double* v2, const double* rho, double* a, int n) {
    for (int j = 0; j < n; ++j) a[j] = v0[j] * rho[i0[j]] + v1[j] * rho[i1[j]] + v2[j] * rho[i2[j]];
}
#ifdef FEDLP_X86
__attribute__((target("avx2,fma"))) inline void ell3_avx2(const int* i0, const int* i1, const int* i2,
                                                            const double* v0, const double* v1, const double* v2,
                                                            const double* rho, double* a, int n) {
    int j = 0;
    for (; j + 4 <= n; j += 4) {
        const __m256d r0 = _mm256_i32gather_pd(rho, _mm_loadu_si128((const __m128i*)(i0 + j)), 8);
        const __m256d r1 = _mm256_i32gather_pd(rho, _mm_loadu_si128((const __m128i*)(i1 + j)), 8);
        const __m256d r2 = _mm256_i32gather_pd(rho, _mm_loadu_si128((const __m128i*)(i2 + j)), 8);
        __m256d s = _mm256_mul_pd(_mm256_loadu_pd(v0 + j), r0);
        s = _mm256_fmadd_pd(_mm256_loadu_pd(v1 + j), r1, s);
        s = _mm256_fmadd_pd(_mm256_loadu_pd(v2 + j), r2, s);
        _mm256_storeu_pd(a + j, s);
    }
    for (; j < n; ++j) a[j] = v0[j] * rho[i0[j]] + v1[j] * rho[i1[j]] + v2[j] * rho[i2[j]];
}
inline bool has_avx2_fma() {
    static const bool b = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    return b;
}
#else
inline bool has_avx2_fma() { return false; }
#endif
inline void ell3(const int* i0, const int* i1, const int* i2, const double* v0, const double* v1, const double* v2,
                 const double* rho, double* a, int n, bool simd) {
#ifdef FEDLP_X86
    if (simd) return ell3_avx2(i0, i1, i2, v0, v1, v2, rho, a, n);
#endif
    (void)simd;
    ell3_scalar(i0, i1, i2, v0, v1, v2, rho, a, n);
}
}  // namespace kern
}  // namespace fedlp
#include "parallel.hpp"
#include "model.hpp"

namespace fedlp {

// Phase timers, compiled in with -DFEDLP_PROFILE (stats()["t_*"], seconds).
#ifdef FEDLP_PROFILE
struct ScopeTimer {
    double& acc;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    explicit ScopeTimer(double& a) : acc(a) {}
    ~ScopeTimer() { acc += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }
};
#define FEDLP_CAT2(a, b) a##b
#define FEDLP_CAT(a, b) FEDLP_CAT2(a, b)
#define FEDLP_TIME(field) ScopeTimer FEDLP_CAT(fedlp_timer_, __LINE__)(stats_.field)
// lap timers: the time since the previous lap goes to field
#define FEDLP_LAP_START auto fedlp_lap_ = std::chrono::steady_clock::now()
#define FEDLP_LAP(field)                                                                  \
    do {                                                                                  \
        const auto fedlp_now_ = std::chrono::steady_clock::now();                         \
        stats_.field += std::chrono::duration<double>(fedlp_now_ - fedlp_lap_).count();   \
        fedlp_lap_ = fedlp_now_;                                                          \
    } while (0)
#else
#define FEDLP_TIME(field) (void)0
#define FEDLP_LAP_START (void)0
#define FEDLP_LAP(field) (void)0
#endif

enum class Status : int { Optimal = 0, Infeasible = 1, Unbounded = 2, IterationLimit = 3, Numerical = 5 };

struct Options {
    double primal_tol = 1e-9;    // primal feasibility
    double feas_accept = 1e-7;   // violations no pivot can repair are accepted (snapped) below this
    double dual_tol = 1e-9;      // optimality, scaled by 1 + |c_j|
    double pivot_tol = 1e-7;     // smallest acceptable pivot element
    double perturb = 5e-7;       // relative size of anti-degeneracy perturbation
    int degenerate_trigger = 10; // degenerate pivots in a row before perturbing
    int refactor_interval = 200; // pivots between refactorisations
    long long max_iter = 10'000'000;
    // phase-1 pivots per solve before giving up with ITERATION_LIMIT (0: no
    // cap): a caller that tries the primal first (Kramer's wide masters,
    // where it beats the dual 2-5x) falls back to the dual when the
    // composite phase 1 wanders (AF_N50_M2_R4: 18.7 k phase-1 pivots, 2461 rows)
    long long phase1_limit = 0;
    // wall-clock seconds per solve (checked every 64 iterations); a solve
    // past it returns ITERATION_LIMIT like one past max_iter
    double time_limit = 1e100;
    // Sifting: pivots work on the basic columns plus a working set of
    // candidate nonbasic columns; the rest stay frozen at their bounds until
    // a pricing pass with the current duals admits them. The working set is
    // capped at max(work_min, work_per_row * rows) nonbasic structurals.
    // Basis factor: 0 auto (the explicit dense inverse up to
    // dense_max_rows rows, else the sparse LU), 1 dense, 2 sparse product
    // form, 3 sparse LU with Forrest-Tomlin updates. On lpfirst's 964-job
    // masters (~1.5k rows) the LU's LP took 0.59 s, the product form's
    // 1.94 s and Gurobi's 1.02 s; on bpc's masters the two sparse ones tie.
    int factor = 0;
    int lu_refactor_interval = 100;   // Forrest-Tomlin updates before refactorising
    int dense_max_rows = 128;
    int sparse_refactor_interval = 64;   // etas appended before refactorising
    bool sifting = true;
    // Partial pricing when the working set exceeds partial_min_cols columns:
    // no pivot row, reduced costs re-priced section by section from the
    // duals (as primal codes do when columns far outnumber rows); below it,
    // full Devex with maintained reduced costs. 0 (the default) disables it:
    // on pos_congestion's lpfirst masters it halved the cost per pivot but
    // took 2.9x the pivots (Dantzig sections against Devex), 15.1 s vs 11.5 s.
    int partial_min_cols = 0;
    // Multiple pricing for large working sets (more than mp_min_cols
    // columns; 0 disables): a major iteration re-prices the whole working
    // set from the duals and keeps the mp_candidates best by Devex score;
    // minor iterations pivot among those only, updating their reduced costs
    // and weights with one dot product each (no pivot row), until none
    // prices out or mp_max_minor pivots have passed.
    int mp_min_cols = 0;   // off: 3x the pivots of full Devex on pos_congestion
    // Threads for the column-wise pivot-row pass when the working set has
    // more than parallel_min_nnz nonzeros (0 threads: serial).
    int threads = 4;
    long long parallel_min_nnz = 100000;
    int mp_candidates = 64;
    int mp_max_minor = 32;
    // Dual simplex when a solve starts primal infeasible but dual feasible
    // (bounds changed, nothing appended: a re-targeted branch-and-price node)
    bool dual = true;
    // Dual pricing: 1 dual steepest edge (Forrest & Goldfarb 1992), 0 dual
    // Dantzig (largest violation)
    int dual_pricing = 1;
    // Dual phase 1 by bound flipping: a start that is dual infeasible only
    // in boxed columns has them moved to their opposite bound, then the dual
    // simplex runs (Koberstein 2005; HiGHS on boxed LPs), on masters of at
    // least dual_flip_min_rows rows. lpfirst's columns are bounded by their
    // job counts: pos_congestion (4.5 k rows) re-solves 9.5 -> 3.1 s; on the
    // 964-job trace (2.4 k rows) the primal's working set keeps its pivots
    // cheap and the dual lost the bench (0.87 vs 0.53 s). Branch and price's
    // columns are unbounded, so its masters keep the primal regardless.
    bool dual_flip = true;
    int dual_flip_min_rows = 3000;
    // the dual ratio test passes breakpoints of boxed columns by flipping them
    bool bfrt = true;
    // the dual simplex perturbs costs against dual degeneracy
    bool dual_perturb = true;
    // exact initial DSE weights when the dual starts from a basis it did
    // not build (else unit weights)
    bool dse_exact = true;
    // above this many rows the dual starts from unit DSE weights instead
    int dse_exact_max_rows = 20000;
    // the dual prices only the sifting working set (every dual-infeasible
    // column joins it first); the final optimality check over all columns
    // hands anything else to the primal
    // Off: on pos_congestion the restricted dual left columns that matter
    // to a long primal clean-up (3.1 -> 6.3 s); on the 964-job trace it
    // brought the dual level with the primal (0.87 -> 0.56 vs 0.53 s).
    bool dual_sift = false;
    // the dual's row-wise PRICE reads a nonbasic-first row copy from this
    // many rows (its O(nnz) build per dual phase does not pay on small ones)
    int dual_row_copy_min_rows = 1000;
    // LU updates between the dual's refactorisations (pos_congestion's
    // lpfirst masters: 3.54 s at 100, 3.24 s at 200, 3.19 s at 400 with
    // FTRAN's R etas growing)
    int dual_refactor_interval = 200;
    double dual_perturb_scale = 1e-6;
    int work_min = 500;
    double work_per_row = 3.0;
};

struct Stats {
    long long iterations = 0, phase1_iterations = 0, flips = 0, degenerate = 0;
    long long refactors = 0, stability_refactors = 0, perturbations = 0, cleanups = 0;
    long long row_pivots = 0, col_pivots = 0;   // how pivot rows were computed
    long long sifts = 0, admitted = 0, evicted = 0;
    long long dual_iterations = 0, dual_solves = 0;
    long long majors = 0;   // multiple pricing's full re-pricings
    long long parallel_passes = 0;
    long long dual_flips = 0, bfrt_flips = 0;
    // why the last dual phase stopped short: 1 no entering column (dual
    // unbounded), 2 refactorisation failed on a tiny pivot, 3 periodic one
    int dual_exit = 0;
    long long dse_resets = 0;
    long long basis_repairs = 0;   // singular bases repaired with logicals   // weights recomputed because violated rows went unscored
    int dual_status = -1;   // the last dual phase's Status
    double post_dual_infeas = -1;
    double t_select = 0, t_ftran = 0, t_ratio = 0, t_pivot = 0, t_refactor = 0, t_sift = 0;
    double t_btran = 0, t_rowwise = 0, t_colwise = 0, t_fupdate = 0;
    // dual phase: row choice, BTRAN, PRICE, ratio test, FTRANs (alpha, DSE
    // tau, flips), updates, factor update + refactorisation
    double t_dseinit = 0, t_dchuzr = 0, t_dbtran = 0, t_dprice = 0, t_dratio = 0, t_dftran = 0, t_dupdate = 0, t_dfactor = 0;
    long long d_rho_nnz = 0, d_row_nnz = 0, d_breaks = 0, d_alpha_nnz = 0, d_tau_nnz = 0;   // dual pivot rows' sizes, summed
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
    // after Model::set_obj(j): a nonbasic column's reduced cost shifts by
    // the change, a basic one's changes the duals (recomputed at the solve)
    void cost_changed(int j, double old) {
        if (st_[j] == DEAD || !basis_valid_) return;
        if (st_[j] == BASIC) refresh_duals_ = true;
        else d_[j] += M_.obj(j) - old;
        cand_valid_ = false;
    }
    void bounds_changed(int j) {
        cand_valid_ = false;
        if (st_[j] == DEAD || st_[j] == BASIC) return;   // a basic one is fixed by phase 1
        const double nx = rest_value(j, st_[j]);
        st_[j] = std::isfinite(M_.lb(j)) && nx == M_.lb(j) ? AT_LB : AT_UB;
        if (basis_valid_ && nx != x_[j]) shift_basics(j, nx - x_[j]);
        x_[j] = nx;
    }

    Status solve() {
        auto t0 = std::chrono::steady_clock::now();
        solve_start_ = t0;
        sync_columns();
        next_valid_ = false;
        degenerate_ = 0;
        if (refresh_duals_ && basis_valid_) compute_y(), compute_d();
        refresh_duals_ = false;
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
    // columns the next crash should make basic (model indices)
    void set_hint(std::vector<int> cols) { hint_ = std::move(cols); }

private:
    Model& M_;
    std::unique_ptr<BasisFactor> fac_;
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
    std::vector<int> anz_, rnz_;   // nonzero positions of alpha_ and rho_
    std::vector<double> dse_, tau_;   // dual steepest-edge weights per basis position
    // DSE weights stay finite, nothing more: an infinite weight scores its
    // row zero, and a dual that cannot see a violated row stops short of
    // feasibility (AF_N100_M10: "optimal" with rows violated by 32). Capping
    // lower changes real weights - pos_congestion's reach past 1e100, and caps
    // at 1e12 / 1e100 took its recording from 43 k pivots to 113 k / 73 k.
    static constexpr double kDseMin = 1e-6, kDseMax = 1e300;
    std::vector<double> wcol_;        // the weights by column, across a refactorisation
    bool dse_exact_ = false;          // dse_ = ||e_i' B^-1||^2, kept by dual pivots only
    // the dual ratio test's breakpoints, compact (SoA over candidates k):
    // column, signed reduced cost (>= 0 when dual feasible), |alpha_rj|,
    // the reduced cost clamped at 0 (breakpoint t_k = ct_k / ca_k); chunk_
    // holds (t_k, k) of the chunk being walked, cidx_ its members, thr_ the
    // chunks' upper thresholds
    std::vector<int> cj_, cidx_;
    std::vector<double> cdd_, ca_, ct_, thr_;
    std::vector<std::pair<double, int>> chunk_;
    std::vector<char> dfix_;   // per dual phase: dead or fixed (lbw == ubw)
    // per dual phase: the columns as a padded 3-wide ELLPACK (SoA; columns
    // with more entries are padded out and summed apart from ell_long_),
    // the dot products' buffer, and rho with a 0 appended for the padding
    std::vector<int> ell_i_[3], ell_long_;
    std::vector<double> ell_v_[3], adot_, rhop_;
    std::vector<double> flipv_;
    std::vector<int> flipped_, fnz_;
    std::vector<double> infeas_;   // the dual's primal infeasibility by position
    std::vector<std::pair<double, int>> harris_;
    std::vector<int> touched_;
    std::vector<char> in_touched_;
    // sifting working set: columns pivots and pricing look at (logicals,
    // new and recently attractive columns); in_work_ marks membership
    std::vector<int> work_;
    std::vector<char> in_work_;

    std::vector<int> cand_;      // attractive working columns (select_devex)
    std::vector<char> in_cand_;
    bool cand_valid_ = false;
    bool perturbed_ = false;
    int since_refactor_ = 0;
    int degenerate_ = 0;        // degenerate pivots in a row, this solve
    bool refresh_duals_ = false;   // a basic column's cost changed
    bool maintain_work_only_ = false;   // in phase 2: refactorisations re-price the working set only
    bool partial_ = false;              // phase 2 is pricing partially
    bool multi_ = false;                // phase 2 uses multiple pricing
    std::vector<int> mp_list_;          // its current candidates
    int mp_minor_ = 0;
    size_t ppos_ = 0;                   // partial pricing's position in the working set
    size_t p1pos_ = 0;                  // phase 1's partial pricing position
    std::vector<int> hint_;        // crash hint (set_hint)
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
    int yarg_ = -1;       // where it is attained
    void update_ymax() {
        double v = 0.0;
        int a = -1;
        for (int k = 0, n = (int)y_.size(); k < n; ++k)
            if (std::fabs(y_[k]) > v) v = std::fabs(y_[k]), a = k;
        ymax_ = v, yarg_ = a;
    }
    // exact after y changed only on nz: a full pass only when the entry
    // holding the maximum shrank
    void update_ymax(const std::vector<int>& nz) {
        bool lost = false;
        for (int k : nz) {
            const double v = std::fabs(y_[k]);
            if (v > ymax_) ymax_ = v, yarg_ = k;
            else if (k == yarg_ && v < ymax_) lost = true;
        }
        if (lost) update_ymax();
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
        acc_.push_back(0.0), in_touched_.push_back(0), in_work_.push_back(0), in_cand_.push_back(0);
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

    // nonzeros of the working set (recounted when its size changes)
    long long work_nnz() {
        if (wnnz_size_ != work_.size()) {
            wnnz_ = 0;
            for (int j : work_) wnnz_ += M_.col_nnz(j);
            wnnz_size_ = work_.size();
        }
        return wnnz_;
    }
    long long wnnz_ = 0;
    size_t wnnz_size_ = (size_t)-1;

    void admit(int j) {
        if (!in_work_[j]) in_work_[j] = 1, work_.push_back(j);
        note_attractive(j);
    }

    // alpha_ = B^-1 a_q and its nonzero positions anz_ (from the factor
    // when it tracks them, else one scan): the ratio test and the primal
    // update then touch only those
    void ftran_alpha(int q) {
        const int m = M_.rows();
        alpha_.resize(m);
        fac_->ftran(M_, q, alpha_.data());
        const std::vector<int>* nz = fac_->last_nz();
        if (nz) {
            anz_ = *nz;
        } else {
            anz_.clear();
            for (int i = 0; i < m; ++i)
                if (alpha_[i] != 0.0) anz_.push_back(i);
        }
    }
    void rho_pattern() {
        rnz_.clear();
        const std::vector<int>* nz = fac_->row_nz();
        if (nz && (int)nz->size() * 8 < M_.rows()) {   // the factor's pattern, when sparse
            for (int k : *nz)
                if (std::fabs(rho_[k]) >= 1e-14) rnz_.push_back(k);
            // ascending, as a scan gives: pivot rows then sum in a fixed
            // order and break ties the same way whatever the factor did
            std::sort(rnz_.begin(), rnz_.end());
            return;
        }
        for (int k = 0, m = M_.rows(); k < m; ++k)
            if (std::fabs(rho_[k]) >= 1e-14) rnz_.push_back(k);
    }

    // nonbasic j moved by delta: x_B -= delta B^-1 a_j
    void shift_basics(int j, double delta) {
        const int m = M_.rows();
        alpha_.resize(m);
        fac_->ftran(M_, j, alpha_.data());
        for (int i = 0; i < m; ++i) xb_[i] -= delta * alpha_[i];
    }

    // ------------------------------------------------------------ basis
    // Logical basis, except that an equality row takes a structural
    // singleton column with a positive coefficient when there is one (the
    // master's artificials and null columns) — usually a feasible start.
    void crash() {
        dse_exact_ = false;
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
        // hinted columns (e.g. a known feasible solution's) take an equality
        // row they touch, before the singleton crash
        for (int j : hint_) {
            if (j < m || j >= n || st_[j] == DEAD) continue;
            const int* ix = M_.col_idx(j);
            const double* v = M_.col_val(j);
            for (int k = 0, nz = M_.col_nnz(j); k < nz; ++k) {
                const int r = ix[k];
                if (M_.sense(r) == 'E' && head_[r] < 0 && v[k] != 0.0) {
                    head_[r] = j;
                    break;
                }
            }
        }
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
        make_factor();
        basis_valid_ = refactor();
    }

    void make_factor() {
        const int m = M_.rows();
        if (opt.factor == 3 || (opt.factor == 0 && m > opt.dense_max_rows))
            fac_ = std::make_unique<LuFT>();
        else if (opt.factor == 2) fac_ = std::make_unique<SparsePFI>();
        else fac_ = std::make_unique<DenseFactor>();
        fac_->reset(m);
    }

public:
    // Start from a given basis (status per model column: BASIC, AT_LB or
    // AT_UB; a nonbasic is placed at the bound it names, or at its finite
    // one). A basis of the wrong size or a singular one falls back to the
    // crash; returns whether the given basis was taken. Branch and bound
    // restores a node's parent basis this way, and a model rebuilt with
    // new rows (cuts) keeps its basis with the new logicals basic.
    bool warm_start(const std::vector<char>& want) {
        sync_columns();
        dse_exact_ = false;
        const int m = M_.rows(), n = M_.cols();
        if ((int)want.size() != n) {
            crash();
            return false;
        }
        head_.clear();
        for (int j = 0; j < n; ++j) {
            if (st_[j] == DEAD) continue;
            pos_[j] = -1, w_[j] = 1.0, pl_[j] = pu_[j] = 0.0;
            if (want[j] == BASIC) {
                head_.push_back(j);
                continue;
            }
            char s = want[j] == AT_UB ? AT_UB : AT_LB;
            if (s == AT_UB && !std::isfinite(M_.ub(j))) s = AT_LB;
            if (s == AT_LB && !std::isfinite(M_.lb(j))) s = AT_UB;
            st_[j] = s, x_[j] = rest_value(j, s);
        }
        if ((int)head_.size() != m) {
            crash();
            return false;
        }
        for (int i = 0; i < m; ++i) st_[head_[i]] = BASIC, pos_[head_[i]] = i;
        for (int j = 0; j < n; ++j)
            if (st_[j] != DEAD) admit(j);
        perturbed_ = false;
        make_factor();
        basis_valid_ = refactor();
        if (!basis_valid_) {
            crash();
            return false;
        }
        return true;
    }
    bool has_basis() const { return basis_valid_; }
    // The DSE weights by model variable (0 for nonbasic), or empty when they
    // are not exact for the current basis. Branch and bound saves them with
    // a node's basis: restoring a basis otherwise costs the next dual solve
    // its exact initialisation (a BTRAN per row, O(m^2)), which on Kramer's
    // arc-flow masters (m ~ 2.5 k) was most of strong branching's time.
    std::vector<double> dual_weights() const {
        const int m = M_.rows();
        if (!basis_valid_ || !dse_exact_ || (int)dse_.size() != m) return {};
        std::vector<double> w(M_.cols(), 0.0);
        for (int i = 0; i < m; ++i) w[head_[i]] = dse_[i];
        return w;
    }
    // weights saved by dual_weights() for this same basis; false (nothing
    // changes) when the sizes differ or a basic variable has none
    bool set_dual_weights(const std::vector<double>& w) {
        const int m = M_.rows();
        if (!basis_valid_ || (int)w.size() != M_.cols()) return false;
        std::vector<double> d(m);
        for (int i = 0; i < m; ++i) {
            const double v = w[head_[i]];
            if (!(v > 0.0) || !std::isfinite(v)) return false;
            d[i] = v;
        }
        dse_ = std::move(d);
        dse_exact_ = true;
        return true;
    }
    // DSE weights ||e_i' B^-1||^2: 1 for a basis of logicals (B = +-I);
    // above dse_exact_max_rows unit weights (each BTRAN clears and copies
    // O(m) vectors: an exact pass is O(m^2), hours on gfd's 250 k-row root);
    // else one BTRAN per row, the time limit checked every 64 rows. Returns
    // whether the weights are exact.
    bool exact_dse_weights() {
        const int m = M_.rows();
        dse_.assign(m, 1.0);
        bool logical = true;
        for (int i = 0; i < m && logical; ++i) logical = head_[i] < m;
        if (logical) return true;
        if (m > opt.dse_exact_max_rows) return false;
        for (int i = 0; i < m; ++i) {
            if ((i & 63) == 63 && opt.time_limit < 1e99 &&
                std::chrono::duration<double>(std::chrono::steady_clock::now() - solve_start_).count() > opt.time_limit) {
                std::fill(dse_.begin(), dse_.end(), 1.0);
                return false;
            }
            const double* rr = fac_->row(i);
            double w = 0.0;
            if (const std::vector<int>* nz = fac_->row_nz()) {
                for (int k : *nz) w += rr[k] * rr[k];
            } else {
                for (int k = 0; k < m; ++k) w += rr[k] * rr[k];
            }
            dse_[i] = std::isfinite(w) ? std::min(std::max(w, kDseMin), kDseMax) : 1.0;
        }
        return true;
    }

    // the clock is read every 64 calls, not every 64 pivots: a loop that
    // refactors without completing a pivot (a tiny pivot element, the same
    // row again) kept the count still and never looked at the time
    // (physiciansched3-3's root LP ran 400 s past a 120 s limit)
    mutable unsigned budget_calls_ = 0;
    bool out_of_budget() const {
        if (stats_.iterations >= opt.max_iter) return true;
        if ((++budget_calls_ & 63) == 0 && opt.time_limit < 1e99)
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - solve_start_).count() > opt.time_limit;
        return false;
    }
    std::chrono::steady_clock::time_point solve_start_ = std::chrono::steady_clock::now();
    // the column basic at position i
    int head(int i) const { return head_[i]; }
    // The nonbasic part of row i of B^-1 A (logicals included), for cut
    // separation: x_head(i) + sum_j abar_j x_j = xbar_i.
    void tableau_row(int i, std::vector<int>& cols, std::vector<double>& vals) {
        cols.clear(), vals.clear();
        if (!basis_valid_) return;
        const int m = M_.rows(), n = M_.cols();
        const double* rr = fac_->row(i);
        rho_.assign(rr, rr + m);
        for (int j = 0; j < n; ++j) {
            const char sj = st_[j];
            if (sj == BASIC || sj == DEAD) continue;
            const double a = M_.dot(rho_.data(), j);
            if (std::fabs(a) > 1e-12) cols.push_back(j), vals.push_back(a);
        }
    }

private:

    // all_d: recompute every column's reduced cost; otherwise only the
    // working set's (primal pivoting looks at no other column, and sifting
    // re-prices the frozen ones before they are considered)
    bool refactor(bool all_d = true) {
        FEDLP_TIME(t_refactor);
        ++stats_.refactors;
        since_refactor_ = 0;
        // DSE weights belong to basic columns: carry them through the
        // factor's reordering of positions
        const int m = M_.rows();
        const bool keep_w = (int)dse_.size() == m;
        if (keep_w) {
            if ((int)wcol_.size() < M_.cols()) wcol_.resize(M_.cols());
            for (int i = 0; i < m; ++i) wcol_[head_[i]] = dse_[i];
        }
        bool ok = fac_->factorize(M_, head_);
        bool repaired = false;
        for (int attempt = 0; !ok && attempt < 3; ++attempt) {
            // basis repair (as HiGHS and Clp): dependent columns leave for
            // their bounds, the logicals of the rows they left uncovered come
            // in; x_B is recomputed below and the running phase restores
            // feasibility
            const auto* sg = fac_->singular();
            if (!sg || sg->empty()) break;
            bool fine = true;
            for (auto [c, r] : *sg)
                if (st_[r] == BASIC) fine = false;
            if (!fine) break;
            for (auto [c, r] : *sg) {
                const int j = head_[c];
                const char sj = std::isfinite(M_.lb(j)) ? AT_LB : AT_UB;
                st_[j] = sj, x_[j] = rest_value(j, sj), pos_[j] = -1;
                head_[c] = r, st_[r] = BASIC;
            }
            ++stats_.basis_repairs;
            repaired = true;
            cand_valid_ = false, next_valid_ = false;
            ok = fac_->factorize(M_, head_);
        }
        if (!ok) {
            dse_exact_ = false;
            return false;
        }
        if (keep_w)
            for (int i = 0; i < m; ++i) dse_[i] = wcol_[head_[i]];
        if (repaired) {   // the repaired positions' weights are not theirs
            if (keep_w) std::fill(dse_.begin(), dse_.end(), 1.0);
            dse_exact_ = false;
        }
        for (int i = 0; i < M_.rows(); ++i) pos_[head_[i]] = i;   // the factor may reorder
        compute_xb();
        compute_y();
        if (all_d || !opt.sifting) compute_d();
        else compute_d_work();
        return true;
    }
    void compute_d_work() {
        for (int j : work_)
            d_[j] = st_[j] == BASIC || st_[j] == DEAD ? 0.0 : M_.obj(j) - M_.dot(y_.data(), j);
        cand_valid_ = false;
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
        fac_->ftran_inplace(r.data());
        xb_ = r;
        (void)m;
    }
    void compute_y() {
        const int m = M_.rows();
        y_.assign(m, 0.0);
        for (int i = 0; i < m; ++i) y_[i] = M_.obj(head_[i]);
        fac_->btran_inplace(y_.data());
        update_ymax();
    }
    void compute_d() {
        for (int j = 0; j < M_.cols(); ++j)
            d_[j] = st_[j] == BASIC || st_[j] == DEAD ? 0.0 : M_.obj(j) - M_.dot(y_.data(), j);
        cand_valid_ = false;
    }

    // largest bound violation of a basic variable, against working bounds
    // the largest bound violation of a basic variable, relative to the
    // bound (v / (1 + |bound|)): the measure every phase's per-row test uses,
    // so a basis one phase accepts the others do too
    double infeasibility() const {
        double v = 0.0;
        for (int i = 0; i < M_.rows(); ++i) {
            const int j = head_[i];
            const double l = lbw(j), u = ubw(j);
            if (xb_[i] < l) v = std::max(v, (l - xb_[i]) / (1.0 + std::fabs(l)));
            else if (xb_[i] > u) v = std::max(v, (xb_[i] - u) / (1.0 + std::fabs(u)));
        }
        return v;
    }

    // ------------------------------------------------------------ pricing
    bool eligible(int j) const {
        // the sign of d_j rejects nearly every column: test it first, the
        // tolerance and the bounds only for the few left
        const char s = st_[j];
        const double d = d_[j];
        if (s == AT_LB) {
            if (d >= 0.0) return false;
        } else if (s == AT_UB) {
            if (d <= 0.0) return false;
        } else {
            return false;   // basic or removed
        }
        const double t = dtol(j);
        if (s == AT_LB ? d >= -t : d <= t) return false;
        return lbw(j) != ubw(j);
    }
    // Attractive-column list: the working-set columns that price out.
    // Selection scans it (dropping those no longer eligible) instead of the
    // whole working set; after a row-wise pivot only the touched columns
    // can change, so only they are checked for admission. Rebuilt whenever
    // the reduced costs are recomputed.
    void note_attractive(int j) {
        if (cand_valid_ && in_work_[j] && !in_cand_[j] && eligible(j))
            in_cand_[j] = 1, cand_.push_back(j);
    }
    void rebuild_cand() {
        for (int j : cand_) in_cand_[j] = 0;
        cand_.clear();
        for (int j : work_)
            if (!in_cand_[j] && eligible(j)) in_cand_[j] = 1, cand_.push_back(j);
        cand_valid_ = true;
    }
    // Partial pricing: from a rotating position, re-price sections of the
    // working set from the current duals and take the best eligible column
    // of the first section that has one (Dantzig, scaled by 1 + |a_j|^2).
    int select_partial() {
        const size_t n = work_.size();
        if (!n) return -1;
        const size_t sec = std::max<size_t>(1000, n / 8);
        size_t scanned = 0, pos = ppos_ % n;
        while (scanned < n) {
            const size_t end = std::min(n, pos + sec);
            int q = -1;
            double best = 0.0;
            for (size_t k = pos; k < end; ++k) {
                const int j = work_[k];
                const char sj = st_[j];
                if (sj == BASIC || sj == DEAD) continue;
                d_[j] = M_.obj(j) - M_.dot(y_.data(), j);
                if (!eligible(j)) continue;
                const double sc = d_[j] * d_[j] / cnorm(j);
                if (sc > best) best = sc, q = j;
            }
            scanned += end - pos;
            pos = end >= n ? 0 : end;
            if (q >= 0) {
                ppos_ = pos;
                return q;
            }
        }
        return -1;
    }
    // Multiple pricing: the best mp_candidates eligible working columns by
    // Devex score, from fresh reduced costs (a major iteration).
    void mp_major() {
        ++stats_.majors;
        mp_minor_ = 0;
        std::vector<std::pair<double, int>> sc;
        for (int j : work_) {
            const char sj = st_[j];
            if (sj == BASIC || sj == DEAD) continue;
            d_[j] = M_.obj(j) - M_.dot(y_.data(), j);
            if (eligible(j)) sc.push_back({d_[j] * d_[j] / w_[j], j});
        }
        const size_t K = std::min(sc.size(), (size_t)std::max(1, opt.mp_candidates));
        std::nth_element(sc.begin(), sc.begin() + K, sc.end(),
                         [](const auto& a, const auto& b) { return a.first > b.first; });
        mp_list_.clear();
        for (size_t k = 0; k < K; ++k) mp_list_.push_back(sc[k].second);
    }
    int select_multi() {
        if (mp_list_.empty() || mp_minor_ >= opt.mp_max_minor) mp_major();
        for (int round = 0; round < 2; ++round) {
            int q = -1;
            double best = 0.0;
            for (int j : mp_list_) {
                if (!eligible(j)) continue;
                const double sc = d_[j] * d_[j] / w_[j];
                if (sc > best) best = sc, q = j;
            }
            if (q >= 0) {
                ++mp_minor_;
                return q;
            }
            if (round == 0) mp_major();   // the candidates ran out: re-price
        }
        return -1;
    }

    double cnorm(int j) const {
        const double* v = M_.col_val(j);
        double s = 1.0;
        for (int k = 0, n = M_.col_nnz(j); k < n; ++k) s += v[k] * v[k];
        return s;
    }

    int select_devex() {
        if (!cand_valid_) rebuild_cand();
        int q = -1;
        double best = 0.0;
        size_t w = 0;
        for (size_t k = 0; k < cand_.size(); ++k) {
            const int j = cand_[k];
            if (!eligible(j)) {
                in_cand_[j] = 0;
                continue;
            }
            cand_[w++] = j;
            const double sc = d_[j] * d_[j] / w_[j];
            if (sc > best) best = sc, q = j;
        }
        cand_.resize(w);
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
        (void)m;
        for (int i : anz_) {
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
        for (int i : anz_) {
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
            // ties (|alpha| = 1 is common in set partitioning) go to the lowest
            // position, independent of the order anz_ lists them in
            const double fa = std::fabs(a);
            if (t <= tmax && (fa > bestpiv || (fa == bestpiv && i < s.r)))
                bestpiv = fa, s.r = i, s.theta = t, s.hit_lo = a > 0;
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
        dse_exact_ = false;   // a basis change the DSE weights do not follow
        const bool partial = maintain && partial_;
        const bool multi = maintain && multi_;
        const int m = M_.rows();
        const int l = head_[r];
        // primal update
        for (int i : anz_) xb_[i] -= dir * theta * alpha_[i];
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
            {
                FEDLP_TIME(t_btran);
                const double* rr = fac_->row(r);
                rho_.assign(rr, rr + m);
                rho_pattern();
            }
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
            for (int k : rnz_) rowcost += (long long)M_.row(k).size();
            if (partial) {
                // partial pricing: no pivot row; only the duals move
                // (below), and pricing re-prices a section from them
                arq_row = M_.dot(rho_.data(), q);
                d_[l] = -td;
            } else if (multi) {
                // multiple pricing: the pivot row only on the candidates
                arq_row = M_.dot(rho_.data(), q);
                d_[l] = -td;
                w_[l] = std::max(wq * inv2, 1.0);
                for (int j : mp_list_) {
                    if (j == q || j == l || st_[j] == BASIC || st_[j] == DEAD) continue;
                    const double a = M_.dot(rho_.data(), j);
                    if (a == 0.0) continue;
                    d_[j] -= td * a;
                    w_[j] = std::max(w_[j], a * a * inv2 * wq);
                    wmax = std::max(wmax, w_[j]);
                }
            } else if (3 * rowcost < M_.nnz()) {
                FEDLP_TIME(t_rowwise);
                ++stats_.row_pivots;
                for (int k : rnz_) {
                    const double rk = rho_[k];
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
                d_[l] = -td;
                w_[l] = std::max(wq * inv2, 1.0);
                for (int j : touched_)
                    if (j != q) note_attractive(j);
                touched_.clear();
            } else {
                FEDLP_TIME(t_colwise);
                ++stats_.col_pivots;
                d_[l] = -td;
                w_[l] = std::max(wq * inv2, 1.0);
                double best = 0.0;
                int bq = -1;
                if (eligible(l)) best = d_[l] * d_[l] / w_[l], bq = l;
                arq_row = M_.dot(rho_.data(), q);
                // one column's share: its pivot-row entry, reduced cost,
                // Devex weight and candidacy (columns are independent)
                // locals, written back once: through the reference
                // parameters the compiler must assume they alias d_ / w_
                auto pass = [&](size_t lo, size_t hi, double& obest, int& obq, double& owmax) {
                    double lbest = obest, lwmax = owmax;
                    int lbq = obq;
                    const double* rho = rho_.data();
                    double* d = d_.data();
                    double* w = w_.data();
                    const char* st = st_.data();
                    const int* wk = work_.data();
                    for (size_t k = lo; k < hi; ++k) {
                        const int j = wk[k];
                        const char sj = st[j];
                        if (sj == BASIC || sj == DEAD || j == l || j == q) continue;
                        const double a = M_.dot(rho, j);
                        if (a != 0.0) {
                            d[j] -= td * a;
                            const double nw2 = a * a * inv2 * wq;
                            if (nw2 > w[j]) w[j] = nw2;
                            if (w[j] > lwmax) lwmax = w[j];
                        }
                        if (eligible(j)) {
                            // ties to the lowest index: the same choice
                            // however the pass is split across threads
                            const double sc = d[j] * d[j] / w[j];
                            if (sc > lbest || (sc == lbest && sc > 0.0 && j < lbq)) lbest = sc, lbq = j;
                        }
                    }
                    obest = lbest, obq = lbq, owmax = lwmax;
                };
                const size_t nw = work_.size();
                if (opt.threads > 1 && work_nnz() > opt.parallel_min_nnz) {
                    ThreadPool& tp = ThreadPool::global(opt.threads - 1);
                    const int chunks = tp.size() * 4;
                    std::vector<double> cb(chunks, 0.0), cw(chunks, 0.0);
                    std::vector<int> cq(chunks, -1);
                    tp.run(chunks, [&](int c) {
                        pass(nw * c / chunks, nw * (c + 1) / chunks, cb[c], cq[c], cw[c]);
                    });
                    for (int c = 0; c < chunks; ++c) {
                        wmax = std::max(wmax, cw[c]);
                        if (cq[c] >= 0 && (cb[c] > best || (cb[c] == best && cq[c] < bq)))
                            best = cb[c], bq = cq[c];
                    }
                    ++stats_.parallel_passes;
                } else {
                    pass(0, nw, best, bq, wmax);
                }
                next_q_ = bq;
                next_valid_ = true;
                cand_valid_ = false;   // every working column was re-priced
            }
            stable = std::fabs(arq_row - arq) <= 1e-9 * (1.0 + std::fabs(arq));
            for (int k : rnz_) y_[k] += td * rho_[k];
            // exact: a stale (only growing) |y|_inf loosens the tolerance
            // and stops solves early (on rm_M8N40_1 the CG went 137 -> 542
            // iterations)
            update_ymax(rnz_);
            if (wmax > 1e8) {   // reset the reference framework
                std::fill(w_.begin(), w_.end(), 1.0);
                next_valid_ = false;
            }
        }
        {
            FEDLP_TIME(t_fupdate);
            if (!fac_->update(r, alpha_.data())) stable = false;
        }
        head_[r] = q, pos_[q] = r, st_[q] = BASIC, xb_[r] = xq, d_[q] = 0.0;
        if (!stable) ++stats_.stability_refactors;
        const int interval = refactor_interval();
        if (!stable || ++since_refactor_ >= interval) {
            next_valid_ = false;   // the reduced costs are recomputed
            if (!refactor(!maintain_work_only_)) return false;
        }
        return true;
    }

    // Optimal on the working set: price the frozen columns with the
    // current duals and admit the eligible ones; evict the least attractive
    // working columns past the cap. Returns whether any column was admitted.
    bool sift() {
        cand_valid_ = false;
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
        cand_valid_ = false;
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
        cand_valid_ = false;
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
        if (partial_ || multi_) compute_d_work();   // a previous phase priced lazily
        partial_ = multi_ = false;
        mp_list_.clear();
        struct Flag {
            bool& f;
            Flag(bool& x) : f(x) { f = true; }
            ~Flag() { f = false; }
        } only_work(maintain_work_only_);
        while (true) {
            if (out_of_budget()) return Status::IterationLimit;
            int q;
            {
                FEDLP_TIME(t_select);
                const bool was = partial_ || multi_;
                partial_ = opt.partial_min_cols > 0 && (int)work_.size() > opt.partial_min_cols;
                multi_ = !partial_ && opt.mp_min_cols > 0 && (int)work_.size() > opt.mp_min_cols;
                if (was && !partial_ && !multi_) compute_d_work(), next_valid_ = false;   // back to maintained d
                q = partial_ ? select_partial() : multi_ ? select_multi()
                    : next_valid_ ? next_q_ : select_devex();
            }
            next_valid_ = false;
            if (q < 0) {
                bool admitted;
                {
                    FEDLP_TIME(t_sift);
                    admitted = opt.sifting && sift();
                }
                if (!admitted) return Status::Optimal;
                const bool was2 = partial_ || multi_;
                partial_ = opt.partial_min_cols > 0 && (int)work_.size() > opt.partial_min_cols;
                multi_ = !partial_ && opt.mp_min_cols > 0 && (int)work_.size() > opt.mp_min_cols;
                if (was2 && !partial_ && !multi_) compute_d_work();
                mp_list_.clear();   // the working set changed: re-price
                q = partial_ ? select_partial() : multi_ ? select_multi() : select_devex();
                if (q < 0) return Status::Optimal;
            }
            const double dir = d_[q] < 0 ? 1.0 : -1.0;
            {
                FEDLP_TIME(t_ftran);
                ftran_alpha(q);
            }
            Step s;
            {
                FEDLP_TIME(t_ratio);
                s = ratio_test(q, dir, [&](int i) { return lbw(head_[i]); },
                               [&](int i) { return ubw(head_[i]); });
            }
            if (s.unbounded) return Status::Unbounded;
            ++stats_.iterations;
            if (s.r < 0) {
                for (int i : anz_) xb_[i] -= dir * s.theta * alpha_[i];
                x_[q] = dir > 0 ? ubw(q) : lbw(q);
                st_[q] = dir > 0 ? AT_UB : AT_LB;
                ++stats_.flips;
                continue;
            }
            const bool to_upper = !s.hit_lo;
            const bool degen = s.theta * std::fabs(alpha_[s.r]) <= opt.primal_tol;
            if (degen) ++stats_.degenerate;
            degenerate = degen ? degenerate + 1 : 0;
            bool pivoted;
            {
                FEDLP_TIME(t_pivot);
                pivoted = pivot(q, s.r, dir, s.theta, true, to_upper);
            }
            if (!pivoted) return Status::Numerical;
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
        const long long p1_start = stats_.phase1_iterations;
        while (true) {
            if (out_of_budget()) return Status::IterationLimit;
            if (opt.phase1_limit > 0 && stats_.phase1_iterations - p1_start >= opt.phase1_limit)
                return Status::IterationLimit;
            bool any = false;
            for (int i = 0; i < m; ++i) {
                const int j = head_[i];
                const double tl = opt.primal_tol * (1.0 + std::fabs(lbw(j)));
                const double tu = opt.primal_tol * (1.0 + std::fabs(ubw(j)));
                c1[i] = xb_[i] < lbw(j) - tl ? -1.0 : xb_[i] > ubw(j) + tu ? 1.0 : 0.0;
                any |= c1[i] != 0.0;
            }
            if (!any) return Status::Optimal;
            y1 = c1;
            fac_->btran_inplace(y1.data());
            int q = -1;
            double best = 0.0, dq = 0.0;
            // pricing: every column, or with partial pricing (wide LPs) a
            // rotating section, extended section by section until it holds a
            // candidate (each full pass priced 163 k columns per phase-1
            // pivot on AF_N50_M2_R9: 0.85 ms a pivot)
            const int n = M_.cols();
            const bool part = opt.partial_min_cols > 0 && n > 4 * opt.partial_min_cols;
            const int sec = part ? std::max(opt.partial_min_cols, n / 16) : n;
            int scanned = 0;
            int j = part ? (int)(p1pos_ % (size_t)n) : 0;
            while (scanned < n) {
                const int stop = std::min(n - scanned, sec);
                for (int t = 0; t < stop; ++t, ++j) {
                    if (j == n) j = 0;
                    const char s = st_[j];
                    if (s == BASIC || s == DEAD || lbw(j) == ubw(j)) continue;
                    const double d = -M_.dot(y1.data(), j);
                    if ((s == AT_LB && d < -1e-12) || (s == AT_UB && d > 1e-12)) {
                        const double sc = d * d / w_[j];
                        if (sc > best) best = sc, q = j, dq = d;
                    }
                }
                scanned += stop;
                if (q >= 0) break;
            }
            p1pos_ = (size_t)j;
            if (q < 0) {
                // nothing reduces the infeasibility: rounding-sized
                // violations are accepted (snapped), larger ones prove
                // infeasibility
                bool tiny = true;
                for (int i = 0; i < m && tiny; ++i) {
                    if (c1[i] == 0.0) continue;
                    const int j = head_[i];
                    const double b = c1[i] < 0 ? lbw(j) : ubw(j);
                    tiny = std::fabs(xb_[i] - b) <= opt.feas_accept * (1.0 + std::fabs(b));
                }
                if (!tiny) return Status::Infeasible;
                for (int i = 0; i < m; ++i)
                    if (c1[i] != 0.0) xb_[i] = c1[i] < 0 ? lbw(head_[i]) : ubw(head_[i]);
                return Status::Optimal;
            }
            const double dir = dq < 0 ? 1.0 : -1.0;
            ftran_alpha(q);
            // an infeasible variable may move up to its violated bound (and
            // leaves the basis there), not past it
            Step s = ratio_test(
                q, dir, [&](int i) { return c1[i] < 0 ? -kInf : c1[i] > 0 ? ubw(head_[i]) : lbw(head_[i]); },
                [&](int i) { return c1[i] > 0 ? kInf : c1[i] < 0 ? lbw(head_[i]) : ubw(head_[i]); });
            if (s.unbounded) return Status::Infeasible;   // cannot reduce the infeasibility further
            ++stats_.iterations, ++stats_.phase1_iterations;
            if (s.r < 0) {
                for (int i : anz_) xb_[i] -= dir * s.theta * alpha_[i];
                x_[q] = dir > 0 ? ubw(q) : lbw(q);
                st_[q] = dir > 0 ? AT_UB : AT_LB;
                continue;
            }
            // below its lower bound, it stops at it (AT_LB); above its upper, AT_UB
            const bool to_upper = c1[s.r] < 0 ? false : c1[s.r] > 0 ? true : !s.hit_lo;
            if (!pivot(q, s.r, dir, s.theta, false, to_upper)) return Status::Numerical;
        }
    }

    // Bounded dual simplex (dual Dantzig row choice, Harris two-pass dual
    // ratio test), from a dual feasible basis with fresh reduced costs over
    // every column; it keeps them all up to date (no sifting). Optimal when
    // no basic variable is out of bounds; a row with no entering candidate
    // proves primal infeasibility.
    // The dual simplex on cost-perturbed data (opt.dual_perturb): each
    // nonbasic column's cost moves away from dual degeneracy in its feasible
    // direction by ~1e-6 (1 + |c_j|), so the many zero reduced costs of a
    // set-partitioning master stop tying at t = 0 (as HiGHS and Clp do).
    // The true costs come back afterwards, with fresh duals and reduced
    // costs, and the primal path cleans up what they leave.
    Status dual_phase() {
        if (!opt.dual_perturb) {
            Status st = dual_phase_core();
            compute_d();   // fixed columns' reduced costs were not maintained
            return st;
        }
        const int n = M_.cols();
        std::vector<std::pair<int, double>> saved;
        unsigned long long h = 0x9E3779B97F4A7C15ull;
        for (int j = M_.rows(); j < n; ++j) {   // structurals (slacks have no cost)
            const char sj = st_[j];
            if (sj == BASIC || sj == DEAD || lbw(j) == ubw(j)) continue;
            h ^= h << 13, h ^= h >> 7, h ^= h << 17;
            const double r = 0.5 + (double)(h >> 11) * (1.0 / 9007199254740992.0);
            const double c = M_.obj(j), dl = opt.dual_perturb_scale * (1.0 + std::fabs(c)) * r;
            saved.push_back({j, c});
            const double sh = sj == AT_LB ? dl : -dl;
            M_.set_obj(j, c + sh);
            d_[j] += sh;   // nonbasic: y is unchanged
        }
        Status st = dual_phase_core();
        stats_.dual_status = (int)st;
        for (auto [j, c] : saved) M_.set_obj(j, c);
        compute_y(), compute_d();
        cand_valid_ = false, next_valid_ = false;
        return st;
    }

    // primal infeasibility of basis position i (0 when within tolerance;
    // negative below the lower bound): the dual's row choice scans this
    double infeas_of(int i) const {
        const int j = head_[i];
        const double l = lbw(j), u = ubw(j), x = xb_[i];
        if (x < l - opt.primal_tol * (1.0 + std::fabs(l))) return x - l;
        if (x > u + opt.primal_tol * (1.0 + std::fabs(u))) return x - u;
        return 0.0;
    }
    void all_infeas() {
        const int m = M_.rows();
        infeas_.resize(m);
        for (int i = 0; i < m; ++i) infeas_[i] = infeas_of(i);
    }

    bool dual_ws() const { return opt.dual_sift && opt.sifting; }

    // Row-wise copy of the matrix with each row's nonbasic entries first
    // (HiGHS's partitioned row copy): the dual's row-wise PRICE reads only
    // those. Built when a dual phase starts, kept by swapping a column's
    // entries across each row's boundary as it enters or leaves the basis.
    // rp_t_ is an entry's index within its column, cpos_ (by column entry)
    // its position here.
    std::vector<int> rp_start_, rp_nb_, rp_col_, rp_t_, cstart_, cpos_;
    std::vector<double> rp_val_;
    void build_nb_rows() {
        const int m = M_.rows(), n = M_.cols();
        cstart_.assign(n + 1, 0);
        for (int j = 0; j < n; ++j) cstart_[j + 1] = cstart_[j] + (st_[j] == DEAD ? 0 : M_.col_nnz(j));
        std::vector<int> cnt(m, 0), nbc(m, 0);
        for (int j = 0; j < n; ++j) {
            if (st_[j] == DEAD) continue;
            const int* ix = M_.col_idx(j);
            for (int t = 0, nz = M_.col_nnz(j); t < nz; ++t) {
                ++cnt[ix[t]];
                if (st_[j] != BASIC) ++nbc[ix[t]];
            }
        }
        rp_start_.assign(m + 1, 0);
        for (int k = 0; k < m; ++k) rp_start_[k + 1] = rp_start_[k] + cnt[k];
        const int nnz = rp_start_[m];
        rp_col_.resize(nnz), rp_t_.resize(nnz), rp_val_.resize(nnz), cpos_.resize(cstart_[n]);
        rp_nb_ = nbc;
        std::vector<int> fn(m), fb(m);
        for (int k = 0; k < m; ++k) fn[k] = rp_start_[k], fb[k] = rp_start_[k] + nbc[k];
        for (int j = 0; j < n; ++j) {
            if (st_[j] == DEAD) continue;
            const bool nb = st_[j] != BASIC;
            const int* ix = M_.col_idx(j);
            const double* v = M_.col_val(j);
            for (int t = 0, nz = M_.col_nnz(j); t < nz; ++t) {
                const int k = ix[t], at = nb ? fn[k]++ : fb[k]++;
                rp_col_[at] = j, rp_t_[at] = t, rp_val_[at] = v[t];
                cpos_[cstart_[j] + t] = at;
            }
        }
    }
    void rp_swap(int p, int b) {
        if (p == b) return;
        std::swap(rp_col_[p], rp_col_[b]), std::swap(rp_t_[p], rp_t_[b]), std::swap(rp_val_[p], rp_val_[b]);
        cpos_[cstart_[rp_col_[p]] + rp_t_[p]] = p;
        cpos_[cstart_[rp_col_[b]] + rp_t_[b]] = b;
    }
    void rp_to_basic(int j) {   // j entered the basis
        const int* ix = M_.col_idx(j);
        for (int t = 0, nz = M_.col_nnz(j); t < nz; ++t) {
            const int k = ix[t];
            rp_swap(cpos_[cstart_[j] + t], rp_start_[k] + --rp_nb_[k]);
        }
    }
    void rp_to_nonbasic(int j) {   // j left it
        const int* ix = M_.col_idx(j);
        for (int t = 0, nz = M_.col_nnz(j); t < nz; ++t) {
            const int k = ix[t];
            rp_swap(cpos_[cstart_[j] + t], rp_start_[k] + rp_nb_[k]++);
        }
    }

    Status dual_phase_core() {
        const int m = M_.rows(), n = M_.cols();
        const bool ws = dual_ws();
        const bool rp = m >= opt.dual_row_copy_min_rows;
        if (rp) build_nb_rows();
        all_infeas();
        // bounds stay put through a dual phase (no bound shifting here)
        dfix_.resize(n);
        for (int j = 0; j < n; ++j) dfix_[j] = st_[j] == DEAD || lbw(j) == ubw(j);
        {
            const int* cs = M_.col_starts();
            const int* ci = M_.idx_data();
            const double* cv = M_.val_data();
            ell_long_.clear();
            for (int t = 0; t < 3; ++t) ell_i_[t].assign(n, m), ell_v_[t].assign(n, 0.0);
            for (int j = 0; j < n; ++j) {
                const int e = cs[j], len = cs[j + 1] - e;
                if (len > 3) {
                    ell_long_.push_back(j);
                    continue;
                }
                for (int t = 0; t < len; ++t) ell_i_[t][j] = ci[e + t], ell_v_[t][j] = cv[e + t];
            }
            adot_.resize(n);
            rhop_.assign(m + 1, 0.0);
        }
        std::vector<double> arow(n, 0.0);
        std::vector<int> nzj;
        int stall_refactors = 0;   // refactorisations in a row without a pivot
        std::vector<char> inz(n, 0);
        ++stats_.dual_solves;
        cand_valid_ = false;
        while (true) {
            if (out_of_budget()) return Status::IterationLimit;
            FEDLP_LAP_START;
            // leaving row: the largest violation^2 / beta_i (dual steepest
            // edge, beta_i ~ |e_i' B^-1|^2) or the largest violation
            const bool dse = opt.dual_pricing == 1;
            if (dse && ((int)dse_.size() != m || (!dse_exact_ && opt.dse_exact))) {
                // exact weights (one BTRAN per row) for a basis the dual did
                // not build: unit weights are right only for a slack basis,
                // and wrong ones pick rows with dense e_r' B^-1
                dse_.assign(m, 1.0);
                if (opt.dse_exact) {
                    FEDLP_TIME(t_dseinit);
                    dse_exact_ = exact_dse_weights();
                } else {
                    dse_exact_ = false;
                }
            }
            int r = -1;
            double worst = 0.0, best = 0.0;
            bool below = false;
            bool any = false;
            {
                const double* inf = infeas_.data();
                const double* w = dse_.data();
                for (int i = 0; i < m; ++i) {
                    const double v = inf[i];
                    if (v == 0.0) continue;
                    any = true;
                    const double sc = dse ? v * v / w[i] : std::fabs(v);
                    if (sc > best) best = sc, worst = std::fabs(v), r = i, below = v < 0;
                }
            }
            if (r < 0 && any) {
                // violated rows, none scored: the weights went bad (overflow
                // or NaN). Never stop on that: exact weights, choose again.
                ++stats_.dse_resets;
                if (dse && opt.dse_exact) {
                    exact_dse_weights();
                } else {
                    std::fill(dse_.begin(), dse_.end(), 1.0);
                }
                const double* inf = infeas_.data();
                for (int i = 0; i < m; ++i) {
                    const double v = inf[i];
                    if (v == 0.0) continue;
                    const double sc = dse ? v * v / dse_[i] : std::fabs(v);
                    if (!(sc > best)) continue;
                    best = sc, worst = std::fabs(v), r = i, below = v < 0;
                }
                if (r < 0) {   // the scores themselves are not numbers
                    for (int i = 0; i < m; ++i)
                        if (inf[i] != 0.0 && std::fabs(inf[i]) > worst) worst = std::fabs(inf[i]), r = i, below = inf[i] < 0;
                }
            }
            if (r < 0) {
                // primal feasible on x_B carried through updates: confirm on
                // a fresh factor before declaring it (drift over a long
                // interval would otherwise hand the primal a basis it sees as
                // infeasible)
                if (since_refactor_ > 0) {
                    if (!refactor(!ws)) { stats_.dual_exit = 2; return Status::Numerical; }
                    all_infeas();
                    continue;
                }
#ifdef FEDLP_DEBUG_DUAL
                for (int i = 0, shown = 0; i < m && shown < 8; ++i) {
                    const int j = head_[i];
                    if (xb_[i] < lbw(j) - 1e-6 || xb_[i] > ubw(j) + 1e-6)
                        fprintf(stderr, "dual-opt violation pos %d col %d xb %g [%g, %g] infeas %g st %d\n", i, j, xb_[i], lbw(j), ubw(j), infeas_[i], st_[j]), ++shown;
                }
#endif
                return Status::Optimal;
            }
            FEDLP_LAP(t_dchuzr);
            // pivot row over the nonbasic columns
            {
                const double* rr = fac_->row(r);
                rho_.assign(rr, rr + m);
                rho_pattern();
            }
            FEDLP_LAP(t_dbtran);
            int fusedK = -1;   // candidates already collected by a fused PRICE
            long long rowcost = 0;
            if (rp) for (int k : rnz_) rowcost += rp_nb_[k];
            else for (int k : rnz_) rowcost += (long long)M_.row(k).size();
            if (rp && 3 * rowcost < M_.nnz()) {
                // the nonbasic part of each row only: no status tests
                for (int k : rnz_) {
                    const double rk = rho_[k];
                    for (int e = rp_start_[k], f = e + rp_nb_[k]; e < f; ++e) {
                        const int j = rp_col_[e];
                        if (ws && !in_work_[j]) continue;
                        if (!inz[j]) inz[j] = 1, nzj.push_back(j), arow[j] = 0.0;
                        arow[j] += rk * rp_val_[e];
                    }
                }
            } else if (3 * rowcost < (ws ? work_nnz() : M_.nnz())) {
                for (int k : rnz_) {
                    const double rk = rho_[k];
                    for (const auto& e : M_.row(k)) {
                        const int j = e.col;
                        if (st_[j] == BASIC || st_[j] == DEAD || (ws && !in_work_[j])) continue;
                        if (!inz[j]) inz[j] = 1, nzj.push_back(j), arow[j] = 0.0;
                        arow[j] += rk * e.val;
                    }
                }
            } else if (ws) {
                for (int j : work_) {
                    if (st_[j] == BASIC || st_[j] == DEAD) continue;
                    const double a = M_.dot(rho_.data(), j);
                    if (a != 0.0) inz[j] = 1, nzj.push_back(j), arow[j] = a;
                }
            } else {
                // Column-wise over every nonbasic column, the dot products
                // inlined on the raw column store and fused with the ratio
                // test's candidate pass: st_, d_ and dfix_ stream in column
                // order here, where the separate pass gathered them at the
                // row's 20-85 k scattered entries (AF_N50_M2_R4: rho is 44 %
                // dense, so this path prices most pivots)
                const int* cs = M_.col_starts();
                const int* ci = M_.idx_data();
                const double* cv = M_.val_data();
                const double* rh = rho_.data();
                const char* st = st_.data();
                const char* fx = dfix_.data();
                const double* dv = d_.data();
                const double sg = below ? 1.0 : -1.0, pv = opt.pivot_tol;
                if ((int)cj_.size() < n) cj_.resize(n), cdd_.resize(n), ca_.resize(n), ct_.resize(n);
                int* oj = cj_.data();
                double* od = cdd_.data();
                double* oa = ca_.data();
                double* ar = arow.data();
                char* iz = inz.data();
                // the dot products: SIMD over the ELLPACK copy, the long
                // columns apart
                std::copy(rho_.begin(), rho_.begin() + m, rhop_.begin());
                double* ad = adot_.data();
                kern::ell3(ell_i_[0].data(), ell_i_[1].data(), ell_i_[2].data(), ell_v_[0].data(), ell_v_[1].data(),
                           ell_v_[2].data(), rhop_.data(), ad, n, kern::has_avx2_fma());
                for (int j : ell_long_) {
                    double a = 0.0;
                    for (int t = cs[j], f = cs[j + 1]; t < f; ++t) a += rh[ci[t]] * cv[t];
                    ad[j] = a;
                }
                int K = 0, cnt = 0;
                nzj.resize(n);
                int* oz = nzj.data();
                for (int j = 0; j < n; ++j) {
                    const char sj = st[j];
                    // fixed columns never enter nor flip: no pivot-row entry
                    // (their d_j goes stale; the phase's end recomputes d)
                    if (sj == BASIC || sj == DEAD || fx[j]) continue;
                    const double a = ad[j];
                    // branch-free appends: about half the rows' entries are 0
                    const bool nz = a != 0.0;
                    ar[j] = a, iz[j] = nz, oz[cnt] = j, cnt += nz;
                    const double as = sg * a, fa = std::fabs(a);
                    const bool lo = sj == AT_LB, ok = nz & (fa >= pv) & ((lo & (as < 0)) | ((sj == AT_UB) & (as > 0)));
                    oj[K] = j, od[K] = lo ? dv[j] : -dv[j], oa[K] = fa;
                    K += ok;
                }
                nzj.resize(cnt);
                fusedK = K;
            }
            stats_.d_rho_nnz += (long long)rnz_.size(), stats_.d_row_nnz += (long long)nzj.size();
            FEDLP_LAP(t_dprice);
            // Bound-flipping (long-step) dual ratio test (Fourer 1994; Maros
            // 2003, ch. 10; Koberstein 2005). The breakpoints t_j, ascending:
            // passing one changes d_j's sign, which a boxed column absorbs by
            // flipping to its other bound, lowering the slope (row r's
            // violation) by |a_j| (u_j - l_j). Flip while the slope stays
            // positive; the breakpoint that ends it enters, chosen Harris-style
            // (largest |a| within the relaxed bound) among those not flipped.
            const double sgn = below ? 1.0 : -1.0, piv = opt.pivot_tol;
            // candidates (sign-eligible, pivotable, not fixed), branch-free
            // into compact arrays: the row has ~20 k entries on Kramer's
            // arc-flow masters, where the scattered bound reads of a
            // per-entry test cost more than the rest of the ratio test
            int K = fusedK > 0 ? fusedK : 0;
            if (fusedK < 0) {
                const int nz0 = (int)nzj.size();
                if ((int)cj_.size() < nz0) cj_.resize(nz0), cdd_.resize(nz0), ca_.resize(nz0), ct_.resize(nz0);
                const char* st = st_.data();
                const char* fx = dfix_.data();
                const double* dv = d_.data();
                const double* ar = arow.data();
                int* oj = cj_.data();
                double* od = cdd_.data();
                double* oa = ca_.data();
                for (int j : nzj) {
                    const double a = ar[j], as = sgn * a, fa = std::fabs(a);
                    const char sj = st[j];
                    const bool lo = sj == AT_LB, ok = (fa >= piv) & !fx[j] & ((lo & (as < 0)) | ((sj == AT_UB) & (as > 0)));
                    oj[K] = j, od[K] = lo ? dv[j] : -dv[j], oa[K] = fa;
                    K += ok;
                }
            }
            {
                // clamped: t_k = cdd_k / ca_k >= 0 (the raw value stays for
                // Harris's bound, which uses it with the tolerance)
                const double* od = cdd_.data();
                double* ot = ct_.data();
                for (int k = 0; k < K; ++k) ot[k] = od[k] > 0.0 ? od[k] : 0.0;
            }
            if (K == 0) {
                for (int j : nzj) inz[j] = 0;
                const int l = head_[r];
                const double bound = below ? lbw(l) : ubw(l);
                if (worst <= opt.feas_accept * (1.0 + std::fabs(bound))) {
                    xb_[r] = bound;   // rounding no pivot can fix: accept it
                    infeas_[r] = 0.0;
                    nzj.clear();
                    continue;
                }
                // Dual unbounded: no nonbasic move within its bounds can
                // reduce row r's violation, a proof that the LP is primal
                // infeasible (its working bounds contain the true ones). On a
                // fresh factor first (x_B and the row exact), then trust it:
                // handing it to phase 1 cycled for millions of pivots on a
                // degenerate infeasible node LP (csched008).
                if (since_refactor_ > 0) {
                    if (!refactor(!ws)) { stats_.dual_exit = 2; return Status::Numerical; }
                    all_infeas();
                    nzj.clear();
                    continue;
                }
                stats_.dual_exit = 1;
                return Status::Infeasible;
            }
            // Breakpoints in ascending order, a chunk at a time. The chunks'
            // thresholds are quantiles of a 64-point sample of the t_k
            // (positions 0, 1, 3, 7, .. of the sorted sample), so the first
            // chunk holds ~K/64 of the smallest and is usually all the walk
            // reads: one pass over the candidates and a sort of a few
            // hundred, where a heap over all of them cost ~70 us a pivot.
            // Membership tests t_k <= theta as cdd_k <= theta |a_k| (no
            // division per candidate; monotone in theta, so the chunks
            // partition the candidates); only a chunk's members get t_k.
            thr_.clear();
            if (K > 256) {
                double smp[64];
                const int stride = K / 64;
                for (int i = 0; i < 64; ++i) smp[i] = ct_[(size_t)i * stride] / ca_[(size_t)i * stride];
                std::sort(smp, smp + 64);
                for (int i = 1; i <= 64; i *= 2) thr_.push_back(smp[i - 1]);
            }
            thr_.push_back(kInf);
            int ci = -1;
            size_t cpos = 0, cend = 0;
            int remaining = K;
            if ((int)chunk_.size() < K) chunk_.resize(K), cidx_.resize(K);
            // the next chunk (thr_[ci-1], thr_[ci]], sorted by (t, k); false
            // past the last
            auto refill = [&]() {
                while (cpos == cend) {
                    if (ci + 1 >= (int)thr_.size()) return false;
                    ++ci;
                    const double lo = ci == 0 ? -kInf : thr_[ci - 1], hi = thr_[ci];
                    const double* ct = ct_.data();
                    const double* ca = ca_.data();
                    int* out = cidx_.data();
                    size_t c = 0;
                    if (ci == 0) {
                        for (int k = 0; k < K; ++k) out[c] = k, c += ct[k] <= hi * ca[k];
                    } else {
                        for (int k = 0; k < K; ++k) {
                            const double v = ct[k], a = ca[k];
                            out[c] = k;
                            c += (v > lo * a) & (v <= hi * a);
                        }
                    }
                    for (size_t i = 0; i < c; ++i) chunk_[i] = {ct[out[i]] / ca[out[i]], out[i]};
                    std::sort(chunk_.begin(), chunk_.begin() + c);
                    cpos = 0, cend = c;
                }
                return true;
            };
            flipped_.clear();
            if (opt.bfrt) {
                double slope = worst;
                while (remaining > 1) {
                    if (!refill()) break;
                    const int k = chunk_[cpos].second, j = cj_[k];
                    const double range = ubw(j) - lbw(j);
                    if (!std::isfinite(range)) break;
                    const double ns = slope - ca_[k] * range;
                    if (ns <= 0.0) break;
                    slope = ns;
                    flipped_.push_back(j);
                    ++cpos, --remaining;
                }
            }
            double tmax = kInf;
            int q = -1;
            double bestpiv = 0.0;
            // Harris over what is left: take in order while within the
            // relaxed bound, which only shrinks
            harris_.clear();
            while (remaining > 0) {
                if (!refill()) break;
                const auto e = chunk_[cpos];
                if (e.first > tmax) break;
                ++cpos, --remaining;
                harris_.push_back(e);
                const int k = e.second;
                tmax = std::min(tmax, (cdd_[k] + dtol(cj_[k])) / ca_[k]);
            }
            for (const auto& e : harris_) {
                if (e.first > tmax) break;   // taken in order
                const double fa = ca_[e.second];
                if (fa > bestpiv) bestpiv = fa, q = cj_[e.second];
            }
            if (q < 0) q = harris_.empty() ? cj_[0] : cj_[harris_.front().second];
            stats_.d_breaks += (long long)remaining;
            FEDLP_LAP(t_dratio);
            const size_t k0 = flipped_.size();
            if (k0 > 0) {
                // the flips' effect on x_B: one FTRAN of their combined column
                flipv_.assign(m, 0.0);
                fnz_.clear();
                for (const int j : flipped_) {
                    const double nx = st_[j] == AT_LB ? ubw(j) : lbw(j);
                    const double dx = nx - x_[j];
                    st_[j] = st_[j] == AT_LB ? AT_UB : AT_LB;
                    x_[j] = nx;
                    const int* ix = M_.col_idx(j);
                    const double* v = M_.col_val(j);
                    for (int e = 0, nz = M_.col_nnz(j); e < nz; ++e) {
                        if (flipv_[ix[e]] == 0.0) fnz_.push_back(ix[e]);
                        flipv_[ix[e]] += v[e] * dx;
                    }
                }
                fac_->ftran_inplace(flipv_.data(), fnz_);
                if (const std::vector<int>* nz = fac_->inplace_nz()) {
                    for (int i : *nz) xb_[i] -= flipv_[i], infeas_[i] = infeas_of(i);
                } else {
                    for (int i = 0; i < m; ++i) xb_[i] -= flipv_[i];
                    all_infeas();
                }
                stats_.bfrt_flips += (long long)k0;
            }
            ftran_alpha(q);
            const double arq = alpha_[r];
            const bool stable = std::fabs(arow[q] - arq) <= 1e-9 * (1.0 + std::fabs(arq));
            if (std::fabs(arq) < piv) {
                for (int j : nzj) inz[j] = 0;
                // a fresh factor, then the row again; the same tiny pivot
                // over and over is no progress: hand the LP to the primal
                if (++stall_refactors > 5) { stats_.dual_exit = 4; return Status::Numerical; }
                if (!refactor(!ws)) { stats_.dual_exit = 2; return Status::Numerical; }
                all_infeas();   // x_B recomputed
                continue;
            }
            stall_refactors = 0;
            if (dse) {
                // tau = B^-1 rho_r, then the Forrest-Goldfarb update over
                // the rows alpha touches
                tau_.assign(rho_.begin(), rho_.end());
                fac_->ftran_inplace(tau_.data(), rnz_);
                stats_.d_alpha_nnz += (long long)anz_.size();
                stats_.d_tau_nnz += fac_->inplace_nz() ? (long long)fac_->inplace_nz()->size() : (long long)m;
                FEDLP_LAP(t_dftran);
                const double br = dse_[r];
                for (int i : anz_) {
                    if (i == r) continue;
                    const double ratio = alpha_[i] / arq;
                    double b = dse_[i] - 2.0 * ratio * tau_[i] + ratio * ratio * br;
                    b = std::max(b, std::max(kDseMin, ratio * ratio));
                    dse_[i] = std::isfinite(b) ? std::min(b, kDseMax) : kDseMax;
                }
                {
                    const double b = std::max(br / (arq * arq), kDseMin);
                    dse_[r] = std::isfinite(b) ? std::min(b, kDseMax) : kDseMax;
                }
            }
            const int l = head_[r];
            const double target = below ? lbw(l) : ubw(l);
            const double tp = (xb_[r] - target) / arq;
            const double td = d_[q] / arq;
            for (int i : anz_) xb_[i] -= tp * alpha_[i];
            const double xq = x_[q] + tp;
            for (int j : nzj) {
                if (j != q) d_[j] -= td * arow[j];
                inz[j] = 0;
            }
            nzj.clear();
            d_[l] = -td;
            for (int k : rnz_) y_[k] += td * rho_[k];
            // exact: a stale (only growing) |y|_inf loosens the tolerance
            // and stops solves early (on rm_M8N40_1 the CG went 137 -> 542
            // iterations)
            update_ymax(rnz_);
            x_[l] = target, st_[l] = below ? AT_LB : AT_UB, pos_[l] = -1;
            admit(l), admit(q);
            if (rp) rp_to_basic(q), rp_to_nonbasic(l);
            FEDLP_LAP(t_dupdate);
            bool ok = fac_->update(r, alpha_.data());
            head_[r] = q, pos_[q] = r, st_[q] = BASIC, xb_[r] = xq, d_[q] = 0.0;
            for (int i : anz_) infeas_[i] = infeas_of(i);
            infeas_[r] = infeas_of(r);
            ++stats_.iterations, ++stats_.dual_iterations;
            const int interval = fac_->name()[0] == 'l' ? opt.dual_refactor_interval : refactor_interval();
            if (!stable || !ok) ++stats_.stability_refactors;
            if (!stable || !ok || ++since_refactor_ >= interval) {
                if (!refactor(!ws)) { stats_.dual_exit = 3; return Status::Numerical; }
                all_infeas();   // x_B recomputed
            }
            FEDLP_LAP(t_dfactor);
        }
    }

    int refactor_interval() const {
        const std::string f = fac_->name();
        return f == "sparse" ? opt.sparse_refactor_interval
             : f == "lu" ? opt.lu_refactor_interval : opt.refactor_interval;
    }

    // every dual-infeasible nonbasic column to its opposite bound, if all
    // have a finite one (else nothing moves); x_B recomputed
    bool flip_to_dual_feasible() {
        std::vector<int> fl;
        for (int j = 0, n = M_.cols(); j < n; ++j) {
            if (!eligible(j)) continue;
            if (dual_ws()) admit(j);
            if (!std::isfinite(st_[j] == AT_LB ? ubw(j) : lbw(j))) return false;
            fl.push_back(j);
        }
        if (fl.empty()) return false;
        for (int j : fl) {
            if (st_[j] == AT_LB) st_[j] = AT_UB, x_[j] = ubw(j);
            else st_[j] = AT_LB, x_[j] = lbw(j);
        }
        stats_.dual_flips += (long long)fl.size();
        compute_xb();
        cand_valid_ = false;
        return true;
    }

    bool dual_feasible() const {
        for (int j = 0, n = M_.cols(); j < n; ++j)
            if (eligible(j)) return false;
        return true;
    }

    Status solve_impl() {
        if (!basis_valid_) {
            crash();
            if (!basis_valid_) return Status::Numerical;
        }
        // a warm basis that lost only primal feasibility: dual simplex
        const bool try_flip = opt.dual_flip && M_.rows() >= opt.dual_flip_min_rows;
        if (opt.dual && (try_flip || infeasibility() > opt.primal_tol)) {
            compute_y(), compute_d();
            bool df = dual_feasible();
            if (!df && opt.dual_flip && M_.rows() >= opt.dual_flip_min_rows) df = flip_to_dual_feasible();
            if (df && infeasibility() > opt.primal_tol) {
                Status s = dual_phase();
                stats_.post_dual_infeas = infeasibility();
                if (s == Status::Infeasible) return s;
                // Optimal, or trouble: the primal path below finishes the job
            }
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
            // fresh duals and reduced costs over every column, from a fresh
            // factor when updates have accumulated (their rounding, amplified
            // by big-M costs in c_B, could otherwise hide an improving
            // column), before declaring optimality
            if (since_refactor_ > 0) {
                if (!refactor()) return Status::Numerical;
                if (infeasibility() > opt.primal_tol) continue;
            } else {
                compute_y(), compute_d();
            }
            if (!admit_eligible_all()) return Status::Optimal;
        }
        return Status::Numerical;
    }
};

}  // namespace fedlp
