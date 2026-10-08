// ILS-SP for Rm || sum w_j C_j + TMUC (Chen, Chu, Sahli & Li, EJOR 2024).
//
// With no release dates each machine runs its jobs in WSPT order (Smith's
// rule), so a solution is just the job -> machine vector.  For every machine
// the WSPT rank of each job is fixed, and the machine keeps two Fenwick trees
// (processing time and weight, indexed by rank) so the exact cost change of
// inserting / removing / swapping a job is O(log n):
//
//   contrib(j on k) = w_j (P_before + p_jk) + p_jk W_after + L_k p_jk
//
// plus U_k when the machine opens or closes.
//
// The search is an Iterated Local Search:
//   VND  = relocate descent -> swap descent -> machine close/open moves
//          (+ open/closed machine exchange on the deep pass),
//   kick = ruin-and-recreate on a random subset (or on one machine),
//   accept = record-to-record travel around the best solution.
// Every per-machine job set of a near-best local optimum is stored as a
// column of the set-partitioning model (paper's Problem P); Python solves
// that model with Gurobi and hands the combined solution back through
// set_solution().
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using i64 = long long;

namespace {

struct Fenwick {
    std::vector<i64> t;
    void init(int n) { t.assign(n + 1, 0); }
    void add(int i, i64 v) {
        for (; i < (int)t.size(); i += i & -i) t[i] += v;
    }
    i64 pref(int i) const {
        i64 s = 0;
        for (; i > 0; i -= i & -i) s += t[i];
        return s;
    }
};

struct Machine {
    Fenwick fp, fw;  // indexed by WSPT rank on this machine (1-based)
    int cnt = 0;
    i64 totW = 0;
};

struct State {
    std::vector<int> mach;  // job -> machine, -1 while unassigned
    std::vector<Machine> ms;
    i64 cost = 0;
    int open = 0;
};

struct Column {
    int machine;
    std::vector<int> jobs;
    i64 cost;
};

class IlsSp {
public:
    IlsSp(const std::vector<std::vector<int>>& p, const std::vector<int>& w,
          const std::vector<int>& U, const std::vector<int>& L, std::uint64_t seed,
          double destroy_min, double destroy_max, double accept_tol, double pool_tol,
          int restart_after, double p_machine_kick, int swap_k, int deep_every)
        : n_((int)w.size()), m_((int)U.size()), rng_(seed),
          destroy_min_(destroy_min), destroy_max_(destroy_max), accept_tol_(accept_tol),
          pool_tol_(pool_tol), restart_after_(restart_after),
          p_machine_kick_(p_machine_kick), deep_every_(deep_every) {
        if (n_ == 0 || m_ == 0) throw std::invalid_argument("empty instance");
        if ((int)p.size() != n_ || (int)L.size() != m_)
            throw std::invalid_argument("inconsistent instance dimensions");
        p_.resize((size_t)n_ * m_);
        for (int j = 0; j < n_; ++j) {
            if ((int)p[j].size() != m_) throw std::invalid_argument("p row length != m");
            for (int k = 0; k < m_; ++k) p_[(size_t)j * m_ + k] = p[j][k];
        }
        w_.assign(w.begin(), w.end());
        U_.assign(U.begin(), U.end());
        L_.assign(L.begin(), L.end());

        // WSPT rank per machine: p_ik / w_i ascending, ties by index.
        rank_.resize((size_t)m_ * n_);
        std::vector<int> ord(n_);
        for (int k = 0; k < m_; ++k) {
            std::iota(ord.begin(), ord.end(), 0);
            std::sort(ord.begin(), ord.end(), [&](int a, int b) {
                i64 l = (i64)P(a, k) * w_[b], r = (i64)P(b, k) * w_[a];
                return l != r ? l < r : a < b;
            });
            for (int r = 0; r < n_; ++r) rank_[(size_t)k * n_ + ord[r]] = r + 1;
        }

        // Swap candidates: j may go to machine k only if k is among j's
        // swap_k fastest machines.
        int K = swap_k <= 0 ? m_ : std::min(swap_k, m_);
        swapok_.assign((size_t)n_ * m_, 0);
        std::vector<int> ks(m_);
        for (int j = 0; j < n_; ++j) {
            std::iota(ks.begin(), ks.end(), 0);
            std::sort(ks.begin(), ks.end(), [&](int a, int b) { return P(j, a) < P(j, b); });
            for (int q = 0; q < K; ++q) swapok_[(size_t)j * m_ + ks[q]] = 1;
        }

        key_job_.resize(n_);
        key_mach_.resize(m_);
        for (auto& x : key_job_) x = rng_();
        for (auto& x : key_mach_) x = rng_();

        cur_ = empty_state();
        best_ = cur_;
        best_.cost = INT64_MAX;
    }

    // ---- public API --------------------------------------------------------

    i64 construct() {
        State s = empty_state();
        std::vector<int> jobs(n_);
        std::iota(jobs.begin(), jobs.end(), 0);
        // Heaviest-per-unit-time first: those jobs most want an early slot.
        std::vector<double> key(n_);
        for (int j = 0; j < n_; ++j) {
            int pmin = P(j, 0);
            for (int k = 1; k < m_; ++k) pmin = std::min(pmin, P(j, k));
            key[j] = (double)w_[j] / pmin;
        }
        std::sort(jobs.begin(), jobs.end(), [&](int a, int b) { return key[a] > key[b]; });
        greedy_insert(s, jobs, -1, 0.0);
        vnd(s, true);
        adopt(s);
        return best_.cost;
    }

    i64 set_solution(const std::vector<int>& assign, bool descend) {
        if ((int)assign.size() != n_) throw std::invalid_argument("assignment length != n");
        State s = empty_state();
        for (int j = 0; j < n_; ++j) {
            if (assign[j] < 0 || assign[j] >= m_) throw std::invalid_argument("bad machine id");
            insert(s, j, assign[j]);
        }
        if (descend) vnd(s, true);
        adopt(s);
        return s.cost;
    }

    // Run ILS iterations until max_iters (<0: unlimited) or time_limit seconds.
    std::pair<i64, i64> run(i64 max_iters, double time_limit) {
        if (best_.cost == INT64_MAX) construct();
        auto t0 = std::chrono::steady_clock::now();
        i64 it = 0;
        for (; max_iters < 0 || it < max_iters; ++it) {
            if ((it & 7) == 0) {
                double el = std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - t0).count();
                if (el >= time_limit) break;
            }
            State t = cur_;
            kick(t);
            vnd(t);
            record(t);
            ++since_best_;
            if (t.cost < best_.cost) {
                vnd(t, true);
                record(t);
                best_ = t;
                since_best_ = 0;
            } else if (deep_every_ > 0 && total_iters_ % deep_every_ == 0) {
                vnd(t, true);
                record(t);
                if (t.cost < best_.cost) {
                    best_ = t;
                    since_best_ = 0;
                }
            }
            if (t.cost < cur_.cost || t.cost <= (double)best_.cost * (1.0 + accept_tol_))
                cur_ = std::move(t);
            if (restart_after_ > 0 && since_best_ > 0 && since_best_ % restart_after_ == 0)
                cur_ = best_;
            ++total_iters_;
        }
        return {it, best_.cost};
    }

    std::vector<std::tuple<int, std::vector<int>, i64>> take_new_columns() {
        std::vector<std::tuple<int, std::vector<int>, i64>> out;
        out.reserve(new_cols_.size());
        for (auto& c : new_cols_) out.emplace_back(c.machine, std::move(c.jobs), c.cost);
        new_cols_.clear();
        return out;
    }

    i64 evaluate(const std::vector<int>& assign) const {
        if ((int)assign.size() != n_) throw std::invalid_argument("assignment length != n");
        std::vector<std::vector<int>> per(m_);
        for (int j = 0; j < n_; ++j) {
            if (assign[j] < 0 || assign[j] >= m_) throw std::invalid_argument("bad machine id");
            per[assign[j]].push_back(j);
        }
        i64 tot = 0;
        for (int k = 0; k < m_; ++k) tot += machine_cost(k, per[k]);
        return tot;
    }

    // Recompute cur/best from scratch and compare with the incremental costs.
    bool check() const {
        return evaluate(cur_.mach) == cur_.cost &&
               (best_.cost == INT64_MAX || evaluate(best_.mach) == best_.cost);
    }

    i64 best_cost() const { return best_.cost; }
    i64 current_cost() const { return cur_.cost; }
    std::vector<int> best_assignment() const { return best_.mach; }
    i64 iterations() const { return total_iters_; }
    size_t pool_size() const { return seen_.size(); }

private:
    int n_, m_;
    std::vector<int> p_, rank_;
    std::vector<i64> w_, U_, L_;
    std::vector<char> swapok_;
    std::vector<std::uint64_t> key_job_, key_mach_;
    std::mt19937_64 rng_;
    double destroy_min_, destroy_max_, accept_tol_, pool_tol_;
    int restart_after_;
    double p_machine_kick_;
    int deep_every_;
    State cur_, best_;
    i64 since_best_ = 0, total_iters_ = 0;
    std::unordered_set<std::uint64_t> seen_;
    std::vector<Column> new_cols_;

    int P(int j, int k) const { return p_[(size_t)j * m_ + k]; }
    int R(int j, int k) const { return rank_[(size_t)k * n_ + j]; }

    State empty_state() const {
        State s;
        s.mach.assign(n_, -1);
        s.ms.resize(m_);
        for (auto& mc : s.ms) {
            mc.fp.init(n_);
            mc.fw.init(n_);
        }
        return s;
    }

    // Cost job j contributes to machine k's set, ignoring `excl` (which is
    // on k) and j itself; the formula is the same whether j is on k or not.
    i64 contrib(const State& s, int k, int j, int excl = -1) const {
        const Machine& mc = s.ms[k];
        int r = R(j, k);
        i64 pj = P(j, k);
        i64 before = mc.fp.pref(r - 1);
        i64 after = mc.totW - mc.fw.pref(r);
        if (excl >= 0) {
            if (R(excl, k) < r) before -= P(excl, k);
            else after -= w_[excl];
        }
        return w_[j] * (before + pj) + pj * after + L_[k] * pj;
    }

    i64 ins_delta(const State& s, int j, int k) const {
        return contrib(s, k, j) + (s.ms[k].cnt == 0 ? U_[k] : 0);
    }

    i64 rem_delta(const State& s, int j) const {
        int a = s.mach[j];
        return -contrib(s, a, j) - (s.ms[a].cnt == 1 ? U_[a] : 0);
    }

    void insert(State& s, int j, int k) {
        s.cost += ins_delta(s, j, k);
        Machine& mc = s.ms[k];
        int r = R(j, k);
        mc.fp.add(r, P(j, k));
        mc.fw.add(r, w_[j]);
        mc.totW += w_[j];
        if (mc.cnt++ == 0) ++s.open;
        s.mach[j] = k;
    }

    void remove(State& s, int j) {
        s.cost += rem_delta(s, j);
        int k = s.mach[j];
        Machine& mc = s.ms[k];
        int r = R(j, k);
        mc.fp.add(r, -P(j, k));
        mc.fw.add(r, -w_[j]);
        mc.totW -= w_[j];
        if (--mc.cnt == 0) --s.open;
        s.mach[j] = -1;
    }

    void move(State& s, int j, int k) {
        remove(s, j);
        insert(s, j, k);
    }

    i64 machine_cost(int k, std::vector<int> jobs) const {
        if (jobs.empty()) return 0;
        std::sort(jobs.begin(), jobs.end(), [&](int a, int b) { return R(a, k) < R(b, k); });
        i64 t = 0, c = U_[k];
        for (int j : jobs) {
            t += P(j, k);
            c += w_[j] * t;
        }
        return c + L_[k] * t;
    }

    std::vector<int> shuffled() {
        std::vector<int> v(n_);
        std::iota(v.begin(), v.end(), 0);
        std::shuffle(v.begin(), v.end(), rng_);
        return v;
    }

    // Insert each (unassigned) job at its cheapest machine; `forbid` is
    // excluded, and with probability `noise` the second-cheapest is used.
    void greedy_insert(State& s, const std::vector<int>& jobs, int forbid, double noise) {
        std::uniform_real_distribution<double> U01(0.0, 1.0);
        for (int j : jobs) {
            i64 b1 = INT64_MAX, b2 = INT64_MAX;
            int k1 = -1, k2 = -1;
            for (int k = 0; k < m_; ++k) {
                if (k == forbid) continue;
                i64 d = ins_delta(s, j, k);
                if (d < b1) {
                    b2 = b1; k2 = k1;
                    b1 = d; k1 = k;
                } else if (d < b2) {
                    b2 = d; k2 = k;
                }
            }
            int k = (k2 >= 0 && noise > 0 && U01(rng_) < noise) ? k2 : k1;
            insert(s, j, k);
        }
    }

    bool relocate_descent(State& s) {
        bool any = false, improved = true;
        while (improved) {
            improved = false;
            for (int j : shuffled()) {
                int a = s.mach[j];
                i64 rem = rem_delta(s, j), best = 0;
                int bk = -1;
                for (int k = 0; k < m_; ++k) {
                    if (k == a) continue;
                    i64 d = rem + ins_delta(s, j, k);
                    if (d < best) {
                        best = d;
                        bk = k;
                    }
                }
                if (bk >= 0) {
                    move(s, j, bk);
                    improved = any = true;
                }
            }
        }
        return any;
    }

    i64 swap_delta(const State& s, int j, int i) const {
        int a = s.mach[j], b = s.mach[i];
        return -contrib(s, a, j) + contrib(s, a, i, j) - contrib(s, b, i) + contrib(s, b, j, i);
    }

    // One first-improvement pass over job pairs on different machines.
    bool swap_pass(State& s) {
        bool any = false;
        std::vector<int> js = shuffled();
        for (int x = 0; x < n_; ++x) {
            int j = js[x];
            for (int y = x + 1; y < n_; ++y) {
                int i = js[y];
                int a = s.mach[j], b = s.mach[i];
                if (a == b || !swapok_[(size_t)j * m_ + b] || !swapok_[(size_t)i * m_ + a])
                    continue;
                if (swap_delta(s, j, i) < 0) {
                    remove(s, j);
                    remove(s, i);
                    insert(s, i, a);
                    insert(s, j, b);
                    any = true;
                }
            }
        }
        return any;
    }

    // Compound moves on the set of open machines: close one (reinserting
    // its jobs elsewhere) or open one (pulling in the jobs that gain from it),
    // each followed by relocate descent.  The fixed cost U_k makes these
    // unreachable by single-job moves when a machine carries many jobs.
    bool machine_moves(State& s) {
        bool any = false;
        for (int k = 0; k < m_; ++k) {
            State t = s;
            if (t.ms[k].cnt > 0) {
                if (t.open <= 1) continue;
                std::vector<int> jobs;
                for (int j = 0; j < n_; ++j)
                    if (t.mach[j] == k) jobs.push_back(j);
                for (int j : jobs) remove(t, j);
                std::shuffle(jobs.begin(), jobs.end(), rng_);
                greedy_insert(t, jobs, k, 0.0);
            } else {
                fill_machine(t, k, n_);
            }
            relocate_descent(t);
            if (t.cost < s.cost) {
                s = std::move(t);
                any = true;
            }
        }
        return any;
    }

    // Open machine k: move the job cheapest to put there (U_k included),
    // then every job whose relocation to k now gains, up to `limit` jobs.
    void fill_machine(State& s, int k, int limit) {
        int bj = -1;
        i64 bd = INT64_MAX;
        for (int j = 0; j < n_; ++j) {
            if (s.mach[j] == k) continue;
            i64 d = rem_delta(s, j) + ins_delta(s, j, k);
            if (d < bd) {
                bd = d;
                bj = j;
            }
        }
        if (bj < 0) return;
        move(s, bj, k);
        int moved = 1;
        for (int j : shuffled()) {
            if (moved >= limit) break;
            if (s.mach[j] == k) continue;
            if (rem_delta(s, j) + ins_delta(s, j, k) < 0) {
                move(s, j, k);
                ++moved;
            }
        }
    }

    // deep: also try every open/closed machine exchange (O(open x closed)
    // descents, so only on construction, new bests and every deep_every).
    void vnd(State& s, bool deep = false) {
        for (;;) {
            relocate_descent(s);
            if (swap_pass(s)) continue;
            if (m_ > 1 && machine_moves(s)) continue;
            if (deep && m_ > 1 && exchange_moves(s)) continue;
            break;
        }
    }

    // Close open machine a and open closed machine b in one step: a's jobs
    // are reinserted greedily with U_b treated as already paid, then relocate
    // descent.  Reaches machine sets that differ by a swap, which neither a
    // lone close nor a lone open can when both are uphill.
    bool exchange_moves(State& s) {
        bool any = false;
        for (int a = 0; a < m_; ++a) {
            if (s.ms[a].cnt == 0) continue;
            for (int b = 0; b < m_; ++b) {
                if (s.ms[b].cnt != 0 || s.ms[a].cnt == 0) continue;
                State t = s;
                std::vector<int> jobs;
                for (int j = 0; j < n_; ++j)
                    if (t.mach[j] == a) jobs.push_back(j);
                for (int j : jobs) remove(t, j);
                std::shuffle(jobs.begin(), jobs.end(), rng_);
                for (int j : jobs) {
                    i64 bd = INT64_MAX;
                    int bk = -1;
                    for (int k = 0; k < m_; ++k) {
                        if (k == a) continue;
                        i64 d = k == b ? contrib(t, k, j) : ins_delta(t, j, k);
                        if (d < bd) {
                            bd = d;
                            bk = k;
                        }
                    }
                    insert(t, j, bk);
                }
                relocate_descent(t);
                if (t.cost < s.cost) {
                    s = std::move(t);
                    any = true;
                }
            }
        }
        return any;
    }

    void kick(State& s) {
        std::uniform_real_distribution<double> U01(0.0, 1.0);
        int lo = std::max(1, (int)(destroy_min_ * n_));
        int hi = std::max(lo, (int)(destroy_max_ * n_));
        int q = std::uniform_int_distribution<int>(lo, hi)(rng_);

        if (m_ > 1 && U01(rng_) < p_machine_kick_) {
            int k = std::uniform_int_distribution<int>(0, m_ - 1)(rng_);
            if (s.ms[k].cnt > 0 && s.open > 1) {
                std::vector<int> jobs;  // force-close k
                for (int j = 0; j < n_; ++j)
                    if (s.mach[j] == k) jobs.push_back(j);
                for (int j : jobs) remove(s, j);
                std::shuffle(jobs.begin(), jobs.end(), rng_);
                greedy_insert(s, jobs, k, 0.2);
            } else if (s.ms[k].cnt == 0) {
                fill_machine(s, k, q);  // force-open k
            } else {
                // single open machine: fall through to a random ruin
                ruin_recreate(s, q);
            }
            return;
        }
        ruin_recreate(s, q);
    }

    void ruin_recreate(State& s, int q) {
        std::vector<int> jobs = shuffled();
        jobs.resize(std::min(q, n_));
        for (int j : jobs) remove(s, j);
        std::shuffle(jobs.begin(), jobs.end(), rng_);
        greedy_insert(s, jobs, -1, 0.1);
    }

    void adopt(State& s) {
        record(s);
        if (s.cost < best_.cost) {
            best_ = s;
            since_best_ = 0;
        }
        cur_ = s;
    }

    // Store every machine's job set of a near-best local optimum as a column.
    void record(const State& s) {
        if (best_.cost != INT64_MAX && (double)s.cost > (double)best_.cost * (1.0 + pool_tol_))
            return;
        std::vector<std::uint64_t> h(key_mach_.begin(), key_mach_.end());
        for (int j = 0; j < n_; ++j) h[s.mach[j]] ^= key_job_[j];
        for (int k = 0; k < m_; ++k) {
            if (s.ms[k].cnt == 0 || !seen_.insert(h[k]).second) continue;
            Column c{k, {}, 0};
            for (int j = 0; j < n_; ++j)
                if (s.mach[j] == k) c.jobs.push_back(j);
            c.cost = machine_cost(k, c.jobs);
            new_cols_.push_back(std::move(c));
        }
    }
};

}  // namespace

PYBIND11_MODULE(_ilssp, mod) {
    mod.doc() = "ILS with set-partitioning column pool for Rm||sum wC + TMUC";
    py::class_<IlsSp>(mod, "IlsSp")
        .def(py::init<const std::vector<std::vector<int>>&, const std::vector<int>&,
                      const std::vector<int>&, const std::vector<int>&, std::uint64_t,
                      double, double, double, double, int, double, int, int>(),
             py::arg("p"), py::arg("w"), py::arg("U"), py::arg("L"), py::arg("seed") = 0,
             py::arg("destroy_min") = 0.02, py::arg("destroy_max") = 0.10,
             py::arg("accept_tol") = 0.002, py::arg("pool_tol") = 0.01,
             py::arg("restart_after") = 200, py::arg("p_machine_kick") = 0.1,
             py::arg("swap_k") = 5, py::arg("deep_every") = 50)
        .def("construct", &IlsSp::construct, py::call_guard<py::gil_scoped_release>())
        .def("set_solution", &IlsSp::set_solution, py::arg("assign"),
             py::arg("descend") = true, py::call_guard<py::gil_scoped_release>())
        .def("run", &IlsSp::run, py::arg("max_iters") = -1, py::arg("time_limit") = 1.0,
             py::call_guard<py::gil_scoped_release>())
        .def("take_new_columns", &IlsSp::take_new_columns)
        .def("evaluate", &IlsSp::evaluate, py::arg("assign"))
        .def("check", &IlsSp::check)
        .def_property_readonly("best_cost", &IlsSp::best_cost)
        .def_property_readonly("current_cost", &IlsSp::current_cost)
        .def_property_readonly("iterations", &IlsSp::iterations)
        .def_property_readonly("pool_size", &IlsSp::pool_size)
        .def("best_assignment", &IlsSp::best_assignment);
}
