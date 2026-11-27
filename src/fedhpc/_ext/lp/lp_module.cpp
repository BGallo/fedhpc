// fedlp Python bindings (module fedhpc._fedlp): the Solver façade over
// Model + PrimalSimplex. Column ids are the user's: structural columns
// numbered from 0 in insertion order (the logicals are hidden).
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>

#include "model.hpp"
#include "primal.hpp"

namespace py = pybind11;
using namespace fedlp;

namespace {

class Solver {
public:
    Solver(const std::vector<std::string>& sense, const std::vector<double>& rhs)
        : M_(std::make_unique<Model>(senses(sense), rhs)), S_(std::make_unique<PrimalSimplex>(*M_)) {}

    // lo <= A x <= hi per row: = when lo == hi, <= / >= for one infinite
    // side, a ranged row (an L row whose logical is bounded) for two finite
    static Solver ranged(const std::vector<double>& lo, const std::vector<double>& hi) {
        if (lo.size() != hi.size()) throw std::invalid_argument("lo and hi lengths differ");
        std::vector<std::string> sense;
        std::vector<double> rhs, width;
        row_senses(lo, hi, sense, rhs, width);
        Solver S(sense, rhs);
        for (size_t r = 0; r < lo.size(); ++r)
            if (width[r] >= 0) S.M_->set_logical_ub((int)r, width[r]);
        return S;
    }

    int rows() const { return M_->rows(); }
    int cols() const { return M_->structurals(); }

    int add_column(double c, double lb, double ub, const std::vector<int>& idx,
                   const std::vector<double>& val) {
        if (idx.size() != val.size()) throw std::invalid_argument("idx and val lengths differ");
        int j = M_->add_column(c, lb, ub, idx.data(), val.data(), (int)idx.size());
        S_->sync_columns();
        return j - M_->rows();
    }

    // CSC batch: column k has entries idx[start[k]:start[k+1]]
    int add_columns(py::array_t<double, py::array::c_style | py::array::forcecast> c,
                    py::array_t<double, py::array::c_style | py::array::forcecast> lb,
                    py::array_t<double, py::array::c_style | py::array::forcecast> ub,
                    py::array_t<int, py::array::c_style | py::array::forcecast> start,
                    py::array_t<int, py::array::c_style | py::array::forcecast> idx,
                    py::array_t<double, py::array::c_style | py::array::forcecast> val) {
        const py::ssize_t n = c.size();
        if (lb.size() != n || ub.size() != n || start.size() != n + 1 || idx.size() != val.size())
            throw std::invalid_argument("inconsistent CSC arrays");
        auto C = c.unchecked<1>(), L = lb.unchecked<1>(), U = ub.unchecked<1>();
        auto S = start.unchecked<1>();
        const int* I = idx.data();
        const double* V = val.data();
        if (S(0) != 0 || S(n) != idx.size()) throw std::invalid_argument("bad column starts");
        const int first = M_->structurals();
        for (py::ssize_t k = 0; k < n; ++k) {
            const int a = S(k), b = S(k + 1);
            if (b < a) throw std::invalid_argument("column starts must be nondecreasing");
            M_->add_column(C(k), L(k), U(k), I + a, V + a, b - a);
        }
        S_->sync_columns();
        return first;
    }

    void remove_columns(const std::vector<int>& ids) {
        for (int id : ids) {
            const int j = col(id);
            if (!M_->alive(j)) continue;
            M_->remove_column(j);
            S_->column_removed(j);
        }
    }

    // columns the next crash (the first solve, or one after a basic column
    // was removed) places in the basis first: a feasible solution's columns
    // give a feasible start
    void set_crash_hint(const std::vector<int>& ids) {
        std::vector<int> cols;
        for (int id : ids) cols.push_back(col(id));
        S_->set_hint(std::move(cols));
    }

    void set_costs(const std::vector<int>& ids, const std::vector<double>& costs) {
        if (ids.size() != costs.size()) throw std::invalid_argument("ids and costs lengths differ");
        for (size_t k = 0; k < ids.size(); ++k) {
            const int j = col(ids[k]);
            const double old = M_->obj(j);
            M_->set_obj(j, costs[k]);
            S_->cost_changed(j, old);
        }
    }

    void set_bounds(int id, double lb, double ub) {
        const int j = col(id);
        M_->set_bounds(j, lb, ub);
        S_->bounds_changed(j);
    }

    int solve() {
        S_->opt = opt_;
        return (int)S_->solve();
    }

    // --- basis --------------------------------------------------------
    // (row statuses, column statuses): 0 basic, 1 at lower, 2 at upper,
    // 3 removed (columns); a row's status is its logical's
    py::tuple get_basis() const {
        const int m = M_->rows(), ns = M_->structurals();
        py::array_t<int8_t> rs(m), cs(ns);
        auto R = rs.mutable_unchecked<1>();
        auto C = cs.mutable_unchecked<1>();
        for (int r = 0; r < m; ++r) R(r) = S_->var_status(r);
        for (int k = 0; k < ns; ++k) C(k) = S_->var_status(k + m);
        return py::make_tuple(rs, cs);
    }
    // install a basis (as get_basis returns it); false when it was not
    // usable (wrong count of basics, singular) and the crash was taken
    bool set_basis(py::array_t<int8_t, py::array::c_style | py::array::forcecast> rs,
                   py::array_t<int8_t, py::array::c_style | py::array::forcecast> cs) {
        const int m = M_->rows(), ns = M_->structurals();
        if (rs.size() != m || cs.size() != ns) throw std::invalid_argument("basis sizes differ from the model");
        std::vector<char> want(m + ns);
        auto R = rs.unchecked<1>();
        auto C = cs.unchecked<1>();
        for (int r = 0; r < m; ++r) want[r] = (char)R(r);
        for (int k = 0; k < ns; ++k) want[k + m] = (char)C(k);
        S_->opt = opt_;
        return S_->warm_start(want);
    }
    // per basis position: >= 0 a structural id, -1 - r the logical of row r
    py::array_t<int> basic_heads() const {
        const int m = M_->rows();
        py::array_t<int> out(m);
        auto o = out.mutable_unchecked<1>();
        for (int i = 0; i < m; ++i) {
            const int j = S_->head(i);
            o(i) = j >= m ? j - m : -1 - j;
        }
        return out;
    }
    // row i of B^-1 A over the nonbasic columns, in the same encoding:
    // (ids, values) with ids >= 0 structurals and -1 - r logicals. The
    // logical of row r enters a row as sigma_r s_r (sigma = -1 for >= rows).
    py::tuple tableau_row(int i) {
        if (i < 0 || i >= M_->rows()) throw std::out_of_range("basis position");
        std::vector<int> cols;
        std::vector<double> vals;
        S_->tableau_row(i, cols, vals);
        const int m = M_->rows();
        py::array_t<int> ids(cols.size());
        py::array_t<double> vs(vals.size());
        auto I = ids.mutable_unchecked<1>();
        auto V = vs.mutable_unchecked<1>();
        for (size_t k = 0; k < cols.size(); ++k) {
            I(k) = cols[k] >= m ? cols[k] - m : -1 - cols[k];
            V(k) = vals[k];
        }
        return py::make_tuple(ids, vs);
    }
    // logical values (s_r) and bounds, for cut separation
    py::array_t<double> logicals() const {
        py::array_t<double> out(M_->rows());
        auto o = out.mutable_unchecked<1>();
        for (int r = 0; r < M_->rows(); ++r) o(r) = S_->value(r);
        return out;
    }

    // --- rows ---------------------------------------------------------
    // Append rows lo <= a'x <= hi (CSR over structural ids); returns the
    // first new row's index. The model is rebuilt with the same column ids,
    // options and basis, the new rows' logicals basic: a dual feasible
    // start, so the dual simplex restores primal feasibility.
    int add_rows(py::array_t<double, py::array::c_style | py::array::forcecast> lo,
                 py::array_t<double, py::array::c_style | py::array::forcecast> hi,
                 py::array_t<int, py::array::c_style | py::array::forcecast> start,
                 py::array_t<int, py::array::c_style | py::array::forcecast> idx,
                 py::array_t<double, py::array::c_style | py::array::forcecast> val) {
        const py::ssize_t k = lo.size();
        if (hi.size() != k || start.size() != k + 1 || idx.size() != val.size())
            throw std::invalid_argument("inconsistent CSR arrays");
        auto Lo = lo.unchecked<1>(), Hi = hi.unchecked<1>();
        auto St = start.unchecked<1>();
        auto I = idx.unchecked<1>();
        auto V = val.unchecked<1>();
        if (St(0) != 0 || St(k) != idx.size()) throw std::invalid_argument("bad row starts");
        NewRows nr;
        for (py::ssize_t r = 0; r < k; ++r) {
            nr.lo.push_back(Lo(r)), nr.hi.push_back(Hi(r));
            std::vector<std::pair<int, double>> e;
            for (int t = St(r); t < St(r + 1); ++t) {
                if (I(t) < 0 || I(t) >= M_->structurals()) throw std::out_of_range("column id");
                e.push_back({I(t), V(t)});
            }
            nr.entries.push_back(std::move(e));
        }
        const int first = M_->rows();
        std::vector<int> keep(first);
        for (int r = 0; r < first; ++r) keep[r] = r;
        rebuild(keep, nr);
        return first;
    }
    // remove rows (cuts no longer binding); the remaining rows renumber in
    // order. A removed row whose logical was nonbasic leaves a basis of the
    // wrong size, and the next solve crashes.
    void remove_rows(const std::vector<int>& rows) {
        const int m = M_->rows();
        std::vector<char> drop(m, 0);
        for (int r : rows) {
            if (r < 0 || r >= m) throw std::out_of_range("row");
            drop[r] = 1;
        }
        std::vector<int> keep;
        for (int r = 0; r < m; ++r)
            if (!drop[r]) keep.push_back(r);
        rebuild(keep, NewRows{});
    }

    double objective() const { return S_->objective(); }

    py::array_t<double> x() const {
        py::array_t<double> out(M_->structurals());
        auto o = out.mutable_unchecked<1>();
        for (int k = 0; k < M_->structurals(); ++k) o(k) = S_->value(k + M_->rows());
        return out;
    }
    py::array_t<double> duals() const {
        const auto& y = S_->duals();
        py::array_t<double> out(y.size());
        std::copy(y.begin(), y.end(), out.mutable_data());
        return out;
    }
    // row activities A x
    py::array_t<double> slacks() const {
        py::array_t<double> out(M_->rows());
        auto o = out.mutable_unchecked<1>();
        for (int r = 0; r < M_->rows(); ++r) o(r) = S_->value(r);
        return out;
    }
    py::array_t<double> reduced_costs() const {
        py::array_t<double> out(M_->structurals());
        auto o = out.mutable_unchecked<1>();
        for (int k = 0; k < M_->structurals(); ++k) o(k) = S_->reduced_cost(k + M_->rows());
        return out;
    }
    // 0 basic, 1 at lower bound, 2 at upper bound, 3 removed
    py::array_t<int8_t> basis_status() const {
        py::array_t<int8_t> out(M_->structurals());
        auto o = out.mutable_unchecked<1>();
        for (int k = 0; k < M_->structurals(); ++k) o(k) = S_->var_status(k + M_->rows());
        return out;
    }

    py::dict stats() const {
        const Stats& s = S_->stats();
        py::dict d;
        d["iterations"] = s.iterations, d["phase1_iterations"] = s.phase1_iterations;
        d["flips"] = s.flips, d["degenerate"] = s.degenerate, d["refactors"] = s.refactors;
        d["stability_refactors"] = s.stability_refactors, d["perturbations"] = s.perturbations;
        d["cleanups"] = s.cleanups, d["seconds"] = s.seconds;
        d["row_pivots"] = s.row_pivots, d["col_pivots"] = s.col_pivots;
        d["sifts"] = s.sifts, d["admitted"] = s.admitted, d["evicted"] = s.evicted;
        d["dual_iterations"] = s.dual_iterations, d["dual_solves"] = s.dual_solves;
        d["majors"] = s.majors, d["parallel_passes"] = s.parallel_passes;
        d["dual_flips"] = s.dual_flips, d["bfrt_flips"] = s.bfrt_flips;
        d["dual_exit"] = s.dual_exit, d["dual_status"] = s.dual_status;
        d["post_dual_infeas"] = s.post_dual_infeas, d["dse_resets"] = s.dse_resets;
        d["basis_repairs"] = s.basis_repairs;
        d["t_select"] = s.t_select, d["t_ftran"] = s.t_ftran, d["t_ratio"] = s.t_ratio;
        d["t_pivot"] = s.t_pivot, d["t_refactor"] = s.t_refactor, d["t_sift"] = s.t_sift;
        d["t_btran"] = s.t_btran, d["t_rowwise"] = s.t_rowwise, d["t_colwise"] = s.t_colwise;
        d["t_fupdate"] = s.t_fupdate;
        d["t_dchuzr"] = s.t_dchuzr, d["t_dbtran"] = s.t_dbtran, d["t_dprice"] = s.t_dprice;
        d["t_dratio"] = s.t_dratio, d["t_dftran"] = s.t_dftran, d["t_dupdate"] = s.t_dupdate;
        d["t_dfactor"] = s.t_dfactor, d["t_dseinit"] = s.t_dseinit;
        d["d_rho_nnz"] = s.d_rho_nnz, d["d_row_nnz"] = s.d_row_nnz, d["d_breaks"] = s.d_breaks;
        d["d_alpha_nnz"] = s.d_alpha_nnz, d["d_tau_nnz"] = s.d_tau_nnz;
        return d;
    }
    void reset_stats() { S_->reset_stats(); }
    Options& options() { return opt_; }

private:
    std::unique_ptr<Model> M_;
    std::unique_ptr<PrimalSimplex> S_;
    Options opt_;   // copied into the simplex at each solve: survives rebuilds

    struct NewRows {
        std::vector<double> lo, hi;
        std::vector<std::vector<std::pair<int, double>>> entries;   // (structural id, value)
    };
    static void row_senses(const std::vector<double>& lo, const std::vector<double>& hi,
                           std::vector<std::string>& sense, std::vector<double>& rhs,
                           std::vector<double>& width) {
        for (size_t r = 0; r < lo.size(); ++r) {
            const double a = lo[r], b = hi[r];
            if (a > b) throw std::invalid_argument("row lower bound above upper");
            if (!std::isfinite(a) && !std::isfinite(b)) throw std::invalid_argument("free rows are not supported");
            if (a == b) sense.push_back("="), rhs.push_back(a), width.push_back(-1);
            else if (!std::isfinite(a)) sense.push_back("<"), rhs.push_back(b), width.push_back(-1);
            else if (!std::isfinite(b)) sense.push_back(">"), rhs.push_back(a), width.push_back(-1);
            else sense.push_back("<"), rhs.push_back(b), width.push_back(b - a);
        }
    }

    // The model again with rows keep (old indices, in order) and then nr,
    // the same structural columns (removed ones stay removed, so ids hold),
    // and the old basis with the new rows' logicals basic.
    void rebuild(const std::vector<int>& keep, const NewRows& nr) {
        const int m = M_->rows(), ns = M_->structurals();
        const bool had = S_->has_basis();
        // rows: (lo, hi) of each kept row, then the new ones
        std::vector<double> lo, hi;
        std::vector<int> newrow(m, -1);
        for (int r : keep) {
            const char sn = M_->sense(r);
            const double b = M_->rhs()[r];
            newrow[r] = (int)lo.size();
            if (sn == 'E') lo.push_back(b), hi.push_back(b);
            else if (sn == 'G') lo.push_back(b), hi.push_back(kInf);
            else lo.push_back(std::isfinite(M_->ub(r)) ? b - M_->ub(r) : -kInf), hi.push_back(b);
        }
        for (size_t k = 0; k < nr.lo.size(); ++k) lo.push_back(nr.lo[k]), hi.push_back(nr.hi[k]);
        const int m2 = (int)lo.size();
        std::vector<std::string> sense;
        std::vector<double> rhs, width;
        row_senses(lo, hi, sense, rhs, width);
        auto M2 = std::make_unique<Model>(senses(sense), rhs);
        for (int r = 0; r < m2; ++r)
            if (width[r] >= 0) M2->set_logical_ub(r, width[r]);
        // the new rows' entries by column
        std::vector<std::vector<std::pair<int, double>>> extra(ns);
        for (size_t k = 0; k < nr.entries.size(); ++k)
            for (auto [id, v] : nr.entries[k]) extra[id].push_back({m2 - (int)nr.lo.size() + (int)k, v});
        std::vector<int> ix;
        std::vector<double> vx;
        for (int id = 0; id < ns; ++id) {
            const int j = id + m;
            ix.clear(), vx.clear();
            const int* ci = M_->col_idx(j);
            const double* cv = M_->col_val(j);
            for (int t = 0, nz = M_->col_nnz(j); t < nz; ++t)
                if (newrow[ci[t]] >= 0) ix.push_back(newrow[ci[t]]), vx.push_back(cv[t]);
            for (auto [r, v] : extra[id]) ix.push_back(r), vx.push_back(v);
            M2->add_column(M_->obj(j), M_->lb(j), M_->ub(j), ix.data(), vx.data(), (int)ix.size());
            if (!M_->alive(j)) M2->remove_column(id + m2);
        }
        // basis: kept logicals and structurals as they were, new logicals basic
        std::vector<char> want(m2 + ns, 1);
        for (int r : keep) want[newrow[r]] = S_->var_status(r);
        for (int r = m2 - (int)nr.lo.size(); r < m2; ++r) want[r] = 0;
        for (int id = 0; id < ns; ++id) want[id + m2] = S_->var_status(id + m);
        auto S2 = std::make_unique<PrimalSimplex>(*M2);
        for (int id = 0; id < ns; ++id)
            if (!M2->alive(id + m2)) S2->column_removed(id + m2);
        S2->opt = opt_;
        M_.swap(M2);
        S_.swap(S2);   // the old simplex (and its reference to the old model) goes first
        S2.reset();
        M2.reset();
        if (had) S_->warm_start(want);
    }

    static std::vector<char> senses(const std::vector<std::string>& s) {
        std::vector<char> out;
        for (const auto& x : s) {
            if (x == "<" || x == "<=" || x == "L") out.push_back('L');
            else if (x == ">" || x == ">=" || x == "G") out.push_back('G');
            else if (x == "=" || x == "==" || x == "E") out.push_back('E');
            else throw std::invalid_argument("row sense must be <, = or >");
        }
        return out;
    }
    int col(int id) const {
        if (id < 0 || id >= M_->structurals()) throw std::out_of_range("column id");
        return id + M_->rows();
    }
};

}  // namespace

PYBIND11_MODULE(_fedlp, mod) {
    mod.doc() = "fedlp: primal simplex for column generation masters (see src/fedhpc/_ext/lp)";
    py::enum_<Status>(mod, "Status")
        .value("OPTIMAL", Status::Optimal)
        .value("INFEASIBLE", Status::Infeasible)
        .value("UNBOUNDED", Status::Unbounded)
        .value("ITERATION_LIMIT", Status::IterationLimit)
        .value("NUMERICAL", Status::Numerical);
    py::class_<Options>(mod, "Options")
        .def_readwrite("primal_tol", &Options::primal_tol)
        .def_readwrite("feas_accept", &Options::feas_accept)
        .def_readwrite("dual_tol", &Options::dual_tol)
        .def_readwrite("pivot_tol", &Options::pivot_tol)
        .def_readwrite("perturb", &Options::perturb)
        .def_readwrite("degenerate_trigger", &Options::degenerate_trigger)
        .def_readwrite("refactor_interval", &Options::refactor_interval)
        .def_readwrite("max_iter", &Options::max_iter)
        .def_readwrite("time_limit", &Options::time_limit)
        .def_readwrite("sifting", &Options::sifting)
        .def_readwrite("partial_min_cols", &Options::partial_min_cols)
        .def_readwrite("mp_min_cols", &Options::mp_min_cols)
        .def_readwrite("mp_candidates", &Options::mp_candidates)
        .def_readwrite("mp_max_minor", &Options::mp_max_minor)
        .def_readwrite("threads", &Options::threads)
        .def_readwrite("parallel_min_nnz", &Options::parallel_min_nnz)
        .def_readwrite("dual", &Options::dual)
        .def_readwrite("dual_pricing", &Options::dual_pricing)
        .def_readwrite("dual_flip", &Options::dual_flip)
        .def_readwrite("dual_flip_min_rows", &Options::dual_flip_min_rows)
        .def_readwrite("dual_sift", &Options::dual_sift)
        .def_readwrite("dual_row_copy_min_rows", &Options::dual_row_copy_min_rows)
        .def_readwrite("bfrt", &Options::bfrt)
        .def_readwrite("dse_exact", &Options::dse_exact)
        .def_readwrite("dual_refactor_interval", &Options::dual_refactor_interval)
        .def_readwrite("dual_perturb", &Options::dual_perturb)
        .def_readwrite("dual_perturb_scale", &Options::dual_perturb_scale)
        .def_readwrite("factor", &Options::factor)
        .def_readwrite("lu_refactor_interval", &Options::lu_refactor_interval)
        .def_readwrite("dense_max_rows", &Options::dense_max_rows)
        .def_readwrite("sparse_refactor_interval", &Options::sparse_refactor_interval)
        .def_readwrite("work_min", &Options::work_min)
        .def_readwrite("work_per_row", &Options::work_per_row);
    py::class_<Solver>(mod, "Solver")
        .def(py::init<const std::vector<std::string>&, const std::vector<double>&>(),
             py::arg("sense"), py::arg("rhs"))
        .def_static("ranged", &Solver::ranged, py::arg("lo"), py::arg("hi"),
                    "rows lo <= A x <= hi (equal, one-sided or ranged)")
        .def("get_basis", &Solver::get_basis)
        .def("set_basis", &Solver::set_basis, py::arg("row_status"), py::arg("col_status"))
        .def("basic_heads", &Solver::basic_heads)
        .def("tableau_row", &Solver::tableau_row, py::arg("position"))
        .def("logicals", &Solver::logicals)
        .def("add_rows", &Solver::add_rows, py::arg("lo"), py::arg("hi"), py::arg("start"),
             py::arg("idx"), py::arg("val"))
        .def("remove_rows", &Solver::remove_rows, py::arg("rows"))
        .def_property_readonly("rows", &Solver::rows)
        .def_property_readonly("cols", &Solver::cols)
        .def("add_column", &Solver::add_column, py::arg("obj"), py::arg("lb"), py::arg("ub"),
             py::arg("idx"), py::arg("val"))
        .def("add_columns", &Solver::add_columns, py::arg("obj"), py::arg("lb"), py::arg("ub"),
             py::arg("start"), py::arg("idx"), py::arg("val"))
        .def("remove_columns", &Solver::remove_columns, py::arg("ids"))
        .def("set_bounds", &Solver::set_bounds, py::arg("id"), py::arg("lb"), py::arg("ub"))
        .def("set_costs", &Solver::set_costs, py::arg("ids"), py::arg("costs"))
        .def("set_crash_hint", &Solver::set_crash_hint, py::arg("ids"))
        .def("solve", &Solver::solve, py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("objective", &Solver::objective)
        .def("x", &Solver::x)
        .def("duals", &Solver::duals)
        .def("slacks", &Solver::slacks)
        .def("reduced_costs", &Solver::reduced_costs)
        .def("basis_status", &Solver::basis_status)
        .def("stats", &Solver::stats)
        .def("reset_stats", &Solver::reset_stats)
        .def_property_readonly("options", &Solver::options, py::return_value_policy::reference_internal);
}
