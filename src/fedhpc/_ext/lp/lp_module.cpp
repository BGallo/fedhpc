// fedlp Python bindings (module fedhpc._fedlp): the Solver façade over
// Model + PrimalSimplex. Column ids are the user's: structural columns
// numbered from 0 in insertion order (the logicals are hidden).
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "model.hpp"
#include "primal.hpp"

namespace py = pybind11;
using namespace fedlp;

namespace {

class Solver {
public:
    Solver(const std::vector<std::string>& sense, const std::vector<double>& rhs)
        : M_(senses(sense), rhs), S_(M_) {}

    int rows() const { return M_.rows(); }
    int cols() const { return M_.structurals(); }

    int add_column(double c, double lb, double ub, const std::vector<int>& idx,
                   const std::vector<double>& val) {
        if (idx.size() != val.size()) throw std::invalid_argument("idx and val lengths differ");
        int j = M_.add_column(c, lb, ub, idx.data(), val.data(), (int)idx.size());
        S_.sync_columns();
        return j - M_.rows();
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
        const int first = M_.structurals();
        for (py::ssize_t k = 0; k < n; ++k) {
            const int a = S(k), b = S(k + 1);
            if (b < a) throw std::invalid_argument("column starts must be nondecreasing");
            M_.add_column(C(k), L(k), U(k), I + a, V + a, b - a);
        }
        S_.sync_columns();
        return first;
    }

    void remove_columns(const std::vector<int>& ids) {
        for (int id : ids) {
            const int j = col(id);
            if (!M_.alive(j)) continue;
            M_.remove_column(j);
            S_.column_removed(j);
        }
    }

    // columns the next crash (the first solve, or one after a basic column
    // was removed) places in the basis first: a feasible solution's columns
    // give a feasible start
    void set_crash_hint(const std::vector<int>& ids) {
        std::vector<int> cols;
        for (int id : ids) cols.push_back(col(id));
        S_.set_hint(std::move(cols));
    }

    void set_costs(const std::vector<int>& ids, const std::vector<double>& costs) {
        if (ids.size() != costs.size()) throw std::invalid_argument("ids and costs lengths differ");
        for (size_t k = 0; k < ids.size(); ++k) {
            const int j = col(ids[k]);
            const double old = M_.obj(j);
            M_.set_obj(j, costs[k]);
            S_.cost_changed(j, old);
        }
    }

    void set_bounds(int id, double lb, double ub) {
        const int j = col(id);
        M_.set_bounds(j, lb, ub);
        S_.bounds_changed(j);
    }

    int solve() { return (int)S_.solve(); }

    double objective() const { return S_.objective(); }

    py::array_t<double> x() const {
        py::array_t<double> out(M_.structurals());
        auto o = out.mutable_unchecked<1>();
        for (int k = 0; k < M_.structurals(); ++k) o(k) = S_.value(k + M_.rows());
        return out;
    }
    py::array_t<double> duals() const {
        const auto& y = S_.duals();
        py::array_t<double> out(y.size());
        std::copy(y.begin(), y.end(), out.mutable_data());
        return out;
    }
    // row activities A x
    py::array_t<double> slacks() const {
        py::array_t<double> out(M_.rows());
        auto o = out.mutable_unchecked<1>();
        for (int r = 0; r < M_.rows(); ++r) o(r) = S_.value(r);
        return out;
    }
    py::array_t<double> reduced_costs() const {
        py::array_t<double> out(M_.structurals());
        auto o = out.mutable_unchecked<1>();
        for (int k = 0; k < M_.structurals(); ++k) o(k) = S_.reduced_cost(k + M_.rows());
        return out;
    }
    // 0 basic, 1 at lower bound, 2 at upper bound, 3 removed
    py::array_t<int8_t> basis_status() const {
        py::array_t<int8_t> out(M_.structurals());
        auto o = out.mutable_unchecked<1>();
        for (int k = 0; k < M_.structurals(); ++k) o(k) = S_.var_status(k + M_.rows());
        return out;
    }

    py::dict stats() const {
        const Stats& s = S_.stats();
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
    void reset_stats() { S_.reset_stats(); }
    Options& options() { return S_.opt; }

private:
    Model M_;
    PrimalSimplex S_;

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
        if (id < 0 || id >= M_.structurals()) throw std::out_of_range("column id");
        return id + M_.rows();
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
