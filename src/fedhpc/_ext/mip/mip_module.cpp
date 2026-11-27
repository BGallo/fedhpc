// fedhpc._fedmip: MIP kernels for fedhpc.mip (domain propagation).
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "propagate.hpp"
#include "fj.hpp"

namespace py = pybind11;
using fedmip::Propagator;

namespace {

template <class T>
using arr = py::array_t<T, py::array::c_style | py::array::forcecast>;

template <class T>
std::vector<T> vec(const arr<T>& a) {
    return std::vector<T>(a.data(), a.data() + a.size());
}

py::array_t<double> out(const std::vector<double>& v) {
    py::array_t<double> a(v.size());
    std::copy(v.begin(), v.end(), a.mutable_data());
    return a;
}

void check(const Propagator& p, const arr<double>& lb, const arr<double>& ub) {
    if (lb.size() != p.cols() || ub.size() != p.cols()) throw std::invalid_argument("bounds of the wrong size");
}

}  // namespace

PYBIND11_MODULE(_fedmip, mod) {
    mod.def(
        "feasibility_jump",
        [](int m, int n, arr<int> indptr, arr<int> indices, arr<double> data, arr<double> row_lo, arr<double> row_hi,
           arr<double> lb, arr<double> ub, arr<int8_t> integer, arr<double> c, py::object x0, long long max_work,
           unsigned long long seed, double obj_weight) {
            if (indptr.size() != m + 1 || row_lo.size() != m || lb.size() != n || c.size() != n)
                throw std::invalid_argument("inconsistent sizes");
            std::vector<char> it(integer.data(), integer.data() + n);
            fedmip::FeasibilityJump fj(m, n, indptr.data(), indices.data(), data.data(), row_lo.data(),
                                       row_hi.data(), lb.data(), ub.data(), it.data(), c.data());
            std::vector<double> start;
            if (!x0.is_none()) {
                arr<double> a = x0.cast<arr<double>>();
                start.assign(a.data(), a.data() + a.size());
            }
            fedmip::FJResult r;
            {
                py::gil_scoped_release nogil;
                r = fj.run(start.empty() ? nullptr : start.data(), max_work, seed, obj_weight);
            }
            return py::make_tuple(r.found, r.found ? out(r.x) : py::array_t<double>(0), r.objective, r.steps, r.work);
        },
        py::arg("m"), py::arg("n"), py::arg("indptr"), py::arg("indices"), py::arg("data"), py::arg("row_lo"),
        py::arg("row_hi"), py::arg("lb"), py::arg("ub"), py::arg("integer"), py::arg("c"), py::arg("x0"),
        py::arg("max_work"), py::arg("seed") = 0, py::arg("obj_weight") = 0.0);
    mod.doc() = "fedhpc.mip kernels: event-driven domain propagation with a bound trail";
    py::class_<Propagator>(mod, "Propagator")
        .def(py::init([](int m, int n, arr<int> indptr, arr<int> indices, arr<double> data, arr<double> row_lo,
                         arr<double> row_hi, arr<int8_t> integer) {
                 if (indptr.size() != m + 1 || row_lo.size() != m || row_hi.size() != m || integer.size() != n)
                     throw std::invalid_argument("inconsistent sizes");
                 std::vector<char> it(integer.data(), integer.data() + n);
                 return new Propagator(m, n, vec(indptr), vec(indices), vec(data), vec(row_lo), vec(row_hi),
                                       std::move(it));
             }),
             py::arg("m"), py::arg("n"), py::arg("indptr"), py::arg("indices"), py::arg("data"), py::arg("row_lo"),
             py::arg("row_hi"), py::arg("integer"))
        .def_property(
            "int_tol", [](const Propagator& p) { return p.opt.int_tol; }, [](Propagator& p, double v) { p.opt.int_tol = v; })
        .def_property(
            "feas_tol", [](const Propagator& p) { return p.opt.feas_tol; },
            [](Propagator& p, double v) { p.opt.feas_tol = v; })
        .def_property(
            "min_rel_change", [](const Propagator& p) { return p.opt.min_rel_change; },
            [](Propagator& p, double v) { p.opt.min_rel_change = v; })
        .def_property(
            "max_abs_bound", [](const Propagator& p) { return p.opt.max_abs_bound; },
            [](Propagator& p, double v) { p.opt.max_abs_bound = v; })
        .def_property(
            "max_work", [](const Propagator& p) { return p.opt.max_work; },
            [](Propagator& p, long long v) { p.opt.max_work = v; })
        // stateless: (feasible, lb, ub, tightened, rounds, infeasible_row)
        .def("propagate",
             [](Propagator& p, arr<double> lb, arr<double> ub, int max_rounds, py::object rows) {
                 check(p, lb, ub);
                 p.reset(lb.data(), ub.data());
                 bool ok;
                 if (rows.is_none()) {
                     ok = p.propagate_all(max_rounds);
                 } else {
                     arr<int> r = rows.cast<arr<int>>();
                     ok = p.propagate_all(max_rounds, r.data(), (int)r.size());
                 }
                 return py::make_tuple(ok, out(p.lb()), out(p.ub()), p.tightened(), p.rounds(), p.infeasible_row());
             },
             py::arg("lb"), py::arg("ub"), py::arg("max_rounds") = 20, py::arg("rows") = py::none())
        // stateful domain with a trail
        .def("reset", [](Propagator& p, arr<double> lb, arr<double> ub) {
            check(p, lb, ub);
            p.reset(lb.data(), ub.data());
        })
        .def("propagate_domain", [](Propagator& p, int max_rounds) { return p.propagate_all(max_rounds); },
             py::arg("max_rounds") = 20)
        .def("level", &Propagator::level)
        .def("undo", &Propagator::undo, py::arg("level"))
        .def("change",
             [](Propagator& p, arr<int> cols, arr<double> nl, arr<double> nu, int max_rounds) {
                 if (nl.size() != cols.size() || nu.size() != cols.size()) throw std::invalid_argument("sizes");
                 for (ssize_t t = 0; t < cols.size(); ++t)
                     if (cols.data()[t] < 0 || cols.data()[t] >= p.cols()) throw std::out_of_range("column");
                 return p.change(cols.data(), nl.data(), nu.data(), (int)cols.size(), max_rounds);
             },
             py::arg("cols"), py::arg("lb"), py::arg("ub"), py::arg("max_rounds") = 20)
        .def("changed_since",
             [](Propagator& p, size_t level) {
                 std::vector<int> js = p.changed_since(level);
                 py::array_t<int> j(js.size());
                 py::array_t<double> l(js.size()), u(js.size());
                 for (size_t t = 0; t < js.size(); ++t)
                     j.mutable_data()[t] = js[t], l.mutable_data()[t] = p.lb()[js[t]], u.mutable_data()[t] = p.ub()[js[t]];
                 return py::make_tuple(j, l, u);
             },
             py::arg("level"))
        .def("lb", [](const Propagator& p) { return out(p.lb()); })
        .def("ub", [](const Propagator& p) { return out(p.ub()); })
        .def("bound", [](const Propagator& p, int j) { return py::make_tuple(p.lb()[j], p.ub()[j]); })
        .def("activity_bounds",
             [](const Propagator& p) {
                 std::vector<double> mn(p.rows()), mx(p.rows());
                 p.activity_bounds(mn.data(), mx.data());
                 return py::make_tuple(out(mn), out(mx));
             })
        .def_property_readonly("infeasible_row", &Propagator::infeasible_row)
        .def_property_readonly("tightened", &Propagator::tightened)
        .def_property_readonly("work", &Propagator::work);
}
