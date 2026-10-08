// Fast normal-pattern / compressed arc-flow graph construction.
//
// Python still owns the Gurobi model.  This module moves the pseudo-polynomial
// closure and graph bookkeeping out of the model-building hot path and returns
// compact, ready-to-consume start lists, node lists and loss arcs.
#include <algorithm>
#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using JobSpec = std::tuple<int, int, int, int, int>;
//                         jid  mid lo  stop p
using Arrival = std::pair<int, int>;  // mid, completion/event time

struct TypeGraph {
    int mid;
    std::vector<int> nodes;
    std::vector<std::pair<int, int>> loss_arcs;
};

static py::tuple build_normal_pattern_graph(
    int horizon, int n_types, const std::vector<JobSpec>& specs,
    const std::vector<Arrival>& arrivals
) {
    if (horizon < 0 || n_types < 0) throw std::invalid_argument("negative graph size");

    std::vector<std::vector<std::uint8_t>> pattern(
        n_types, std::vector<std::uint8_t>(static_cast<std::size_t>(horizon) + 1, 0));
    std::vector<std::map<int, int>> min_release(n_types);  // p -> minimum lo
    std::vector<std::set<int>> events(n_types);
    for (int m = 0; m < n_types; ++m) {
        events[m].insert(0);
        events[m].insert(horizon);
    }

    for (const auto& [jid, mid, lo, stop, p] : specs) {
        (void)jid;
        if (mid < 0 || mid >= n_types || lo < 0 || lo > horizon || p <= 0 || stop < lo)
            throw std::invalid_argument("invalid job/type graph specification");
        pattern[mid][lo] = 1;
        events[mid].insert(lo);
        auto [it, inserted] = min_release[mid].emplace(p, lo);
        if (!inserted) it->second = std::min(it->second, lo);
    }
    for (const auto& [mid, v] : arrivals) {
        if (mid < 0 || mid >= n_types || v < 0 || v > horizon)
            throw std::invalid_argument("invalid running-job arrival");
        pattern[mid][v] = 1;
        events[mid].insert(v);
    }

    // Algorithm 1 closure. Grouping equal processing times by minimum release
    // is an exact superset reduction and avoids repeating identical updates.
    for (int m = 0; m < n_types; ++m) {
        for (int q = 0; q <= horizon; ++q) {
            if (!pattern[m][q]) continue;
            for (const auto& [p, lo] : min_release[m])
                if (q >= lo && q + p <= horizon) pattern[m][q + p] = 1;
        }
    }

    std::vector<std::tuple<int, int, std::vector<int>>> starts;
    starts.reserve(specs.size());
    for (const auto& [jid, mid, lo, stop, p] : specs) {
        (void)p;
        std::vector<int> allowed;
        for (int t = lo; t < stop; ++t)
            if (pattern[mid][t]) allowed.push_back(t);
        starts.emplace_back(jid, mid, std::move(allowed));
    }

    std::vector<TypeGraph> graphs;
    graphs.reserve(n_types);
    for (int m = 0; m < n_types; ++m) {
        TypeGraph g;
        g.mid = m;
        for (int q = 0; q <= horizon; ++q)
            if (pattern[m][q] || q == 0 || q == horizon) g.nodes.push_back(q);

        const std::vector<int> ev(events[m].begin(), events[m].end());
        for (int q : g.nodes) {
            if (q >= horizon) continue;
            auto it = std::upper_bound(ev.begin(), ev.end(), q);
            if (it == ev.end()) throw std::logic_error("loss arc has no terminal event");
            g.loss_arcs.emplace_back(q, *it);
        }
        graphs.push_back(std::move(g));
    }

    py::list graph_out;
    for (const auto& g : graphs)
        graph_out.append(py::make_tuple(g.mid, g.nodes, g.loss_arcs));
    return py::make_tuple(starts, graph_out);
}

PYBIND11_MODULE(_graph, m) {
    m.doc() = "Normal-pattern and compressed arc-flow graph builder";
    m.def("build_normal_pattern_graph", &build_normal_pattern_graph,
          py::arg("horizon"), py::arg("n_types"), py::arg("job_specs"),
          py::arg("running_arrivals"));
}
