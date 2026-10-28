"""The permutative instance classes of fedhpc.instance_gen.

Structural checks run on class A's full factorial (162 instances), class B's
pairwise-covering set (70) and a seeded sample with counts anywhere in
[100, 500]; exact checks pin the properties the design guarantees:

* min turnaround is load- and count-independent (unlimited cloud lets every
  job start at its release), and greedy earliest-completion attains it;
* raising one type's load only removes capacity (the half-full running set is
  a prefix of the full one), so the cheapest way to reach min turnaround, and
  the best turnaround at minimum cost, can only get worse;
* equivalent specs (interchangeable or irrelevant on-prem types) describe the
  same problem, and enumeration yields one representative per class.
"""
import hashlib
import itertools
import json
import math
from pathlib import Path

import pytest

from fedhpc import baselines
from fedhpc import instance_gen as g
from fedhpc.model import solve_f1, solve_f2

_GUROBI = {"OutputFlag": 0}

ALL_A = list(g.iter_specs("A"))
COVER_B = g.covering_specs("B")
SAMPLED = g.sample_specs("A", 15, seed=1) + g.sample_specs("B", 25, seed=2)
STRUCTURAL = ALL_A + COVER_B + SAMPLED


def _ids(specs):
    return [s.name for s in specs]


@pytest.fixture(scope="module")
def built():
    cache = {}

    def get(spec):
        if spec.name not in cache:
            cache[spec.name] = (g.generate(spec), g.build(spec))
        return cache[spec.name]
    return get


def _min_turnaround(inst):
    """Σ w_j (⌈a_j⌉ + min_m p_jm − a_j): every job starting at its release."""
    return sum(j.weight * (math.ceil(j.arrival) + min(inst.p_occ[j.id, m] for m in inst.F[j.id])
                           - j.arrival)
               for j in inst.jobs)


# ── the design space ─────────────────────────────────────────────────────────

def test_space_sizes():
    assert len(ALL_A) == g.space_size("A") == 2 * 9 * 9
    assert g.space_size("B", distinct=False) == 2 * 9 ** 5
    distinct = {g.canonical(s).name for s in g.iter_specs("B", distinct=False)}
    assert len(distinct) == g.space_size("B") == sum(1 for _ in g.iter_specs("B"))
    assert len({s.name for s in ALL_A}) == len(ALL_A)


@pytest.mark.parametrize("n_factors", [4, 10])
def test_orthogonal_array_covers_every_factor_pair(n_factors):
    rows = g._orthogonal_array(n_factors)
    for a in range(n_factors):
        for b in range(a + 1, n_factors):
            assert {(r[a], r[b]) for r in rows} == {(x, y) for x in range(3) for y in range(3)}


@pytest.mark.parametrize("cls", sorted(g.CLASSES))
def test_covering_set_represents_every_array_row(cls):
    k = len(g.CLASSES[cls])
    names = {s.name for s in g.covering_specs(cls)}
    for w in g.WORKLOADS:
        for row in g._orthogonal_array(2 * k):
            spec = g.InstanceSpec(cls, w, tuple(g.COUNT_LEVELS[i] for i in row[:k]),
                                  tuple("FHE"[i] for i in row[k:]))
            assert g.canonical(spec).name in names


def test_catalogue_matches_the_request():
    a, b = g.CLASSES["A"], g.CLASSES["B"]
    assert len(a) == 2 and all(t.clouds == ((("aws", t.clouds[0][1]),)) for t in a)
    assert len(b) == 5 and all([p for p, _ in t.clouds] == ["aws", "azure"] for t in b)


def test_cloud_equivalents_cover_their_node():
    prices = g.load_prices()
    for types in g.CLASSES.values():
        for t in types:
            for _, sku in t.clouds:
                assert prices[sku]["vcpu"] // 2 >= t.cpu and prices[sku]["mem_gib"] >= t.mem


def test_price_file_is_complete():
    prices = g.load_prices()
    assert {sku for _, sku in g.CLOUD_SKUS} <= prices.keys()
    for provider, sku in g.CLOUD_SKUS:
        p = prices[sku]
        assert p["provider"] == provider and 0 < p["usd_per_hour"] < 50


@pytest.mark.parametrize("bad", [
    dict(cls="C"), dict(workload="x"), dict(counts=(99, 300)), dict(counts=(300, 501)),
    dict(counts=(300,)), dict(loads=("F", "Q")), dict(counts=(300.0, 300)),
])
def test_spec_validation(bad):
    kw = dict(cls="A", workload="pos", counts=(300, 300), loads=("F", "H")) | bad
    with pytest.raises(ValueError):
        g.InstanceSpec(**kw)


# ── per instance ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("spec", STRUCTURAL, ids=_ids(STRUCTURAL))
def test_structure(spec, built):
    data, inst = built(spec)
    assert g.InstanceSpec.parse(spec.name) == spec
    onprem = g.CLASSES[spec.cls]
    k = len(onprem)
    types = inst.instance_types
    assert [m.capacity for m in types[:k]] == list(spec.counts)
    assert all(m.capacity is None and m.kind == "cloud" for m in types[k:])
    assert len(types) == k + sum(len(t.clouds) for t in onprem)
    # running jobs realise the load: none / ⌊N/2⌋ / N per type
    expect = {i: n for i, n in enumerate(spec.running_counts()) if n}
    assert inst.n_running == expect
    for i, (n, load) in enumerate(zip(spec.counts, spec.loads)):
        assert inst.occupied.get((i, 0), 0) == {"F": 0, "H": n // 2, "E": n}[load]
    # cost per slot from the price file
    slot_h = inst.slot_size_seconds / 3600
    names = data["_info"]["instance_type_names"]
    for m in types[k:]:
        assert m.cost_vm == pytest.approx(names[str(m.id)]["usd_per_hour"] * slot_h)
    # jobs and horizon are the workload's, unchanged
    src = g._workload(spec.workload)
    assert inst.horizon == src["horizon"] and len(inst.jobs) == len(src["jobs"])
    # build() already rejects jobs without a feasible type/start; on top of
    # that every job fits some on-prem type, and every on-prem fit has a cloud
    for j in inst.jobs:
        on = [m for m in inst.F[j.id] if m < k]
        assert on
        for m in on:
            clouds = {mid for mid in inst.F[j.id] if names[str(mid)].get("equivalent_of") == m}
            assert len(clouds) == len(onprem[m].clouds)


@pytest.mark.parametrize("spec", STRUCTURAL, ids=_ids(STRUCTURAL))
def test_reproducible(spec, built):
    data, _ = built(spec)
    assert g.generate(spec) == data


@pytest.mark.parametrize("spec", STRUCTURAL, ids=_ids(STRUCTURAL))
def test_baselines_are_feasible_and_greedy_hits_min_turnaround(spec, built):
    _, inst = built(spec)
    greedy = baselines.greedy_earliest_completion(inst)
    inst.check_schedule(greedy.assignment)
    assert greedy.f1 == pytest.approx(_min_turnaround(inst))
    onprem = baselines.onprem_only_spt(inst)
    inst.check_schedule(onprem.assignment, complete=False)
    assert onprem.f2 == 0
    for theta in (0, 6, 36):
        tb = baselines.threshold_bursting(inst, theta)
        inst.check_schedule(tb.assignment)


@pytest.mark.parametrize("spec", [s for s in STRUCTURAL if "E" in s.loads][:40],
                         ids=_ids([s for s in STRUCTURAL if "E" in s.loads][:40]))
def test_running_ends_follow_the_workload_and_nest_across_loads(spec):
    pool = {r["end"] for r in g._workload(spec.workload)["running_jobs"]}
    for i, load in enumerate(spec.loads):
        full = g.running_ends(g.InstanceSpec(spec.cls, spec.workload, spec.counts,
                                             spec.loads[:i] + ("E",) + spec.loads[i + 1:]), i)
        half = g.running_ends(g.InstanceSpec(spec.cls, spec.workload, spec.counts,
                                             spec.loads[:i] + ("H",) + spec.loads[i + 1:]), i)
        assert len(full) == spec.counts[i] and set(full) <= pool
        assert half == full[:spec.counts[i] // 2]


# ── equivalent instances ─────────────────────────────────────────────────────

def test_class_b_equivalences():
    """No fedhpc job fits c16; c32 and m32 fit the same jobs in both workloads."""
    assert g.type_groups("B", "fedhpc") == ((0,), ((1, 4), (2,), (3,)))
    assert g.type_groups("B", "pos") == ((), ((0,), (1, 4), (2,), (3,)))
    assert g.type_groups("A", "fedhpc") == g.type_groups("A", "pos") == ((), ((0,), (1,)))


def _variants(spec):
    """Specs equivalent to ``spec``: every interchangeable group permuted,
    every irrelevant type moved to another count and load."""
    irrelevant, groups = g.type_groups(spec.cls, spec.workload)
    out = []
    for perm in itertools.product(*(itertools.permutations(gr) for gr in groups)):
        counts, loads = list(spec.counts), list(spec.loads)
        for gr, p in zip(groups, perm):
            for i, j in zip(gr, p):
                counts[i], loads[i] = spec.counts[j], spec.loads[j]
        for i in irrelevant:
            counts[i], loads[i] = 437, "E"
        out.append(g.InstanceSpec(spec.cls, spec.workload, tuple(counts), tuple(loads)))
    return out


EQUIV = [s for s in g.sample_specs("B", 30, seed=3)
         if any(len(gr) > 1 for gr in g.type_groups(s.cls, s.workload)[1])][:12]


def _signature(inst, k):
    """The instance as an optimisation problem, up to relabelling the on-prem
    types: per on-prem type some job fits (jobs it fits, capacity, busy
    profile), sorted; the cloud types and jobs in order."""
    onprem = sorted(
        (fit, m.capacity, tuple(inst.occupied.get((m.id, t), 0) for t in range(inst.horizon)))
        for m in inst.instance_types[:k]
        if (fit := tuple(sorted(j.id for j in inst.jobs if m.id in inst.F[j.id]))))
    clouds = [(m.cpu, m.mem, m.cost_vm) for m in inst.instance_types[k:]]
    return onprem, clouds, [(j.id, j.arrival, j.exec_time) for j in inst.jobs]


@pytest.mark.parametrize("spec", EQUIV, ids=_ids(EQUIV))
def test_equivalent_specs_share_a_canonical_form_and_a_problem(spec):
    assert g.canonical(spec) == spec
    k = len(g.CLASSES[spec.cls])
    sig = _signature(g.build(spec), k)
    for v in _variants(spec):
        assert g.canonical(v) == spec
        assert _signature(g.build(v), k) == sig


@pytest.mark.needs_full_license
@pytest.mark.parametrize("spec", EQUIV[:3], ids=_ids(EQUIV[:3]))
def test_equivalent_specs_have_equal_optima(spec):
    ref = solve_f1(g.build(spec), MIPGap=0, **_GUROBI)
    for v in _variants(spec)[1:]:
        s = solve_f1(g.build(v), MIPGap=0, **_GUROBI)
        assert (s.f1, s.f2) == pytest.approx((ref.f1, ref.f2))


# ── files on disk ────────────────────────────────────────────────────────────

_MANIFEST = g.DATA / "generated" / "manifest.json"


@pytest.mark.skipif(not _MANIFEST.exists(), reason="run scripts/build_generated_instances.py")
def test_written_files_match_the_generator():
    manifest = json.loads(_MANIFEST.read_text())
    assert {s.name for s in ALL_A + COVER_B} <= manifest.keys()
    for name in sorted(manifest)[::7]:
        path = _MANIFEST.parent / f"{name}.json"
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == manifest[name]["sha256"]
        assert json.loads(raw) == g.generate(g.InstanceSpec.parse(name))


# ── exact properties (Gurobi) ────────────────────────────────────────────────

def _load_chains(cls, n_chains):
    """Per chain: one base spec and its type-``i`` load raised F → H → E."""
    k = len(g.CLASSES[cls])
    chains = []
    for s in g.sample_specs(cls, n_chains, seed=7):
        i = g._seed(s.name) % k
        chains.append([g.InstanceSpec(s.cls, s.workload, s.counts,
                                      s.loads[:i] + (lvl,) + s.loads[i + 1:]) for lvl in "FHE"])
    return chains


F1_CHAINS = _load_chains("A", 3) + _load_chains("B", 3)


@pytest.mark.needs_full_license
@pytest.mark.parametrize("chain", F1_CHAINS, ids=[c[0].name for c in F1_CHAINS])
def test_cost_of_min_turnaround_grows_with_load(chain):
    sols = []
    for spec in chain:
        inst = g.build(spec)
        s = solve_f1(inst, MIPGap=0, **_GUROBI)
        assert s.status == "optimal"
        inst.check_schedule(s.assignment)
        assert s.f1 == pytest.approx(_min_turnaround(inst))
        assert s.f2 <= baselines.greedy_earliest_completion(inst).f2 + 1e-6
        sols.append(s)
    assert sols[0].f2 <= sols[1].f2 + 1e-6 <= sols[2].f2 + 2e-6


F2_CHAINS = _load_chains("A", 2) + _load_chains("B", 1)


@pytest.mark.slow
@pytest.mark.needs_full_license
@pytest.mark.parametrize("chain", F2_CHAINS, ids=[c[0].name for c in F2_CHAINS])
def test_turnaround_at_min_cost_grows_with_load(chain):
    sols = []
    for spec in chain:
        inst = g.build(spec)
        s = solve_f2(inst, MIPGap=0, **_GUROBI)
        assert s.status == "optimal"
        inst.check_schedule(s.assignment)
        free = baselines.onprem_only_spt(inst)
        if len(free.assignment) == len(inst.jobs):
            assert s.f2 == 0 and s.f1 <= free.f1 + 1e-6
        sols.append(s)
    assert [s.f2 for s in sols] == sorted(s.f2 for s in sols)
    if sols[0].f2 == sols[-1].f2 == 0:
        assert sols[0].f1 <= sols[1].f1 + 1e-6 <= sols[2].f1 + 2e-6
