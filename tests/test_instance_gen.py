"""The permutative instance classes of fedhpc.instance_gen.

Structural checks run on class A's full factorial (162 instances), class B's
pairwise-covering set and a seeded sample with counts anywhere in
[100, 500]; exact checks pin the properties the design guarantees:

* unlimited cloud lets every job start at its release, so greedy
  earliest-completion lies between "every job at release on its fastest
  type" and "every job at release in the cloud" (cloud is slower than
  on-prem by the result transfer τ^IO·ι_j);
* raising the load only removes capacity (the half-full running set is a
  prefix of the full one) and raising the io level only lengthens and
  prices up cloud options (ι_j grows per job), so the optimal turnaround and
  the optimal cost can only get worse;
* equivalent specs (interchangeable on-prem types) describe the same problem,
  and enumeration yields one representative per class;
* every type an instance carries is one some job would be placed on.
"""
import hashlib
import itertools
import json
import math
from pathlib import Path

import pytest

from fedhpc import baselines
from fedhpc import instance_gen as g
from fedhpc import io_model
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


def _min_turnaround(inst, types=None):
    """Σ w_j (⌈a_j⌉ + min_m p_jm − a_j): every job starting at its release on
    its fastest type (among ``types``, default all)."""
    return sum(j.weight * (math.ceil(j.arrival) - j.arrival
                           + min(inst.p_occ[j.id, m] for m in inst.F[j.id]
                                 if types is None or m in types))
               for j in inst.jobs)


def _cloud_turnaround(inst):
    return _min_turnaround(inst, {m.id for m in inst.instance_types if m.capacity is None})


# ── the design space ─────────────────────────────────────────────────────────

def test_space_sizes():
    assert len(ALL_A) == g.space_size("A") == 2 * 9 * 3 * 3
    assert g.space_size("B", distinct=False) == (3 ** 4 + 3 ** 5) * 3 * 3
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
    names = {s.name for s in g.covering_specs(cls)}
    for w in g.WORKLOADS:
        k = len(g.onprem_types(cls, w))
        for row in g._orthogonal_array(k + 2):
            spec = g.InstanceSpec(cls, w, tuple(g.COUNT_LEVELS[i] for i in row[:k]),
                                  "FHE"[row[k]], "LMH"[row[k + 1]])
            assert g.canonical(spec).name in names


def test_catalogue_matches_the_request():
    a, b = g.CLASSES["A"], g.CLASSES["B"]
    assert len(a) == 2 and all(t.clouds == ((("aws", t.clouds[0][1]),)) for t in a)
    assert len(b) == 5 and all([p for p, _ in t.clouds] == ["aws", "azure"] for t in b)


def test_only_pickable_types_are_kept():
    """No fedhpc job fits c16; m32's equivalents lose to c32's (no job needs
    more than 256 GB on 32 cores); D128s_v6 stays though m6i.32xlarge is
    cheaper, since dominance is checked within a provider."""
    names = lambda c, w: [t.name for t in g.onprem_types(c, w)]  # noqa: E731
    skus = lambda c, w: [sku for _, sku, _ in g.cloud_types(c, w)]  # noqa: E731
    assert names("A", "fedhpc") == names("A", "pos") == ["c40", "c64"]
    assert names("B", "fedhpc") == ["c32", "c40", "c64", "m32"]
    assert names("B", "pos") == ["c16", "c32", "c40", "c64", "m32"]
    assert skus("A", "fedhpc") == skus("A", "pos") == ["m6i.24xlarge", "m6i.32xlarge"]
    big = ["m6i.16xlarge", "Standard_D64s_v5", "m6i.24xlarge", "Standard_D96s_v5",
           "m6i.32xlarge", "Standard_D128s_v6"]
    assert skus("B", "fedhpc") == big
    assert skus("B", "pos") == ["m6i.8xlarge", "Standard_D32s_v5"] + big


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
    dict(counts=(300,)), dict(load="Q"), dict(load=("F", "H")), dict(io="X"),
    dict(counts=(300.0, 300)),
])
def test_spec_validation(bad):
    kw = dict(cls="A", workload="pos", counts=(300, 300), load="H", io="M") | bad
    with pytest.raises(ValueError):
        g.InstanceSpec(**kw)


# ── per instance ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("spec", STRUCTURAL, ids=_ids(STRUCTURAL))
def test_structure(spec, built):
    data, inst = built(spec)
    assert g.InstanceSpec.parse(spec.name) == spec
    onprem = g.onprem_types(spec.cls, spec.workload)
    k = len(onprem)
    types = inst.instance_types
    assert [m.capacity for m in types[:k]] == list(spec.counts)
    assert all(m.capacity is None and m.kind == "cloud" for m in types[k:])
    assert len(types) == k + len(g.cloud_types(spec.cls, spec.workload))
    # running jobs realise the load: none / ⌊N/2⌋ / N per type
    expect = {i: n for i, n in enumerate(spec.running_counts()) if n}
    assert inst.n_running == expect
    for i, n in enumerate(spec.counts):
        assert inst.occupied.get((i, 0), 0) == {"F": 0, "H": n // 2, "E": n}[spec.load]
    # cost per slot from the price file; egress price and transfer time per provider
    slot_h = inst.slot_size_seconds / 3600
    names = data["_info"]["instance_type_names"]
    for m in types[:k]:
        assert m.io_time == m.cost_io == 0
    for m in types[k:]:
        provider = names[str(m.id)]["provider"]
        assert m.cost_vm == pytest.approx(names[str(m.id)]["usd_per_hour"] * slot_h)
        assert m.cost_io == io_model.EGRESS_USD_PER_GB[provider]
        assert m.io_time == io_model.io_time(provider, inst.slot_size_seconds)
    assert all(j.io_volume >= 0 for j in inst.jobs)
    assert data["_info"]["io"]["total_gb"] == pytest.approx(sum(j.io_volume for j in inst.jobs))
    # jobs and horizon are the workload's, unchanged
    src = g._workload(spec.workload)
    assert inst.horizon == src["horizon"] and len(inst.jobs) == len(src["jobs"])
    # build() already rejects jobs without a feasible type/start; on top of
    # that every job fits some on-prem type and some size of every provider,
    # and every type is the job's (cheapest-in-provider) choice for some job
    providers = {p for t in onprem for p, _ in t.clouds}
    picked = set()
    for j in inst.jobs:
        on = [m for m in inst.F[j.id] if m < k]
        assert on
        # the horizon guard keeps every cloud option open
        assert all(len(inst.T[j.id, m]) > 0 for m in inst.F[j.id] if m >= k)
        picked.update(on)
        for p in providers:
            fit = [m for m in inst.F[j.id] if m >= k and names[str(m)]["provider"] == p]
            assert fit
            best = min(types[m].cost_vm for m in fit)
            picked.update(m for m in fit if types[m].cost_vm == best)
    assert picked == {m.id for m in types}


@pytest.mark.parametrize("spec", STRUCTURAL, ids=_ids(STRUCTURAL))
def test_reproducible(spec, built):
    data, _ = built(spec)
    assert g.generate(spec) == data


@pytest.mark.parametrize("spec", STRUCTURAL, ids=_ids(STRUCTURAL))
def test_baselines_are_feasible_and_greedy_hits_min_turnaround(spec, built):
    _, inst = built(spec)
    greedy = baselines.greedy_earliest_completion(inst)
    inst.check_schedule(greedy.assignment)
    assert _min_turnaround(inst) - 1e-6 <= greedy.f1 <= _cloud_turnaround(inst) + 1e-6
    onprem = baselines.onprem_only_spt(inst)
    inst.check_schedule(onprem.assignment, complete=False)
    assert onprem.f2 == 0
    for theta in (0, 6, 36):
        tb = baselines.threshold_bursting(inst, theta)
        inst.check_schedule(tb.assignment)


NEST = [s for s in STRUCTURAL if s.load == "E"][:40]


@pytest.mark.parametrize("spec", NEST, ids=_ids(NEST))
def test_running_ends_follow_the_workload_and_nest_across_loads(spec):
    pool = {r["end"] for r in g._workload(spec.workload)["running_jobs"]}
    half_spec = g.InstanceSpec(spec.cls, spec.workload, spec.counts, "H", spec.io)
    for i in range(len(spec.counts)):
        full, half = g.running_ends(spec, i), g.running_ends(half_spec, i)
        assert len(full) == spec.counts[i] and set(full) <= pool
        assert half == full[:spec.counts[i] // 2]


# ── result data ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("workload", sorted(g.WORKLOADS))
def test_io_volume_grows_with_the_level_and_follows_the_policies(workload):
    vols = {io: [j["io_volume"] for j in g.generate(g.InstanceSpec(
        "A", workload, (300, 300), "F", io))["jobs"]] for io in "LMH"}
    assert all(a <= b <= c for a, b, c in zip(vols["L"], vols["M"], vols["H"]))
    assert any(a < c for a, c in zip(vols["L"], vols["H"]))
    # without noise and caps a 40-core job's volume is rate × core-hours
    src, flags = g._workload(workload), io_model.ensemble_flags(workload)
    slot = src["_info"]["fedhpc_settings"]["time_scale_seconds"]
    for j, v in zip(src["jobs"], vols["H"]):
        if j["cpu"] == io_model.C_REF:
            z = math.log10(v / (io_model.RATES["F"] * 40 * j["exec_time"] * slot / 3600))
            assert abs(z) <= io_model.SIGMA * io_model.Z_CLIP + 1e-9
    assert len(flags) == len(src["jobs"]) and 0.9 < sum(flags) / len(flags) < 1


def test_io_time_is_the_disk_bound_rate():
    """1 GB at the default cloud disk throughput: 7.63 s (gp3, 125 MiB/s) and
    8 s (Premium SSD v2, 125 MB/s)."""
    assert io_model.io_time("aws", 600) * 600 == pytest.approx(1000 / 131.072)
    assert io_model.io_time("azure", 600) * 600 == pytest.approx(8.0)
    assert io_model.EGRESS_USD_PER_GB["azure"] < io_model.EGRESS_USD_PER_GB["aws"]


# ── equivalent instances ─────────────────────────────────────────────────────

def test_class_b_equivalences():
    """c32 and m32 fit the same jobs in both workloads."""
    assert g.type_groups("B", "fedhpc") == ((0, 3), (1,), (2,))
    assert g.type_groups("B", "pos") == ((0,), (1, 4), (2,), (3,))
    assert g.type_groups("A", "fedhpc") == g.type_groups("A", "pos") == ((0,), (1,))


def _variants(spec):
    """Specs equivalent to ``spec``: every interchangeable group permuted."""
    groups = g.type_groups(spec.cls, spec.workload)
    out = []
    for perm in itertools.product(*(itertools.permutations(gr) for gr in groups)):
        counts = list(spec.counts)
        for gr, p in zip(groups, perm):
            for i, j in zip(gr, p):
                counts[i] = spec.counts[j]
        out.append(g.InstanceSpec(spec.cls, spec.workload, tuple(counts), spec.load, spec.io))
    return out


EQUIV = [s for s in g.sample_specs("B", 30, seed=3)
         if any(len(gr) > 1 for gr in g.type_groups(s.cls, s.workload))][:12]


def _signature(inst, k):
    """The instance as an optimisation problem, up to relabelling the on-prem
    types: per on-prem type some job fits (jobs it fits, capacity, busy
    profile), sorted; the cloud types and jobs in order."""
    onprem = sorted(
        (fit, m.capacity, tuple(inst.occupied.get((m.id, t), 0) for t in range(inst.horizon)))
        for m in inst.instance_types[:k]
        if (fit := tuple(sorted(j.id for j in inst.jobs if m.id in inst.F[j.id]))))
    clouds = [(m.cpu, m.mem, m.cost_vm, m.cost_io, m.io_time) for m in inst.instance_types[k:]]
    return onprem, clouds, [(j.id, j.arrival, j.exec_time, j.io_volume) for j in inst.jobs]


@pytest.mark.parametrize("spec", EQUIV, ids=_ids(EQUIV))
def test_equivalent_specs_share_a_canonical_form_and_a_problem(spec):
    assert g.canonical(spec) == spec
    k = len(g.onprem_types(spec.cls, spec.workload))
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

def _chains(cls, n_chains, factor):
    """Per chain: one base spec with its load raised F → H → E, or its io
    level raised L → M → H."""
    levels = {"load": "FHE", "io": "LMH"}[factor]
    return [[g.InstanceSpec(s.cls, s.workload, s.counts, **({"load": s.load, "io": s.io}
                                                            | {factor: lvl}))
             for lvl in levels]
            for s in g.sample_specs(cls, n_chains, seed=7)]


F1_CHAINS = (_chains("A", 2, "load") + _chains("B", 2, "load")
             + _chains("A", 2, "io") + _chains("B", 1, "io"))


@pytest.mark.needs_full_license
@pytest.mark.parametrize("chain", F1_CHAINS, ids=[f"{c[0].name}->{c[-1].name}" for c in F1_CHAINS])
def test_min_turnaround_grows_along_the_chain(chain):
    sols = []
    for spec in chain:
        inst = g.build(spec)
        s = solve_f1(inst, MIPGap=0, **_GUROBI)
        assert s.status == "optimal"
        inst.check_schedule(s.assignment)
        assert _min_turnaround(inst) - 1e-6 <= s.f1 <= _cloud_turnaround(inst) + 1e-6
        assert s.f1 <= baselines.greedy_earliest_completion(inst).f1 + 1e-6
        sols.append(s)
    assert sols[0].f1 <= sols[1].f1 + 1e-6 <= sols[2].f1 + 2e-6


F2_CHAINS = _chains("A", 1, "load") + _chains("B", 1, "load") + _chains("A", 1, "io")


@pytest.mark.slow
@pytest.mark.needs_full_license
@pytest.mark.parametrize("chain", F2_CHAINS, ids=[f"{c[0].name}->{c[-1].name}" for c in F2_CHAINS])
def test_min_cost_grows_along_the_chain(chain):
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
