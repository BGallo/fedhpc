"""Permutative instance generator: two machine catalogues × two real workloads ×
per-type machine counts × per-type initial load.

Every instance is named by its :class:`InstanceSpec`, and the name alone
determines the instance (``generate(InstanceSpec.parse(name))`` is
reproducible bit for bit), so the full factorial space can be enumerated
lazily and any point of it rebuilt on demand without storing it.

Classes
-------
``A`` — two on-prem types modelled on the real csr1 (40 cores / 300 GB) and
        csr3 (64 cores / 480 GB) nodes, each with one AWS equivalent.
``B`` — five on-prem types (16c/128, 32c/256, 40c/300, 64c/480 and a
        memory-heavy 32c/512), each with an AWS *and* an Azure equivalent.

A cloud equivalent is the smallest instance of the same Intel generation
(Ice Lake: AWS m6i/r6i, Azure Dsv5/Esv5) whose physical cores (vCPU / 2,
SMT) and memory cover the on-prem node. Azure has no 128-vCPU Dsv5, so the
64-core node maps to Dsv6. Prices are on-demand Linux list prices from
``data/cloud_prices.json`` (``scripts/fetch_cloud_prices.py``), converted to
cost per slot. Same-shape Ice Lake sizes are priced identically by both
providers; only D128s_v6 differs from m6i.32xlarge.

Factors (per on-prem type)
--------------------------
count  ``N_m`` — any integer in [100, 500]; the default levels are 100/300/500.
load   at t = 0 — ``F`` free (no running jobs), ``H`` half full (⌊N_m/2⌋
       running jobs) or ``E`` entirely full (N_m running jobs).

Running-job end times are drawn from the empirical end-time distribution of
the workload's own ``running_jobs`` (the real cluster state of that
snapshot), with an RNG seeded by (class, workload, type group, count) only —
not by the load — so the half-full set is the first half of the full one:
raising a type's load only ever removes capacity.

Equivalent instances
--------------------
On-prem types differ only in which jobs they can run (all are free, perf 1,
no deploy; the cloud types do not depend on the factors). Per workload:

* a type no job fits is *irrelevant*: its count and load change nothing;
* types that fit exactly the same jobs are *interchangeable*: swapping their
  (count, load) pairs relabels the instance (their running jobs are drawn per
  group, so the swap is exact).

In class B, no ``fedhpc`` job fits c16, and c32 and m32 fit the same jobs in
both workloads. :func:`canonical` maps a spec to its class representative
(irrelevant types at the lowest count and free, interchangeable ones sorted),
and the enumerations below yield representatives only: class B's raw 118 098
combinations collapse to 36 450 distinct instances.

Benchmark sets: class A's full factorial (162) and, for class B, the
pairwise-covering set :func:`covering_specs` (70) — the on-prem types differ
in shape, so which type carries which load changes the optimum and no further
collapse is exact; the remaining B instances stay reachable by name.

Workloads
---------
``fedhpc`` — data/fedhpc_known_runtime_offline_10min.json (964 jobs, H=289)
``pos``    — data/pos_congestion_known_runtime_offline_10min.json (3340 jobs, H=288)
Jobs, horizon and slot length are copied unchanged.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .data import Instance

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
PRICES = DATA / "cloud_prices.json"

WORKLOADS = {
    "fedhpc": DATA / "fedhpc_known_runtime_offline_10min.json",
    "pos": DATA / "pos_congestion_known_runtime_offline_10min.json",
}
LOADS = {"F": 0.0, "H": 0.5, "E": 1.0}
COUNT_RANGE = (100, 500)
COUNT_LEVELS = (100, 300, 500)
ONPREM_STOR = CLOUD_STOR = 10000


@dataclass(frozen=True)
class OnPremType:
    name: str
    cpu: int                                 # cores per node
    mem: int                                 # GB per node
    clouds: tuple[tuple[str, str], ...]      # (provider, sku) equivalents


CLASSES: dict[str, tuple[OnPremType, ...]] = {
    "A": (
        OnPremType("c40", 40, 300, (("aws", "m6i.24xlarge"),)),
        OnPremType("c64", 64, 480, (("aws", "m6i.32xlarge"),)),
    ),
    "B": (
        OnPremType("c16", 16, 128, (("aws", "m6i.8xlarge"), ("azure", "Standard_D32s_v5"))),
        OnPremType("c32", 32, 256, (("aws", "m6i.16xlarge"), ("azure", "Standard_D64s_v5"))),
        OnPremType("c40", 40, 300, (("aws", "m6i.24xlarge"), ("azure", "Standard_D96s_v5"))),
        OnPremType("c64", 64, 480, (("aws", "m6i.32xlarge"), ("azure", "Standard_D128s_v6"))),
        OnPremType("m32", 32, 512, (("aws", "r6i.16xlarge"), ("azure", "Standard_E64s_v5"))),
    ),
}

CLOUD_SKUS: list[tuple[str, str]] = sorted(
    {c for types in CLASSES.values() for t in types for c in t.clouds})


@dataclass(frozen=True)
class InstanceSpec:
    cls: str
    workload: str
    counts: tuple[int, ...]   # N_m per on-prem type, in CLASSES[cls] order
    loads: tuple[str, ...]    # "F" / "H" / "E" per on-prem type

    def __post_init__(self):
        if self.cls not in CLASSES:
            raise ValueError(f"unknown class {self.cls!r}")
        if self.workload not in WORKLOADS:
            raise ValueError(f"unknown workload {self.workload!r}")
        k = len(CLASSES[self.cls])
        if len(self.counts) != k or len(self.loads) != k:
            raise ValueError(f"class {self.cls} needs {k} counts and {k} loads")
        lo, hi = COUNT_RANGE
        if not all(isinstance(n, int) and lo <= n <= hi for n in self.counts):
            raise ValueError(f"counts must be integers in [{lo}, {hi}], got {self.counts}")
        if not all(l in LOADS for l in self.loads):
            raise ValueError(f"loads must be among {''.join(LOADS)}, got {self.loads}")

    @property
    def name(self) -> str:
        """e.g. ``A_pos_n100-500_lFE`` — round-trips through :meth:`parse`."""
        return (f"{self.cls}_{self.workload}_n{'-'.join(map(str, self.counts))}"
                f"_l{''.join(self.loads)}")

    _NAME = re.compile(r"([A-Z])_([a-z]+)_n([\d-]+)_l([A-Z]+)")

    @classmethod
    def parse(cls, name: str) -> InstanceSpec:
        m = cls._NAME.fullmatch(name)
        if not m:
            raise ValueError(f"not an instance name: {name!r}")
        c, w, counts, loads = m.groups()
        return cls(c, w, tuple(int(n) for n in counts.split("-")), tuple(loads))

    def running_counts(self) -> tuple[int, ...]:
        return tuple(int(n * LOADS[l]) for n, l in zip(self.counts, self.loads))


# ─────────────────────────────────────────────────────────────────────────────
# Equivalence
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=None)
def type_groups(cls: str, workload: str) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    """``(irrelevant, groups)``: the on-prem type indices no job of the
    workload fits, and the partition of the others into types that fit
    exactly the same jobs (each group sorted, groups by first index)."""
    jobs = _workload(workload)["jobs"]
    by_fit: dict[frozenset, list[int]] = {}
    for i, t in enumerate(CLASSES[cls]):
        fit = frozenset(j["id"] for j in jobs
                        if j["cpu"] <= t.cpu and j["mem"] <= t.mem and j["stor"] <= ONPREM_STOR)
        by_fit.setdefault(fit, []).append(i)
    irrelevant = tuple(by_fit.pop(frozenset(), ()))
    return irrelevant, tuple(sorted(tuple(g) for g in by_fit.values()))


def _group_of(cls: str, workload: str, i: int) -> int:
    """First index of type ``i``'s group (``i`` itself when irrelevant)."""
    return next((g[0] for g in type_groups(cls, workload)[1] if i in g), i)


def canonical(spec: InstanceSpec) -> InstanceSpec:
    """The representative of ``spec``'s equivalence class: irrelevant types at
    the lowest count and free, each interchangeable group's (count, load)
    pairs sorted over the group's indices."""
    irrelevant, groups = type_groups(spec.cls, spec.workload)
    counts, loads = list(spec.counts), list(spec.loads)
    for i in irrelevant:
        counts[i], loads[i] = COUNT_RANGE[0], "F"
    order = {l: k for k, l in enumerate(LOADS)}
    for g in groups:
        pairs = sorted(((counts[i], loads[i]) for i in g), key=lambda p: (p[0], order[p[1]]))
        for i, (n, l) in zip(g, pairs):
            counts[i], loads[i] = n, l
    return InstanceSpec(spec.cls, spec.workload, tuple(counts), tuple(loads))


# ─────────────────────────────────────────────────────────────────────────────
# Enumeration
# ─────────────────────────────────────────────────────────────────────────────

def space_size(cls: str, *, workloads=tuple(WORKLOADS), count_levels=COUNT_LEVELS,
               load_levels=tuple(LOADS), distinct: bool = True) -> int:
    """Size of :func:`iter_specs`. With ``distinct`` (the default) one per
    equivalence class: an irrelevant type contributes 1, a group of g
    interchangeable types the C(L+g−1, g) multisets of its L level pairs
    (assumes ``count_levels`` contains the lowest count, 100)."""
    L = len(count_levels) * len(load_levels)
    if not distinct:
        return len(workloads) * L ** len(CLASSES[cls])
    total = 0
    for w in workloads:
        n = 1
        for g in type_groups(cls, w)[1]:
            n *= math.comb(L + len(g) - 1, len(g))
        total += n
    return total


def iter_specs(cls: str, *, workloads=tuple(WORKLOADS), count_levels=COUNT_LEVELS,
               load_levels=tuple(LOADS), distinct: bool = True):
    """The full factorial: every workload × count per type × load per type —
    with ``distinct`` (the default) only the canonical representative of each
    equivalence class (see :func:`canonical`)."""
    k = len(CLASSES[cls])
    for w in workloads:
        for counts in itertools.product(count_levels, repeat=k):
            for loads in itertools.product(load_levels, repeat=k):
                spec = InstanceSpec(cls, w, counts, loads)
                if not distinct or canonical(spec) == spec:
                    yield spec


def _orthogonal_array(n_factors: int) -> list[tuple[int, ...]]:
    """Strength-2 orthogonal array over 3 levels (Rao–Hamming): rows are all
    r ∈ Z₃ˢ, columns the projective points v of PG(s−1, 3), entry r·v mod 3.
    Every pair of columns shows all 9 level pairs equally often."""
    s = 1
    while (3 ** s - 1) // 2 < n_factors:
        s += 1
    cols = [v for v in itertools.product(range(3), repeat=s)
            if any(v) and next(x for x in v if x) == 1][:n_factors]
    return [tuple(sum(a * b for a, b in zip(r, v)) % 3 for v in cols)
            for r in itertools.product(range(3), repeat=s)]


def covering_specs(cls: str, *, workloads=tuple(WORKLOADS), count_levels=COUNT_LEVELS,
                   load_levels=tuple(LOADS), corners: bool = True) -> list[InstanceSpec]:
    """A pairwise-covering subset of :func:`iter_specs`: per workload, an
    orthogonal array over the 2k factors (every pair of factors meets every
    pair of levels), plus — with ``corners`` — the uniform configurations
    (all types at the same count and the same load), each row replaced by its
    canonical representative and duplicates dropped. Coverage then holds up
    to equivalence: every array row's class is represented."""
    if len(count_levels) != 3 or len(load_levels) != 3:
        raise ValueError("covering_specs needs exactly 3 count and 3 load levels")
    k = len(CLASSES[cls])
    out: dict[str, InstanceSpec] = {}
    for w in workloads:
        rows = _orthogonal_array(2 * k)
        if corners:
            rows += [(i,) * k + (j,) * k for i in range(3) for j in range(3)]
        for row in rows:
            spec = canonical(InstanceSpec(cls, w, tuple(count_levels[i] for i in row[:k]),
                                          tuple(load_levels[i] for i in row[k:])))
            out.setdefault(spec.name, spec)
    return list(out.values())


def sample_specs(cls: str, n: int, seed: int = 0, *, workloads=tuple(WORKLOADS)) -> list[InstanceSpec]:
    """``n`` specs with counts drawn uniformly from the whole range [100, 500]
    (not just the default levels) and loads uniformly from F/H/E, mapped to
    their canonical representatives."""
    rng = random.Random(seed)
    k = len(CLASSES[cls])
    lo, hi = COUNT_RANGE
    return [canonical(InstanceSpec(cls, rng.choice(workloads),
                                   tuple(rng.randint(lo, hi) for _ in range(k)),
                                   tuple(rng.choice(tuple(LOADS)) for _ in range(k))))
            for _ in range(n)]


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=None)
def _workload(name: str) -> dict:
    with open(WORKLOADS[name]) as f:
        return json.load(f)


@lru_cache(maxsize=None)
def load_prices(path: Path = PRICES) -> dict[str, dict]:
    with open(path) as f:
        return json.load(f)["prices"]


def _seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big")


def running_ends(spec: InstanceSpec, type_index: int) -> list[int]:
    """End slots of the running jobs on on-prem type ``type_index``: the first
    ``running_counts()[type_index]`` of N_m draws from the workload's
    empirical running-job end times (nested across loads)."""
    pool = sorted(r["end"] for r in _workload(spec.workload)["running_jobs"])
    # seeded by the type's group, not the type, so interchangeable types with
    # equal counts draw equal ends and swapping them is an exact relabelling
    group = _group_of(spec.cls, spec.workload, type_index)
    rng = random.Random(_seed(spec.cls, spec.workload, group, spec.counts[type_index]))
    draws = [rng.choice(pool) for _ in range(spec.counts[type_index])]
    return draws[:spec.running_counts()[type_index]]


def generate(spec: InstanceSpec, prices: dict[str, dict] | None = None) -> dict:
    """The instance as a JSON-ready dict (the schema of data/*.json)."""
    prices = load_prices() if prices is None else prices
    src = _workload(spec.workload)
    slot_s = float(src["_info"]["fedhpc_settings"]["time_scale_seconds"])
    onprem = CLASSES[spec.cls]

    types, names = [], {}
    for i, (t, n) in enumerate(zip(onprem, spec.counts)):
        types.append(dict(id=i, kind="on-prem", perf=1.0, io_time=0.0, deploy=0.0,
                          cost_io=0.0, cost_vm=0.0, cost_stor=0.0,
                          cpu=t.cpu, mem=t.mem, stor=ONPREM_STOR, capacity=n))
        names[str(i)] = dict(name=f"onprem_{t.name}", count=n, load=spec.loads[i])
    for i, t in enumerate(onprem):
        for provider, sku in t.clouds:
            p = prices[sku]
            mid = len(types)
            cost_vm = p["usd_per_hour"] * slot_s / 3600
            types.append(dict(id=mid, kind="cloud", perf=1.0, io_time=0.0, deploy=0.0,
                              cost_io=0.0, cost_vm=cost_vm, cost_stor=0.0,
                              cpu=p["vcpu"] // 2, mem=p["mem_gib"], stor=CLOUD_STOR,
                              capacity=None))
            names[str(mid)] = dict(name=f"{provider}_{sku}", provider=provider,
                                   equivalent_of=i, usd_per_hour=p["usd_per_hour"],
                                   vcpu=p["vcpu"])

    running = [dict(id=1_000_000 * (i + 1) + k, type_id=i, end=end)
               for i in range(len(onprem))
               for k, end in enumerate(running_ends(spec, i))]

    return dict(
        _info=dict(
            description=f"Generated by fedhpc.instance_gen: class {spec.cls}, workload "
                        f"{spec.workload}, counts {list(spec.counts)}, loads "
                        f"{''.join(spec.loads)} (F free / H half / E full at t=0).",
            spec=dict(name=spec.name, cls=spec.cls, workload=spec.workload,
                      counts=list(spec.counts), loads=list(spec.loads)),
            source_workload=str(WORKLOADS[spec.workload].relative_to(ROOT)),
            fedhpc_settings=dict(time_scale_seconds=slot_s,
                                 horizon_slots=src["horizon"], onprem_cost=0.0),
            instance_type_names=names,
        ),
        horizon=src["horizon"],
        instance_types=types,
        jobs=[dict(j) for j in src["jobs"]],
        running_jobs=running,
    )


def build(spec: InstanceSpec | str, prices: dict[str, dict] | None = None) -> Instance:
    """The built :class:`Instance` for a spec or an instance name."""
    if isinstance(spec, str):
        spec = InstanceSpec.parse(spec)
    return Instance.from_dict(generate(spec, prices))
