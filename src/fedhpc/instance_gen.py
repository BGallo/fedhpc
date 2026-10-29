"""Permutative instance generator: two machine catalogues × two real workloads ×
per-type machine counts × initial load × result-data level.

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

Factors
-------
count  ``N_m`` per on-prem type — any integer in [100, 500]; the default
       levels are 100/300/500.
load   at t = 0, one level for every on-prem type — ``F`` free (no running
       jobs), ``H`` half full (⌊N_m/2⌋ running jobs per type) or ``E``
       entirely full (N_m running jobs per type).
io     result-data level ``L`` / ``M`` / ``H`` (:mod:`fedhpc.io_model`): the
       output volume ι_j each job must bring home if it runs in the cloud,
       priced at the provider's internet egress (κ^IO) and moved at the
       cloud disk's baseline rate (τ^IO). On-prem types have κ^IO = τ^IO = 0.

Load and io apply to the whole instance rather than per type, which keeps the
space small enough to cover every io level (the real cluster's load is a
snapshot of one queue, not set per partition).

Running-job end times are drawn from the empirical end-time distribution of
the workload's own ``running_jobs`` (the real cluster state of that
snapshot), with an RNG seeded by (class, workload, type group, count) only —
not by the load — so the half-full set is the first half of the full one:
raising a type's load only ever removes capacity.

Pickable types only
-------------------
A type no job of the workload would ever be placed on is left out of that
workload's instances (:func:`onprem_types`, :func:`cloud_types`):

* an on-prem type no job fits (class B, ``fedhpc``: no job fits c16);
* a cloud size that fits no job, or that is strictly dearer, for every job it
  fits, than another size of the *same provider* (all types run at perf 1, so
  price decides): r6i.16xlarge and E64s_v5 (m32's equivalents) lose to
  m6i.16xlarge / D64s_v5 because no job needs more than 256 GB on 32 cores.
  Dominance is checked within a provider so each provider keeps a complete
  offer (D128s_v6, dearer than m6i.32xlarge, stays).

So the number of counts in a spec depends on the workload: class B
has 4 on-prem types for ``fedhpc`` (c32, c40, c64, m32) and 5 for ``pos``.

Equivalent instances
--------------------
On-prem types differ only in which jobs they can run (all are free, perf 1,
no deploy; the cloud types depend only on the io level). Types that fit
exactly the same jobs are *interchangeable*: swapping their counts relabels
the instance (their running jobs are drawn per group, so the swap is exact).
In class B, c32 and m32 fit the same jobs in both workloads.
:func:`canonical` maps a spec to its class representative (interchangeable
types sorted), and the enumerations below yield representatives only.

Benchmark sets: class A's full factorial (2 × 9 × 3 × 3 = 162) and, for
class B (1,944 distinct), the pairwise-covering set :func:`covering_specs` —
the on-prem types differ in shape, so which type has which count changes the
optimum and no further collapse is exact; the remaining B instances stay
reachable by name.

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

from . import io_model
from .data import Instance

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
PRICES = DATA / "cloud_prices.json"

WORKLOADS = {
    "fedhpc": DATA / "fedhpc_known_runtime_offline_10min.json",
    "pos": DATA / "pos_congestion_known_runtime_offline_10min.json",
}
LOADS = {"F": 0.0, "H": 0.5, "E": 1.0}
IO_LEVELS = tuple(io_model.IO_LEVELS)
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
    counts: tuple[int, ...]   # N_m per on-prem type, in onprem_types() order
    load: str                 # "F" / "H" / "E", every on-prem type
    io: str                   # "L" / "M" / "H" result-data level

    def __post_init__(self):
        if self.cls not in CLASSES:
            raise ValueError(f"unknown class {self.cls!r}")
        if self.workload not in WORKLOADS:
            raise ValueError(f"unknown workload {self.workload!r}")
        k = len(onprem_types(self.cls, self.workload))
        if len(self.counts) != k:
            raise ValueError(f"class {self.cls} / {self.workload} needs {k} counts")
        lo, hi = COUNT_RANGE
        if not all(isinstance(n, int) and lo <= n <= hi for n in self.counts):
            raise ValueError(f"counts must be integers in [{lo}, {hi}], got {self.counts}")
        if self.load not in LOADS:
            raise ValueError(f"load must be one of {''.join(LOADS)}, got {self.load!r}")
        if self.io not in IO_LEVELS:
            raise ValueError(f"io must be one of {''.join(IO_LEVELS)}, got {self.io!r}")

    @property
    def name(self) -> str:
        """e.g. ``A_pos_n100-500_lE_iM`` — round-trips through :meth:`parse`."""
        return (f"{self.cls}_{self.workload}_n{'-'.join(map(str, self.counts))}"
                f"_l{self.load}_i{self.io}")

    _NAME = re.compile(r"([A-Z])_([a-z]+)_n([\d-]+)_l([A-Z])_i([A-Z])")

    @classmethod
    def parse(cls, name: str) -> InstanceSpec:
        m = cls._NAME.fullmatch(name)
        if not m:
            raise ValueError(f"not an instance name: {name!r}")
        c, w, counts, load, io = m.groups()
        return cls(c, w, tuple(int(n) for n in counts.split("-")), load, io)

    def running_counts(self) -> tuple[int, ...]:
        return tuple(int(n * LOADS[self.load]) for n in self.counts)


# ─────────────────────────────────────────────────────────────────────────────
# Pickable types and equivalence
# ─────────────────────────────────────────────────────────────────────────────

def _fits(job: dict, cpu: float, mem: float, stor: float) -> bool:
    return job["cpu"] <= cpu and job["mem"] <= mem and job["stor"] <= stor


@lru_cache(maxsize=None)
def onprem_types(cls: str, workload: str) -> tuple[OnPremType, ...]:
    """``CLASSES[cls]`` without the types no job of the workload fits."""
    jobs = _workload(workload)["jobs"]
    return tuple(t for t in CLASSES[cls]
                 if any(_fits(j, t.cpu, t.mem, ONPREM_STOR) for j in jobs))


def cloud_types(cls: str, workload: str,
                prices: dict[str, dict] | None = None) -> tuple[tuple[str, str, str], ...]:
    """``(provider, sku, equivalent on-prem type)`` of every cloud equivalent
    in the class that some job of the workload would pick: it fits the job
    and no size of the same provider that fits it is strictly cheaper."""
    prices = load_prices() if prices is None else prices
    jobs = _workload(workload)["jobs"]
    offer = [(p, sku, t.name) for t in CLASSES[cls] for p, sku in t.clouds]

    def fits(j, sku):
        return _fits(j, prices[sku]["vcpu"] // 2, prices[sku]["mem_gib"], CLOUD_STOR)

    picked = set()
    for j in jobs:
        for provider in {p for p, _, _ in offer}:
            fit = [sku for p, sku, _ in offer if p == provider and fits(j, sku)]
            if fit:
                best = min(prices[sku]["usd_per_hour"] for sku in fit)
                picked.update(sku for sku in fit if prices[sku]["usd_per_hour"] == best)
    return tuple(c for c in offer if c[1] in picked)


@lru_cache(maxsize=None)
def type_groups(cls: str, workload: str) -> tuple[tuple[int, ...], ...]:
    """The partition of :func:`onprem_types` (by index) into types that fit
    exactly the same jobs of the workload (each group sorted, groups by first
    index)."""
    jobs = _workload(workload)["jobs"]
    by_fit: dict[frozenset, list[int]] = {}
    for i, t in enumerate(onprem_types(cls, workload)):
        fit = frozenset(j["id"] for j in jobs if _fits(j, t.cpu, t.mem, ONPREM_STOR))
        by_fit.setdefault(fit, []).append(i)
    return tuple(sorted(tuple(g) for g in by_fit.values()))


def _group_of(cls: str, workload: str, i: int) -> int:
    """First index of type ``i``'s group."""
    return next(g[0] for g in type_groups(cls, workload) if i in g)


def canonical(spec: InstanceSpec) -> InstanceSpec:
    """The representative of ``spec``'s equivalence class: each
    interchangeable group's counts sorted over the group's indices."""
    counts = list(spec.counts)
    for g in type_groups(spec.cls, spec.workload):
        for i, n in zip(g, sorted(counts[i] for i in g)):
            counts[i] = n
    return InstanceSpec(spec.cls, spec.workload, tuple(counts), spec.load, spec.io)


# ─────────────────────────────────────────────────────────────────────────────
# Enumeration
# ─────────────────────────────────────────────────────────────────────────────

def space_size(cls: str, *, workloads=tuple(WORKLOADS), count_levels=COUNT_LEVELS,
               load_levels=tuple(LOADS), io_levels=IO_LEVELS, distinct: bool = True) -> int:
    """Size of :func:`iter_specs`. With ``distinct`` (the default) one per
    equivalence class: a group of g interchangeable types contributes the
    C(L+g−1, g) multisets of its L count levels."""
    L = len(count_levels)
    total = 0
    for w in workloads:
        if distinct:
            n = math.prod(math.comb(L + len(g) - 1, len(g)) for g in type_groups(cls, w))
        else:
            n = L ** len(onprem_types(cls, w))
        total += n * len(load_levels) * len(io_levels)
    return total


def iter_specs(cls: str, *, workloads=tuple(WORKLOADS), count_levels=COUNT_LEVELS,
               load_levels=tuple(LOADS), io_levels=IO_LEVELS, distinct: bool = True):
    """The full factorial: every workload × count per type × load × io level
    — with ``distinct`` (the default) only the canonical representative of
    each equivalence class (see :func:`canonical`)."""
    for w in workloads:
        k = len(onprem_types(cls, w))
        for counts in itertools.product(count_levels, repeat=k):
            for load in load_levels:
                for io in io_levels:
                    spec = InstanceSpec(cls, w, counts, load, io)
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
                   load_levels=tuple(LOADS), io_levels=IO_LEVELS,
                   corners: bool = True) -> list[InstanceSpec]:
    """A pairwise-covering subset of :func:`iter_specs`: per workload, an
    orthogonal array over the k + 2 factors (k counts, load, io: every pair
    of factors meets every pair of levels), plus — with ``corners`` — the
    uniform configurations (all types at the same count, for every load and
    io level), each row replaced by its canonical representative and
    duplicates dropped. Coverage then holds up to equivalence: every array
    row's class is represented. k follows the workload's
    :func:`onprem_types`."""
    if len(count_levels) != 3 or len(load_levels) != 3 or len(io_levels) != 3:
        raise ValueError("covering_specs needs exactly 3 count, 3 load and 3 io levels")
    out: dict[str, InstanceSpec] = {}
    for w in workloads:
        k = len(onprem_types(cls, w))
        rows = _orthogonal_array(k + 2)
        if corners:
            rows += [(i,) * k + (j, l) for i in range(3) for j in range(3) for l in range(3)]
        for row in rows:
            spec = canonical(InstanceSpec(cls, w, tuple(count_levels[i] for i in row[:k]),
                                          load_levels[row[k]], io_levels[row[k + 1]]))
            out.setdefault(spec.name, spec)
    return list(out.values())


def sample_specs(cls: str, n: int, seed: int = 0, *, workloads=tuple(WORKLOADS)) -> list[InstanceSpec]:
    """``n`` specs with counts drawn uniformly from the whole range [100, 500]
    (not just the default levels), load and io level uniformly from their
    levels, mapped to their canonical representatives."""
    rng = random.Random(seed)
    lo, hi = COUNT_RANGE
    out = []
    for _ in range(n):
        w = rng.choice(workloads)
        k = len(onprem_types(cls, w))
        out.append(canonical(InstanceSpec(cls, w, tuple(rng.randint(lo, hi) for _ in range(k)),
                                          rng.choice(tuple(LOADS)), rng.choice(IO_LEVELS))))
    return out


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
    onprem = onprem_types(spec.cls, spec.workload)

    types, names = [], {}
    for i, (t, n) in enumerate(zip(onprem, spec.counts)):
        types.append(dict(id=i, kind="on-prem", perf=1.0, io_time=0.0, deploy=0.0,
                          cost_io=0.0, cost_vm=0.0, cost_stor=0.0,
                          cpu=t.cpu, mem=t.mem, stor=ONPREM_STOR, capacity=n))
        names[str(i)] = dict(name=f"onprem_{t.name}", count=n, load=spec.load)
    clouds = cloud_types(spec.cls, spec.workload, prices)
    for provider, sku, equivalent in clouds:
        p = prices[sku]
        mid = len(types)
        cost_vm = p["usd_per_hour"] * slot_s / 3600
        types.append(dict(id=mid, kind="cloud", perf=1.0,
                          io_time=io_model.io_time(provider, slot_s), deploy=0.0,
                          cost_io=io_model.EGRESS_USD_PER_GB[provider], cost_vm=cost_vm,
                          cost_stor=0.0, cpu=p["vcpu"] // 2, mem=p["mem_gib"],
                          stor=CLOUD_STOR, capacity=None))
        names[str(mid)] = dict(name=f"{provider}_{sku}", provider=provider,
                               equivalent_of=f"onprem_{equivalent}",
                               usd_per_hour=p["usd_per_hour"], vcpu=p["vcpu"])

    tau_max = max(io_model.io_time(provider, slot_s) for provider, _, _ in clouds)
    io_volume, capped = io_model.volumes(spec.workload, src["jobs"], spec.io, slot_s,
                                         src["horizon"], tau_max)
    jobs = [dict(j, io_volume=v) for j, v in zip(src["jobs"], io_volume)]

    running = [dict(id=1_000_000 * (i + 1) + k, type_id=i, end=end)
               for i in range(len(onprem))
               for k, end in enumerate(running_ends(spec, i))]

    return dict(
        _info=dict(
            description=f"Generated by fedhpc.instance_gen: class {spec.cls}, workload "
                        f"{spec.workload}, counts {list(spec.counts)}, load {spec.load} "
                        f"(F free / H half / E full at t=0), io level {spec.io} "
                        f"(fedhpc.io_model; io_volume in GB).",
            spec=dict(name=spec.name, cls=spec.cls, workload=spec.workload,
                      counts=list(spec.counts), load=spec.load, io=spec.io),
            io=dict(level=spec.io, total_gb=sum(io_volume), capped=capped),
            source_workload=str(WORKLOADS[spec.workload].relative_to(ROOT)),
            fedhpc_settings=dict(time_scale_seconds=slot_s,
                                 horizon_slots=src["horizon"], onprem_cost=0.0),
            instance_type_names=names,
        ),
        horizon=src["horizon"],
        instance_types=types,
        jobs=jobs,
        running_jobs=running,
    )


def build(spec: InstanceSpec | str, prices: dict[str, dict] | None = None) -> Instance:
    """The built :class:`Instance` for a spec or an instance name."""
    if isinstance(spec, str):
        spec = InstanceSpec.parse(spec)
    return Instance.from_dict(generate(spec, prices))
