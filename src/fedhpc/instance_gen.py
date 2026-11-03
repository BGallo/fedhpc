"""Permutative instance generator: two machine catalogues × two real workloads ×
node count × initial load × result-data level.

Every instance is named by its :class:`InstanceSpec`, and the name alone
determines the instance (``generate(InstanceSpec.parse(name))`` is
reproducible bit for bit), so the full factorial space can be enumerated
lazily and any point of it rebuilt on demand without storing it.

Classes
-------
``A`` — two on-prem types modelled on the real csr1 (40 cores / 300 GB) and
        csr3 (64 cores / 480 GB) nodes, each with one AWS equivalent.
``B`` — four on-prem types (16c/128, 32c/256, 40c/300, 64c/480), each with an
        AWS *and* an Azure equivalent.

A cloud equivalent is the smallest instance of the same Intel generation
(Ice Lake: AWS m6i, Azure Dsv5) whose physical cores (vCPU / 2, SMT) and
memory cover the on-prem node. Azure has no 128-vCPU Dsv5, so the 64-core
node maps to Dsv6. Prices are on-demand Linux list prices from
``data/cloud_prices.json`` (``scripts/fetch_cloud_prices.py``), converted to
cost per slot. Same-shape Ice Lake sizes are priced identically by both
providers; only D128s_v6 differs from m6i.32xlarge.

A memory-heavy 32c/512 type was dropped from B: no job needs more than
256 GB on 32 cores, so it fit exactly the same jobs as c32 and only split
c32's capacity in two.

Factors — one level each, for the whole instance
-------------------------------------------------
count  ``N`` nodes of *every* on-prem type — any integer in [100, 500]; the
       default levels are 100/300/500.
load   at t = 0 — ``F`` free (no running jobs), ``H`` half full (⌊N/2⌋
       running jobs per type) or ``E`` entirely full (N per type).
io     result-data level ``L`` / ``M`` / ``H`` (:mod:`fedhpc.io_model`): the
       output volume ι_j each job must bring home if it runs in the cloud,
       priced at the provider's internet egress (κ^IO) and moved at the
       cloud disk's baseline rate (τ^IO). On-prem types have κ^IO = τ^IO = 0.

Per-type counts made the space grow as 3^k in the number of on-prem types k
while the jobs only tell apart a few nested fit classes (every job fits c64,
c40 fits all but a few dozen); one count, load and io level per instance
gives 3 × 3 × 3 = 27 instances per workload, small enough to run both
classes as full factorials (54 each).

Running-job end times are drawn from the empirical end-time distribution of
the workload's own ``running_jobs`` (the real cluster state of that
snapshot), with an RNG seeded by (class, workload, type, count) only — not by
the load — so the half-full set is the first half of the full one: raising
the load only ever removes capacity.

Pickable types only
-------------------
A type no job of the workload would ever be placed on is left out of that
workload's instances (:func:`onprem_types`, :func:`cloud_types`):

* an on-prem type no job fits (class B, ``fedhpc``: no job fits c16);
* a cloud size that fits no job, or that is strictly dearer, for every job it
  fits, than another size of the *same provider* (all types run at perf 1, so
  price decides). Dominance is checked within a provider so each provider
  keeps a complete offer (D128s_v6, dearer than m6i.32xlarge, stays).

So class B has 3 on-prem types for ``fedhpc`` (c32, c40, c64) and 4 for
``pos``.

Workloads
---------
``fedhpc`` — data/fedhpc_known_runtime_offline_10min.json (964 jobs, H=289)
``pos``    — data/pos_congestion_known_runtime_offline_10min.json (3340 jobs, H=288)
Jobs, horizon and slot length are copied unchanged.
"""
from __future__ import annotations

import hashlib
import json
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
    ),
}

CLOUD_SKUS: list[tuple[str, str]] = sorted(
    {c for types in CLASSES.values() for t in types for c in t.clouds})


@dataclass(frozen=True)
class InstanceSpec:
    cls: str
    workload: str
    count: int                # N nodes of every on-prem type
    load: str                 # "F" / "H" / "E"
    io: str                   # "L" / "M" / "H" result-data level

    def __post_init__(self):
        if self.cls not in CLASSES:
            raise ValueError(f"unknown class {self.cls!r}")
        if self.workload not in WORKLOADS:
            raise ValueError(f"unknown workload {self.workload!r}")
        lo, hi = COUNT_RANGE
        if not (isinstance(self.count, int) and lo <= self.count <= hi):
            raise ValueError(f"count must be an integer in [{lo}, {hi}], got {self.count!r}")
        if self.load not in LOADS:
            raise ValueError(f"load must be one of {''.join(LOADS)}, got {self.load!r}")
        if self.io not in IO_LEVELS:
            raise ValueError(f"io must be one of {''.join(IO_LEVELS)}, got {self.io!r}")

    @property
    def name(self) -> str:
        """e.g. ``A_pos_n300_lE_iM`` — round-trips through :meth:`parse`."""
        return f"{self.cls}_{self.workload}_n{self.count}_l{self.load}_i{self.io}"

    _NAME = re.compile(r"([A-Z])_([a-z]+)_n(\d+)_l([A-Z])_i([A-Z])")

    @classmethod
    def parse(cls, name: str) -> InstanceSpec:
        m = cls._NAME.fullmatch(name)
        if not m:
            raise ValueError(f"not an instance name: {name!r}")
        c, w, count, load, io = m.groups()
        return cls(c, w, int(count), load, io)

    def running_count(self) -> int:
        """Running jobs at t = 0 on each on-prem type."""
        return int(self.count * LOADS[self.load])


# ─────────────────────────────────────────────────────────────────────────────
# Pickable types
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


# ─────────────────────────────────────────────────────────────────────────────
# Enumeration
# ─────────────────────────────────────────────────────────────────────────────

def space_size(cls: str, *, workloads=tuple(WORKLOADS), count_levels=COUNT_LEVELS,
               load_levels=tuple(LOADS), io_levels=IO_LEVELS) -> int:
    """Size of :func:`iter_specs`."""
    return len(workloads) * len(count_levels) * len(load_levels) * len(io_levels)


def iter_specs(cls: str, *, workloads=tuple(WORKLOADS), count_levels=COUNT_LEVELS,
               load_levels=tuple(LOADS), io_levels=IO_LEVELS):
    """The full factorial: every workload × count × load × io level."""
    for w in workloads:
        for n in count_levels:
            for load in load_levels:
                for io in io_levels:
                    yield InstanceSpec(cls, w, n, load, io)


def sample_specs(cls: str, n: int, seed: int = 0, *, workloads=tuple(WORKLOADS)) -> list[InstanceSpec]:
    """``n`` specs with the count drawn uniformly from the whole range
    [100, 500] (not just the default levels), load and io level uniformly
    from their levels."""
    rng = random.Random(seed)
    lo, hi = COUNT_RANGE
    return [InstanceSpec(cls, rng.choice(workloads), rng.randint(lo, hi),
                         rng.choice(tuple(LOADS)), rng.choice(IO_LEVELS))
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
    ``running_count()`` of N draws from the workload's empirical running-job
    end times (nested across loads)."""
    pool = sorted(r["end"] for r in _workload(spec.workload)["running_jobs"])
    name = onprem_types(spec.cls, spec.workload)[type_index].name
    rng = random.Random(_seed(spec.cls, spec.workload, name, spec.count))
    draws = [rng.choice(pool) for _ in range(spec.count)]
    return draws[:spec.running_count()]


def generate(spec: InstanceSpec, prices: dict[str, dict] | None = None) -> dict:
    """The instance as a JSON-ready dict (the schema of data/*.json)."""
    prices = load_prices() if prices is None else prices
    src = _workload(spec.workload)
    slot_s = float(src["_info"]["fedhpc_settings"]["time_scale_seconds"])
    onprem = onprem_types(spec.cls, spec.workload)

    types, names = [], {}
    for i, t in enumerate(onprem):
        types.append(dict(id=i, kind="on-prem", perf=1.0, io_time=0.0, deploy=0.0,
                          cost_io=0.0, cost_vm=0.0, cost_stor=0.0,
                          cpu=t.cpu, mem=t.mem, stor=ONPREM_STOR, capacity=spec.count))
        names[str(i)] = dict(name=f"onprem_{t.name}", count=spec.count, load=spec.load)
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
                        f"{spec.workload}, {spec.count} nodes per on-prem type, load {spec.load} "
                        f"(F free / H half / E full at t=0), io level {spec.io} "
                        f"(fedhpc.io_model; io_volume in GB).",
            spec=dict(name=spec.name, cls=spec.cls, workload=spec.workload,
                      count=spec.count, load=spec.load, io=spec.io),
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
