"""Two-centre federation instances: site-aware data transfer (Level 1 of
docs/geo_distributed_transfer.md), built from the class-A catalogue of
:mod:`fedhpc.instance_gen`.

The review asks for a controlled multi-centre scenario (critique §2.2: separate
on-prem capacities, distinct job compatibility sets, site-dependent transfer
overheads) and a physically grounded I/O model (§3.4: input and output split,
an effective bandwidth B_eff = min(read, network, write)). This module builds
that scenario from the same real workloads, so that every effect it adds can
be switched off and compared against a paired control.

Federation
----------
Two HPC centres, ``S1`` and ``S2``, each with ``N`` nodes of every class-A
on-prem type (c40, c64), plus the AWS equivalents at site ``aws``. Total local
capacity is 2·N per type, so ``n50``/``n150`` match class A's ``n100``/``n300``.
Each centre's running jobs at t = 0 are drawn as in instance_gen (load F/H/E,
end times from the workload's own snapshot), seeded per (site, type, N).

Jobs
----
* **Origin.** A job's inputs live at, and its results return to, one centre:
  ``S1`` for ``ORIGIN_SHARE`` (65 %) of the jobs, ``S2`` for the rest, drawn
  per job id. The skew makes S1 the busier centre, so federating has something
  to balance. (The traces carry no user, so origins cannot follow users.)
* **Output** ι^out_j: :func:`fedhpc.io_model.volumes` at level L/M/H, unchanged.
* **Input** ι^in_j = IN_GB_REF · (c_j / 40)^0.54 · 10^(0.5·z'_j) GB: the
  realisation-specific part of a reservoir deck (grid properties, schedule),
  with the same size exponent and spread as the output model and its own
  common-random-number draw z'_j. IN_GB_REF = 2 GB is an assumption, not data.
* **Compatibility.** Single runs (non-ensemble jobs, ≈ 3 %) depend on a
  site-installed software stack (e.g. a licensed simulator), so they may run
  only at their origin centre — no remote centre, no cloud.

Links — slots per GB from B_eff = min(source read, network, sink write)
-----------------------------------------------------------------------
  S1 ↔ S2    min(PFS 5 GB/s, WAN ``wan`` Gbps per flow, PFS)  — research network, $0
  Sk → aws   min(PFS, 5 Gbps internet, gp3 write 131 MB/s)    — ingress, $0
  aws → Sk   min(gp3 read 131 MB/s, 5 Gbps egress, PFS)        — egress, $0.09/GB
``wan = 0`` is the paired control: the centres are co-located (transfer between
them is instant and free), so the instance differs from the federated one in
transfer only.

Modes
-----
``fed`` — MOSPACE schedules every job on any centre (and the cloud).
``iso`` — isolated centres: each job only at its origin centre (and the cloud),
          i.e. two independent schedulers sharing one cloud account.
``C`` / ``N`` — with or without the cloud types.

Horizon guard: a job's two volumes are scaled down together so that its
slowest transfer option still finishes inside the horizon when started at its
release; the number of scaled jobs is reported as ``capped``.

Names: ``G_<workload>_n<N>_l<load>_i<io>_w<Gbps>_<mode><cloud>``, e.g.
``G_pos_n150_lH_iM_w10_fedC``; ``generate(GeoSpec.parse(name))`` is
reproducible bit for bit.
"""
from __future__ import annotations

import hashlib
import math
import random
import re
from dataclasses import dataclass

from . import instance_gen as ig
from . import io_model
from .data import Instance

SITES = ("S1", "S2")
CLOUD_SITE = "aws"
CLASS = "A"
ORIGIN_SHARE = 0.65          # share of jobs whose data live at S1
IN_GB_REF = 2.0              # median input GB of a 40-core job
PFS_MB_S = 5000.0            # parallel file system read / write, per transfer
INTERNET_MB_S = 625.0        # 5 Gbps provider egress / ingress cap (io_model)
MODES = ("fed", "iso")
CLOUD = ("C", "N")
COUNT_RANGE = (25, 250)


@dataclass(frozen=True)
class GeoSpec:
    workload: str
    count: int                # N nodes of every on-prem type at each centre
    load: str                 # F / H / E
    io: str                   # L / M / H output level
    wan: int                  # inter-centre bandwidth per flow, Gbps; 0 = co-located
    mode: str                 # fed / iso
    cloud: str                # C / N

    def __post_init__(self):
        if self.workload not in ig.WORKLOADS:
            raise ValueError(f"unknown workload {self.workload!r}")
        lo, hi = COUNT_RANGE
        if not (isinstance(self.count, int) and lo <= self.count <= hi):
            raise ValueError(f"count must be an integer in [{lo}, {hi}], got {self.count!r}")
        if self.load not in ig.LOADS:
            raise ValueError(f"load must be one of {''.join(ig.LOADS)}, got {self.load!r}")
        if self.io not in ig.IO_LEVELS:
            raise ValueError(f"io must be one of {''.join(ig.IO_LEVELS)}, got {self.io!r}")
        if not (isinstance(self.wan, int) and self.wan >= 0):
            raise ValueError(f"wan must be a non-negative integer (Gbps), got {self.wan!r}")
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.cloud not in CLOUD:
            raise ValueError(f"cloud must be C or N, got {self.cloud!r}")

    @property
    def name(self) -> str:
        return (f"G_{self.workload}_n{self.count}_l{self.load}_i{self.io}"
                f"_w{self.wan}_{self.mode}{self.cloud}")

    _NAME = re.compile(r"G_([a-z]+)_n(\d+)_l([A-Z])_i([A-Z])_w(\d+)_(fed|iso)([CN])")

    @classmethod
    def parse(cls, name: str) -> GeoSpec:
        m = cls._NAME.fullmatch(name)
        if not m:
            raise ValueError(f"not a geo instance name: {name!r}")
        w, n, load, io, wan, mode, cloud = m.groups()
        return cls(w, int(n), load, io, int(wan), mode, cloud)

    def running_count(self) -> int:
        return int(self.count * ig.LOADS[self.load])


def _u(*parts) -> float:
    """A uniform [0, 1) draw fixed by ``parts`` (common random numbers)."""
    return random.Random(ig._seed(*parts)).random()


def _z(*parts) -> float:
    rng = random.Random(ig._seed(*parts))
    return max(-io_model.Z_CLIP, min(io_model.Z_CLIP, rng.gauss(0.0, 1.0)))


def origin(workload: str, job_id: int) -> str:
    return SITES[0] if _u("origin", workload, job_id) < ORIGIN_SHARE else SITES[1]


def input_gb(workload: str, job: dict) -> float:
    return (IN_GB_REF * (job["cpu"] / io_model.C_REF) ** io_model.BETA_P
            * 10 ** (io_model.SIGMA * _z("io_in", workload, job["id"])))


def slots_per_gb(mb_s: float, slot_s: float) -> float:
    return 1000.0 / mb_s / slot_s


def links(spec: GeoSpec, slot_s: float) -> list[dict]:
    """The directed links (B_eff = min over read, network, write)."""
    disk = io_model.TRANSFER_MB_PER_S["aws"]
    egress = io_model.EGRESS_USD_PER_GB["aws"]
    out = []
    a, b = SITES
    wan_t = 0.0 if spec.wan == 0 else slots_per_gb(min(PFS_MB_S, spec.wan * 125.0, PFS_MB_S), slot_s)
    out += [dict(src=a, dst=b, time=wan_t, cost=0.0), dict(src=b, dst=a, time=wan_t, cost=0.0)]
    for s in SITES:
        out.append(dict(src=s, dst=CLOUD_SITE, cost=0.0,
                        time=slots_per_gb(min(PFS_MB_S, INTERNET_MB_S, disk), slot_s)))
        out.append(dict(src=CLOUD_SITE, dst=s, cost=egress,
                        time=slots_per_gb(min(disk, INTERNET_MB_S, PFS_MB_S), slot_s)))
    return out


def generate(spec: GeoSpec, prices: dict[str, dict] | None = None) -> dict:
    """The instance as a JSON-ready dict (the schema of data/*.json, plus
    ``links`` and the site fields)."""
    prices = ig.load_prices() if prices is None else prices
    src = ig._workload(spec.workload)
    slot_s = float(src["_info"]["fedhpc_settings"]["time_scale_seconds"])
    H = src["horizon"]
    onprem = ig.onprem_types(CLASS, spec.workload)
    pool = sorted(r["end"] for r in src["running_jobs"])

    types, names, running = [], {}, []
    for s in SITES:
        for t in onprem:
            mid = len(types)
            types.append(dict(id=mid, kind="on-prem", perf=1.0, io_time=0.0, deploy=0.0,
                              cost_io=0.0, cost_vm=0.0, cost_stor=0.0, cpu=t.cpu, mem=t.mem,
                              stor=ig.ONPREM_STOR, capacity=spec.count, site=s))
            names[str(mid)] = dict(name=f"{s}_{t.name}", site=s, count=spec.count, load=spec.load)
            rng = random.Random(ig._seed("G", spec.workload, s, t.name, spec.count))
            ends = [rng.choice(pool) for _ in range(spec.count)][:spec.running_count()]
            running += [dict(id=1_000_000 * (mid + 1) + k, type_id=mid, end=e)
                        for k, e in enumerate(ends)]
    if spec.cloud == "C":
        for provider, sku, equivalent in ig.cloud_types(CLASS, spec.workload, prices):
            p = prices[sku]
            mid = len(types)
            types.append(dict(id=mid, kind="cloud", perf=1.0, io_time=0.0, deploy=0.0,
                              cost_io=0.0, cost_vm=p["usd_per_hour"] * slot_s / 3600,
                              cost_stor=0.0, cpu=p["vcpu"] // 2, mem=p["mem_gib"],
                              stor=ig.CLOUD_STOR, capacity=None, site=CLOUD_SITE))
            names[str(mid)] = dict(name=f"{provider}_{sku}", site=CLOUD_SITE, provider=provider,
                                   equivalent_of=f"onprem_{equivalent}",
                                   usd_per_hour=p["usd_per_hour"], vcpu=p["vcpu"])

    lk = links(spec, slot_s)
    t_of = {(ln["src"], ln["dst"]): ln["time"] for ln in lk}
    t_of.update({(s, s): 0.0 for s in (*SITES, CLOUD_SITE)})
    tau_out_cloud = t_of[CLOUD_SITE, SITES[0]]
    out_gb, _ = io_model.volumes(spec.workload, src["jobs"], spec.io, slot_s, H, tau_out_cloud)
    flags = io_model.ensemble_flags(spec.workload)

    jobs, capped = [], 0
    cloud_sites = [CLOUD_SITE] if spec.cloud == "C" else []
    for j, v_out in zip(src["jobs"], out_gb):
        o = origin(spec.workload, j["id"])
        if not flags[j["id"]]:
            allowed = [o]                                  # site-licensed single run
        elif spec.mode == "iso":
            allowed = [o, *cloud_sites]
        else:
            allowed = None
        v_in = input_gb(spec.workload, j)
        reach = allowed or [*SITES, *cloud_sites]
        worst = max(t_of[o, s] * v_in + t_of[s, o] * v_out for s in reach)
        room = H - max(0, math.ceil(j["arrival"])) - j["exec_time"]
        if worst > room:                                   # horizon guard
            k = max(0.0, room / worst * (1 - 1e-9))
            v_in, v_out, capped = v_in * k, v_out * k, capped + 1
        jobs.append(dict(j, io_volume=v_out, io_in=v_in, origin=o, sites=allowed))

    return dict(
        _info=dict(
            description=f"Generated by fedhpc.geo_gen: two centres {SITES} with {spec.count} "
                        f"nodes per class-{CLASS} type each, load {spec.load}, output level "
                        f"{spec.io}, inter-centre WAN {spec.wan or 'co-located'} Gbps, mode "
                        f"{spec.mode}, cloud {'on' if spec.cloud == 'C' else 'off'}.",
            spec=dict(name=spec.name, workload=spec.workload, count=spec.count, load=spec.load,
                      io=spec.io, wan=spec.wan, mode=spec.mode, cloud=spec.cloud),
            io=dict(level=spec.io, out_gb=sum(j["io_volume"] for j in jobs),
                    in_gb=sum(j["io_in"] for j in jobs), capped=capped,
                    origin_S1=sum(j["origin"] == SITES[0] for j in jobs),
                    pinned=sum(j["sites"] == [j["origin"]] for j in jobs)),
            source_workload=str(ig.WORKLOADS[spec.workload].relative_to(ig.ROOT)),
            fedhpc_settings=dict(time_scale_seconds=slot_s, horizon_slots=H, onprem_cost=0.0),
            instance_type_names=names,
        ),
        horizon=H,
        instance_types=types,
        jobs=jobs,
        running_jobs=running,
        links=lk,
    )


def build(spec: GeoSpec | str, prices: dict[str, dict] | None = None) -> Instance:
    if isinstance(spec, str):
        spec = GeoSpec.parse(spec)
    return Instance.from_dict(generate(spec, prices))
