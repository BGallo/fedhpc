"""Result-data model for generated instances: per-job output volume ι_j (GB)
and the cloud transfer parameters κ^IO (egress price) and τ^IO (transfer time).

The Slurm logs behind the workloads record no I/O, so ι_j comes from a model
plus a scenario factor, not from data. Only cloud placements move data: a job
run on-prem leaves its results on-prem (κ^IO = τ^IO = 0), a cloud job must
bring them home through the provider's internet egress.

Volume — class intercept, shared power law in size and time
-----------------------------------------------------------
    ι_j = r(policy) · C_REF · (c_j / C_REF)^β_p · h_j · 10^(σ·z_j)

``c_j`` cores, ``h_j`` runtime in hours, ``r`` GB per core-hour of the job's
output-retention policy (the median: the noise below lifts the mean ≈ 1.9×).
This is the class-intercept / shared-slope form
(ANCOVA) that generalised best out of sample on ALCF Theta Darshan logs
(test R² 0.52–0.66 vs −0.33 for size and time alone, −0.09 for a global
mean). Exponents: β_t = 1 (output records grow with simulated time; Theta
fit 1.05) and β_p = 0.54 (Theta fit within classes). σ = 0.5 decades is the
median within-class spread on Theta; z_j ~ N(0, 1), clipped to ±3, is drawn
once per (workload, job), so a job keeps its draw at every level (common
random numbers) and ι_j only grows from level L to H.

Policies, measured with OPM Flow 2026.04 + Darshan on open reservoir decks
(SPE, Norne, Drogon, UNISIM-I-D; ../overseer_diagnostics/egress):
  S  summary vectors only (UNSMRY/SMSPEC)                     ≈ 0.02 GB/core-h
  R  + grid, init and a final restart (UNISIM-I-D as shipped)  ≈ 0.13
  F  + a restart every report step (Norne, Drogon, SPE10)     ≈ 1.5

Classes (scripts/build_job_classes.py): an *ensemble member* shares its
(user, job-name stem) with ≥ 10 jobs (history matching / optimisation runs;
96–98 % of both workloads), anything else is a *single run*. The level
shifts each class up one policy:

  level  ensemble  single     reading
  L      S         R          results stay in the cloud, only curves return
  M      R         F          production decks as shipped
  H      F         F          full restart history brought home

The Azure-vs-AWS break-even for 64-core jobs (D128s_v6's dearer VM against
its cheaper egress) is 0.48 GB/core-h, inside this range.

Egress price κ^IO ($/GB) — first paid tier, internet egress, North America
(AWS price list API us-east-1; Azure Retail Prices API eastus, *routing
preference: internet*, i.e. ISP transit as AWS). Volume tiers and the free
100 GB/month are ignored.

Transfer time τ^IO (slots/GB) — the maximum achievable end-to-end rate is
the minimum over the subsystems a transfer crosses (Liu et al., "Explaining
wide area data transfer performance", HPDC'17, eq. 1):
    R_max ≤ min(DR_max source disk read, MM_max network, DW_max sink write)
  DR  the cloud VM's default block volume: EBS gp3 baseline 125 MiB/s
      (131 MB/s); Azure Premium SSD v2 baseline 125 MB/s — the throughput
      included in the storage price on each provider.
  MM  provider egress limits: AWS caps internet-gateway traffic at 5 Gbps
      (< 32 vCPU) or 50 % of the instance bandwidth (m6i 8–32xlarge:
      12.5–50 Gbps) and a single flow at 5 Gbps; Azure caps all egress at
      the VM bandwidth (Dsv5/Dsv6 16–54 Gbps). Skyplane (NSDI'23) measured
      the same caps. ≥ 5 Gbps = 625 MB/s everywhere here.
  DW  the on-prem parallel file system — not binding.
So the disk binds (≈ 1 Gbps) on both providers: 7.6 s/GB on AWS, 8.0 s/GB
on Azure. Azure is the cheaper and slightly slower egress — a cost/time
trade-off, not a dominance. τ^IO enters billed and occupied time (the VM
stays up while uploading) and so both objectives.

Horizon guard: a volume is capped so its cloud transfer still ends inside
the horizon when the job starts at its release slot (any cloud type is then
still feasible for it); the cap is reported as ``capped`` per instance.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data"

RATES = {"S": 0.02, "R": 0.13, "F": 1.5}               # GB per core-hour
IO_LEVELS = {"L": ("S", "R"), "M": ("R", "F"), "H": ("F", "F")}   # (ensemble, single)
BETA_P = 0.54
C_REF = 40
SIGMA = 0.5
Z_CLIP = 3.0

EGRESS_USD_PER_GB = {"aws": 0.090, "azure": 0.080}
TRANSFER_MB_PER_S = {"aws": 125 * 2 ** 20 / 1e6, "azure": 125.0}


def io_time(provider: str, slot_s: float) -> float:
    """τ^IO: slots per GB moved out of ``provider``."""
    return 1000.0 / TRANSFER_MB_PER_S[provider] / slot_s


@lru_cache(maxsize=None)
def ensemble_flags(workload: str) -> tuple[int, ...]:
    with open(DATA / f"job_classes_{workload}.json") as f:
        return tuple(json.load(f)["ensemble"])


def _z(workload: str, job_id: int) -> float:
    seed = int.from_bytes(hashlib.sha256(f"io|{workload}|{job_id}".encode()).digest()[:8], "big")
    return max(-Z_CLIP, min(Z_CLIP, random.Random(seed).gauss(0.0, 1.0)))


def volumes(workload: str, jobs: list[dict], level: str, slot_s: float,
            horizon: int, tau_max: float) -> tuple[list[float], int]:
    """ι_j (GB) for every job at ``level``, and how many were capped by the
    horizon guard (``tau_max``: the slowest cloud τ^IO of the instance)."""
    flags = ensemble_flags(workload)
    if len(flags) != len(jobs):
        raise ValueError(f"job_classes_{workload}.json has {len(flags)} jobs, workload {len(jobs)}")
    ens_policy, single_policy = IO_LEVELS[level]
    out, capped = [], 0
    for j in jobs:
        rate = RATES[ens_policy if flags[j["id"]] else single_policy]
        hours = j["exec_time"] * slot_s / 3600
        v = rate * C_REF * (j["cpu"] / C_REF) ** BETA_P * hours * 10 ** (SIGMA * _z(workload, j["id"]))
        # cloud occupation ceil(e + τ·ι) ≤ horizon − release ⟺ e + τ·ι ≤ that room
        room = horizon - max(0, math.ceil(j["arrival"])) - j["exec_time"]
        cap = max(0.0, room / tau_max * (1 - 1e-9))
        if v > cap:
            v, capped = cap, capped + 1
        out.append(v)
    return out, capped
