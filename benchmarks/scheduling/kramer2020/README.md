# Kramer et al. PrwC 2020

Problem: `P | r_j | sum w_j C_j`, nonpreemptive scheduling on identical
parallel machines with release dates. Original archive:
[`data/prwc_kramer2020/Instances_PrwC_KDFI2020.rar`](../../../data/prwc_kramer2020/Instances_PrwC_KDFI2020.rar).
The archive was already in the repository; it was not moved or replaced.
Upstream: <https://github.com/ahkramer/PrwC>; paper:
<https://arxiv.org/abs/2005.10097>.

Existing runners: `scripts/bench_kramer_fair.py` and
`scripts/kramer_lpfirst_pilot.py`. Earlier results: `results/kramer_fair/`.
The latter pilot used 12 `p_max_100`, 20-job instances, two LP backends,
HiGHS for any MIP stage, one thread, and a 30-second per-solve limit. Its
JSONL is in `results/kramer_pilot/`; it is not a general MIP comparison.

For the pilot, extract the archive to an external working directory, then run:

```sh
mkdir -p /tmp/fedhpc_kramer
unrar x -o+ data/prwc_kramer2020/Instances_PrwC_KDFI2020.rar /tmp/fedhpc_kramer/
.venv/bin/python scripts/kramer_lpfirst_pilot.py /tmp/fedhpc_kramer results/kramer_pilot/lpfirst.jsonl
```

The pilot skips already recorded instance/backend pairs. The returned `bound`
is the LP-derived bound even when `how` records a later MIP proof.
