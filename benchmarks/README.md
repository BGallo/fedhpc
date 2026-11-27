# Optimization benchmark catalogue

This directory organizes external benchmark data by problem family. Existing
`data/`, `Rm_(TWCT+TMUC)/`, scripts and results remain at their original paths.
Downloaded payloads, rosters, reference metadata and checksums are kept here
for reproducibility. See upstream terms
before redistributing any third-party data.

| Family | Collection | Local location | Purpose |
|---|---|---|---|
| Scheduling, identical parallel machines with release dates | Kramer et al. PrwC 2020 | [`kramer2020`](scheduling/kramer2020/README.md); original archive in [`data/`](../data/prwc_kramer2020/) | Primary solver development and comparison |
| Scheduling, unrelated machines with usage costs | Chen et al. Rm | [`rm_twct_tmuc`](scheduling/rm_twct_tmuc/README.md); original files in [`Rm_(TWCT+TMUC)/`](../Rm_%28TWCT%2BTMUC%29/) | Existing exact and heuristic engines |
| General mixed-integer programming | MIPLIB 2017 | [`miplib2017`](mip/miplib2017/README.md) | 240-instance general benchmark and selected open challenges |
| Capacitated vehicle routing | CVRPLIB A and X | [`cvrplib`](routing/cvrplib/README.md) | Small validation case and 100 X instances |
| Traveling salesman | DIMACS TSPLIB and National TSP | [`tsp`](tsp/README.md) | Longer-term branch-and-cut challenges |

The [`SHA256SUMS`](SHA256SUMS) manifest covers the downloaded archives,
metadata, and individual open or validation instances. Run `sha256sum -c
benchmarks/SHA256SUMS` from the repository root to verify the local files.
Hashes identify this snapshot, not an upstream authenticity guarantee.

Experiment results belong in `results/` and experiment runners in `scripts/`;
do not mix solver outputs with source instances. Existing historical results
and the user data partition have not been moved.
