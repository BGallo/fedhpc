# MIPLIB 2017

Source: <https://miplib.zib.de/download>. Downloaded 2026-11-26.

- `benchmark-v2.zip` and extracted `benchmark-v2/`: official 240-instance
  benchmark, not the full MIPLIB collection.
- `benchmark-v2.test`: official 240-name roster.
- `easy-v19.test`, `hard-v31.test`, `open-v30.test`: official status lists
  dated 2026-09-24. They refer to the **whole collection**, not just the
  240-instance benchmark. Status may have changed since this snapshot.
- `miplib2017-v37.solu`: upstream objective/status reference, not a full
  set of independently verifiable primal solutions.
- `open-selected/`: four collection instances listed as open in that snapshot:
  `fhnw-schedule-paira100`, `fhnw-binschedule0`, `bppc6-06`, and
  `cvrpa-n64k9vrpi`. They are not represented as a curated or representative
  sample of all open instances.

The complete collection archive is 3.5 GB upstream and is **not** downloaded.
MPS files are compressed individually; preserve the originals when extracting.
Use the official roster and test scripts for full benchmark claims.

URLs: `https://miplib.zib.de/downloads/benchmark.zip`, the corresponding
versioned metadata filenames under `/downloads/`, and
`https://miplib.zib.de/WebData/instances/<instance>.mps.gz`.
