"""Write generated instances (fedhpc.instance_gen) to data/generated/.

Default set: both classes' full factorials (2 workloads × 3 node counts ×
3 loads × 3 io levels = 54 instances each, 108 in all).

    uv run python scripts/build_generated_instances.py                # default set
    uv run python scripts/build_generated_instances.py --full B       # one class
    uv run python scripts/build_generated_instances.py --name B_pos_n250_lH_iM

The files are deterministic functions of their names and of
data/cloud_prices.json and data/job_classes_<workload>.json; manifest.json
records each file's sha256 so a rebuild can be checked against it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fedhpc import instance_gen as g  # noqa: E402

OUT = ROOT / "data" / "generated"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="append", default=[], choices=sorted(g.CLASSES),
                    help="write this class's full factorial")
    ap.add_argument("--name", action="append", default=[], help="write one instance by name")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    if not (args.full or args.name):
        args.full = sorted(g.CLASSES)

    specs: dict[str, g.InstanceSpec] = {}
    for c in args.full:
        specs.update((s.name, s) for s in g.iter_specs(c))
    for n in args.name:
        s = g.InstanceSpec.parse(n)
        specs[s.name] = s

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name, spec in sorted(specs.items()):
        raw = json.dumps(g.generate(spec), separators=(",", ":")).encode()
        (args.out / f"{name}.json").write_bytes(raw)
        manifest[name] = dict(cls=spec.cls, workload=spec.workload, count=spec.count,
                              load=spec.load, io=spec.io, sha256=hashlib.sha256(raw).hexdigest())
    manifest_path.write_text(json.dumps(dict(sorted(manifest.items())), indent=1) + "\n")
    print(f"wrote {len(specs)} instances to {args.out} ({len(manifest)} in manifest)")


if __name__ == "__main__":
    main()
