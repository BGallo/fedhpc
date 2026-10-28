"""Write generated instances (fedhpc.instance_gen) to data/generated/.

Default set: class A's full factorial (2 workloads × 3² counts × 3² loads =
162) and class B's pairwise-covering set (an orthogonal array over its 10
factors plus the uniform corners, per workload = 70) — B's full factorial
(118 098 instances) is reachable by name through ``instance_gen.build``.

    uv run python scripts/build_generated_instances.py                # default set
    uv run python scripts/build_generated_instances.py --full B       # B's full factorial
    uv run python scripts/build_generated_instances.py --name B_pos_n100-200-300-400-500_lFHEFH

The files are deterministic functions of their names and of
data/cloud_prices.json; manifest.json records each file's sha256 so a
rebuild can be checked against it.
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
    ap.add_argument("--covering", action="append", default=[], choices=sorted(g.CLASSES),
                    help="write this class's pairwise-covering set")
    ap.add_argument("--name", action="append", default=[], help="write one instance by name")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    if not (args.full or args.covering or args.name):
        args.full, args.covering = ["A"], ["B"]

    specs: dict[str, g.InstanceSpec] = {}
    for c in args.full:
        specs.update((s.name, s) for s in g.iter_specs(c))
    for c in args.covering:
        specs.update((s.name, s) for s in g.covering_specs(c))
    for n in args.name:
        s = g.InstanceSpec.parse(n)
        specs[s.name] = s

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name, spec in sorted(specs.items()):
        raw = json.dumps(g.generate(spec), separators=(",", ":")).encode()
        (args.out / f"{name}.json").write_bytes(raw)
        manifest[name] = dict(cls=spec.cls, workload=spec.workload, counts=list(spec.counts),
                              loads="".join(spec.loads), sha256=hashlib.sha256(raw).hexdigest())
    manifest_path.write_text(json.dumps(dict(sorted(manifest.items())), indent=1) + "\n")
    print(f"wrote {len(specs)} instances to {args.out} ({len(manifest)} in manifest)")


if __name__ == "__main__":
    main()
