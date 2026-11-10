"""Run dispatch_validate.py arms one after another from a spec file of lines
``tag | solver | kw-json | extra args`` (``#`` comments), skipping finished
tags (the validator itself resumes per instance).

    uv run python scripts/run_arms.py SPEC --split tune [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--split", choices=("tune", "test"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    for line in Path(a.spec).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = [x.strip() for x in line.split("|")]
        tag, solver, kw = parts[0], parts[1], parts[2] or "{}"
        json.loads(kw)
        tag = f"{tag}_s{a.seed}"
        cmd = [sys.executable, str(ROOT / "scripts" / "dispatch_validate.py"), "--solver", solver,
               "--tag", tag, "--kw", kw, "--seed", str(a.seed), "--workers", str(a.workers)]
        if a.split:
            cmd += ["--split", a.split]
        if len(parts) > 3 and parts[3]:
            cmd += parts[3].split()
        print(f"=== {tag}", flush=True)
        subprocess.run(cmd, check=False)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
