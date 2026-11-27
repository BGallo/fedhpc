"""Profiling harness for fedhpc.mip and fedlp.

Subcommands
-----------
build    standalone _fedlp / _fedmip builds in results/profile/build/<variant>:
         ``timers``  -DFEDLP_PROFILE (fedlp's phase timers t_* and row sizes)
         ``perf``    -O3 -g -fno-omit-frame-pointer (symbols and frame-pointer
                     stacks for perf and py-spy --native)
         ``native``  -O3 -march=native (what the CPU could do with AVX2/AVX-512)
         Selected at run time with --build <variant> (sets FEDLP_DIR; the MIP
         layer imports _fedlp from there, _fedmip from the package).
mip      one MIP solve: per-phase wall time and pivots (MipSolver's
         stats["time"]), with --build timers also fedlp's per-phase timers and
         per-pivot averages summed over every LP solve.
lp       the root LP alone under fedlp option sets (k=v,k=v ...), with timers.
perf     perf record of ``mip`` (Python 3.12+ -X perf trampolines, so Python
         frames appear), then: time by shared object (the Python interpreter,
         _fedlp, _fedmip, NumPy/SciPy, libc, kernel) - the Python / C++ split -
         and the top symbols.
pyspy    py-spy record --native of ``mip``: a speedscope profile (open it at
         https://www.speedscope.app) and the split by leaf frame (Python
         code vs native code) from the collapsed stacks.

Examples
--------
    uv run python scripts/profile.py build perf
    uv run python scripts/profile.py mip results/mip_bench/mps/AF_N50_M2_R4_1.mps.gz -t 60 --build timers
    uv run python scripts/profile.py perf results/mip_bench/mps/TI_N50_M2_R0_1.mps.gz -t 60 --build perf
    uv run python scripts/profile.py lp results/mip_bench/mps/AF_N50_M5_R0_1.mps.gz "" "dual=0,partial_min_cols=2000" --build timers

Runs are pinned to one core (--cpu, default 0), with fedlp single-threaded,
as in scripts/mip_bench.py. Outputs go to results/profile/<label>/.
"""
from __future__ import annotations

import argparse
import ast
import collections
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "profile"
EXT = ROOT / "src" / "fedhpc" / "_ext"
VARIANTS = {
    "timers": ["-O3", "-DNDEBUG", "-DFEDLP_PROFILE"],
    "perf": ["-O3", "-DNDEBUG", "-g", "-fno-omit-frame-pointer"],
    "native": ["-O3", "-DNDEBUG", "-march=native"],
}


# ------------------------------------------------------------------- build
def build(variant: str, quiet: bool = False) -> Path:
    import pybind11
    out = OUT / "build" / variant
    out.mkdir(parents=True, exist_ok=True)
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    inc = ["-I" + sysconfig.get_paths()["include"], "-I" + pybind11.get_include()]
    for name, src in (("_fedlp", EXT / "lp" / "lp_module.cpp"), ("_fedmip", EXT / "mip" / "mip_module.cpp")):
        target = out / f"{name}{suffix}"
        deps = list(src.parent.glob("*"))
        if target.exists() and target.stat().st_mtime > max(d.stat().st_mtime for d in deps):
            continue
        cmd = ["g++", "-std=c++17", "-shared", "-fPIC", *VARIANTS[variant], *inc, str(src), "-o", str(target),
               "-lpthread"]
        if not quiet:
            print("build", variant, name, flush=True)
        subprocess.run(cmd, check=True)
    return out


def _use_build(variant: str | None) -> None:
    if variant:
        os.environ["FEDLP_DIR"] = str(build(variant, quiet=True))


# --------------------------------------------------------------------- mip
def _parse_opts(kvs):
    kw = {}
    for kv in kvs:
        k, v = kv.split("=", 1)
        try:
            kw[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            kw[k] = v
    return kw


def run_mip(path: str, tl: float, opts: dict, label: str | None) -> dict:
    from fedhpc.mip import read_mps, solve
    t0 = time.perf_counter()
    P = read_mps(path)
    t_read = time.perf_counter() - t0
    r = solve(P, time_limit=tl, lp_options={"threads": 1}, verbose=False, **opts)
    st = r.stats
    rep = {"instance": Path(path).name, "status": r.status, "objective": r.objective, "bound": r.bound,
           "nodes": r.nodes, "seconds": r.seconds, "read_seconds": t_read, "options": opts,
           "time": st.get("time", {}), "pivots": st.get("pivots", {}), "fedlp": st.get("fedlp", {}),
           "lp_seconds": st.get("lp_seconds"), "lp_iterations": st.get("lp_iterations")}
    print_mip(rep)
    if label:
        d = OUT / label
        d.mkdir(parents=True, exist_ok=True)
        (d / "mip.json").write_text(json.dumps(rep, indent=1, default=str))
    return rep


def print_mip(rep: dict) -> None:
    tot = rep["seconds"]
    print(f"{rep['instance']}: {rep['status']} obj {rep['objective']} bound {rep['bound']} "
          f"nodes {rep['nodes']} {tot:.1f}s (read {rep['read_seconds']:.1f}s)")
    tt, pv = rep["time"], rep["pivots"]
    acc = sum(tt.values())
    print(f"  {'phase':<20}{'seconds':>9}{'share':>8}{'pivots':>10}{'us/pivot':>10}")
    for k, v in sorted(tt.items(), key=lambda kv: -kv[1]):
        p = pv.get(k, 0)
        print(f"  {k:<20}{v:9.2f}{100 * v / tot:7.1f}%{p:10d}{(1e6 * v / p if p else 0):10.0f}")
    print(f"  {'(untimed)':<20}{tot - acc:9.2f}{100 * (tot - acc) / tot:7.1f}%")
    f = rep["fedlp"]
    lps = rep.get("lp_seconds") or 0.0
    print(f"  LP solves: {f.get('iterations', 0)} pivots in {lps:.2f}s "
          f"(dual {f.get('dual_iterations', 0)}, phase 1 {f.get('phase1_iterations', 0)}, "
          f"refactors {f.get('refactors', 0)}, bfrt flips {f.get('bfrt_flips', 0)})")
    timers = {k: v for k, v in f.items() if k.startswith("t_")}
    if timers:
        di = max(1, f.get("dual_iterations", 0))
        print("  fedlp timers (summed over solves):")
        for k, v in sorted(timers.items(), key=lambda kv: -kv[1]):
            if v >= 0.005 * max(lps, 1e-9):
                print(f"    {k:<14}{v:8.2f}s {100 * v / max(lps, 1e-9):5.1f}% of LP")
        sizes = {k: f[k] / di for k in ("d_rho_nnz", "d_row_nnz", "d_breaks", "d_alpha_nnz", "d_tau_nnz") if k in f}
        if sizes:
            print("  per dual pivot: " + " ".join(f"{k[2:]}={v:.0f}" for k, v in sizes.items()))


# ---------------------------------------------------------------------- lp
def run_lp(path: str, specs: list[str], tl: float) -> None:
    import numpy as np  # noqa: F401
    from fedhpc.mip import read_mps
    from fedhpc.mip.lp import STATUS_NAME, LpRelaxation
    P = read_mps(path)
    for spec in specs or [""]:
        o = {"threads": 1}
        for kv in filter(None, spec.split(",")):
            k, v = kv.split("=")
            o[k] = ast.literal_eval(v)
        lp = LpRelaxation(P, o)
        if lp.net is not None:
            lp.set_basis(lp.slack_basis())
        lp.deadline = time.perf_counter() + tl
        t = time.perf_counter()
        st = lp.solve()
        dt = time.perf_counter() - t
        s = lp.stats.fedlp
        tim = " ".join(f"{k}={v:.2f}" for k, v in sorted(s.items(), key=lambda kv: -kv[1])
                       if k.startswith("t_") and v > 0.03 * dt)
        print(f"[{spec}] {STATUS_NAME.get(st)} obj {lp.objective:.6g} {dt:.2f}s pivots {s.get('iterations', 0)} "
              f"(dual {s.get('dual_iterations', 0)}, p1 {s.get('phase1_iterations', 0)}) "
              f"{1e6 * dt / max(1, s.get('iterations', 0)):.0f} us/pivot | {tim}", flush=True)


# -------------------------------------------------------------- perf/pyspy
def _inner_cmd(args) -> list[str]:
    cmd = [sys.executable, "-X", "perf", str(Path(__file__)), "mip", args.file, "-t", str(args.time_limit),
           "--cpu", str(args.cpu)]
    if args.build:
        cmd += ["--build", args.build]
    return cmd + list(args.opts)


DSO_CLASSES = [
    ("python interpreter", re.compile(r"python3|libpython")),
    ("fedlp (C++)", re.compile(r"_fedlp|primal\.hpp|lu_ft\.hpp|model\.hpp|factor\.hpp|dense_inverse|lp_module|parallel\.hpp")),
    ("fedmip (C++)", re.compile(r"_fedmip|propagate\.hpp|mip_module")),
    ("numpy/scipy", re.compile(r"numpy|scipy|_multiarray|_umath|openblas|lapack|_sparsetools|libgfortran")),
    ("python JIT/trampolines", re.compile(r"perf-\d+\.map|\[JIT\]|anon")),
    ("libc/libm/libstdc++", re.compile(r"libc\.|libm\.|libstdc\+\+|ld-linux|libgcc")),
    ("kernel", re.compile(r"kernel|vmlinux|\[k\]")),
]


def run_perf(args) -> None:
    if not shutil.which("perf"):
        raise SystemExit("perf not found (install linux-tools / perf)")
    d = OUT / (args.label or "perf")
    d.mkdir(parents=True, exist_ok=True)
    data = d / "perf.data"
    if args.build:
        build(args.build)
    env = dict(os.environ, PYTHONPERFSUPPORT="1")
    cmd = ["perf", "record", "-F", str(args.freq), "-g", "-o", str(data), "--"] + _inner_cmd(args)
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)
    rep = subprocess.run(["perf", "report", "-i", str(data), "--no-children", "--sort", "dso,sym", "--stdio",
                          "-g", "none", "--percent-limit", "0.05"], capture_output=True, text=True).stdout
    (d / "perf_report.txt").write_text(rep)
    by_class = collections.Counter()
    syms = []
    for line in rep.splitlines():
        m = re.match(r"\s*([\d.]+)%\s+(\S+)\s+\[[.k]\]\s+(.*)", line)
        if not m:
            continue
        pct, dso, sym = float(m.group(1)), m.group(2), m.group(3).strip()
        cls = next((c for c, rx in DSO_CLASSES if rx.search(dso)), "other: " + dso)
        if sym.startswith("py::"):
            cls = "python code (frames)"
        by_class[cls] += pct
        syms.append((pct, dso, sym))
    print("\nTime by component (self samples):")
    for c, v in by_class.most_common():
        print(f"  {v:6.1f}%  {c}")
    print("\nTop symbols:")
    for pct, dso, sym in syms[:args.top]:
        print(f"  {pct:6.2f}%  {dso:<28} {sym[:110]}")
    print(f"\nfull report: {d / 'perf_report.txt'} (perf report -i {data})")


def run_pyspy(args) -> None:
    pyspy = shutil.which("py-spy") or str(Path(sys.executable).parent / "py-spy")
    d = OUT / (args.label or "pyspy")
    d.mkdir(parents=True, exist_ok=True)
    if args.build:
        build(args.build)
    raw = d / "stacks.txt"
    cmd = [pyspy, "record", "--native", "-r", str(args.freq), "-f", "raw", "-o", str(raw), "--"] + _inner_cmd(args)
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    leaf = collections.Counter()
    pyfun = collections.Counter()
    total = 0
    for line in raw.read_text().splitlines():
        stack, _, n = line.rpartition(" ")
        n = int(n)
        total += n
        frames = stack.split(";")
        top = frames[-1]
        # with --native the leaf is always a C function: class it by its
        # shared object (the interpreter itself is Python overhead)
        dso = top[top.rfind("(") + 1:-1] if top.endswith(")") else "?"
        if ".py:" in dso:
            cls = "python code"
        else:
            cls = next((c for c, rx in DSO_CLASSES if rx.search(dso)), "other: " + dso)
        leaf[cls] += n
        # the innermost Python frame: where Python code spends (incl. its native callees)
        for f in reversed(frames):
            if ".py:" in f or ".py)" in f:
                pyfun[f.split(" (")[0] + " " + f[f.find("("):][:80]] += n
                break
    print(f"\n{total} samples; by the leaf frame's code:")
    for k, v in leaf.most_common():
        print(f"  {100 * v / total:6.1f}%  {k}")
    print("Innermost Python frames (with the native time under them):")
    for f, v in pyfun.most_common(args.top):
        print(f"  {100 * v / total:6.2f}%  {f}")
    print(f"\nraw stacks: {raw} (convert: py-spy can also write -f speedscope)")


# -------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("variant", choices=sorted(VARIANTS))
    for name in ("mip", "perf", "pyspy"):
        m = sub.add_parser(name)
        m.add_argument("file")
        m.add_argument("opts", nargs="*", help="MipOptions as k=v")
        m.add_argument("-t", "--time-limit", type=float, default=60.0)
        m.add_argument("--cpu", type=int, default=0)
        m.add_argument("--build", choices=sorted(VARIANTS))
        m.add_argument("--label")
        if name != "mip":
            m.add_argument("--freq", type=int, default=499)
            m.add_argument("--top", type=int, default=30)
    lpp = sub.add_parser("lp")
    lpp.add_argument("file")
    lpp.add_argument("specs", nargs="*")
    lpp.add_argument("-t", "--time-limit", type=float, default=120.0)
    lpp.add_argument("--cpu", type=int, default=0)
    lpp.add_argument("--build", choices=sorted(VARIANTS))
    args = ap.parse_args()
    if args.cmd == "build":
        print(build(args.variant))
        return
    try:
        os.sched_setaffinity(0, {args.cpu})
    except (AttributeError, OSError):
        pass
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    if args.cmd in ("mip", "lp"):
        os.environ["FEDHPC_LP_STATS"] = "1"    # fedlp's timers summed over every solve
    if args.cmd == "mip":
        _use_build(args.build)
        run_mip(args.file, args.time_limit, _parse_opts(args.opts), args.label)
    elif args.cmd == "lp":
        _use_build(args.build)
        run_lp(args.file, args.specs, args.time_limit)
    elif args.cmd == "perf":
        run_perf(args)
    else:
        run_pyspy(args)


if __name__ == "__main__":
    main()
