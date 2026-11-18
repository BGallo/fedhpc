"""Re-run the heuristic arms on the instances whose lex anchors changed
(presolve0_rerun.json), with each file's original configuration
(dispatch_study/tuning/final.spec; T_* = GA-mid; validate_ils_s42 = dispatch defaults)."""
import json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
GAMID = {"flags": 12352, "shortlist": 48, "pop_size": 16, "n_gen": 120, "adapt": 215}
ARMS = {"ils": ("dispatch", {}), "ilslong": ("dispatch", {"n_kicks": 120, "kick_patience": 0}),
        "gafast": ("memetic", {"flags": 12352, "adapt": 215, "shortlist": 48, "pop_size": 16}),
        "gamid": ("memetic", GAMID),
        "gaqual": ("memetic", {"flags": 12352, "adapt": 215, "shortlist": 48, "n_gen": 120})}
rerun = json.loads((ROOT / "results/generated_bench/presolve0_rerun.json").read_text())
for f, names in rerun.items():
    tag = f[len("validate_"):-len(".jsonl")]
    base, seed = tag.rsplit("_s", 1)
    if base.startswith("F_"):
        solver, kw = ARMS[base[2:]]
    elif base.startswith("T_"):
        solver, kw = "memetic", GAMID
    elif base == "ils":
        solver, kw = "dispatch", {}
    else:
        sys.exit(f"no config for {f}")
    cmd = [sys.executable, str(ROOT / "scripts/dispatch_validate.py"), "--solver", solver, "--tag", tag,
           "--kw", json.dumps(kw), "--seed", seed, "--workers", "3", "--threads", "4", "--names", *names]
    print("===", tag, names, flush=True)
    subprocess.run(cmd, check=True)
print("ALLDONE", flush=True)
