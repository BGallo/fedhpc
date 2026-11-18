"""Fold the lex phase-2 Presolve=0 re-run into the generated-bench results.

Phase 2 of lex f1 (f1 fixed by a dense row with slack) can stall in Gurobi's
presolve far past TimeLimit (B_pos_n300_lE_iH: 3013 s, uncertified; 26 s
optimal with Presolve=0). Presolve=0 on every solve costs the hard λ = 0.25
weighted sums their certificate; in lex f2 phase 2 (f2 = 0 fixed, no slack)
presolve helps: off, 10 lex f2 hit the limit instead of 8 and B_pos_n100_lE_iM
f1_0 went 108217 -> 286969. So only lex f1 phase 2 runs with Presolve=0;
lex f2 keeps the presolve-default results.

Pipeline (bench_generated.py; P = results/generated_bench/bench_io_noreduc):
  1. P_p2lex.jsonl   --lex-only --phase2-presolve 0          (all 108)
  2. `changed`       anchors = lex f1 from step 1 + lex f2 from P.jsonl ->
                     P_p2ref.jsonl; instances whose anchors moved ->
                     presolve0_changed.json
  3. P_p2ws.jsonl    --no-dispatch --ref-from P_p2ref.jsonl NAMES...
                     (weighted sums + baselines on the changed ones only)
  4. `merge`         archive P.jsonl + the validate files the reports read to
                     archive_presolve_default/, then write P.jsonl:
                     unchanged anchors -> old record, lex_f1 from step 1
                     (same values, new status/time); changed -> step-3 record,
                     the old dispatch schedule re-scored under the new anchors
  5. `validate`      per archived validate file: unchanged-anchor instances get
                     gap / old_gap / trace re-computed against the new g*;
                     changed-anchor records are dropped for dispatch_validate.py
                     to re-run (it resumes per instance) ->
                     presolve0_rerun.json {file: [names]}

    uv run python scripts/presolve0_merge.py changed|merge|validate
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GB = ROOT / "results" / "generated_bench"
D = GB / "dispatch_study"
ARCH = GB / "archive_presolve_default"
OLD = GB / "bench_io_noreduc.jsonl"
LEX, WS = GB / "bench_io_noreduc_p2lex.jsonl", GB / "bench_io_noreduc_p2ws.jsonl"
REF = GB / "bench_io_noreduc_p2ref.jsonl"
CHANGED, RERUN = GB / "presolve0_changed.json", GB / "presolve0_rerun.json"
VALIDATE = [f"validate_F_{a}_s{s}.jsonl" for a in ("ils", "ilslong", "gafast", "gamid", "gaqual")
            for s in (42, 1, 7)] + ["validate_T_sl48p16g120_s42.jsonl", "validate_T_gamid_s1.jsonl",
                                    "validate_T_gamid_s7.jsonl", "validate_fast_default_trivial.jsonl",
                                    "validate_ils_s42.jsonl"]
LAMS = ("0.25", "0.5", "0.75")


def _g(lam, f1, f2, ref):
    f1_T, f2_T, f1_0 = ref
    if f1_0 <= f1_T + 1e-9 or f2_T <= 1e-9:
        return f2
    return lam * (f1 - f1_T) / (f1_0 - f1_T) + (1 - lam) * f2 / f2_T


def ref_of(r):
    return (r["lex_f1"]["f1"], r["lex_f1"]["f2"], r["lex_f2"]["f1"])


def load(p):
    return {(x := json.loads(l))["name"]: x for l in p.read_text().splitlines() if l.strip()}


def old_bench():
    return load(ARCH / OLD.name if (ARCH / OLD.name).exists() else OLD)


def changed():
    old, new = old_bench(), load(LEX)
    assert set(old) == set(new), set(old) ^ set(new)
    lex = {n: dict(name=n, lex_f1=new[n]["lex_f1"], lex_f2=old[n]["lex_f2"]) for n in old}
    with open(REF, "w") as fo:
        for r in lex.values():
            fo.write(json.dumps(r) + "\n")
    ch = sorted(n for n in lex if any(abs(a - b) > 1e-6 * max(1, abs(a))
                                      for a, b in zip(ref_of(old[n]), ref_of(lex[n]))))
    print(f"anchors changed on {len(ch)} of {len(lex)} instances:")
    for n in ch:
        o, x = old[n], lex[n]
        print(f"  {n}: lex_f1 {o['lex_f1']['status']}->{x['lex_f1']['status']}  "
              f"{ref_of(o)} -> {ref_of(x)}")
    CHANGED.write_text(json.dumps(ch, indent=1))


def merge():
    ARCH.mkdir(exist_ok=True)
    if not (ARCH / OLD.name).exists():
        shutil.copy2(OLD, ARCH / OLD.name)
        for f in VALIDATE:
            shutil.copy2(D / f, ARCH / f)
    old, lex = load(ARCH / OLD.name), load(REF)
    ch = set(json.loads(CHANGED.read_text()))
    ws = load(WS) if ch else {}
    assert ch <= set(ws), ch - set(ws)
    with open(OLD, "w") as fo:
        for n, o in old.items():  # keep the old file's order
            if n in ch:
                r = ws[n]
                ref = ref_of(r)
                for lam in LAMS:
                    od = o["lams"][lam]["dispatch"]
                    r["lams"][lam]["dispatch"] = dict(
                        od, g=_g(float(lam), od["f1"], od["f2"], ref),
                        note="presolve-default bench schedule (optimised for its anchors), re-scored")
            else:
                r = dict(o, lex_f1=lex[n]["lex_f1"])
                assert all(abs(a - b) <= 1e-6 * max(1, abs(a)) for a, b in zip(ref_of(r), ref_of(o)))
                r["lex_f1"] = dict(r["lex_f1"], f1=o["lex_f1"]["f1"], f2=o["lex_f1"]["f2"])
            r["lex_f1_phase2_presolve"] = 0
            r.pop("phase2_presolve", None)
            r["rerun"] = "anchors+weighted" if n in ch else "anchors"
            fo.write(json.dumps(r) + "\n")
    print(f"wrote {OLD.name}: {len(ch)} instances with new anchors and weighted sums, "
          f"{len(old) - len(ch)} with new lex f1 status/time only")


def validate():
    new = load(OLD)
    ch = set(json.loads(CHANGED.read_text()))
    rerun = {}
    for f in VALIDATE:
        kept = dropped = 0
        with open(D / f, "w") as fo:
            for l in (ARCH / f).read_text().splitlines():
                rec = json.loads(l)
                if rec["name"] in ch:
                    dropped += 1
                    rerun.setdefault(f, []).append(rec["name"])
                    continue
                for lam, x in rec["lams"].items():
                    gs_old, gs = x["gstar"], new[rec["name"]]["lams"][lam]["gurobi"]["g"]
                    rel = (lambda v: 100 * (v - gs) / gs) if gs > 1e-12 else (lambda v: None)
                    x["gstar"], x["gap"], x["old_gap"] = gs, rel(x["g"]), rel(x["old_g"])
                    if x.get("trace") and gs_old > 1e-12:
                        x["trace"] = [(t, round(rel(gs_old * (1 + gp / 100)), 4))
                                      for t, gp in x["trace"]]
                fo.write(json.dumps(rec) + "\n")
                kept += 1
        print(f"{f}: kept {kept}, dropped {dropped}")
    RERUN.write_text(json.dumps(rerun, indent=1))
    print("re-run per file:", json.dumps(rerun, indent=1))


if __name__ == "__main__":
    {"changed": changed, "merge": merge, "validate": validate}[sys.argv[1]]()
