"""Re-solve class-B Gurobi weighted sums (stored refs, no reductions, as in
bench_io_noreduc) and record the cloud provider / VM mix of each optimum, plus
the same schedule with every cloud job swapped to the other provider."""
import json, sys, time, collections
from concurrent.futures import ProcessPoolExecutor
BENCH = "results/generated_bench/bench_io_noreduc.jsonl"
OUT = sys.argv[1]

def run(name):
    from fedhpc.data import Instance
    from fedhpc.formulations import SpaceTimeFormulation
    from fedhpc.model import solve_weighted_sum
    rec = next(json.loads(l) for l in open(BENCH) if json.loads(l)["name"] == name)
    d = json.load(open(f"data/generated/{name}.json")); inst = Instance.from_dict(d)
    names = d["_info"]["instance_type_names"]
    kind = {m.id: m.kind for m in inst.instance_types}
    prov = {int(k): v.get("provider", "onprem") for k, v in names.items()}
    eqv = {int(k): v.get("equivalent_of", v["name"]) for k, v in names.items()}
    other = {m: next((n for n in prov if prov[n] not in ("onprem", prov[m]) and eqv[n] == eqv[m]), None)
             for m in prov if prov[m] != "onprem"}
    arr = {j.id: j.arrival for j in inst.jobs}
    fmt = SpaceTimeFormulation(collapse_unbounded=False, pareto_unbounded=False,
                               rejection_deadlines=False, lex_deadlines=False)
    ref = (rec["lex_f1"]["f1"], rec["lex_f1"]["f2"], rec["lex_f2"]["f1"])
    out = []
    for lam in (0.25, 0.5, 0.75):
        t = time.perf_counter()
        s = solve_weighted_sum(inst, lam, *ref, formulation=fmt, f2_0=rec["lex_f2"]["f2"],
                               OutputFlag=0, Threads=4, TimeLimit=600, MIPGap=0)
        inst.check_schedule(s.assignment)
        n = collections.Counter(); cost = collections.Counter(); jobs_cloud = 0
        swap_df1 = swap_df2 = 0.0; swap_ok = True
        for j, (m, st) in s.assignment.items():
            if kind[m] != "cloud": n["onprem"] += 1; continue
            jobs_cloud += 1
            key = names[str(m)]["name"]; n[key] += 1; cost[key] += inst.c[j, m]
            o = other[m]
            if o is None or (j, o) not in inst.c: swap_ok = False; continue
            swap_df2 += inst.c[j, o] - inst.c[j, m]
            swap_df1 += inst.p_occ[j, o] - inst.p_occ[j, m]
        out.append(dict(name=name, lam=lam, status=s.status, t=time.perf_counter() - t,
                        f1=s.f1, f2=s.f2, n_cloud=jobs_cloud, n=dict(n), cost=dict(cost),
                        swap_df1=swap_df1, swap_df2=swap_df2, swap_ok=swap_ok,
                        ref=ref))
    return out

if __name__ == "__main__":
    recs = [json.loads(l) for l in open(BENCH)]
    nontriv = [r["name"] for r in recs if r["cls"] == "B" and r["lex_f1"]["f2"] > 1e-9]
    todo = [n for n in nontriv if "_fedhpc_" in n] + \
           ["B_pos_n300_lE_iL", "B_pos_n300_lE_iM", "B_pos_n300_lE_iH", "B_pos_n100_lE_iH"]
    with ProcessPoolExecutor(3) as ex, open(OUT, "a") as f:
        for res in ex.map(run, todo):
            for r in res: f.write(json.dumps(r) + "\n")
            f.flush(); print(res[0]["name"], "done", flush=True)
