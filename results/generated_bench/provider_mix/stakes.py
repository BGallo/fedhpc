import json, collections, statistics as st
from fedhpc.data import Instance
for w in ("fedhpc", "pos"):
    for io in "LMH":
        name = f"B_{w}_n300_lE_i{io}"
        d = json.load(open(f"data/generated/{name}.json")); inst = Instance.from_dict(d)
        nm = d["_info"]["instance_type_names"]
        cl = {m.id: nm[str(m.id)] for m in inst.instance_types if m.kind == "cloud"}
        by_eq = collections.defaultdict(dict)
        for m, v in cl.items(): by_eq[v["equivalent_of"]][v["provider"]] = m
        size = collections.Counter(); dc = collections.defaultdict(list); rel = collections.defaultdict(list)
        slot = collections.Counter(); caz = cav = 0.0
        for j in inst.jobs:
            # cheapest-fitting size (all providers agree on the size)
            eqs = [e for e, p in by_eq.items() if p["aws"] in inst.F[j.id]]
            e = min(eqs, key=lambda e: inst.c[j.id, by_eq[e]["aws"]])
            size[e] += 1
            a, z = by_eq[e]["aws"], by_eq[e]["azure"]
            dc[e].append(inst.c[j.id, a] - inst.c[j.id, z]); rel[e].append((inst.c[j.id, a] - inst.c[j.id, z]) / inst.c[j.id, z])
            slot[inst.p_occ[j.id, z] - inst.p_occ[j.id, a]] += 1
            caz += inst.c[j.id, z]; cav += inst.c[j.id, a]
        print(f"\n{name}: jobs by size {dict(size)}  all-cloud cost Azure={caz:.0f} AWS={cav:.0f} (AWS {100*(cav/caz-1):+.2f}%)  p_occ(Az)-p_occ(AWS): {dict(slot)}")
        for e in sorted(dc):
            print(f"  {e}: AWS-Azure $ median {st.median(dc[e]):+.3f} max {max(dc[e]):+.2f} min {min(dc[e]):+.2f}; rel median {100*st.median(rel[e]):+.2f}% range [{100*min(rel[e]):+.1f},{100*max(rel[e]):+.1f}]%")
