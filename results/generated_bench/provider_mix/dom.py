"""Per job: Pareto set of its cloud options over (p_occ, cost). Cloud is unlimited,
so a bursted job starts at release and only (p_occ, c) matter."""
import json, glob, os, collections
from fedhpc.data import Instance
out = []
for f in sorted(glob.glob("data/generated/*_i*.json")):
    d = json.load(open(f)); inst = Instance.from_dict(d)
    names = d["_info"]["instance_type_names"]
    cloud = [m.id for m in inst.instance_types if m.kind == "cloud"]
    single = collections.Counter(); multi = collections.Counter(); nmulti = 0
    tie = 0
    for j in inst.jobs:
        opts = [(inst.p_occ[j.id, m], inst.c_units[j.id, m], m) for m in inst.F[j.id] if m in cloud]
        nd = [o for o in opts if not any((p[0] <= o[0] and p[1] <= o[1]) and (p[0] < o[0] or p[1] < o[1]) for p in opts)]
        # exact ties (same p_occ & cost) on different types
        keys = collections.Counter((o[0], o[1]) for o in nd)
        if any(v > 1 for v in keys.values()): tie += 1
        if len(keys) == 1:
            single[tuple(sorted(names[str(o[2])]["name"] for o in nd))] += 1
        else:
            nmulti += 1
            multi[tuple(sorted(names[str(o[2])]["name"] for o in nd))] += 1
    out.append(dict(name=os.path.basename(f)[:-5], single={"+".join(k): v for k, v in single.items()},
                    multi={"+".join(k): v for k, v in multi.items()}, n_multi=nmulti, ties=tie))
    print(out[-1], flush=True)
json.dump(out, open(os.environ["OUT"], "w"), indent=1)
