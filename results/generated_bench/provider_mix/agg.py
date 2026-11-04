import json, collections, sys
R = [json.loads(l) for l in open(sys.argv[1])]
print(f"{'instance':24s} lam  st   cloud  aws_jobs  aws$%   by size(provider: jobs)                          swap→other: Δf2%  Δf1")
tot = collections.Counter(); totc = collections.Counter()
for r in R:
    aws = sum(v for k, v in r["n"].items() if k.startswith("aws")); awsc = sum(v for k, v in r["cost"].items() if k.startswith("aws"))
    for k, v in r["n"].items():
        if k != "onprem": tot[k] += v; totc[k] += r["cost"][k]
    sz = " ".join(f"{k.split('_',1)[1]}:{v}" for k, v in sorted(r["n"].items()) if k != "onprem")
    print(f"{r['name']:24s} {r['lam']:.2f} {r['status'][:4]} {r['n_cloud']:5d} {aws:6d} {100*awsc/max(r['f2'],1e-9):6.2f}  {sz:52s} {100*r['swap_df2']/max(r['f2'],1e-9):+6.2f} {r['swap_df1']:+5.0f}")
print("\nTOTAL jobs by type:", dict(tot)); print("TOTAL $ by type:", {k: round(v) for k, v in totc.items()})
