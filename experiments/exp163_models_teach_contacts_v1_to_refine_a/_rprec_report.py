import gcsfs, numpy as np, pyarrow as pa, pyarrow.parquet as pq, csv
fs = gcsfs.GCSFileSystem()
R = "marin-us-east5/MarinFold/exp163/eval554"
def load(d):
    f = [p for p in fs.find(d) if p.endswith(".parquet")]
    return pa.concat_tables([pq.read_table(fs.open(p, "rb")) for p in f]).to_pydict()
B = load(R + "/rprec_base"); F = load(R + "/rprec_tpuF")
nn = lambda t, c: np.array([x for x in t[c] if x is not None], float)
print("TEACHER-FORCED R-PRECISION, plain <contacts-v1>, no drafts")
print("%-14s%8s%10s%10s%10s%10s" % ("model", "n", "R_all", "R_short", "R_med", "R_long"))
for lab, t in (("base E8", B), ("arm F", F)):
    print("%-14s%8d%10.4f%10.4f%10.4f%10.4f" % (lab, len(t["entry_id"]), np.nanmean(nn(t,"R0_all")), np.nanmean(nn(t,"R0_short")), np.nanmean(nn(t,"R0_medium")), np.nanmean(nn(t,"R0_long"))))
kb = dict(zip(B["entry_id"], B["R0_all"])); kbl = dict(zip(B["entry_id"], B["R0_long"]))
ids = [e for e in F["entry_id"] if e in kb]
fa = dict(zip(F["entry_id"], F["R0_all"])); fl = dict(zip(F["entry_id"], F["R0_long"]))
for name, cur, ref in (("R_all", fa, kb), ("R_long", fl, kbl)):
    d = np.array([cur[e] - ref[e] for e in ids], float); d = d[~np.isnan(d)]
    s = d.std(ddof=1) / np.sqrt(len(d))
    print("paired %-7s arm F - base = %+.4f +/- %.4f (%+.1f sigma, win %.1f%%, n=%d)" % (name, d.mean(), s, d.mean()/s, 100*(d>0).mean(), len(d)))
