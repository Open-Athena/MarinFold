import csv, sys, gcsfs, numpy as np, pyarrow.parquet as pq
TPU = "marin-us-east5/MarinFold/exp163/eval554/rprec_validate/shard-0-of-1.parquet"
fs = gcsfs.GCSFileSystem()
fh = fs.open(TPU, "rb")
t = pq.read_table(fh).to_pydict()
tpu = dict(zip(t["entry_id"], t["R0_all"]))
tpul = dict(zip(t["entry_id"], t["R0_long"]))
ref = {r["entry_id"]: r for r in csv.DictReader(open("data_md/eval554_base.csv"))}
common = [e for e in tpu if e in ref]
a = np.array([tpu[e] for e in common], float)
b = np.array([float(ref[e]["R0_all"]) for e in common])
al = np.array([tpul[e] for e in common], float)
bl = np.array([float(ref[e]["R0_long"]) for e in common])
print("n =", len(common))
print("R0_all  TPU %.4f  CUDA %.4f  mean|d| %.5f  max|d| %.5f  corr %.5f" % (a.mean(), b.mean(), np.abs(a-b).mean(), np.abs(a-b).max(), np.corrcoef(a,b)[0,1]))
m = ~(np.isnan(al) | np.isnan(bl))
print("R0_long TPU %.4f  CUDA %.4f  mean|d| %.5f  max|d| %.5f  corr %.5f" % (al[m].mean(), bl[m].mean(), np.abs(al-bl)[m].mean(), np.abs(al-bl)[m].max(), np.corrcoef(al[m],bl[m])[0,1]))
print("exact matches: %d/%d" % (int((a==b).sum()), len(common)))
