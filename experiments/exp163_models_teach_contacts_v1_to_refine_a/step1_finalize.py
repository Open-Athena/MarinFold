"""
Wait for the base-matrix scoring run to finish, then produce a PAIRED, same-proteins
head-to-head: base calibrated matrix vs consensus voting vs single rollout vs oracle
best-of-K, on the exact proteins the matrix run scored.  Usage: step1_finalize.py <producer_pid> <matrix_csv>
"""
import os, sys, time
import numpy as np, pandas as pd
from collections import Counter
from sklearn.metrics import roc_auc_score

PID = int(sys.argv[1]); MAT_CSV = sys.argv[2]
B = "hf://buckets/open-athena/MarinFold/data/contacts-v1-train-rollouts-exp98"
MIN_SEP, K, DRAWS, TIES = 6, 16, 3, 3

# ---- wait for the producer process to exit (poll /proc) ----
waited = 0
while os.path.exists(f"/proc/{PID}"):
    time.sleep(20); waited += 20
    if waited % 120 == 0:
        try:
            n = sum(1 for _ in open(MAT_CSV)) - 1
        except FileNotFoundError:
            n = 0
        print(f"[wait {waited}s] matrix rows so far: {n}", flush=True)
print(f"producer {PID} exited after ~{waited}s", flush=True)

def canon(flat):
    a = np.asarray(flat).reshape(-1, 2)
    if a.size == 0: return set()
    lo = np.minimum(a[:, 0], a[:, 1]); hi = np.maximum(a[:, 0], a[:, 1])
    keep = (hi - lo) >= MIN_SEP
    return set(zip(lo[keep].tolist(), hi[keep].tolist()))

mat = pd.read_csv(MAT_CSV)
eids = mat.entry_id.tolist()
print(f"\n=== base calibrated matrix ({len(mat)} proteins) ===", flush=True)
print(f"  all_R={mat.all_R.mean():.4f}  long_R={mat.long_R.mean():.4f}  "
      f"all_AUC={mat.all_AUC.mean():.4f}  long_AUC={mat.long_AUC.mean():.4f}", flush=True)

print("\nloading rollouts for the same proteins ...", flush=True)
tgt = pd.read_parquet(f"{B}/targets.parquet").set_index("entry_id")
mp = pd.read_parquet(f"{B}/rollout_metrics_all.parquet",
                     columns=["entry_id", "r", "pred", "all_f1", "nll_per_tok"],
                     filters=[("entry_id", "in", eids)])
pg = {e: g.reset_index(drop=True) for e, g in mp.groupby("entry_id")}
rng = np.random.default_rng(0)

rows = []
for eid in eids:
    if eid not in pg: continue
    g = pg[eid]; L = int(tgt.loc[eid, "L"]); n_gt = int(tgt.loc[eid, "n_gt"])
    if len(g) < K or n_gt < 5: continue
    gt = canon(np.concatenate([np.asarray(p).ravel() for p in tgt.loc[eid, "gt_contacts"]]))
    if not gt: continue
    gt_keys = np.array([i * L + j for i, j in gt])
    preds = [canon(p) for p in g["pred"].to_numpy()]
    f1 = g.all_f1.to_numpy(); nll = g.nll_per_tok.to_numpy()
    iu, ju = np.triu_indices(L, k=MIN_SEP); uni = iu * L + ju
    yt = np.zeros(L * L, bool); yt[gt_keys] = True; yt_u = yt[uni]
    v_auc, v_rp, u_rec, s_mean, s_oracle = [], [], [], [], []
    for d in range(DRAWS):
        idx = rng.choice(len(g), size=K, replace=False)
        votes = Counter(); union = set()
        for t in idx: votes.update(preds[t]); union |= preds[t]
        dense = np.zeros(L * L, np.float32)
        for (i, j), c in votes.items(): dense[i * L + j] = c
        sc = dense[uni]
        v_auc.append(roc_auc_score(yt_u, sc) if 0 < yt_u.sum() < len(yt_u) else np.nan)
        rp = []
        for s in range(TIES):
            noise = np.random.default_rng(1000 * d + s).random(sc.size) * 1e-3
            rp.append(yt_u[np.argsort(-(sc + noise))[:n_gt]].mean())
        v_rp.append(np.mean(rp)); u_rec.append(len(union & gt) / len(gt))
        s_mean.append(f1[idx].mean()); s_oracle.append(f1[idx].max())
    rows.append(dict(entry_id=eid, mat_R=float(mat.set_index("entry_id").loc[eid, "all_R"]),
                     vote_R=np.mean(v_rp), vote_AUC=np.nanmean(v_auc),
                     single_f1=np.mean(s_mean), oracle_f1=np.mean(s_oracle),
                     union_rec=np.mean(u_rec)))
d = pd.DataFrame(rows)
d.to_csv("step1_headtohead.csv", index=False)
print(f"\n=== PAIRED head-to-head, SAME {len(d)} proteins (all-band, K={K}) ===", flush=True)
print(f"  base calibrated matrix R-prec : {d.mat_R.mean():.4f}   (AUC {mat.set_index('entry_id').loc[d.entry_id,'all_AUC'].mean():.3f})", flush=True)
print(f"  consensus vote        R-prec : {d.vote_R.mean():.4f}   (AUC {d.vote_AUC.mean():.3f})", flush=True)
print(f"  single rollout        F1     : {d.single_f1.mean():.4f}", flush=True)
print(f"  oracle best-of-K      F1     : {d.oracle_f1.mean():.4f}", flush=True)
print(f"  union-recall ceiling         : {d.union_rec.mean():.3f}", flush=True)
print(f"\n  paired: vote_R - mat_R  mean={ (d.vote_R-d.mat_R).mean():+.4f}  "
      f"(vote wins on {(d.vote_R>d.mat_R).mean()*100:.0f}% of proteins)", flush=True)
print("\nDONE", flush=True)
