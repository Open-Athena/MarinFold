"""
Phase-0 feasibility probe for the rollout-refinement idea, on exp98 data.

Answers three questions, all training-free, from existing rollouts:
  B. Proxy viability: does base-model confidence (nll_per_tok) rank rollouts by F1
     within a protein?  (decides whether the *deployable sorted ramp* is possible)
  C. Selection headroom: best-of-K vs mean, oracle vs proxy selection.
  D. Aggregation crux: does a training-free consensus-VOTE over K rollouts beat
     (i) the mean single rollout, (ii) deployable proxy-best-of-K, and approach
     (iii) the oracle best-of-K?  Consensus AUC/R-precision are what a trained
     refiner must beat to be worth building.
"""
import sys
import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score

B = "hf://buckets/open-athena/MarinFold/data/contacts-v1-train-rollouts-exp98"
RNG = np.random.default_rng(0)
MIN_SEP = 6

def canon_pairs(flat):
    """flat [i0,j0,i1,j1,...] -> set of (min,max) with sep>=6."""
    a = np.asarray(flat).reshape(-1, 2)
    if a.size == 0:
        return set()
    lo = np.minimum(a[:, 0], a[:, 1]); hi = np.maximum(a[:, 0], a[:, 1])
    keep = (hi - lo) >= MIN_SEP
    return set(zip(lo[keep].tolist(), hi[keep].tolist()))

# ---------------------------------------------------------------- load
print("loading targets + metrics ...", flush=True)
tgt = pd.read_parquet(f"{B}/targets.parquet")
tgt = tgt.set_index("entry_id")
cols = ["entry_id", "r", "nll_per_tok", "all_f1", "long_f1", "n_pred", "finished"]
m = pd.read_parquet(f"{B}/rollout_metrics_all.parquet", columns=cols)
print(f"  metrics rows={len(m)} proteins={m.entry_id.nunique()}", flush=True)

# ================================================================ B. proxy
print("\n================ B. PROXY VIABILITY (nll_per_tok vs F1) ================", flush=True)
p_all = pearsonr(m.nll_per_tok, m.all_f1)[0]
s_all = spearmanr(m.nll_per_tok, m.all_f1)[0]
s_long = spearmanr(m.nll_per_tok, m.long_f1)[0]
print(f"pooled Pearson(nll, all_f1)  = {p_all:+.3f}", flush=True)
print(f"pooled Spearman(nll, all_f1) = {s_all:+.3f}  (negative = confident->accurate)", flush=True)
print(f"pooled Spearman(nll, long_f1)= {s_long:+.3f}", flush=True)

# within-protein spearman + proxy pick quality
per = []
for eid, g in m.groupby("entry_id"):
    if g.all_f1.nunique() < 2:
        continue
    sp = spearmanr(g.nll_per_tok, g.all_f1)[0]
    # deployable proxy = pick lowest nll; where does its F1 land?
    pick = g.loc[g.nll_per_tok.idxmin()]
    pct = (g.all_f1 < pick.all_f1).mean()  # percentile of picked rollout's F1
    per.append((sp, pct, pick.all_f1, g.all_f1.mean(), g.all_f1.max()))
per = np.array(per)
print(f"within-protein Spearman(nll,all_f1): median={np.nanmedian(per[:,0]):+.3f} "
      f"[{np.nanpercentile(per[:,0],25):+.3f}, {np.nanpercentile(per[:,0],75):+.3f}]  "
      f"frac<0={np.mean(per[:,0]<0):.2f}", flush=True)
print(f"proxy pick (min-nll) mean F1-percentile within protein = {per[:,1].mean():.3f} "
      f"(0.5=random, 1.0=always best)", flush=True)
print(f"F1 of proxy-pick={per[:,2].mean():.4f}  vs mean={per[:,3].mean():.4f}  "
      f"vs oracle-max={per[:,4].mean():.4f}", flush=True)

# ================================================================ C. best-of-K
print("\n================ C. SELECTION HEADROOM (best-of-K) ================", flush=True)
groups = {eid: g.reset_index(drop=True) for eid, g in m.groupby("entry_id")}
Ks = [1, 2, 4, 8, 16, 32, 64, 128]
DRAWS = 8
print(f"{'K':>4} | {'mean':>7} {'proxy':>7} {'oracle':>7}  (all_f1, avg over {DRAWS} draws x proteins)", flush=True)
for K in Ks:
    mean_v, proxy_v, oracle_v = [], [], []
    for eid, g in groups.items():
        n = len(g)
        f1 = g.all_f1.to_numpy(); nll = g.nll_per_tok.to_numpy()
        for d in range(DRAWS):
            idx = RNG.choice(n, size=min(K, n), replace=False)
            sub_f1 = f1[idx]; sub_nll = nll[idx]
            mean_v.append(sub_f1.mean())
            proxy_v.append(sub_f1[np.argmin(sub_nll)])
            oracle_v.append(sub_f1.max())
    print(f"{K:>4} | {np.mean(mean_v):7.4f} {np.mean(proxy_v):7.4f} {np.mean(oracle_v):7.4f}", flush=True)

# ================================================================ D. aggregation
print("\n================ D. AGGREGATION CRUX (consensus vote over K) ================", flush=True)
S = int(sys.argv[1]) if len(sys.argv) > 1 else 200
sample_eids = RNG.choice(sorted(groups.keys()), size=min(S, len(groups)), replace=False)
print(f"reading pred for {len(sample_eids)} sampled proteins ...", flush=True)
mp = pd.read_parquet(
    f"{B}/rollout_metrics_all.parquet",
    columns=["entry_id", "r", "pred", "all_f1", "nll_per_tok"],
    filters=[("entry_id", "in", list(sample_eids))],
)
print(f"  got {len(mp)} rows", flush=True)
pg = {eid: g.reset_index(drop=True) for eid, g in mp.groupby("entry_id")}

def eval_protein(eid, K, draws=3, tie_seeds=3):
    g = pg[eid]; L = int(tgt.loc[eid, "L"]); n_gt = int(tgt.loc[eid, "n_gt"])
    if n_gt < 5 or len(g) < K:
        return None
    gt = canon_pairs(np.concatenate([np.asarray(p).ravel() for p in tgt.loc[eid, "gt_contacts"]]))
    if len(gt) == 0:
        return None
    gt_keys = np.array([i * L + j for (i, j) in gt])
    preds = [canon_pairs(p) for p in g["pred"].to_numpy()]
    f1 = g.all_f1.to_numpy(); nll = g.nll_per_tok.to_numpy()
    # universe
    iu, ju = np.triu_indices(L, k=MIN_SEP)
    uni = iu * L + ju
    y_true = np.zeros(L * L, bool); y_true[gt_keys] = True
    y_true_u = y_true[uni]
    out = {k: [] for k in ["vote_auc", "vote_rp", "union_rec", "mean_f1", "proxy_f1", "oracle_f1"]}
    n = len(g)
    for d in range(draws):
        idx = RNG.choice(n, size=K, replace=False)
        votes = Counter()
        union = set()
        for t in idx:
            votes.update(preds[t]); union |= preds[t]
        dense = np.zeros(L * L, np.float32)
        for (i, j), c in votes.items():
            dense[i * L + j] = c
        score_u = dense[uni]
        out["vote_auc"].append(roc_auc_score(y_true_u, score_u) if 0 < y_true_u.sum() < len(y_true_u) else np.nan)
        # R-precision with random tiebreak
        rps = []
        for s in range(tie_seeds):
            noise = np.random.default_rng(1000 * d + s).random(score_u.size) * 1e-3
            top = np.argsort(-(score_u + noise))[:n_gt]
            rps.append(y_true_u[top].mean())
        out["vote_rp"].append(np.mean(rps))
        out["union_rec"].append(len(union & gt) / len(gt))
        out["mean_f1"].append(f1[idx].mean())
        out["proxy_f1"].append(f1[idx][np.argmin(nll[idx])])
        out["oracle_f1"].append(f1[idx].max())
    return {k: np.nanmean(v) for k, v in out.items()}

for K in [8, 16, 32]:
    rows = [r for eid in pg if (r := eval_protein(eid, K)) is not None]
    if not rows:
        continue
    df = pd.DataFrame(rows)
    print(f"\n--- K={K}  (n={len(df)} proteins) ---", flush=True)
    print(f"  DEPLOYABLE consensus vote : AUC={df.vote_auc.mean():.3f}   "
          f"R-precision={df.vote_rp.mean():.4f}", flush=True)
    print(f"  deployable proxy-best-of-K: F1 ={df.proxy_f1.mean():.4f}", flush=True)
    print(f"  mean single rollout       : F1 ={df.mean_f1.mean():.4f}", flush=True)
    print(f"  ORACLE best-of-K          : F1 ={df.oracle_f1.mean():.4f}  (non-deployable ceiling)", flush=True)
    print(f"  union recall of K         :     ={df.union_rec.mean():.3f}  (coverage ceiling)", flush=True)
print("\nDONE", flush=True)
