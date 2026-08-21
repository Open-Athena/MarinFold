"""Pooled-8-multi-rollouts vs plain-100, paired per protein — issue #237.

`pool_across_rollouts.py` prints means only and keys its .npy by dict order, so
the arrays it saves cannot be aligned to another predictor's. This recomputes the
pooled vote WITH the (dataset, stem) key attached and pairs it against #230's own
Gate A per-protein table, which is where plain-100's 0.6058 comes from.
"""
import glob, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
sys.path.insert(0, str(Path.home()/"MarinFold"/"experiments"/"exp230_models_contacts_v1_multi_from_exp199"))
from score_gate_a import metrics_for

tgt = {(r["dataset"], r["stem"]): r for r in
       pq.read_table(str(Path.home()/"exp230_data"/"eval577_targets.parquet")).to_pylist()}
plain = {(r["dataset"], r["stem"]): r["all:R"] for r in
         pq.read_table("/home/ubuntu/exp230_data/eval/gate_a/gate_a_per_protein.parquet").to_pylist()
         if r["label"] == "finetune"}

def pooled(root):
    by = defaultdict(list)
    for p in sorted(glob.glob(f"{root}/**/*.parquet", recursive=True)):
        for r in pq.read_table(p).to_pylist():
            if r["sec_idx"] < 0: continue
            by[(r["dataset"], r["stem"])].append({(int(i), int(j)) for i, j in r["contacts"]})
    out = {}
    for key, secs in by.items():
        rec = tgt.get(key)
        if rec is None or not rec["in_legacy554"]: continue
        L = int(rec["L"]); gt = {(int(i), int(j)) for i, j in rec["gt_contacts"]}
        if not gt: continue
        M = np.zeros((L, L), np.float32)
        for s in secs:
            for i, j in s:
                if 0 <= i < L and 0 <= j < L: M[i, j] += 1; M[j, i] += 1
        out[key] = metrics_for(M.astype(np.float16), gt, L).get("all:R", np.nan)
    return out

rng = np.random.default_rng(237)
for label, root in [(a.split("=")[0], a.split("=")[1]) for a in sys.argv[1:]]:
    P = pooled(root)
    keys = [k for k in P if k in plain and np.isfinite(P[k]) and np.isfinite(plain[k])]
    a = np.array([P[k] for k in keys]); b = np.array([plain[k] for k in keys])
    d = a - b
    idx = rng.integers(0, len(d), size=(10000, len(d)))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    star = "*" if lo * hi > 0 else " "
    print(f"{label:<10} n={len(d):<4} pooled-8 {a.mean():.4f}  plain-100 {b.mean():.4f}  "
          f"Δ {d.mean():+.4f}  [{lo:+.4f}, {hi:+.4f}] {np.sum(d>0)}/{np.sum(d<0)} {star}")
