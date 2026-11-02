"""Does M-B gain compound when you pool sections ACROSS rollouts? — issue #237.

M-B optimises an ORACLE quantity, and its oracle-best (0.5663) has essentially
caught what 22 independent plain rollouts offer (0.5680). So the question is no
longer whether the candidates are good -- it is whether the gain survives being
CASHED OUT by an aggregator. The eval already generated 8 multi rollouts per
protein; pooling every section of all 8 into one vote gives ~160 candidates and
costs nothing but a re-score.
"""
import glob, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
sys.path.insert(0, str(Path.home()/"MarinFold"/"experiments"/"exp230_models_contacts_v1_multi_from_exp199"))
from score_gate_a import metrics_for

def load(root):
    by = defaultdict(lambda: defaultdict(dict))
    for p in sorted(glob.glob(f"{root}/**/*.parquet", recursive=True)):
        for r in pq.read_table(p).to_pylist():
            if r["sec_idx"] < 0: continue
            by[(r["dataset"], r["stem"])][r["r"]][r["sec_idx"]] = {(int(i), int(j)) for i, j in r["contacts"]}
    return {k: {r: [s[i] for i in sorted(s)] for r, s in v.items()} for k, v in by.items()}

tgt = {(r["dataset"], r["stem"]): r for r in pq.read_table(str(Path.home()/"exp230_data"/"eval577_targets.parquet")).to_pylist()}
for label, root in [(a.split("=")[0], a.split("=")[1]) for a in sys.argv[1:]]:
    sec = load(root)
    one, pooled, n = [], [], 0
    for key, rolls in sec.items():
        rec = tgt.get(key)
        if rec is None or not rec["in_legacy554"]: continue
        L = int(rec["L"]); gt = {(int(i), int(j)) for i, j in rec["gt_contacts"]}
        if not gt: continue
        n += 1
        # per rollout, then averaged: the number already reported
        per = []
        for r, secs in rolls.items():
            M = np.zeros((L, L), np.float32)
            for s in secs:
                for i, j in s:
                    if 0 <= i < L and 0 <= j < L: M[i, j] += 1; M[j, i] += 1
            per.append(metrics_for(M.astype(np.float16), gt, L).get("all:R", np.nan))
        one.append(np.nanmean(per))
        # every section of every rollout, one vote
        M = np.zeros((L, L), np.float32)
        for secs in rolls.values():
            for s in secs:
                for i, j in s:
                    if 0 <= i < L and 0 <= j < L: M[i, j] += 1; M[j, i] += 1
        pooled.append(metrics_for(M.astype(np.float16), gt, L).get("all:R", np.nan))
    np.save(f"/tmp/pooled_{label.replace(chr(32),chr(95)).replace(chr(35),chr(78))}.npy", np.array(pooled)); print(f"{label:<24} n={n:<4} 1 rollout {np.nanmean(one):.4f}   pooled 8 {np.nanmean(pooled):.4f}")
