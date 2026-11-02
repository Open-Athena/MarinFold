"""Calibrate arm M-KS's `beta_shape` against a measured spread — issue #237.

`beta_shape` multiplies `(m_k - mean_k m)`, where m is the causal prefix
marginal, and it is added to a base that is GRPO-standardised (unit spread by
construction). So the only question is: how big is the shaping term's own spread
in raw units? #208 mis-set `lam_doc` twice by guessing this, in both directions.
"""
import glob, sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
sys.path.insert(0, "/home/ubuntu/exp237/skyrl")
import section_rewards as sr

tgt = {(r["dataset"], r["stem"]): r for r in
       pq.read_table(str(Path.home()/"exp230_data"/"eval577_targets.parquet")).to_pylist()}
root = sys.argv[1]
rows = {}
for p in sorted(glob.glob(f"{root}/**/*.parquet", recursive=True))[:6]:
    for r in pq.read_table(p).to_pylist():
        if r["sec_idx"] < 0: continue
        rows.setdefault((r["dataset"], r["stem"], r["r"]), {})[r["sec_idx"]] = \
            {(int(i), int(j)) for i, j in r["contacts"]}

spreads, ranges, ks, totals = [], [], [], []
for (ds, stem, _), secs in list(rows.items())[:400]:
    rec = tgt.get((ds, stem))
    if rec is None: continue
    gt = {(int(i), int(j)) for i, j in rec["gt_contacts"]}
    if not gt: continue
    sections = [secs[i] for i in sorted(secs)]
    m = sr.prefix_marginals(sections, gt, int(rec["L"]))
    if len(m) < 2: continue
    c = m - m.mean()
    spreads.append(float(c.std())); ranges.append(float(c.max() - c.min()))
    ks.append(len(m)); totals.append(float(m.sum()))

s = np.array(spreads)
print(f"n_rollouts {len(s)}   sections/rollout {np.mean(ks):.1f}")
print(f"centred prefix marginal:  sd {np.median(s):.4f} (median)  {s.mean():.4f} (mean)")
print(f"                          range {np.median(ranges):.4f} (median)")
print(f"telescoped sum = C(all):  {np.mean(totals):.4f}   <- sanity, should be ~0.5")
for target in (0.25, 0.5, 1.0):
    print(f"beta for a shaping sd of {target:.2f} base-units:  {target/np.median(s):6.1f}")
