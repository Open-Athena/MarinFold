"""Calibrate arm M-KP's beta against the measured per-token spread — issue #237.

The base is GRPO-standardised (unit spread by construction), so the only question
is the spread of the shaping vector in the units the loss actually reads: over
RESPONSE TOKENS, which is what `loss_reduction=token_mean` averages.
"""
import glob, sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
sys.path.insert(0, "/home/ubuntu/exp237/skyrl")
import section_rewards as sr, contact_rewards as cr

v = cr.DEFAULT_VOCAB
tgt = {(r["dataset"], r["stem"]): r for r in
       pq.read_table(str(Path.home()/"exp230_data"/"eval577_targets.parquet")).to_pylist()}
rows = {}
for p in sorted(glob.glob(f"{sys.argv[1]}/**/*.parquet", recursive=True))[:6]:
    for r in pq.read_table(p).to_pylist():
        if r["sec_idx"] < 0: continue
        rows.setdefault((r["dataset"], r["stem"], r["r"]), {})[r["sec_idx"]] = \
            {(int(i), int(j)) for i, j in r["contacts"]}

sds, fracs, means = [], [], []
for (ds, stem, _), secs in list(rows.items())[:400]:
    rec = tgt.get((ds, stem))
    if rec is None: continue
    gt = {(int(i), int(j)) for i, j in rec["gt_contacts"]}
    if not gt: continue
    # Rebuild a response from the sections: <begin> then triples, then <end>.
    ids, m = [], {i: i for i in range(int(rec["L"]))}
    for k, s in enumerate(sorted(secs)):
        if k: ids.append(v.begin_id)
        for (a, b) in sorted(secs[s]):
            if a < v.n_positions and b < v.n_positions:
                ids += [v.contact_id, v.p0_id + a, v.p0_id + b]
    ids.append(v.end_id)
    adv = sr.pair_token_advantages(ids, m, gt)
    if not len(adv): continue
    sds.append(float(adv.std()))
    fracs.append(float(np.mean(adv != 0)))
    means.append(float(adv.mean()))

sd = np.median(sds)
print(f"n_rollouts {len(sds)}   tokens carrying shaping: {100*np.median(fracs):.0f}%")
print(f"per-token shaping sd (median): {sd:.5f}   mean {np.median(means):+.2e} (should be ~0)")
for t in (0.25, 0.5, 1.0):
    print(f"beta for a shaping sd of {t:.2f} base-units: {t/sd:7.1f}")
