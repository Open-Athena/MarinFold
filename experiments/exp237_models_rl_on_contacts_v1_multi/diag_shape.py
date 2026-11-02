"""Is arm M-KS's shaping term a disguised 'stop early' signal? — issue #237.

The zero-sum construction guarantees the shaping cannot move the rollout's TOTAL
advantage at any section count. It says nothing about the shape *within* the
rollout — and a section owns the `<begin_statements>` token that OPENS it, so a
term that decreases in k penalises the decision to write another candidate.
"""
import glob, sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
sys.path.insert(0, "/home/ubuntu/exp237/skyrl")
import section_rewards as sr

tgt = {(r["dataset"], r["stem"]): r for r in
       pq.read_table(str(Path.home()/"exp230_data"/"eval577_targets.parquet")).to_pylist()}
rows = {}
for p in sorted(glob.glob(f"{sys.argv[1]}/**/*.parquet", recursive=True))[:6]:
    for r in pq.read_table(p).to_pylist():
        if r["sec_idx"] < 0: continue
        rows.setdefault((r["dataset"], r["stem"], r["r"]), {})[r["sec_idx"]] = \
            {(int(i), int(j)) for i, j in r["contacts"]}

by_pos, slopes = {}, []
for (ds, stem, _), secs in list(rows.items())[:600]:
    rec = tgt.get((ds, stem))
    if rec is None: continue
    gt = {(int(i), int(j)) for i, j in rec["gt_contacts"]}
    if not gt: continue
    sections = [secs[i] for i in sorted(secs)]
    m = sr.prefix_marginals(sections, gt, int(rec["L"]))
    if len(m) < 6: continue
    c = m - m.mean()                       # exactly what beta multiplies
    for k, v in enumerate(c):
        by_pos.setdefault(k, []).append(float(v))
    slopes.append(np.polyfit(np.arange(len(c)), c, 1)[0])

print("centred shaping term  (m_k - mean_k m)  by section position k:")
for k in [0, 1, 2, 4, 8, 12, 16, 20, 24]:
    v = by_pos.get(k)
    if v: print(f"  k={k:3d}   mean {np.mean(v):+.4f}   (n={len(v)})")
sl = np.array(slopes)
print(f"\nper-rollout slope in k: mean {sl.mean():+.5f}  median {np.median(sl):+.5f}  "
      f"negative in {100*np.mean(sl<0):.0f}% of rollouts")
print(f"beta=3 turns that into {3*np.mean(by_pos[0]):+.3f} on the FIRST section's tokens and "
      f"{3*np.mean(by_pos[20]):+.3f} on the 21st's — against a base of ~1 unit.")
