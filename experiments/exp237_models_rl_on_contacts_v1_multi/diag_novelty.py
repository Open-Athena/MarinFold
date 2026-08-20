"""Does the prefix marginal actually reward NOVELTY? — issue #237.

Arm M-KS2's whole motivation is that a section should be paid for covering what
its predecessors missed. M-KS2's Jaccard came out consistently HIGHER than
M-K's, which is the opposite. This checks the premise directly: correlate each
section's prefix marginal against how novel it actually is.
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

nov, rep_true, marg = [], [], []
for (ds, stem, _), secs in list(rows.items())[:500]:
    rec = tgt.get((ds, stem))
    if rec is None: continue
    gt = {(int(i), int(j)) for i, j in rec["gt_contacts"]}
    if not gt: continue
    sections = [secs[i] for i in sorted(secs)]
    m = sr.prefix_marginals(sections, gt, int(rec["L"]))
    seen = set()
    for k, s in enumerate(sections):
        if k == 0 or not s:
            seen |= s; continue
        new = s - seen
        # novelty: share of this section's pairs never emitted before
        nov.append(len(new) / len(s))
        # "piling on": share that are TRUE pairs already emitted before
        rep_true.append(len(s & seen & gt) / len(s))
        marg.append(float(m[k]))
        seen |= s

nov, rep_true, marg = map(np.array, (nov, rep_true, marg))
ok = np.isfinite(marg)
print(f"n_sections {ok.sum()}")
print(f"corr(prefix marginal, NOVELTY)              {np.corrcoef(marg[ok], nov[ok])[0,1]:+.3f}")
print(f"corr(prefix marginal, REPEATED TRUE pairs)  {np.corrcoef(marg[ok], rep_true[ok])[0,1]:+.3f}")
print()
hi = marg > np.percentile(marg[ok], 90)
lo = marg < np.percentile(marg[ok], 10)
print(f"top-decile-marginal sections: novelty {nov[hi].mean():.3f}  repeated-true {rep_true[hi].mean():.3f}")
print(f"bot-decile-marginal sections: novelty {nov[lo].mean():.3f}  repeated-true {rep_true[lo].mean():.3f}")
