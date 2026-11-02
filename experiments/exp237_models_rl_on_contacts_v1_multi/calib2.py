import glob, sys
from pathlib import Path
from collections import defaultdict
import numpy as np, pyarrow.parquet as pq
sys.path.insert(0, "/home/ubuntu/exp237/skyrl")
import section_rewards as sr
tgt = {(r["dataset"], r["stem"]): r for r in
       pq.read_table(str(Path.home()/"exp230_data"/"eval577_targets.parquet")).to_pylist()}
groups = defaultdict(dict)
for p in sorted(glob.glob(f"{sys.argv[1]}/**/*.parquet", recursive=True))[:6]:
    for r in pq.read_table(p).to_pylist():
        if r["sec_idx"] < 0: continue
        groups[(r["dataset"], r["stem"])].setdefault(r["r"], {})[r["sec_idx"]] = \
            {(int(i), int(j)) for i, j in r["contacts"]}
plain_sd, fixed_sd, fixed_slope = [], [], []
for key, reps in list(groups.items())[:120]:
    rec = tgt.get(key)
    if rec is None: continue
    gt = {(int(i), int(j)) for i, j in rec["gt_contacts"]}
    if not gt or len(reps) < 2: continue
    marg = {r: sr.prefix_marginals([s[i] for i in sorted(s)], gt, int(rec["L"]))
            for r, s in reps.items()}
    base = sr.positional_baseline(marg)
    for r, m in marg.items():
        if len(m) < 6: continue
        plain_sd.append(float((m - m.mean()).std()))
        f = m - base[:len(m)]
        f = f - f.mean()
        fixed_sd.append(float(f.std()))
        fixed_slope.append(float(np.polyfit(np.arange(len(f)), f, 1)[0]))
ps, fs = np.median(plain_sd), np.median(fixed_sd)
print(f"n={len(fixed_sd)}  plain sd {ps:.4f}   positional-corrected sd {fs:.4f}  ({ps/fs:.1f}x smaller)")
print(f"residual slope in k: mean {np.mean(fixed_slope):+.6f}  negative in "
      f"{100*np.mean(np.array(fixed_slope)<0):.0f}% of rollouts   (was 100%)")
for t in (0.25, 0.5):
    print(f"beta for shaping sd {t:.2f} base-units:  {t/fs:5.1f}   (old term: {t/ps:5.1f})")
