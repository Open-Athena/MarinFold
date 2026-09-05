# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Pool the ESMFold2 refold shards: design arm against the native control.

The design pass rate is only interpretable next to the native one. Native
sequences refolded onto their OWN AFDB backbones are the ceiling this
measurement can reach, and it is far below 100% because ESMFold2 at 1 sample
/ 100 steps against a whole-chain 2 A gate is a strict test — not because
AFDB backbones are wrong. Report the ratio, not the absolute.
"""

import glob, collections, statistics as st
import pyarrow.parquet as pq

def load(pat):
    rows = []
    for f in sorted(glob.glob(pat)):
        rows += pq.read_table(f).to_pylist()
    return rows

d = load("/data/exp266/design_refold-*.parquet")
n = load("/data/exp266/native_refold-*.parquet")
print(f"design arm: {len(d)} refolds over {len({r['entry_id'] for r in d})} backbones")
print(f"native ctrl: {len(n)} refolds over {len({r['entry_id'] for r in n})} backbones")

def rate(rows, key, thr, gt):
    v = [r[key] for r in rows]
    return (sum((x > thr) if gt else (x < thr) for x in v) / len(v)) if v else float('nan')

print(f"\n{'arm':<8} {'n':>6} {'scRMSD<2A':>10} {'scTM>0.5':>9} {'med RMSD':>9} {'med TM':>7}")
for name, rows in (("design", d), ("native", n)):
    print(f"{name:<8} {len(rows):6d} {rate(rows,'sc_rmsd',2.0,False):9.1%} "
          f"{rate(rows,'sc_tm',0.5,True):8.1%} "
          f"{st.median(r['sc_rmsd'] for r in rows):9.2f} "
          f"{st.median(r['sc_tm'] for r in rows):7.3f}")

if d and n:
    print(f"\ndesign / native ratio: scRMSD {rate(d,'sc_rmsd',2.0,False)/rate(n,'sc_rmsd',2.0,False):.3f}, "
          f"scTM {rate(d,'sc_tm',0.5,True)/rate(n,'sc_tm',0.5,True):.3f}")

# per-backbone best-of-8 for the design arm
by = collections.defaultdict(list)
for r in d: by[r['entry_id']].append(r)
print(f"per-backbone designability (any of 8): "
      f"{sum(any(x['sc_rmsd']<2.0 for x in v) for v in by.values())/len(by):.1%} over {len(by)}")

# by temperature and by length
print(f"\n{'T':>5} {'n':>6} {'scRMSD<2A':>10} {'scTM>0.5':>9}")
bt = collections.defaultdict(list)
for r in d: bt[r['mpnn_temperature']].append(r)
for t in sorted(bt):
    print(f"{t:5.1f} {len(bt[t]):6d} {rate(bt[t],'sc_rmsd',2.0,False):9.1%} {rate(bt[t],'sc_tm',0.5,True):8.1%}")

print(f"\n{'length':>10} {'n_des':>6} {'des TM>0.5':>11} {'n_nat':>6} {'nat TM>0.5':>11}")
def binof(L):
    for lo,hi in ((0,100),(100,200),(200,400),(400,10**6)):
        if lo<=L<hi: return f"{lo}-{hi if hi<10**6 else '+'}"
bd, bn = collections.defaultdict(list), collections.defaultdict(list)
for r in d: bd[binof(r['seq_len'])].append(r)
for r in n: bn[binof(r['seq_len'])].append(r)
for k in sorted(set(bd)|set(bn), key=lambda x: int(x.split('-')[0])):
    print(f"{k:>10} {len(bd.get(k,[])):6d} {rate(bd[k],'sc_tm',0.5,True) if bd.get(k) else float('nan'):10.1%} "
          f"{len(bn.get(k,[])):6d} {rate(bn[k],'sc_tm',0.5,True) if bn.get(k) else float('nan'):10.1%}")
