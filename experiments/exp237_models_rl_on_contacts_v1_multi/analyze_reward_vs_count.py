"""How does each arm reward depend on the NUMBER of sections a rollout emits?"""
import glob, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
sys.path.insert(0, str(Path.home()/"exp237"/"skyrl"))
import consensus as cs

SIZES=[1,2,4,8,16,22]; rng=np.random.default_rng(237)
def load(root):
    by=defaultdict(lambda: defaultdict(dict))
    for p in sorted(glob.glob(f"{root}/**/*.parquet", recursive=True)):
        for r in pq.read_table(p).to_pylist():
            if r["sec_idx"]<0: continue
            by[(r["dataset"],r["stem"])][r["r"]][r["sec_idx"]]={(int(i),int(j)) for i,j in r["contacts"]}
    return {k:{r:[s[i] for i in sorted(s)] for r,s in v.items()} for k,v in by.items()}
def f1(pred,gt):
    if not pred or not gt: return 0.0
    tp=len(pred&gt); p,r=tp/len(pred),tp/len(gt)
    return 2*p*r/(p+r) if p+r else 0.0
def C(secs,is_true,pos,n,nt):
    if not secs: return cs.rprecision(np.zeros(n,np.int32),is_true,nt)
    return cs.rprecision(cs.vote_counts(secs,pos,n).sum(0),is_true,nt)

tgt={(r["dataset"],r["stem"]):r for r in pq.read_table(str(Path.home()/"exp230_data"/"eval577_targets.parquet")).to_pylist()}
sec=load(str(Path.home()/"exp230_data"/"eval"/"agg_sections"))
keys=sorted(sec); rng.shuffle(keys)
raw={k:defaultdict(list) for k in ["M-B max_k F1","M-F F1(last)","M-C mean m_k"]}
adv={k:defaultdict(list) for k in raw}
ng=0
for key in keys:
    if ng>=120: break
    rec=tgt.get(key)
    if rec is None: continue
    L=int(rec["L"]); gt={(int(i),int(j)) for i,j in rec["gt_contacts"]}
    if not gt: continue
    pairs,pos=cs.candidate_index(L); is_true=cs.truth_mask(pairs,gt); nt=int(is_true.sum())
    if nt<=0: continue
    rolls=[s for s in sec[key].values() if len(s)>=max(SIZES)]
    if len(rolls)<len(SIZES): continue
    per={}
    ok=True
    for s_,k in zip(rolls,SIZES):
        s=s_[:k]; ca=C(s,is_true,pos,len(pairs),nt)
        if np.isnan(ca): ok=False; break
        m=np.array([ca-C(s[:i]+s[i+1:],is_true,pos,len(pairs),nt) for i in range(k)])
        f=[f1(x,gt) for x in s]
        per[k]=dict(b=max(f), l=f[-1], c=float(np.nan_to_num(m).mean()))
    if not ok or len(per)<len(SIZES): continue
    ng+=1
    for name,fld in [("M-B max_k F1","b"),("M-F F1(last)","l"),("M-C mean m_k","c")]:
        v=np.array([per[k][fld] for k in SIZES])
        for k,x in zip(SIZES,v): raw[name][k].append(float(x))
        if v.std()>0:
            z=(v-v.mean())/v.std(ddof=1)
            for k,x in zip(SIZES,z): adv[name][k].append(float(x))
print(f"{ng} rollouts truncated to each size\n")
print(f"{chr(39)+chr(39):>16}"+ "".join(f"{k:>9}" for k in SIZES) + "   direction")
for name in raw:
    r=[np.mean(raw[name][k]) for k in SIZES]
    d="RISES with count" if r[-1]>1.15*r[0] else ("FALLS with count" if r[-1]<0.85*r[0] else "FLAT in count")
    print(f"{name:>16}"+"".join(f"{x:>9.4f}" for x in r)+f"   {d}")
print("\ngroup-centred advantage (groups differing ONLY in section count):")
for name in adv:
    a=[np.mean(adv[name][k]) for k in SIZES]
    print(f"{name:>16}"+"".join(f"{x:>+9.3f}" for x in a))
