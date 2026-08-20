"""Step-2 zero-shot conditional probe: does conditioning the BASE model on a
partial set of contacts improve its ranking of the REMAINING contacts?  Tests
whether exploitable JOINT/structural signal exists beyond the per-pair marginal
(Step-1 tie: matrix 0.221 == vote 0.224).

Per protein, cond vs uncond on the *identical* reduced task:
  remaining universe = triu pairs (sep>=6) MINUS given set G;  positives P = GT\G
  R-precision = precision@|P| over remaining; + AUC.
Given-sets:  A. ORACLE-partial (random 50% of TRUE contacts; mechanism ceiling)
             B. NOISY-candidate (one real exp98 rollout's predicted contacts; deployable)
Conditioning = extend the scoring prefix with G emitted as <contact> <pi> <pj>
(shuffled + random orientation), then score remaining pairs.
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np, pandas as pd

REPO = Path("/home/bizon/git/MarinFold/.claude/worktrees/protein-rollout-post-training-e203d8")
EXP82 = REPO / "experiments/exp82_evals_contacts_v1_contact_prediction"
EXP89 = REPO / "experiments/exp89_evals_contacts_v1_model_on_eval_set"
sys.path.insert(0, str(EXP82)); sys.path.insert(0, str(EXP89))
from eval_contact_prediction import Scorer
from score_eval_set import prefix_and_positions
import torch

MODEL = "/home/bizon/exp89_export/hf_step35679"
TARGETS = str(REPO / "experiments/exp98_data_generate_rollouts_contacts_v1_train/data/targets.parquet")
STEP1_CSV = "step1_base_matrix_exp98.csv"
NOISY_PREDS = "noisy_preds.parquet"
N_PROT = int(sys.argv[1]) if len(sys.argv) > 1 else 50
MIN_SEP = 6

def canon(flat):
    a = np.asarray(flat).reshape(-1, 2)
    if a.size == 0: return set()
    lo = np.minimum(a[:,0], a[:,1]); hi = np.maximum(a[:,0], a[:,1])
    k = (hi - lo) >= MIN_SEP
    return set(zip(lo[k].tolist(), hi[k].tolist()))

def auc_np(labels, scores):
    labels = np.asarray(labels, bool)
    npos = int(labels.sum()); nneg = len(labels) - npos
    if npos == 0 or nneg == 0: return np.nan
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), float); ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels].sum() - npos * (npos + 1) / 2) / (npos * nneg))

def sym_from_ids(scorer, ids, positions):
    lp1, lp2 = scorer.contact_logprob_matrix(ids, positions)
    fwd = lp1[:, None] + lp2
    return (0.5 * (fwd + fwd.T)).astype(np.float32)

def ext_ids(scorer, prefix_ids, given, positions, rng):
    order = list(given); rng.shuffle(order)
    ids = list(prefix_ids)
    for (i, j) in order:
        a, b = (i, j) if rng.random() < 0.5 else (j, i)
        ids += [scorer.contact_id, scorer.ptoken(positions[a]), scorer.ptoken(positions[b])]
    return ids

def eval_cond(sym, L, P, G):
    iu, ju = np.triu_indices(L, k=MIN_SEP)
    gd = np.zeros(L * L, bool)
    for (i, j) in G: gd[i * L + j] = True
    pdz = np.zeros(L * L, bool)
    for (i, j) in P: pdz[i * L + j] = True
    keys = iu * L + ju
    rem = ~gd[keys]
    ii, jj, kk = iu[rem], ju[rem], keys[rem]
    scores = sym[ii, jj]
    labels = pdz[kk]
    R = int(labels.sum())
    if R == 0 or R == len(labels): return np.nan, np.nan
    noise = np.random.default_rng(0).random(scores.size) * 1e-4
    order = np.argsort(-(scores + noise))
    return float(labels[order[:R]].mean()), auc_np(labels, scores)

tgt = pd.read_parquet(TARGETS).set_index("entry_id")
eids = [e for e in pd.read_csv(STEP1_CSV).entry_id.tolist() if e in tgt.index][:N_PROT]
print(f"probing {len(eids)} proteins", flush=True)
mp = pd.read_parquet(NOISY_PREDS)
preds_by = {e: [canon(p) for p in g["pred"].to_numpy()] for e, g in mp.groupby("entry_id")}

scorer = Scorer(MODEL)
print(f"model loaded (device={scorer.device})", flush=True)
rng = np.random.default_rng(0)
rows = []; t0 = time.time()
for n, eid in enumerate(eids):
    seq = tgt.loc[eid, "sequence"]; L = int(tgt.loc[eid, "L"])
    built = prefix_and_positions(eid, seq)
    if built is None: continue
    prefix, positions, L = built
    prefix_ids = scorer.tok(prefix, add_special_tokens=False).input_ids
    gt = canon(np.concatenate([np.asarray(p).ravel() for p in tgt.loc[eid, "gt_contacts"]]))
    if len(gt) < 8: continue
    try:
        sym_base = sym_from_ids(scorer, prefix_ids, positions)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache(); continue
    gt_list = sorted(gt)
    for d in range(2):  # A. oracle-partial 50%
        idx = rng.permutation(len(gt_list)); m = len(gt_list) // 2
        G = {gt_list[t] for t in idx[:m]}; P = set(gt_list) - G
        u_r, u_a = eval_cond(sym_base, L, P, G)
        sym_c = sym_from_ids(scorer, ext_ids(scorer, prefix_ids, G, positions, rng), positions)
        c_r, c_a = eval_cond(sym_c, L, P, G)
        rows.append(dict(entry_id=eid, L=L, cond="oracle50", u_R=u_r, c_R=c_r, u_AUC=u_a, c_AUC=c_a, nG=len(G), nP=len(P)))
    cands = preds_by.get(eid, [])
    for d in range(2):  # B. noisy-candidate
        if not cands: break
        G = cands[rng.integers(len(cands))]
        if len(G) < 3: continue
        P = gt - G
        if len(P) < 3: continue
        u_r, u_a = eval_cond(sym_base, L, P, G)
        sym_c = sym_from_ids(scorer, ext_ids(scorer, prefix_ids, G, positions, rng), positions)
        c_r, c_a = eval_cond(sym_c, L, P, G)
        tp = len(G & gt)
        rows.append(dict(entry_id=eid, L=L, cond="noisycand", u_R=u_r, c_R=c_r, u_AUC=u_a, c_AUC=c_a,
                         nG=len(G), nP=len(P), cand_prec=tp / max(1, len(G))))
    torch.cuda.empty_cache()
    if (n + 1) % 10 == 0:
        print(f"  {n+1}/{len(eids)} ({time.time()-t0:.0f}s)", flush=True)

df = pd.DataFrame(rows); df.to_csv("probe_conditional.csv", index=False)
print(f"\nwrote {len(df)} rows -> probe_conditional.csv  (wall {time.time()-t0:.0f}s)\n", flush=True)
for cond, g in df.groupby("cond"):
    dR = g.c_R - g.u_R; dA = g.c_AUC - g.u_AUC
    print(f"=== {cond}  (n={len(g)} evals, {g.entry_id.nunique()} proteins) ===", flush=True)
    print(f"  uncond R={g.u_R.mean():.4f}  ->  cond R={g.c_R.mean():.4f}   dR={dR.mean():+.4f}  (cond wins {100*(dR>0).mean():.0f}%)", flush=True)
    print(f"  uncond AUC={g.u_AUC.mean():.4f} -> cond AUC={g.c_AUC.mean():.4f}  dAUC={dA.mean():+.4f}  (cond wins {100*(dA>0).mean():.0f}%)", flush=True)
    if cond == "noisycand":
        print(f"  (mean candidate precision={g.cand_prec.mean():.3f}, nG={g.nG.mean():.0f})", flush=True)
print("\nDONE", flush=True)
