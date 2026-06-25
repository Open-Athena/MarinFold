# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Probe: does ensembling over resampled sequence definitions help?

contacts-v1 randomizes two nuisances per document: the n-term **start position**
and the **order** of the ``<pX> <AA>`` statements. The eval uses one realization
(seed = entry_id). Here we draw K realizations per protein, score each, and
compare a single realization to the **ensemble mean** P(contact) — measuring
(a) how sensitive the model is to the nuisance and (b) whether averaging helps
AUC / R-precision.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from eval_contact_prediction import MIN_SEP, NUM_POS, BEGIN, Scorer
from marinfold.document_structures.contacts_v1 import (
    GenerationConfig, build_document, residues_from_sequence,
)

EXP78 = Path("/home/bizon/git/MarinFold-exp78/experiments/exp78_evals_esmfold_contacts")


def seqs():
    out = {}
    for m in ("eval_manifest_foldbench.csv", "eval_manifest_exp65.csv"):
        for _, r in pd.read_csv(EXP78 / "data" / m).iterrows():
            out[(r["dataset"], r["stem"])] = r["input_seq"]
    return out


def pcontact(scorer, seq, entry_id):
    """Model P(contact)[i,j] (seq-index space) for one resampled definition."""
    res = residues_from_sequence(seq)
    r = build_document(entry_id, res, [], config=GenerationConfig())
    L = r.seq_len
    pos = [(r.n_term_index + k) % NUM_POS for k in range(L)]
    prefix = r.document[: r.document.index(BEGIN) + len(BEGIN)]
    pid = scorer.tok(prefix, add_special_tokens=False).input_ids
    lp1, lp2 = scorer.contact_logprob_matrix(pid, pos)
    fwd = lp1[:, None] + lp2
    return np.exp(fwd) + np.exp(fwd.T)


def true_mat(L, contacts):
    m = np.zeros((L, L), bool)
    for i, j, d in contacts:
        if d >= 0.001 and (j - i) >= MIN_SEP and i < j < L:
            m[i, j] = m[j, i] = True
    return m


def metrics(P, tmat, resolved):
    a, b = np.triu_indices(len(resolved), k=1)
    i, j = resolved[a], resolved[b]
    sep = j - i
    s, g = P[i, j], tmat[i, j].astype(int)
    out = {}
    for name, lo in (("all", 6), ("long", 24)):
        inr = sep >= lo
        ss, gg = s[inr], g[inr]
        if gg.sum() == 0 or gg.sum() == gg.size:
            out[f"auc_{name}"] = np.nan; out[f"rprec_{name}"] = np.nan; continue
        out[f"auc_{name}"] = roc_auc_score(gg, ss)
        R = int(gg.sum())
        out[f"rprec_{name}"] = gg[np.argsort(-ss, kind="mergesort")][:R].sum() / R
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gt", type=Path, default=Path("data/gt_universe.jsonl"))
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("-n", type=int, default=10)
    ap.add_argument("--lmin", type=int, default=50)
    ap.add_argument("--lmax", type=int, default=120)
    args = ap.parse_args()

    S = seqs()
    gt = [json.loads(l) for l in args.gt.open()]
    gt = [r for r in gt if args.lmin <= r["L"] <= args.lmax
          and sum(1 for i, j, d in r["contacts"] if d >= 0.001 and j - i >= 24) >= 5]
    gt = gt[: args.n]
    scorer = Scorer(args.model)
    print(f"K={args.k} resamples; {len(gt)} proteins (L in [{args.lmin},{args.lmax}])\n")

    rows = []
    for rec in gt:
        stem, L = rec["stem"], rec["L"]
        seq = S[(rec["dataset"], stem)]
        resolved = np.asarray(rec["resolved"], np.int64)
        tmat = true_mat(L, rec["contacts"])
        Ps = [pcontact(scorer, seq, f"{stem}#{k}") for k in range(args.k)]
        per = [metrics(P, tmat, resolved) for P in Ps]
        ens = metrics(np.mean(Ps, axis=0), tmat, resolved)
        for key in ("auc_all", "auc_long", "rprec_long"):
            vals = np.array([p[key] for p in per])
            rows.append(dict(stem=stem, L=L, metric=key,
                             single_mean=np.nanmean(vals), single_std=np.nanstd(vals),
                             ensemble=ens[key]))
        a = [r for r in rows if r["stem"] == stem]
        print(f"{stem:>10} L={L:>3}  " + "  ".join(
            f"{r['metric']}: single {r['single_mean']:.3f}±{r['single_std']:.3f} -> ens {r['ensemble']:.3f}"
            for r in a))

    df = pd.DataFrame(rows)
    print("\n=== AGGREGATE (mean over proteins) ===")
    for key in ("auc_all", "auc_long", "rprec_long"):
        d = df[df.metric == key]
        print(f"{key:>11}: single-realization {d.single_mean.mean():.3f} "
              f"(within-protein std {d.single_std.mean():.3f})  ->  ensemble {d.ensemble.mean():.3f}  "
              f"(Δ {d.ensemble.mean() - d.single_mean.mean():+.3f})")


if __name__ == "__main__":
    raise SystemExit(main())
