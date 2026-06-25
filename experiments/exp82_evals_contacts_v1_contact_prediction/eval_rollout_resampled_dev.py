# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Resampled rollout (rollout × document-resampling) on the FoldBench dev set.

Simple rollout draws N sampled completions from **one** contacts-v1 document
realization (a fixed N-terminus start + statement order). exp89 showed P(contact)
also varies with that nuisance, and averaging over it (K=10 TTA) added +0.05
R-precision to *pairwise*. This stacks the two ensembling effects: each rollout
gets a **fresh realization** (``build_document`` with a per-rollout entry_id), so
the frequency vote averages over **both** the sampling noise and the document
nuisance.

~No extra GPU cost vs simple rollout: every realization of a protein has the
*same* prefix length, so batched generation is just as efficient — only the
per-rollout ``build_document`` + tokenize is added (~seconds total). See
``Scorer.sample_completions``.

Compares pairwise / rollout / rollout+resample on the 16 dev proteins at the
default sampling settled on by the sweep (T=1.0, p=0.95, k=50). Run::

    PYTHONPATH=<repo>/marinfold uv run python eval_rollout_resampled_dev.py \
        --model /home/bizon/exp89_export/hf_step35679 --n-rollouts 100
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np
import torch

from eval_contact_prediction import (
    BEGIN,
    CONTACT_RE,
    MIN_SEP,
    NUM_POS,
    Protein,
    Scorer,
    aggregate,
    candidate_pairs,
    metrics,
    rank_pairwise,
    rank_rollout,
)
from eval_search_foldbench_dev import DEV_JSONL
from marinfold.document_structures.contacts_v1 import (
    GenerationConfig,
    build_document,
    residues_from_sequence,
)

COLS = ["long_P@L", "long_P@L2", "long_P@L5", "long_R", "medlong_P@L", "medlong_R", "P@ngt"]
METHODS = ["pairwise", "rollout", "rollout+resample"]


def realization(stem, residues, tag):
    """One contacts-v1 realization: prefix string + per-seq-index position list."""
    res = build_document(f"{stem}:{tag}", residues, [], config=GenerationConfig())
    L = res.seq_len
    nterm = res.n_term_index
    seq_positions = [(nterm + k) % NUM_POS for k in range(L)]
    prefix = res.document[: res.document.index(BEGIN) + len(BEGIN)]
    return prefix, seq_positions, L


def rank_rollout_resampled(scorer, stem, residues, resolved, n_rollouts,
                           temperature, top_p, top_k, batch):
    """Rollout where every sampled completion uses a fresh document realization."""
    prefixes, maps, L = [], [], None
    for r in range(n_rollouts):
        prefix, seq_positions, L = realization(stem, residues, f"r{r}")
        prefixes.append(scorer.tok(prefix, add_special_tokens=False).input_ids)
        maps.append({pos: i for i, pos in enumerate(seq_positions)})
    max_new = min(8192 - len(prefixes[0]), 4 * L + 64)
    texts = scorer.sample_completions(prefixes, temperature, top_p, max_new,
                                      batch=batch, top_k=top_k)
    counts = Counter()
    for text, seqidx in zip(texts, maps):                  # parse each with ITS realization's map
        seen = set()
        for a, b in CONTACT_RE.findall(text):
            ia, ib = seqidx.get(int(a)), seqidx.get(int(b))
            if ia is None or ib is None or ia == ib:
                continue
            key = (min(ia, ib), max(ia, ib))
            if abs(ia - ib) >= MIN_SEP and key not in seen:
                seen.add(key)
                counts[key] += 1
    return sorted(candidate_pairs(L, resolved), key=lambda pr: -counts.get(pr, 0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-rollouts", type=int, default=100)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    records = [json.loads(line) for line in DEV_JSONL.open()]
    scorer = Scorer(args.model)
    print(f"{len(records)} FoldBench dev proteins; n_rollouts={args.n_rollouts} "
          f"batch={args.batch} T={args.temperature} p={args.top_p} k={args.top_k}\n", flush=True)

    res = {m: [] for m in METHODS}
    for d in records:
        residues = residues_from_sequence(d["input_seq"])
        prefix, seq_positions, L = realization(d["stem"], residues, "r0")  # canonical single realization
        resolved = frozenset(d["resolved"])
        gt = {frozenset((i, j)) for (i, j) in d["contacts"]
              if abs(i - j) >= MIN_SEP and i in resolved and j in resolved}
        p = Protein(d["stem"], prefix, L, seq_positions[0], seq_positions, gt, [], resolved)

        res["pairwise"].append(dict(metrics(rank_pairwise(scorer, p), gt, L)))
        torch.manual_seed(args.seed)
        rk_simple = rank_rollout(scorer, p, args.n_rollouts, args.temperature, args.top_p,
                                 top_k=args.top_k, batch=args.batch)
        res["rollout"].append(dict(metrics(rk_simple, gt, L)))
        torch.manual_seed(args.seed)
        rk_resamp = rank_rollout_resampled(scorer, d["stem"], residues, resolved, args.n_rollouts,
                                           args.temperature, args.top_p, args.top_k, args.batch)
        res["rollout+resample"].append(dict(metrics(rk_resamp, gt, L)))
        print(f"  {d['stem']:>9} L={L:>3}  simple_lR={res['rollout'][-1]['long_R']:.3f}  "
              f"resample_lR={res['rollout+resample'][-1]['long_R']:.3f}", flush=True)
        torch.cuda.empty_cache()

    print(f"\n=== AGGREGATE (mean over {len(records)} dev proteins) ===")
    print(f"{'method':<18} " + " ".join(f"{c:>11}" for c in COLS))
    for m in METHODS:
        a = aggregate(res[m])
        print(f"{m:<18} " + " ".join(f"{a.get(c, float('nan')):>11.3f}" for c in COLS))
    print("\n(long = sep>=24, medlong = >=12; R = R-precision = precision at top-#GT in band)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
