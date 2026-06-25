# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Head-to-head (exp82): the PRIOR contacts-and-distances-v1 1.5B model's contact
predictions on the SAME held-out test proteins, for comparison with the
contacts-v1 model (see `eval_contact_prediction.py` and the README).

The prior model uses the contacts-and-distances-v1 format + the protein-docs
tokenizer (HF export in the bucket at
`checkpoints/protein-contacts-1_5b-distance-masked-70f8f5/step-49999/`). It was
distance-masked, so it predicts contacts two ways — we try BOTH:

* ``statements`` — its native CONTACT statement: for each candidate pair (i, j),
  score P(`<{range}-range-contact> <pi> <pj>` | sequence), where the range token
  is set by the pair's sequence separation and positions are 0-based sequence
  indices. (Geo-mean symmetrized, like the contacts-v1 ``pairwise`` method.)
* ``distance`` — its distance readout: P(CA–CA distance ≤ 8 Å) for each pair,
  i.e. the summed probability of the first 16 distance bins after the tail
  `<distance> <pi> <pj> <CA> <CA>`.

Same proteins / ground truth / metrics as `eval_contact_prediction.py`.

CAVEAT: the two models use different contact *definitions* (contacts-v1 =
pyconfind side-chain contact; the prior model = CB–CB ≤ 8 Å). We score both
against the contacts-v1 ground truth, so the contacts-v1 model has a
home-field edge — read the comparison as directional.

Run on GPU via the same venv as the contacts-v1 eval; fetch the prior model::

    hf buckets cp -r hf://buckets/open-athena/MarinFold/checkpoints/\
protein-contacts-1_5b-distance-masked-70f8f5/step-49999 ./prior_model
    uv run python eval_prior_model_contacts.py --model ./prior_model -n 24
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from eval_contact_prediction import (
    LONG, MED, MIN_SEP, Scorer, aggregate, candidate_pairs, load_proteins, metrics,
)

CD_DOC, BEGIN_SEQ, BEGIN_ST = "<contacts-and-distances-v1>", "<begin_sequence>", "<begin_statements>"
# 64 distance bins <d0.5>..<d32.0>; contact = CA-CA <= 8 A = first 16 bins.
DIST_TOKENS = [f"<d{i * 0.5:.1f}>" for i in range(1, 65)]
CONTACT_BINS = 16


def range_token(s):
    return "<long-range-contact>" if s >= LONG else ("<medium-range-contact>" if s >= MED else "<short-range-contact>")


def prior_prefix_ids(scorer, p):
    """contacts-and-distances-v1 prefix: doc-type, begin_sequence, ordered
    residue names, begin_statements (positions are implicit 0-based seq order)."""
    toks = [CD_DOC, BEGIN_SEQ, *p.residues, BEGIN_ST]
    return scorer.tok.convert_tokens_to_ids(toks)


def rank_statements(scorer, p):
    prefix_ids = prior_prefix_ids(scorer, p)
    positions = list(range(p.L))                      # ptoken(i) -> <pi>, 0-based
    mats = {}
    for rt in ("<long-range-contact>", "<medium-range-contact>", "<short-range-contact>"):
        rid = scorer.tok.convert_tokens_to_ids(rt)
        lp1, lp2 = scorer.contact_logprob_matrix(prefix_ids, positions, contact_id=rid)
        fwd = lp1[:, None] + lp2
        mats[rt] = 0.5 * (fwd + fwd.T)                 # symmetrize
    scored = [(mats[range_token(abs(i - j))][i, j], (i, j)) for (i, j) in candidate_pairs(p.L)]
    scored.sort(reverse=True)
    return [pr for _, pr in scored]


@torch.no_grad()
def rank_distance(scorer, p):
    """Rank pairs by P(CA-CA <= 8 A) from the <distance> readout."""
    prefix_ids = prior_prefix_ids(scorer, p)
    dist_id = scorer.tok.convert_tokens_to_ids("<distance>")
    ca_id = scorer.tok.convert_tokens_to_ids("<CA>")
    dbin_ids = [scorer.tok.convert_tokens_to_ids(t) for t in DIST_TOKENS]
    pairs = candidate_pairs(p.L)
    # tail = <distance> <pi> <pj> <CA> <CA>; score dist over <d*> at last position
    seqs = [list(prefix_ids) + [dist_id, scorer.ptoken(i), scorer.ptoken(j), ca_id, ca_id]
            for (i, j) in pairs]
    probs = np.empty(len(pairs), np.float32)
    for s in range(0, len(seqs), scorer.batch):
        chunk = seqs[s:s + scorer.batch]
        X = torch.tensor(chunk, device=scorer.device)
        logits = scorer.model(X).logits[:, -1].float()
        p_all = F.softmax(logits, -1)[:, dbin_ids].cpu().numpy()   # [b, 64]
        probs[s:s + len(chunk)] = p_all[:, :CONTACT_BINS].sum(1)
    order = np.argsort(-probs)
    return [pairs[k] for k in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("-n", type=int, default=24)
    ap.add_argument("--min-len", type=int, default=50)
    ap.add_argument("--max-len", type=int, default=150)
    ap.add_argument("--methods", default="statements,distance")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    methods = args.methods.split(",")
    scorer = Scorer(args.model)
    # sanity: the prior format's special tokens must exist in this tokenizer
    for t in (CD_DOC, "<long-range-contact>", "<distance>", "<CA>", DIST_TOKENS[0]):
        assert scorer.tok.convert_tokens_to_ids(t) != scorer.tok.unk_token_id, f"missing token {t}"
    proteins = load_proteins(args.n, args.min_len, args.max_len)
    print(f"PRIOR contacts-and-distances-v1 1.5B on {len(proteins)} held-out test proteins; "
          f"methods={methods}\n", flush=True)

    results = {m: [] for m in [*methods, "random"]}
    for p in proteins:
        rng = np.random.default_rng(args.seed)
        rankings = {}
        try:
            if "statements" in methods:
                rankings["statements"] = rank_statements(scorer, p)
            if "distance" in methods:
                rankings["distance"] = rank_distance(scorer, p)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); print(f"{p.entry:>22} L={p.L:>3} -- SKIPPED (OOM)", flush=True); continue
        rand = candidate_pairs(p.L); rng.shuffle(rand)
        rankings["random"] = rand
        line = [f"{p.entry:>22} L={p.L:>3} gt={len(p.gt):>3}"]
        for m, rk in rankings.items():
            mm = metrics(rk, p.gt, p.L); mm["entry"] = p.entry
            results[m].append(mm)
            line.append(f"{m}:long_P@L=" + (f"{mm['long_P@L']:.2f}" if not np.isnan(mm["long_P@L"]) else "n/a"))
        print("  ".join(line), flush=True)
        torch.cuda.empty_cache()

    print("\n=== PRIOR 1.5B AGGREGATE (mean over proteins) ===")
    cols = ["long_P@L", "long_P@L2", "long_P@L5", "medlong_P@L", "medlong_P@L2", "medlong_P@L5", "P@ngt"]
    print(f"{'method':<14} " + " ".join(f"{c:>12}" for c in cols))
    for m in [*methods, "random"]:
        if not results[m]:
            continue
        a = aggregate(results[m])
        print(f"{m:<14} " + " ".join(f"{a.get(c, float('nan')):>12.3f}" for c in cols))


if __name__ == "__main__":
    main()
