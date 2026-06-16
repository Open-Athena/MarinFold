# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Contact-prediction eval harness (exp82) for the contacts-v1 1.5B model.

Evaluates the model trained in exp67 (issue #67;
``protein-contacts-1_5b-3.5e-4-contacts-v1-unmasked-3b5cf2``, published to the
open-athena bucket at ``…/hf/step-11999/``). See issue #82.

Given a contacts-v1 document's *sequence section* (the residues + termini, which
fully specify the protein) the model must predict the *contacts*. Free-running
generation is degenerate for an unordered set (greedy loops, sampling ≈ random),
so we instead score/rank candidate residue pairs and report standard
contact-prediction metrics (precision of the top-{L, L/2, L/5} long-range
contacts). Ground truth comes straight from the document text — no pyconfind.

Three inference variants (``--methods``):

* ``pairwise`` — for every candidate pair (i, j) score the autoregressive
  probability of the contact statement and rank by it. Symmetrized as the
  geometric mean of P(i)·P(j|i) and P(j)·P(i|j) (the document randomizes pair
  orientation, so neither direction is privileged).

* ``rollout`` — draw ``--n-rollouts`` sampled completions of the contact
  section, parse the contacts from each, and rank pairs by how often they occur
  (frequency / ensemble voting).

* ``iterative`` — exp27-style growing-K refinement: rank pairs, *commit* the
  top-K as a prefix, re-score the remaining pairs conditioned on the committed
  contacts (joint structure the marginals miss), repeat over a growing-K
  schedule. (exp27 did this for distogram readout; here it's contacts-only.)

A ``random`` baseline (shuffle the candidate ranking) is always reported for
context.

Run on a CUDA box via this experiment's venv (``uv venv && uv sync`` against the
``pyproject.toml`` here, which pins a +cu121 torch). See the README "Running"
section for fetching the model::

    uv run python eval_contact_prediction.py --model ./model \
        --methods pairwise,rollout,iterative -n 25 --n-rollouts 100
"""
from __future__ import annotations

import argparse
import os
import re
from collections import Counter
from dataclasses import dataclass

# Reduce CUDA fragmentation (the iterative method's growing committed prefix
# makes the activation footprint vary a lot between forwards). Must be set
# before torch initializes CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gcsfs
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# contacts-v1 constants (see marinfold .../contacts_v1/{vocab,generate}.py).
NUM_POS = 2000
MIN_SEP = 6  # contacts-v1 min_seq_separation: pairs closer than this are never contacts.
LONG, MED = 24, 12  # CASP sequence-separation ranges (long ≥24, medium 12-23).
TEST_SHARD = (
    "marin-us-east5/protein-structure/MarinFold/exp53_contacts_v1_5x/"
    "documents/test/contacts_v1-00000-of-00022.parquet"
)
CONTACT_RE = re.compile(r"<contact>\s+<p(\d+)>\s+<p(\d+)>")
BEGIN = "<begin_statements>"


@dataclass
class Protein:
    entry: str
    prefix: str          # sequence section incl. <begin_statements> (contacts-v1 format)
    L: int               # number of residues
    nterm: int           # position index of the N-terminus
    seq_positions: list[int]   # contacts-v1 position index per sequence index 0..L-1
    gt: set              # ground-truth contacts as frozenset({seq_i, seq_j})
    residues: list[str]  # residue token (e.g. "<GLY>") per sequence index 0..L-1
                         # (lets a different format, e.g. contacts-and-distances-v1,
                         #  rebuild the same protein's prefix — see the prior-model eval)


# --------------------------------------------------------------------------- #
# Parsing                                                                      #
# --------------------------------------------------------------------------- #
def parse_protein(entry: str, doc: str) -> Protein | None:
    cut = doc.index(BEGIN) + len(BEGIN)
    prefix, struct = doc[:cut], doc[cut:]
    m = re.search(r"<n-term>\s+<p(\d+)>", prefix)
    if not m:
        return None
    nterm = int(m.group(1))
    pos_in_seq = sorted({int(p) for p in re.findall(r"<p(\d+)>", prefix)},
                        key=lambda p: (p - nterm) % NUM_POS)
    L = len(pos_in_seq)
    seqidx = {p: (p - nterm) % NUM_POS for p in pos_in_seq}
    # residue at each position (from `<pX> <RES>` statements; 3-letter AA tokens).
    res_of_pos = {int(p): aa for p, aa in re.findall(r"<p(\d+)>\s+<([A-Z]{3})>", prefix)}
    residues = [f"<{res_of_pos[p]}>" for p in pos_in_seq] if all(p in res_of_pos for p in pos_in_seq) else None
    if residues is None:
        return None
    gt = set()
    for a, b in CONTACT_RE.findall(struct):
        ia, ib = seqidx.get(int(a)), seqidx.get(int(b))
        if ia is not None and ib is not None and ia != ib:
            gt.add(frozenset((ia, ib)))
    return Protein(entry, prefix, L, nterm, pos_in_seq, gt, residues)


def load_proteins(n, lo, hi):
    fs = gcsfs.GCSFileSystem()
    with fs.open(TEST_SHARD, "rb") as fh:
        t = pq.read_table(fh, columns=["document", "seq_len", "entry_id"])
    docs, sl, ids = (t.column(c).to_pylist() for c in ("document", "seq_len", "entry_id"))
    out = []
    for i in range(len(docs)):
        if not (lo <= sl[i] <= hi):
            continue
        p = parse_protein(ids[i], docs[i])
        if p is not None and len(p.gt) >= 5:
            out.append(p)
        if len(out) >= n:
            break
    return out


# --------------------------------------------------------------------------- #
# Scoring engine (self-contained transformers; mirrors marinfold.inference)    #
# --------------------------------------------------------------------------- #
class Scorer:
    def __init__(self, model_path, device="cuda", dtype=torch.bfloat16, batch=16):
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype).to(device).eval()
        self.device, self.batch = device, batch
        self.contact_id = self.tok.convert_tokens_to_ids("<contact>")
        self.end_id = self.tok.convert_tokens_to_ids("<end>")

    def ptoken(self, pos):
        return self.tok.convert_tokens_to_ids(f"<p{pos}>")

    @torch.no_grad()
    def contact_logprob_matrix(self, prefix_ids, positions, contact_id=None):
        """log P(<contact-tok> pi pj | prefix) for all i,j over `positions`.

        Returns (lp1[i], lp2[i, j]) where lp1 = log P(pi | prefix,<contact-tok>)
        and lp2[i,j] = log P(pj | prefix,<contact-tok>,pi). ``contact_id``
        defaults to ``<contact>`` (contacts-v1); pass a ``<{range}-range-contact>``
        id for the contacts-and-distances-v1 prior model. Batched: one pass for
        lp1, chunked passes for lp2.
        """
        if contact_id is None:
            contact_id = self.contact_id
        pos_ids = [self.ptoken(p) for p in positions]
        base = list(prefix_ids) + [contact_id]
        # lp1: distribution right after prefix+<contact>
        X = torch.tensor([base], device=self.device)
        logits = self.model(X).logits[0, -1].float()
        lp_all = F.log_softmax(logits, -1)
        lp1 = lp_all[pos_ids].cpu().numpy()                      # [L]
        # lp2[i, :]: distribution after prefix+<contact>+pi
        lp2 = np.empty((len(positions), len(positions)), np.float32)
        seqs = [base + [pid] for pid in pos_ids]
        for s in range(0, len(seqs), self.batch):
            chunk = seqs[s:s + self.batch]
            X = torch.tensor(chunk, device=self.device)
            logits = self.model(X).logits[:, -1].float()         # [b, V]
            lp = F.log_softmax(logits, -1)[:, pos_ids].cpu().numpy()
            lp2[s:s + len(chunk)] = lp
        return lp1, lp2

    @torch.no_grad()
    def rollouts(self, prefix_ids, n_rollouts, temperature, top_p, max_new, batch=10):
        """Sample contact-section completions; return list of decoded strings."""
        X = torch.tensor([prefix_ids], device=self.device)
        texts = []
        done = 0
        while done < n_rollouts:
            b = min(batch, n_rollouts - done)
            out = self.model.generate(
                X.repeat(b, 1), do_sample=True, temperature=temperature, top_p=top_p,
                max_new_tokens=max_new, eos_token_id=self.end_id, pad_token_id=self.end_id,
            )
            for row in out:
                texts.append(self.tok.decode(row[X.shape[1]:], skip_special_tokens=False))
            done += b
        return texts


# --------------------------------------------------------------------------- #
# Methods → ranked candidate list                                              #
# --------------------------------------------------------------------------- #
def candidate_pairs(L):
    return [(i, j) for i in range(L) for j in range(i + MIN_SEP, L)]


def rank_pairwise(scorer, p: Protein, committed=()):
    """Symmetrized geo-mean log-score per candidate pair, under an optional set
    of already-committed contacts (used by the iterative method)."""
    prefix_ids = scorer.tok(p.prefix, add_special_tokens=False).input_ids
    for (i, j) in committed:                       # commit contacts into the prefix
        prefix_ids = prefix_ids + [scorer.contact_id, scorer.ptoken(p.seq_positions[i]),
                                   scorer.ptoken(p.seq_positions[j])]
    lp1, lp2 = scorer.contact_logprob_matrix(prefix_ids, p.seq_positions)
    # score(i,j) for orientation i->j = lp1[i] + lp2[i,j]; symmetrize.
    fwd = lp1[:, None] + lp2                        # [i, j]
    sym = 0.5 * (fwd + fwd.T)                       # geo-mean in log space
    pairs = candidate_pairs(p.L)
    scored = [(sym[i, j], (i, j)) for (i, j) in pairs]
    scored.sort(reverse=True)
    return [pr for _, pr in scored]


def rank_rollout(scorer, p: Protein, n_rollouts, temperature, top_p):
    prefix_ids = scorer.tok(p.prefix, add_special_tokens=False).input_ids
    seqidx = {pos: i for i, pos in enumerate(p.seq_positions)}
    counts = Counter()
    max_new = min(8192 - len(prefix_ids), 3 * len(p.gt) + 60)
    for text in scorer.rollouts(prefix_ids, n_rollouts, temperature, top_p, max_new):
        seen = set()
        for a, b in CONTACT_RE.findall(text):
            ia, ib = seqidx.get(int(a)), seqidx.get(int(b))
            if ia is None or ib is None or ia == ib:
                continue
            key = (min(ia, ib), max(ia, ib))
            if abs(ia - ib) >= MIN_SEP and key not in seen:
                seen.add(key); counts[key] += 1
    pairs = candidate_pairs(p.L)
    # rank by frequency desc; unseen pairs keep candidate order (count 0)
    return sorted(pairs, key=lambda pr: -counts.get(pr, 0))


def rank_iterative(scorer, p: Protein, schedule=(0.5, 1.0, 1.5, 2.5)):
    """exp27-style growing-K: commit top-K, re-score the rest, repeat."""
    ranking = rank_pairwise(scorer, p)              # round 0 baseline
    committed = []
    for frac in schedule:
        k = max(1, round(frac * p.L))
        committed = ranking[:k]
        re_ranked = rank_pairwise(scorer, p, committed=committed)
        # keep committed on top (in their order), then the rest by new score
        cset = set(map(tuple, committed))
        rest = [pr for pr in re_ranked if pr not in cset]
        ranking = list(map(tuple, committed)) + rest
    return ranking


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #
def sep(pair):
    return abs(pair[0] - pair[1])


def metrics(ranking, gt, L):
    gt_pairs = {tuple(sorted(g)) for g in gt}
    def prec_at(rank_list, k):
        if k == 0:
            return float("nan")
        top = rank_list[:k]
        return sum(1 for pr in top if tuple(sorted(pr)) in gt_pairs) / len(top)

    out = {}
    for band, lo in (("long", LONG), ("medlong", MED)):
        cand = [pr for pr in ranking if sep(pr) >= lo]
        gtb = {g for g in gt_pairs if sep(g) >= lo}
        out[f"{band}_ngt"] = len(gtb)
        for name, k in (("L", L), ("L2", L // 2), ("L5", L // 5)):
            kk = min(k, len(cand)) if len(gtb) else 0
            # precision restricted to this separation band
            top = [pr for pr in cand[:k]]
            out[f"{band}_P@{name}"] = (sum(1 for pr in top if tuple(sorted(pr)) in gtb) / len(top)) if top and gtb else float("nan")
    # overall P@(#gt) and a cheap AUC vs gt over all candidates
    out["P@ngt"] = prec_at(ranking, len(gt_pairs))
    return out


def aggregate(rows):
    keys = [k for k in rows[0] if k != "entry"]
    agg = {}
    for k in keys:
        vals = [r[k] for r in rows if isinstance(r[k], (int, float)) and not (isinstance(r[k], float) and np.isnan(r[k]))]
        agg[k] = float(np.mean(vals)) if vals else float("nan")
    return agg


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("-n", type=int, default=15)
    ap.add_argument("--min-len", type=int, default=50)
    ap.add_argument("--max-len", type=int, default=150)
    ap.add_argument("--methods", default="pairwise,rollout,iterative")
    ap.add_argument("--n-rollouts", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    methods = args.methods.split(",")
    scorer = Scorer(args.model)
    proteins = load_proteins(args.n, args.min_len, args.max_len)
    print(f"{len(proteins)} held-out test proteins (seq_len {args.min_len}-{args.max_len}); "
          f"methods={methods}\n", flush=True)

    results = {m: [] for m in [*methods, "random"]}
    for p in proteins:
        rng = np.random.default_rng(args.seed)
        rankings = {}
        try:
            if "pairwise" in methods:
                rankings["pairwise"] = rank_pairwise(scorer, p)
            if "rollout" in methods:
                rankings["rollout"] = rank_rollout(scorer, p, args.n_rollouts, args.temperature, args.top_p)
            if "iterative" in methods:
                rankings["iterative"] = rank_iterative(scorer, p)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"{p.entry:>22} L={p.L:>3} -- SKIPPED (CUDA OOM)", flush=True)
            continue
        rand = candidate_pairs(p.L); rng.shuffle(rand)
        rankings["random"] = rand
        line = [f"{p.entry:>22} L={p.L:>3} gt={len(p.gt):>3}"]
        for m, rk in rankings.items():
            mm = metrics(rk, p.gt, p.L)
            mm["entry"] = p.entry
            results[m].append(mm)
            line.append(f"{m}:long_P@L={mm['long_P@L']:.2f}" if not np.isnan(mm["long_P@L"]) else f"{m}:long_P@L=n/a")
        print("  ".join(line), flush=True)
        torch.cuda.empty_cache()

    print("\n=== AGGREGATE (mean over proteins) ===")
    cols = ["long_P@L", "long_P@L2", "long_P@L5", "medlong_P@L", "medlong_P@L2", "medlong_P@L5", "P@ngt"]
    print(f"{'method':<12} " + " ".join(f"{c:>12}" for c in cols))
    for m in [*methods, "random"]:
        if not results[m]:
            continue
        a = aggregate(results[m])
        print(f"{m:<12} " + " ".join(f"{a.get(c, float('nan')):>12.3f}" for c in cols))
    print("\n(precision of the top-{L,L/2,L/5} ranked contacts; long = seq-sep>=24, "
          "medlong = seq-sep>=12. chance ≈ contact density in each band.)")


if __name__ == "__main__":
    main()
