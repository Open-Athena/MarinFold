# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark analysis (exp82) on canonical proteins (1QYS / 7BNY / 1UBQ).

Two artifacts, for the contacts-v1 1.5B model (exp67) and the prior
contacts-and-distances-v1 1.5B:

1. **Contact-probability heatmaps** — the model's per-pair contact score over
   all (i, j), next to the pyconfind ground-truth contact map. Saved to
   ``plots/heatmap_<PDB>.png``.

2. **AUC vs #seeded ground-truth contacts** — seed N true contacts into the
   prompt and measure how well the model ranks the *remaining* contacts
   (long+medium range) against decoys; sweep N. Tests whether conditioning on
   known contacts helps predict the rest (the iterative intuition as a curve).
   Saved to ``plots/auc_vs_seeded.png``.

Inputs: a local parquet of contacts-v1 documents for the benchmark proteins
(built with `contacts-v1 generate` — see the README). Run on GPU::

    uv run python benchmark_analysis.py \
        --cv1-model ./cv1_model --prior-model ./prior_model \
        --docs ./benchmark.parquet
"""
from __future__ import annotations

import argparse
import os
import re
from dataclasses import replace

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import torch

from eval_contact_prediction import LONG, MED, MIN_SEP, Scorer, candidate_pairs, parse_protein
from eval_prior_model_contacts import prior_prefix_ids, range_token


# --------------------------------------------------------------------------- #
def load_local_proteins(path):
    t = pq.read_table(path, columns=["document", "seq_len", "entry_id"])
    docs, ids = t.column("document").to_pylist(), t.column("entry_id").to_pylist()
    out = []
    for entry, doc in zip(ids, docs):
        p = parse_protein(entry, doc)
        if p is not None:
            out.append(p)
    return out


def poly_ala(p):
    """Same protein with every residue identity replaced by ALA in the prompt.

    Only the `<pX> <RES>` residue assignments are rewritten (n-term/c-term
    position markers and all positions/contacts are untouched), so the model
    sees the protein's *length* and the seeded contacts but no sequence to
    reason about — a control for "is the model using the sequence at all?"."""
    ala_prefix = re.sub(r"(<p\d+>\s+)<[A-Z]{3}>", r"\1<ALA>", p.prefix)
    return replace(p, entry=f"{p.entry}-polyAla", prefix=ala_prefix)


def gt_matrix(p):
    M = np.zeros((p.L, p.L), bool)
    for g in p.gt:
        i, j = sorted(g)
        M[i, j] = M[j, i] = True
    return M


def cv1_score_matrix(scorer, p, seeded=()):
    """Symmetrized contacts-v1 pairwise log-score over all pairs, optionally
    conditioned on `seeded` true contacts committed into the prefix."""
    prefix_ids = scorer.tok(p.prefix, add_special_tokens=False).input_ids
    for (i, j) in seeded:
        prefix_ids = prefix_ids + [scorer.contact_id, scorer.ptoken(p.seq_positions[i]),
                                   scorer.ptoken(p.seq_positions[j])]
    lp1, lp2 = scorer.contact_logprob_matrix(prefix_ids, p.seq_positions)
    fwd = lp1[:, None] + lp2
    return 0.5 * (fwd + fwd.T)


def prior_distance_matrix(scorer, p):
    """Prior model P(CA-CA <= 8 A) over all pairs (true contact probability)."""
    import torch.nn.functional as F
    from eval_prior_model_contacts import CONTACT_BINS, DIST_TOKENS
    prefix_ids = prior_prefix_ids(scorer, p)
    dist_id = scorer.tok.convert_tokens_to_ids("<distance>")
    ca = scorer.tok.convert_tokens_to_ids("<CA>")
    dbin = [scorer.tok.convert_tokens_to_ids(t) for t in DIST_TOKENS]
    M = np.zeros((p.L, p.L), np.float32)
    pairs = candidate_pairs(p.L)
    seqs = [list(prefix_ids) + [dist_id, scorer.ptoken(i), scorer.ptoken(j), ca, ca] for (i, j) in pairs]
    with torch.no_grad():
        for s in range(0, len(seqs), scorer.batch):
            X = torch.tensor(seqs[s:s + scorer.batch], device=scorer.device)
            pr = F.softmax(scorer.model(X).logits[:, -1].float(), -1)[:, dbin].cpu().numpy()
            for k, (i, j) in enumerate(pairs[s:s + scorer.batch]):
                v = pr[k, :CONTACT_BINS].sum()
                M[i, j] = M[j, i] = v
    return M


# --------------------------------------------------------------------------- #
def auc_remaining(score_mat, p, seeded_set, n_neg_mult=1, rng=None, min_sep=MED):
    """AUC: do remaining true contacts (sep>=min_sep) outscore decoys?"""
    rng = rng or np.random.default_rng(0)
    gt = {tuple(sorted(g)) for g in p.gt if abs(list(g)[0] - list(g)[1]) >= min_sep}
    pos = [pr for pr in gt if pr not in seeded_set]
    if not pos:
        return float("nan")
    cand = [pr for pr in candidate_pairs(p.L) if abs(pr[0] - pr[1]) >= min_sep
            and pr not in gt and pr not in seeded_set]
    if not cand:
        return float("nan")
    neg = [cand[k] for k in rng.choice(len(cand), min(len(pos) * n_neg_mult, len(cand)), replace=False)]
    sp = np.array([score_mat[i, j] for i, j in pos])
    sn = np.array([score_mat[i, j] for i, j in neg])
    return float(np.mean(sp[:, None] > sn[None, :]))


def auc_vs_seeding(scorer, p, ns, n_reps=3):
    """Mean AUC over n_reps random seedings, for each N in `ns`."""
    gt_pairs = [tuple(sorted(g)) for g in p.gt]
    means = []
    for N in ns:
        if N >= len(gt_pairs):
            means.append(float("nan")); continue
        aucs = []
        for rep in range(n_reps if N > 0 else 1):
            rng = np.random.default_rng(100 * rep + N)
            seeded = [] if N == 0 else [gt_pairs[k] for k in rng.choice(len(gt_pairs), N, replace=False)]
            mat = cv1_score_matrix(scorer, p, seeded=seeded)
            aucs.append(auc_remaining(mat, p, set(seeded), rng=rng))
        means.append(float(np.nanmean(aucs)))
    return means


# --------------------------------------------------------------------------- #
def plot_heatmaps(proteins, cv1, prior, outdir):
    for p in proteins:
        gt = gt_matrix(p)
        cv1_m = cv1_score_matrix(cv1, p)
        prior_m = prior_distance_matrix(prior, p) if prior else None
        ncols = 3 if prior else 2
        fig, ax = plt.subplots(1, ncols, figsize=(5 * ncols, 4.6))
        # normalize scores to [0,1] for display (mask the unscored band)
        def norm(M):
            m = M.copy().astype(float)
            band = np.abs(np.subtract.outer(np.arange(p.L), np.arange(p.L))) < MIN_SEP
            m[band] = np.nan
            lo, hi = np.nanmin(m), np.nanmax(m)
            return (m - lo) / (hi - lo + 1e-9)
        ax[0].imshow(gt, cmap="Greys", origin="lower"); ax[0].set_title(f"{p.entry}  ground truth\n(L={p.L}, {len(p.gt)} contacts)")
        im = ax[1].imshow(norm(cv1_m), cmap="viridis", origin="lower"); ax[1].set_title("contacts-v1 1.5B\n(pairwise score, norm.)")
        fig.colorbar(im, ax=ax[1], fraction=0.046)
        if prior:
            im2 = ax[2].imshow(prior_m, cmap="viridis", origin="lower", vmin=0, vmax=max(0.05, np.nanmax(prior_m)))
            ax[2].set_title("prior 1.5B\nP(CA-CA ≤ 8Å)")
            fig.colorbar(im2, ax=ax[2], fraction=0.046)
        for a in ax:
            a.set_xlabel("residue j"); a.set_ylabel("residue i")
        fig.tight_layout()
        out = os.path.join(outdir, f"heatmap_{p.entry}.png")
        fig.savefig(out, dpi=120); plt.close(fig)
        print(f"  wrote {out}", flush=True)


def plot_auc_vs_seeding(proteins, cv1, ns, outdir):
    fig, ax = plt.subplots(figsize=(7, 5))
    for p in proteins:
        means = auc_vs_seeding(cv1, p, ns)
        xs = [n for n, m in zip(ns, means) if not np.isnan(m)]
        ys = [m for m in means if not np.isnan(m)]
        line, = ax.plot(xs, ys, marker="o", label=f"{p.entry} (L={p.L}, {len(p.gt)} gt)")
        print(f"  {p.entry}: AUC vs seeded {list(zip(ns, [round(m,3) for m in means]))}", flush=True)
        # poly-ALA control: same length + seeded contacts, no real sequence.
        pa = poly_ala(p)
        means_a = auc_vs_seeding(cv1, pa, ns)
        xs_a = [n for n, m in zip(ns, means_a) if not np.isnan(m)]
        ys_a = [m for m in means_a if not np.isnan(m)]
        ax.plot(xs_a, ys_a, ls="--", marker="x", color=line.get_color())
        print(f"  {p.entry} (poly-Ala): AUC vs seeded {list(zip(ns, [round(m,3) for m in means_a]))}", flush=True)
    ax.axhline(0.5, ls=":", c="grey", label="chance")
    # style proxies so the dashed-vs-solid meaning is explicit in the legend
    ax.plot([], [], c="black", marker="o", label="native sequence")
    ax.plot([], [], c="black", ls="--", marker="x", label="poly-Ala (no sequence)")
    ax.set_xlabel("# ground-truth contacts seeded into the prompt")
    ax.set_ylabel("AUC of predicting the remaining contacts (sep ≥ 12)")
    ax.set_title("contacts-v1 1.5B: does seeding known contacts help predict the rest?")
    ax.legend(); ax.set_ylim(0.4, 1.0); fig.tight_layout()
    out = os.path.join(outdir, "auc_vs_seeded.png")
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"  wrote {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv1-model", required=True)
    ap.add_argument("--prior-model", default=None)
    ap.add_argument("--docs", required=True)
    ap.add_argument("--outdir", default="plots")
    ap.add_argument("--seed-ns", default="0,1,2,4,8,16,32")
    ap.add_argument("--skip-heatmaps", action="store_true",
                    help="only regenerate the AUC-vs-seeding plot (no prior model needed)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    ns = [int(x) for x in args.seed_ns.split(",")]

    proteins = load_local_proteins(args.docs)
    print(f"{len(proteins)} benchmark proteins: {[p.entry for p in proteins]}\n", flush=True)
    cv1 = Scorer(args.cv1_model)
    prior = Scorer(args.prior_model) if args.prior_model else None

    if not args.skip_heatmaps:
        print("=== heatmaps ===", flush=True)
        plot_heatmaps(proteins, cv1, prior, args.outdir)
    print("=== AUC vs #seeded GT contacts (contacts-v1 model) ===", flush=True)
    plot_auc_vs_seeding(proteins, cv1, ns, args.outdir)


if __name__ == "__main__":
    main()
