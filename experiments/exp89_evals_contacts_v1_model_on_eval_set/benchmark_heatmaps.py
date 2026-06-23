# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Contact-map heatmaps for the three canonical test proteins — 1QYS (Top7),
7BNY, 1UBQ — for the #61/#75 contacts-v1 1.5B model (eval loss 2.7566).

For each protein, one figure with two panels:

* **Ground truth** — pyconfind side-chain contacts (degree ≥ 0.001, sep ≥ 6).
* **Model P(contact)** — the model's probability that its *next* emitted contact
  statement is the pair {i, j}: ``P[i,j] = P(i)·P(j|i) + P(j)·P(i|j)`` from the
  pairwise readout (the #82 best inference approach). This is a genuine
  distribution over candidate contacts (sums to ≈1 over pairs); the near-diagonal
  band (|i−j| < 6, never a contact in contacts-v1) is masked.

Docs (sequence + GT contacts) are exp82's ``benchmark_docs.parquet``. Run in the
exp89 venv on a CUDA box::

    uv run python benchmark_heatmaps.py --model /path/to/hf_step35679 \
        --docs _scratch/benchmark_docs.parquet --out plots
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

from eval_contact_prediction import MIN_SEP, Scorer, parse_protein


def gt_matrix(p) -> np.ndarray:
    M = np.zeros((p.L, p.L), bool)
    for g in p.gt:
        i, j = sorted(g)
        M[i, j] = M[j, i] = True
    return M


def model_pcontact(scorer: Scorer, p) -> np.ndarray:
    """P(model's next contact statement is {i,j}) over all pairs, sep-band masked."""
    prefix_ids = scorer.tok(p.prefix, add_special_tokens=False).input_ids
    lp1, lp2 = scorer.contact_logprob_matrix(prefix_ids, p.seq_positions)
    fwd = lp1[:, None] + lp2                      # log P(i)·P(j|i)
    P = np.exp(fwd) + np.exp(fwd.T)               # unordered contact probability
    band = np.abs(np.subtract.outer(np.arange(p.L), np.arange(p.L))) < MIN_SEP
    P[band] = np.nan                              # sep<6 is never a contact
    np.fill_diagonal(P, np.nan)
    return P


def plot_protein(p, gt: np.ndarray, P: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.9))
    ax[0].imshow(gt, cmap="Greys", origin="lower", interpolation="none")
    ax[0].set_title(f"{p.entry} · ground-truth contacts\n(L={p.L}, {len(p.gt)} contacts, pyconfind)")
    vmax = float(np.nanpercentile(P, 99.5)) or float(np.nanmax(P))
    im = ax[1].imshow(P, cmap="magma", origin="lower", vmin=0, vmax=vmax, interpolation="none")
    ax[1].set_title("MarinFold-cv1 1.5B · model P(contact)\n(pairwise readout, from sequence)")
    fig.colorbar(im, ax=ax[1], fraction=0.046, label="P(contact)")
    for a in ax:
        a.set_xlabel("residue j")
        a.set_ylabel("residue i")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    out.with_suffix(out.suffix + ".meta.json").write_text(
        '{"script": "benchmark_heatmaps.py", "caption": "%s"}'
        % (f"{p.entry}: ground-truth pyconfind contacts (left) vs the contacts-v1 1.5B model's "
           f"P(contact) from sequence (right) — the probability the model emits each pair as "
           f"its next contact statement. Near-diagonal band (sep<6) masked."))
    plt.close(fig)
    print(f"  wrote {out}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--docs", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("plots"))
    args = ap.parse_args()

    t = pq.read_table(args.docs).to_pylist()
    scorer = Scorer(args.model)
    for row in t:
        p = parse_protein(row["entry_id"], row["document"])
        if p is None:
            print(f"  {row['entry_id']}: parse failed; skipping")
            continue
        gt = gt_matrix(p)
        P = model_pcontact(scorer, p)
        plot_protein(p, gt, P, args.out / f"heatmap_{p.entry}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
