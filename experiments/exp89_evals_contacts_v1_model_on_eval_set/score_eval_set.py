# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step B — MarinFold pairwise contact scores for every eval-set protein.

For each protein in the exp74/exp78 eval manifests we build a contacts-v1
**sequence-section prefix** from the input sequence (the official, deterministic
``build_document`` with empty contacts), then score every candidate residue pair
with the model exactly as exp82's *pairwise* method did — the symmetrized
geometric-mean log-probability of the ``<contact> <pi> <pj>`` statement (the
"best inference approach identified in #82"; iterative/rollout did not beat it).

Output: one ``scores/<stem>.npz`` per protein holding the ``[L, L]`` symmetrized
score matrix in **input-sequence coordinates** (so it aligns 1:1 with the GT
universe from ``prepare_gt_universe.py``). Scoring is backend-agnostic; this
runs the transformers/CUDA scorer locally. The metric step consumes the saved
matrices, so a later vLLM/TPU run can drop in the same ``.npz`` layout.

Run in the exp89 venv (torch + transformers); needs ``marinfold`` on the path::

    PYTHONPATH=<repo>/marinfold uv run python score_eval_set.py \
        --model /home/bizon/exp89_export/hf_step35679 \
        --out-dir _scratch/scores
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from eval_contact_prediction import BEGIN, NUM_POS, Scorer

from marinfold.document_structures.contacts_v1 import (
    GenerationConfig,
    build_document,
    residues_from_sequence,
)

EXP78 = Path("/home/bizon/git/MarinFold-exp78/experiments/exp78_evals_esmfold_contacts")
MANIFESTS = (
    EXP78 / "data/eval_manifest_foldbench.csv",
    EXP78 / "data/eval_manifest_exp65.csv",
)


def load_eval_proteins() -> list[tuple[str, str, str]]:
    """Return [(dataset, stem, input_seq)] for every manifest row.

    Two stems (7ur7_A, 8ah9_A) appear in both FoldBench-100 and exp65 with
    *different* sequences/structures, so we key by (dataset, stem), not stem.
    """
    rows: list[tuple[str, str, str]] = []
    for m in MANIFESTS:
        df = pd.read_csv(m)
        for _, r in df.iterrows():
            rows.append((r["dataset"], r["stem"], r["input_seq"]))
    return rows


def prefix_and_positions(stem: str, input_seq: str):
    """Deterministic contacts-v1 sequence prefix + per-seq-index position ids."""
    residues = residues_from_sequence(input_seq)
    result = build_document(stem, residues, [], config=GenerationConfig())
    if result is None:
        return None
    L = result.seq_len
    nterm = result.n_term_index
    seq_positions = [(nterm + k) % NUM_POS for k in range(L)]
    doc = result.document
    prefix = doc[: doc.index(BEGIN) + len(BEGIN)]
    return prefix, seq_positions, L


def score_matrix(scorer: Scorer, prefix: str, seq_positions: list[int]) -> np.ndarray:
    """Symmetrized geo-mean log-score [L,L] (exp82 pairwise), input-seq coords."""
    prefix_ids = scorer.tok(prefix, add_special_tokens=False).input_ids
    lp1, lp2 = scorer.contact_logprob_matrix(prefix_ids, seq_positions)
    fwd = lp1[:, None] + lp2          # score(i->j) = logP(pi)+logP(pj|pi)
    sym = 0.5 * (fwd + fwd.T)         # symmetrize (geo-mean in log space)
    return sym.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None, help="score only the first N (smoke).")
    ap.add_argument("--timings", type=Path, default=None)
    args = ap.parse_args()

    proteins = load_eval_proteins()
    if args.limit:
        proteins = proteins[: args.limit]
    print(f"scoring {len(proteins)} eval-set proteins -> {args.out_dir}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    scorer = Scorer(args.model)
    gpu_name = ""
    try:
        import torch
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        pass

    timings: list[dict] = []
    n_ok = n_skip = 0
    for k, (dataset, stem, seq) in enumerate(proteins):
        out = args.out_dir / f"{dataset}__{stem}.npz"
        if out.exists():
            n_ok += 1
            continue
        built = prefix_and_positions(stem, seq)
        if built is None:
            print(f"  {stem}: build_document returned None (len={len(seq)}); skipping", flush=True)
            n_skip += 1
            continue
        prefix, seq_positions, L = built
        t0 = time.time()
        try:
            sym = score_matrix(scorer, prefix, seq_positions)
        except Exception as e:  # noqa: BLE001
            print(f"  {stem}: scoring FAILED: {e!r}", flush=True)
            n_skip += 1
            continue
        dt = time.time() - t0
        np.savez_compressed(out, score=sym.astype(np.float16))
        timings.append(dict(model="marinfold-contacts-v1", stem=stem, n_residues=L,
                            elapsed_seconds=round(dt, 4), gpu_name=gpu_name))
        n_ok += 1
        if (k + 1) % 25 == 0:
            print(f"  ...{k + 1}/{len(proteins)}  last {stem} L={L} {dt:.1f}s", flush=True)

    if args.timings and timings:
        pd.DataFrame(timings).to_csv(args.timings, index=False)
    print(f"[score] {n_ok} scored, {n_skip} skipped -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
