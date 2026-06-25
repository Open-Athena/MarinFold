# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Save per-pair rollout+resample vote-score matrices for the curated eval set.

To compare rollout+resample against the other predictors in #89's tidy table, its
metrics must be computed by #89's *exact* ``compute_metrics.py`` — our exp82
metric impl disagrees with #89's by up to 0.4/protein (float16 tie-breaking +
small proteins), so merging our numbers would be wrong. #89 scores a model from an
``[L, L]`` score matrix per protein (``scores/<dataset>__<stem>.npz``, key
``score``), so we emit rollout+resample in that layout and re-use #89's metrics.

For each of the 554 proteins: draw N resampled rollouts (a fresh contacts-v1
realization per rollout — resampled N-terminus start + statement order), accumulate
the per-pair occurrence frequency into a symmetric ``[L, L]`` matrix in
input-sequence coordinates (so it aligns with the GT universe), save it as
float16. **Resumable** (skip stems already scored); **adaptive batch** keyed on L
(the A5000 sat at ~5 GB of 24 at batch 24). Run::

    PYTHONPATH=<repo>/marinfold uv run python score_rollout_resample_eval.py \
        --model /home/bizon/exp89_export/hf_step35679 \
        --out-dir _scratch/scores_rollout_resample --n-rollouts 100
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from eval_contact_prediction import CONTACT_RE, MIN_SEP, NUM_POS, Scorer
from eval_full_curated_set import load_eval_records
from eval_rollout_resampled_dev import realization
from marinfold.document_structures.contacts_v1 import residues_from_sequence


def adaptive_batch(L: int, cap: int = 64, floor: int = 8, budget: int = 20000) -> int:
    """Bigger batch for short proteins, smaller for long, to bound KV-cache."""
    return max(floor, min(cap, budget // max(L, 1)))


def vote_matrix(scorer, stem, residues, L, n_rollouts, temperature, top_p, top_k, batch):
    """[L,L] symmetric per-pair occurrence frequency over n_rollouts resampled rollouts."""
    prefixes, maps = [], []
    for r in range(n_rollouts):
        prefix, seq_positions, _ = realization(stem, residues, f"r{r}")
        prefixes.append(scorer.tok(prefix, add_special_tokens=False).input_ids)
        maps.append({pos: i for i, pos in enumerate(seq_positions)})
    max_new = min(8192 - len(prefixes[0]), 4 * L + 64)
    texts = scorer.sample_completions(prefixes, temperature, top_p, max_new, batch=batch, top_k=top_k)
    M = np.zeros((L, L), np.float32)
    for text, seqidx in zip(texts, maps):
        seen = set()
        for a, b in CONTACT_RE.findall(text):
            ia, ib = seqidx.get(int(a)), seqidx.get(int(b))
            if ia is None or ib is None or ia == ib:
                continue
            key = (min(ia, ib), max(ia, ib))
            if abs(ia - ib) >= MIN_SEP and key not in seen:
                seen.add(key)
                M[key[0], key[1]] += 1
                M[key[1], key[0]] += 1
    return M


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-rollouts", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cap-batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="smoke: only the first N (by length)")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = load_eval_records()
    todo = [r for r in records
            if not (args.out_dir / f"{r['dataset']}__{r['stem']}.npz").exists()]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(records)} eval proteins | {len(records) - len(todo)} already scored | "
          f"{len(todo)} to do | n_rollouts={args.n_rollouts}", flush=True)

    scorer = Scorer(args.model)
    t0 = time.time()
    for k, r in enumerate(todo):
        residues, L = residues_from_sequence(r["input_seq"]), r["L"]
        batch = adaptive_batch(L, cap=args.cap_batch)
        torch.manual_seed(args.seed)  # match the recipe's per-protein RNG handling
        ts = time.time()
        M = vote_matrix(scorer, r["stem"], residues, L, args.n_rollouts,
                        args.temperature, args.top_p, args.top_k, batch)
        dt = time.time() - ts
        np.savez_compressed(args.out_dir / f"{r['dataset']}__{r['stem']}.npz",
                            score=M.astype(np.float16))
        print(f"  [{k + 1}/{len(todo)}] {r['stem']:>9} L={L:>3} batch={batch:>2} "
              f"{dt:5.1f}s (elapsed {(time.time() - t0) / 60:.1f}m)", flush=True)
        torch.cuda.empty_cache()
    print(f"[done] {len(todo)} proteins in {(time.time() - t0) / 60:.1f} min -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
