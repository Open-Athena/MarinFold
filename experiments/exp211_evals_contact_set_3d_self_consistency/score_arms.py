# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Steps 2-3 (issue #211) — build every arm and score it.

Reads the per-rollout contact sets produced by ``gen_rollouts_worker.py``,
constructs the seven arms of ``arms.py`` for each protein, and scores them all
with ``consistency.py``. One row per (protein, arm, replicate).

The primary contrast is **arm 3 (within-rollout) vs arm 4 (marginal-matched
chimera)**: same model, same protein, same per-pair marginals, same set size,
differing only in whether the contacts were drawn jointly in one autoregressive
pass or independently from the pooled vote distribution.

Per-rollout **accuracy** is recorded alongside (precision / recall / F1 against
the pyconfind ground truth) so the secondary question — does consistency rank
rollouts without ground truth? — is answerable from the same table.

    uv run python score_arms.py --rollouts _scratch/rollouts \\
        --gt-dir _scratch/gt --bounds data/bounds.json --out data/arm_scores.csv

**Batching is memory-bounded, not set-bounded.** The non-contact constraint list
is O(L^2) per row, so a fixed batch size that fits at L=150 will OOM at L=761.
``--max-pairs`` caps ``rows x pairs_per_row`` instead, which keeps the batch as
wide as the device allows at every length — and width is what makes this
affordable, since the embedder is kernel-launch bound (see ``consistency``).
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from arms import (
    decoy_protein, ground_truth, marginal_chimera, separation_matched_random,
    splice_chimera, subsample,
)
from calibrate_bounds import load_bundle
from consistency import contact_matrix, embed_residual, packing_score
from run_gt_gate import bounds_from_json, chain_break_count, gt_pairs


def load_rollouts(path: Path) -> dict[tuple[str, str], list[list[tuple[int, int]]]]:
    """Per-protein list of per-rollout distinct contact sets, in rollout order."""
    files = sorted((Path(path) / "contacts").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet under {path}/contacts")
    df = pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)
    # An empty part (a protein whose rollouts emitted nothing) makes pandas widen
    # the bool column to float on concat, and `~` on a float column raises.
    df["duplicate"] = df["duplicate"].fillna(False).astype(bool)
    df = df[~df["duplicate"]]  # distinct contacts only, matching exp82's readout
    out: dict[tuple[str, str], list[list[tuple[int, int]]]] = {}
    for (ds, stem), g in df.groupby(["dataset", "stem"], sort=False):
        n = int(g["rollout"].max()) + 1
        per = [[] for _ in range(n)]
        for k, i, j in zip(g["rollout"].to_numpy(), g["i"].to_numpy(), g["j"].to_numpy()):
            per[int(k)].append((int(i), int(j)))
        out[(ds, stem)] = per
    print(f"[score] loaded rollouts for {len(out)} proteins from {len(files)} parts "
          f"({len(df):,} contact rows)")
    return out


def accuracy(pred, gt_set):
    """Per-rollout precision / recall / F1 against the ground-truth contact set."""
    p = set(pred)
    if not p:
        return dict(precision=np.nan, recall=0.0, f1=0.0, n_pred=0)
    hit = len(p & gt_set)
    prec = hit / len(p)
    rec = hit / max(len(gt_set), 1)
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return dict(precision=prec, recall=rec, f1=f1, n_pred=len(p))


def chunk_by_pairs(n_sets: int, length: int, n_restarts: int, max_pairs: int) -> int:
    """How many sets can go in one embed call before the O(L^2) lists blow memory."""
    per_row = max(length * length // 2, 1)
    rows = max(int(max_pairs // per_row), 1)
    return max(rows // max(n_restarts, 1), 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", type=Path, default=Path("_scratch/rollouts"))
    ap.add_argument("--gt-dir", type=Path, default=Path("_scratch/gt"))
    ap.add_argument("--bounds", type=Path, default=Path("data/bounds.json"))
    ap.add_argument("--out", type=Path, default=Path("data/arm_scores.csv"))
    ap.add_argument("--n-replicates", type=int, default=40,
                    help="rollouts scored per protein (arms 3/4/5 each get this many)")
    ap.add_argument("--n-restarts", type=int, default=4)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--max-pairs", type=int, default=40_000_000)
    ap.add_argument("--min-length", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None,
                    help="torch device; defaults to cuda when available")
    args = ap.parse_args()

    bounds = bounds_from_json(args.bounds)
    rollouts = load_rollouts(args.rollouts)

    meta = {}
    for record_id, m, xyz, raw in load_bundle(args.gt_dir):
        ds, stem = record_id.split("/", 1)
        meta[(ds, stem)] = dict(record_id=record_id, dataset=ds, L=int(m["L"]),
                                gt=gt_pairs(raw), breaks=chain_break_count(xyz))

    keys = [k for k in rollouts if k in meta and meta[k]["L"] >= args.min_length]
    keys.sort(key=lambda k: meta[k]["L"])
    if args.limit:
        keys = keys[: args.limit]
    print(f"[score] {len(keys)} proteins, {args.n_replicates} replicates/arm, "
          f"restarts={args.n_restarts}")

    rows, t0 = [], time.time()
    for n, key in enumerate(keys):
        info = meta[key]
        length, gt = info["L"], info["gt"]
        gt_set = set(gt)
        per_rollout = rollouts[key]
        rng = np.random.default_rng(n)

        # The model's own per-pair marginal: how many rollouts emitted each pair.
        votes = Counter()
        for r in per_rollout:
            votes.update(set(r))
        if not votes:
            continue

        pick = rng.permutation(len(per_rollout))[: args.n_replicates]
        named: list[tuple[str, int, list[tuple[int, int]]]] = []
        for rep, ri in enumerate(pick):
            own = sorted(set(per_rollout[ri]))
            if len(own) < 5:
                continue
            named.append(("rollout", rep, own))
            named.append(("chimera_marginal", rep,
                          marginal_chimera(dict(votes), len(own), rng)))
            other = per_rollout[int(rng.integers(len(per_rollout)))]
            named.append(("chimera_splice", rep,
                          splice_chimera(own, other, len(own), rng,
                                         pool=sorted(votes))))
        if not named:
            continue

        # Reference arms, sized to the median rollout so nothing is size-confounded.
        med = int(np.median([len(s) for a, _, s in named if a == "rollout"]))
        named.append(("gt", 0, ground_truth(gt)))
        named.append(("gt_subsampled", 0, subsample(gt, med, rng)))
        named.append(("random", 0, separation_matched_random(
            subsample(gt, med, rng), length, rng)))
        donor = meta[keys[(n + 1) % len(keys)]]["gt"]
        named.append(("decoy", 0, decoy_protein(donor, length, med, rng)))

        named = [(a, r, s) for a, r, s in named if len(s) >= 3]
        step = chunk_by_pairs(len(named), length, args.n_restarts, args.max_pairs)
        for s0 in range(0, len(named), step):
            block = named[s0:s0 + step]
            masks = np.stack([contact_matrix(s, length) for _, _, s in block])
            emb = embed_residual(masks, bounds, n_restarts=args.n_restarts,
                                 iters=args.iters, seed=n * 977 + s0,
                                 device=args.device)
            for (arm, rep, s), mask, e in zip(block, masks, emb):
                rows.append(dict(
                    record_id=info["record_id"], dataset=info["dataset"], L=length,
                    arm=arm, replicate=rep,
                    n_chain_breaks=info["breaks"],
                    has_chain_break=info["breaks"] > 0,
                    n_gt_contacts=len(gt_set),
                    **packing_score(mask), **e, **accuracy(s, gt_set),
                ))
        if (n + 1) % 25 == 0:
            el = time.time() - t0
            print(f"[score] {n + 1}/{len(keys)}  {el / 60:.1f} min  "
                  f"({el / (n + 1):.1f} s/protein)  {len(rows):,} rows", flush=True)

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\n[score] wrote {args.out}: {len(df):,} rows, "
          f"{df['record_id'].nunique()} proteins, {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
