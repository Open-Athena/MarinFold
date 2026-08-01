# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Oracle best-of-N rollout R-precision: for each protein, the single BEST of
the N=100 sampled rollouts, instead of the votes-aggregated ranking.

Matches exp89's own "R" cut definition exactly (RANGES/true_matrix/
resolved_pairs below are copied verbatim from exp89 compute_metrics.py, same
convention as build_rollout_rows.py — keep identical):
R = number of true contacts in a range, restricted to the resolved-residue
candidate universe. The only difference from the standard metric is the
ranking source -- instead of ranking ALL candidate pairs by vote count, each
rollout supplies its own short, ORDER-PRESERVING list of emitted contacts
(already sep>=6 filtered + deduped by the worker); precision is computed on
that rollout's first R (resolved-filtered) contacts, and the reported value
is the max over the 100 rollouts. This is the "how good is the single best
sample" upper bound the standard vote-based aggregation is compared against
(see the README's "oracle_best_of_100" section).

**Not a deployable recipe.** You cannot know which of the 100 rollouts is
best without ground truth -- this is a headroom diagnostic, kept in its own
inference category (see exp180's build_dataset.py ORACLE_BEST100), never
mixed into the pairwise/rollout frontier.

Reads score_rollout_worker_oracle.py's *_detail parquet output (dataset,
stem, L, rollout, rank, i, j), not the votes triplets.

    uv run python build_oracle_best_rollout.py --gt gt_universe.jsonl \
        --detail gs://.../rollout_scores/LABEL_detail --label LABEL \
        --out oracle_rows.csv --summary oracle_summary.csv
"""
import argparse
import json
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# --- verbatim from exp89 compute_metrics.py (DO NOT EDIT — must stay identical) ---
RANGES = {"all": (6, None), "short": (6, 11), "medium": (12, 23), "long": (24, None)}
MIN_DEG, MIN_SEP = 0.001, 6


def true_matrix(L, contacts):
    m = np.zeros((L, L), bool)
    for i, j, d in contacts:
        i, j = int(i), int(j)
        if d >= MIN_DEG and (j - i) >= MIN_SEP and i < j < L:
            m[i, j] = True
    return m


def resolved_pairs(resolved):
    a, b = np.triu_indices(len(resolved), k=1)
    i, j = resolved[a], resolved[b]
    return i, j, (j - i)
# --- end verbatim ---


def load_detail(prefix: str) -> pd.DataFrame:
    fs, root = fsspec.core.url_to_fs(prefix)
    parts = fs.glob(f"{root.rstrip('/')}/shard-*-part-*.parquet")
    assert parts, f"no detail parts under {prefix}"
    dfs = []
    for p in parts:
        with fsspec.open(fs.unstrip_protocol(p), "rb") as fh:
            dfs.append(pq.read_table(fh).to_pandas())
    df = pd.concat(dfs, ignore_index=True)
    print(f"[oracle] loaded {len(df)} detail rows from {len(parts)} parts")
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--detail", required=True, help="gs:// prefix with shard-*-part-*.parquet")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    args = ap.parse_args()

    gt = {(r["dataset"], r["stem"]): r for r in
          (json.loads(line) for line in args.gt.open())}
    detail = load_detail(args.detail)

    rows = []
    n_scored = 0
    for (dataset, stem), g in detail.groupby(["dataset", "stem"], sort=False):
        rec = gt.get((dataset, stem))
        if rec is None:
            continue
        L = rec["L"]
        resolved = np.asarray(rec["resolved"], dtype=np.int64)
        resolved_mask = np.zeros(L, dtype=bool)
        resolved_mask[resolved] = True
        tmat = true_matrix(L, rec["contacts"])
        pi, pj, psep = resolved_pairs(resolved)
        cg_all = tmat[pi, pj].astype(int)

        ii_all = g["i"].to_numpy()
        jj_all = g["j"].to_numpy()
        roll_all = g["rollout"].to_numpy()
        rank_all = g["rank"].to_numpy()
        in_resolved = resolved_mask[ii_all] & resolved_mask[jj_all]
        sep_all = np.abs(jj_all.astype(np.int64) - ii_all.astype(np.int64))

        for rng, (lo, hi) in RANGES.items():
            inr_gt = psep >= lo
            if hi is not None:
                inr_gt = inr_gt & (psep <= hi)
            nt = int(cg_all[inr_gt].sum())
            if nt <= 0:
                rows.append(dict(dataset=dataset, stem=stem, L=L, range=rng,
                                 r_oracle=float("nan"), r_oracle_rollout=-1, n_true=nt))
                continue

            inr = sep_all >= lo
            if hi is not None:
                inr = inr & (sep_all <= hi)
            keep = in_resolved & inr
            if not keep.any():
                rows.append(dict(dataset=dataset, stem=stem, L=L, range=rng,
                                 r_oracle=0.0, r_oracle_rollout=-1, n_true=nt))
                continue

            sub = pd.DataFrame({"rollout": roll_all[keep], "rank": rank_all[keep],
                                "i": ii_all[keep], "j": jj_all[keep]})
            best, best_k = 0.0, -1
            for k, gk in sub.groupby("rollout", sort=False):
                gk = gk.sort_values("rank")
                top = min(nt, len(gk))
                if top == 0:
                    continue
                fi = gk["i"].to_numpy()[:top]
                fj = gk["j"].to_numpy()[:top]
                prec = float(tmat[fi, fj].sum()) / top
                if prec > best:
                    best, best_k = prec, int(k)
            rows.append(dict(dataset=dataset, stem=stem, L=L, range=rng,
                             r_oracle=best, r_oracle_rollout=best_k, n_true=nt))
        n_scored += 1

    df = pd.DataFrame(rows)
    df["model"] = args.label
    df.to_csv(args.out, index=False)
    print(f"[oracle] scored {n_scored}/{len(gt)} proteins -> {args.out}")

    summary = df.groupby("range")["r_oracle"].mean().rename("mean_r_oracle").reset_index()
    summary["model"] = args.label
    summary.to_csv(args.summary, index=False)
    print("[oracle] mean best-of-N R-precision by range:")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
