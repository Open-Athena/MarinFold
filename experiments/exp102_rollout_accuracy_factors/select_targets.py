# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage A (local, CPU): pick a length-stratified subset of the exp98 rollout
targets for exp102.

exp98 already selected + published 1000 contacts-v1 **train** targets (round-0,
L<=512, >=5 GT contacts) with their sequences and ground-truth contacts on the
public HF bucket. exp102 studies the *structure* of rollouts rather than scale,
so we reuse that exact target set and take a length-stratified subset (default
200): the accuracy-factor questions in issue #102 (is length a driver? do good
rollouts front-load long-range contacts?) need coverage across the length range,
not all 1000 targets. Sampling from exp98's targets keeps GT identical and lets
the new ordered metrics join 1:1 to exp98's ``rollout_metrics_all``.

Stratification: sort by L, split into ``--n`` equal-count bins, pick one target
per bin (seeded RNG) — an even spread across the length-rank, reproducible.

Output ``data/targets.parquet`` — same schema as exp98's targets.parquet
(``entry_id, L, sequence, n_gt, gt_contacts, global_plddt, struct_cluster_id,
round, shard``), so gen_prompts.py / the worker are drop-in.

    uv run python select_targets.py --n 200 --seed 102
"""
from __future__ import annotations

import argparse
import os
import random
import urllib.request

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

EXP98 = ("https://huggingface.co/buckets/open-athena/MarinFold/resolve/"
         "data/contacts-v1-train-rollouts-exp98")


def _download(url: str, dest: str) -> str:
    if not os.path.exists(dest):
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        urllib.request.urlretrieve(url, dest)
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="number of targets to keep")
    ap.add_argument("--seed", type=int, default=102)
    ap.add_argument("--src", default=None,
                    help="local exp98 targets.parquet (default: download from HF bucket)")
    ap.add_argument("--out", default="data/targets.parquet")
    args = ap.parse_args()

    src = args.src or _download(f"{EXP98}/targets.parquet", "data/exp98_targets.parquet")
    df = pd.read_parquet(src).reset_index(drop=True)
    print(f"exp98 targets: {len(df)}  L[min/median/max]="
          f"{df.L.min()}/{int(df.L.median())}/{df.L.max()}", flush=True)
    if args.n >= len(df):
        raise SystemExit(f"--n {args.n} >= available {len(df)}; nothing to stratify")

    # length-stratified: equal-count bins by L, one seeded pick per bin.
    df = df.sort_values("L", kind="stable").reset_index(drop=True)
    rng = random.Random(args.seed)
    bins = pd.cut(df.index, bins=args.n, labels=False)  # equal-width over rank -> equal count
    picks = []
    for b in range(args.n):
        idx = df.index[bins == b].tolist()
        if idx:
            picks.append(rng.choice(idx))
    sub = df.loc[picks].drop_duplicates("entry_id").sort_values("entry_id").reset_index(drop=True)

    print(f"selected {len(sub)} targets", flush=True)
    print("  L:    ", sub["L"].describe()[["min", "mean", "50%", "max"]].round(1).to_dict())
    print("  n_gt: ", sub["n_gt"].describe()[["min", "mean", "50%", "max"]].round(1).to_dict())
    if "struct_cluster_id" in sub:
        print("  uniq clusters:", sub["struct_cluster_id"].nunique())

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pq.write_table(pa.Table.from_pandas(sub, preserve_index=False), args.out)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
