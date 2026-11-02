# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the 50:50 backtracking:clean training mix for #160.

Two halves, both contacts-v1 documents over ESM-Atlas proteins:

- **backtracking** — the #159 corpus (1,023,997 documents that emit contacts,
  retract the wrong ones, and end at exactly ground truth).
- **clean** — ordinary contacts-v1 documents sampled from exp139's ESM-Atlas
  corpus, so the model keeps its ability to answer without retracting and only
  backtracks when it helps.

The two halves are drawn from **disjoint proteins**. exp139 has 66.76M
documents and #159 used ~1M of them, so there is no reason to make the model
see the same protein in both forms — that would confound "learns to retract"
with "memorises this protein".

Mixing at the *document* level (rather than via a marin mixture config) keeps
training a single-source run, which is markedly simpler on the GPU path.

    uv run --with boto3 python build_mix.py --out data/mix --clean-frac 0.5
"""

from __future__ import annotations

import argparse
import io
import random
from pathlib import Path

import pandas as pd

BACKTRACKING = (
    "hf://buckets/open-athena/MarinFold/data/document_structures/"
    "contacts_v1_backtracking/train"
)
CLEAN = (
    "hf://buckets/open-athena/MarinFold/data/document_structures/"
    "contacts_v1_esm_atlas/train"
)
CLEAN_SHARDS = 3338
DOCS_PER_OUT_SHARD = 64_000


def _hf_get(path: str) -> bytes:
    """Fetch one bucket object via its resolve URL (pyarrow can't route hf://buckets)."""
    import requests
    from huggingface_hub import get_token

    assert path.startswith("hf://buckets/")
    org, repo, rest = path[len("hf://buckets/"):].split("/", 2)
    url = f"https://huggingface.co/buckets/{org}/{repo}/resolve/{rest}"
    token = get_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    response = requests.get(url, headers=headers, timeout=900)
    response.raise_for_status()
    return response.content


def load_backtracking(n_shards: int | None) -> pd.DataFrame:
    frames = []
    for i in range(16 if n_shards is None else n_shards):
        raw = _hf_get(f"{BACKTRACKING}/shard-{i:05d}.parquet")
        df = pd.read_parquet(io.BytesIO(raw), columns=["entry_id", "document", "num_tokens"])
        df["kind"] = "backtracking"
        frames.append(df)
        print(f"  backtracking shard {i}: {len(df):,}", flush=True)
    return pd.concat(frames, ignore_index=True)


def load_clean(n_docs: int, exclude: set[str], seed: int) -> pd.DataFrame:
    """Sample ``n_docs`` clean documents from proteins not in ``exclude``."""
    rng = random.Random(seed)
    order = list(range(CLEAN_SHARDS))
    rng.shuffle(order)
    frames, have = [], 0
    for shard in order:
        raw = _hf_get(f"{CLEAN}/shard-{shard:05d}-of-{CLEAN_SHARDS:05d}.parquet")
        df = pd.read_parquet(io.BytesIO(raw), columns=["entry_id", "document", "num_tokens"])
        df = df[~df.entry_id.isin(exclude)]
        if have + len(df) > n_docs:
            df = df.iloc[: n_docs - have]
        df = df.copy()
        df["kind"] = "clean"
        frames.append(df)
        have += len(df)
        print(f"  clean shard {shard}: +{len(df):,} (total {have:,}/{n_docs:,})", flush=True)
        if have >= n_docs:
            break
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/mix"))
    ap.add_argument("--clean-frac", type=float, default=0.5)
    ap.add_argument("--backtracking-shards", type=int, default=None,
                    help="limit for a smoke run (default: all 16)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("loading backtracking half ...", flush=True)
    bt = load_backtracking(args.backtracking_shards)
    n_clean = int(len(bt) * args.clean_frac / (1 - args.clean_frac))
    print(f"backtracking: {len(bt):,} -> sampling {n_clean:,} clean documents", flush=True)

    clean = load_clean(n_clean, set(bt.entry_id), args.seed)
    mixed = pd.concat([bt, clean], ignore_index=True)
    # Shuffle so the two kinds interleave; training sees them mixed from step 0.
    mixed = mixed.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    overlap = len(set(bt.entry_id) & set(clean.entry_id))
    assert overlap == 0, f"halves share {overlap} proteins — they must be disjoint"

    out = args.out / "train"
    out.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(mixed), DOCS_PER_OUT_SHARD):
        chunk = mixed.iloc[i : i + DOCS_PER_OUT_SHARD]
        path = out / f"shard-{i // DOCS_PER_OUT_SHARD:05d}.parquet"
        chunk.to_parquet(path, index=False, compression="zstd")
        print(f"  wrote {path.name}: {len(chunk):,}", flush=True)

    print(f"\nmix: {len(mixed):,} documents "
          f"({(mixed.kind == 'backtracking').mean():.1%} backtracking), "
          f"{int(mixed.num_tokens.sum()):,} tokens, protein overlap 0")


if __name__ == "__main__":
    main()
