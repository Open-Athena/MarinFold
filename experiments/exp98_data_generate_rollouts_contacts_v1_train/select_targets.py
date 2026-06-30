# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage A (local, CPU): sample the exp98 rollout targets from the contacts-v1
**train** split.

The contacts-v1 corpus (exp53) lives at
``gs://marin-us-east5/protein-structure/MarinFold/exp53_contacts_v1_5x/documents/train/``
as ``contacts_v1-{NNNNN}-of-02067.parquet``, physically ordered round-descending
(round-4 shards first … round-0 last), each shard a single pLDDT round.

We sample **1000 round-0 targets** (highest pLDDT == cleanest GT) with
``L <= --max-len`` (512), ``not truncated``, and ``>= --min-contacts`` (5)
ground-truth contacts (seq-sep >= 6 — the contacts-v1 contact definition). GT
contacts come straight from the document text (no pyconfind). We store a
one-letter ``sequence`` so the rollout worker can rebuild fresh contacts-v1
realizations via ``residues_from_sequence`` (a faithful round-trip — verified
here on the sample).

Output ``data/targets.parquet`` (+ optional GCS copy): one row per target with
``entry_id, L, sequence, n_gt, gt_contacts ([[i,j],…], seq-index space),
global_plddt, struct_cluster_id, round, shard``.

    uv run python select_targets.py --n 1000 --max-len 512 --min-contacts 5 \
        --out data/targets.parquet \
        --gcs-out gs://marin-us-east5/protein-structure/MarinFold/exp98_rollouts_contacts_v1_train/targets.parquet
"""
from __future__ import annotations

import argparse
import random
import re

import gcsfs
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from marinfold.document_structures.contacts_v1 import residues_from_sequence
from marinfold.document_structures.contacts_v1.parse import _ONE_LETTER_TO_THREE

NUM_POS = 2000          # contacts-v1 position-token wrap
MIN_SEP = 6             # contacts-v1 min_seq_separation
TRAIN_DIR = ("marin-us-east5/protein-structure/MarinFold/exp53_contacts_v1_5x/"
             "documents/train")
N_SHARDS = 2067
BEGIN = "<begin_statements>"
CONTACT_RE = re.compile(r"<contact>\s+<p(\d+)>\s+<p(\d+)>")
NTERM_RE = re.compile(r"<n-term>\s+<p(\d+)>")
RES_RE = re.compile(r"<p(\d+)>\s+<([A-Z]{3})>")

THREE_TO_ONE = {three: one for one, three in _ONE_LETTER_TO_THREE.items()}


def shard_path(i: int) -> str:
    return f"{TRAIN_DIR}/contacts_v1-{i:05d}-of-{N_SHARDS:05d}.parquet"


def parse_doc(doc: str):
    """(L, one-letter sequence, sorted GT pairs in seq-index space) or None."""
    cut = doc.index(BEGIN) + len(BEGIN)
    prefix, struct = doc[:cut], doc[cut:]
    m = NTERM_RE.search(prefix)
    if not m:
        return None
    nterm = int(m.group(1))
    pos_in_seq = sorted({int(p) for p in re.findall(r"<p(\d+)>", prefix)},
                        key=lambda p: (p - nterm) % NUM_POS)
    seqidx = {p: (p - nterm) % NUM_POS for p in pos_in_seq}
    res_of_pos = {int(p): aa for p, aa in RES_RE.findall(prefix)}
    if not all(p in res_of_pos for p in pos_in_seq):
        return None
    seq = "".join(THREE_TO_ONE.get(res_of_pos[p], "X") for p in pos_in_seq)
    gt = set()
    for a, b in CONTACT_RE.findall(struct):
        ia, ib = seqidx.get(int(a)), seqidx.get(int(b))
        if ia is None or ib is None or ia == ib or abs(ia - ib) < MIN_SEP:
            continue
        gt.add((min(ia, ib), max(ia, ib)))
    return len(pos_in_seq), seq, sorted(gt)


def shard_round(fs, i: int) -> int:
    """The (single) pLDDT round of shard ``i`` — read just the round column."""
    with fs.open(shard_path(i), "rb") as fh:
        col = pq.read_table(fh, columns=["round"]).column("round").to_pylist()
    return col[0]


def first_round0_shard(fs) -> int:
    """Binary-search the first shard whose round == 0 (rounds are descending)."""
    lo, hi = 0, N_SHARDS - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if shard_round(fs, mid) == 0:
            hi = mid
        else:
            lo = mid + 1
    assert shard_round(fs, lo) == 0, "no round-0 shards found"
    return lo


def verify_roundtrip(seq: str, doc_resnames: list[str]) -> bool:
    """residues_from_sequence(seq) must reproduce the document's 3-letter names."""
    rebuilt = [r.resname for r in residues_from_sequence(seq)]
    return rebuilt == doc_resnames


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--min-contacts", type=int, default=5)
    ap.add_argument("--pool-mult", type=int, default=8,
                    help="collect ~pool-mult*n candidates before sampling")
    ap.add_argument("--seed", type=int, default=98)
    ap.add_argument("--out", default="data/targets.parquet")
    ap.add_argument("--gcs-out", default=None)
    args = ap.parse_args()

    fs = gcsfs.GCSFileSystem()
    start0 = first_round0_shard(fs)
    round0_shards = list(range(start0, N_SHARDS))
    print(f"round-0 shards: [{start0}, {N_SHARDS - 1}]  ({len(round0_shards)} shards)",
          flush=True)

    rng = random.Random(args.seed)
    order = round0_shards[:]
    rng.shuffle(order)

    pool: list[dict] = []
    want_pool = args.pool_mult * args.n
    cols = ["document", "seq_len", "entry_id", "global_plddt",
            "struct_cluster_id", "round", "truncated"]
    n_shards_read = 0
    for si in order:
        if len(pool) >= want_pool:
            break
        with fs.open(shard_path(si), "rb") as fh:
            df = pq.read_table(fh, columns=cols).to_pandas()
        n_shards_read += 1
        df = df[(df["round"] == 0) & (~df["truncated"]) & (df["seq_len"] <= args.max_len)]
        for _, r in df.iterrows():
            parsed = parse_doc(r["document"])
            if parsed is None:
                continue
            L, seq, gt = parsed
            if len(gt) < args.min_contacts:
                continue
            pool.append(dict(
                entry_id=r["entry_id"], L=L, sequence=seq, n_gt=len(gt),
                gt_contacts=[[i, j] for (i, j) in gt],
                global_plddt=float(r["global_plddt"]),
                struct_cluster_id=r["struct_cluster_id"], round=0, shard=si,
            ))
    print(f"read {n_shards_read} round-0 shards -> pool {len(pool)} candidates", flush=True)
    if len(pool) < args.n:
        raise SystemExit(f"only {len(pool)} candidates < requested {args.n}")

    # dedup by entry_id (paranoia), then sample.
    seen, dedup = set(), []
    for c in pool:
        if c["entry_id"] not in seen:
            seen.add(c["entry_id"]); dedup.append(c)
    sample = rng.sample(dedup, args.n)
    sample.sort(key=lambda c: c["entry_id"])
    df = pd.DataFrame(sample)

    # round-trip sanity on a handful: residues_from_sequence(seq) must match the
    # document's 3-letter resnames (else the worker would feed a different chain
    # than the GT was computed on).
    for c in rng.sample(sample, min(20, len(sample))):
        with fs.open(shard_path(c["shard"]), "rb") as fh:
            doc = (pq.read_table(fh, columns=["document", "entry_id"]).to_pandas()
                   .set_index("entry_id").loc[c["entry_id"], "document"])
        cut = doc.index(BEGIN) + len(BEGIN)
        prefix = doc[:cut]
        nterm = int(NTERM_RE.search(prefix).group(1))
        pos = sorted({int(p) for p in re.findall(r"<p(\d+)>", prefix)},
                     key=lambda p: (p - nterm) % NUM_POS)
        rmap = {int(p): aa for p, aa in RES_RE.findall(prefix)}
        doc_resnames = [rmap[p] for p in pos]
        assert verify_roundtrip(c["sequence"], doc_resnames), f"roundtrip fail {c['entry_id']}"
    print("round-trip verified on sample", flush=True)

    # summary
    print(f"\nselected {len(df)} targets")
    print("  L:        ", df["L"].describe()[["min", "mean", "50%", "max"]].round(1).to_dict())
    print("  n_gt:     ", df["n_gt"].describe()[["min", "mean", "50%", "max"]].round(1).to_dict())
    print("  plddt:    ", df["global_plddt"].describe()[["min", "mean", "max"]].round(1).to_dict())
    print("  uniq clusters:", df["struct_cluster_id"].nunique())

    table = pa.Table.from_pandas(df, preserve_index=False)
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pq.write_table(table, args.out)
    print(f"wrote {args.out}")
    if args.gcs_out:
        with fs.open(args.gcs_out, "wb") as fh:
            pq.write_table(table, fh)
        print(f"wrote {args.gcs_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
