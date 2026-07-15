# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the Arm A (baseline / re-epoch) corpus: the ORIGINAL contacts-v1 round-0
documents for **exactly** the proteins in the exp100 regenerated set.

The two arms must differ only in document content, so Arm A is aligned to Arm B's
protein set (the 941,004 ``entry_id``s that survived exp100's correctness gate —
NOT all 941,028 round-0 proteins). We inner-join the round-0 train shards against
the mirrored regen ``entry_id``s and keep the original ``document`` verbatim.

This is also where we MEASURE the matched training budget: it sums ``num_tokens``
over the aligned corpus and prints ``EXP120_STEPS_PER_EPOCH`` for a given batch x
seq. Arm B has the same per-protein token count by construction (same proteins,
same true-contact set, order-only difference), so both arms use this value.

    uv run python build_arm_a_aligned.py            # reads regen entry_ids from the mirror

Prereq: mirror_regen_train.py has written the regen shards (we read their
``entry_id`` column as the target protein set).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import gcsfs
import pyarrow as pa
import pyarrow.parquet as pq

PREFIX = ("marin-us-east5/protein-structure/MarinFold/"
          "exp120_regen_vs_reepoch_contacts_v1/data")
REGEN_DIR = f"gs://{PREFIX}/regen_train"
DEST_DIR = f"gs://{PREFIX}/orig_r0_train"

TRAIN_DIR = ("marin-us-east5/protein-structure/MarinFold/exp53_contacts_v1_5x/"
             "documents/train")
N_SHARDS = 2067


def shard_path(i: int) -> str:
    return f"{TRAIN_DIR}/contacts_v1-{i:05d}-of-{N_SHARDS:05d}.parquet"


def shard_round(fs, i: int) -> int:
    with fs.open(shard_path(i), "rb") as fh:
        return pq.read_table(fh, columns=["round"]).column("round").to_pylist()[0]


def first_round0_shard(fs) -> int:
    """Binary-search the first round-0 shard (shards are round-descending)."""
    lo, hi = 0, N_SHARDS - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if shard_round(fs, mid) == 0:
            hi = mid
        else:
            lo = mid + 1
    assert shard_round(fs, lo) == 0, "no round-0 shards found"
    return lo


def load_regen_entry_ids(fs, regen_dir: str) -> set[str]:
    ids: set[str] = set()
    paths = [p for p in fs.ls(regen_dir, detail=False) if p.endswith(".parquet")]
    if not paths:
        raise SystemExit(f"no regen shards under {regen_dir}; run mirror_regen_train.py first")
    for p in paths:
        with fs.open(p, "rb") as fh:
            ids.update(pq.read_table(fh, columns=["entry_id"]).column("entry_id").to_pylist())
    return ids


def process_shard(fs, si: int, want: set[str]) -> tuple[list[dict], int]:
    with fs.open(shard_path(si), "rb") as fh:
        tbl = pq.read_table(fh, columns=["entry_id", "document", "round", "num_tokens"]).to_pylist()
    rows, toks = [], 0
    for r in tbl:
        if r["round"] != 0 or r["entry_id"] not in want:
            continue
        rows.append({"entry_id": r["entry_id"], "document": r["document"]})
        toks += int(r["num_tokens"]) if r["num_tokens"] is not None else 0
    return rows, toks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regen-dir", default=REGEN_DIR)
    ap.add_argument("--dest", default=DEST_DIR)
    ap.add_argument("--shards-out", type=int, default=64)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seq", type=int, default=8192)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()

    fs = gcsfs.GCSFileSystem()
    want = load_regen_entry_ids(fs, a.regen_dir)
    print(f"regen target proteins: {len(want)}", flush=True)

    start0 = first_round0_shard(fs)
    shards = list(range(start0, N_SHARDS))
    print(f"round-0 train shards: [{start0}, {N_SHARDS-1}] ({len(shards)})", flush=True)

    rows: list[dict] = []
    total_tokens = 0
    done = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(process_shard, fs, si, want): si for si in shards}
        for fut in as_completed(futs):
            r, t = fut.result()
            rows.extend(r); total_tokens += t
            done += 1
            if done % 50 == 0 or done == len(shards):
                print(f"  {done}/{len(shards)} shards, {len(rows)} matched", flush=True)

    found_ids = {r["entry_id"] for r in rows}
    missing = want - found_ids
    print(f"\nmatched {len(rows)} / {len(want)} regen proteins in round-0 train", flush=True)
    if missing:
        print(f"WARNING: {len(missing)} regen entry_ids not found in round-0 train "
              f"(e.g. {sorted(missing)[:3]})", flush=True)

    rows.sort(key=lambda c: c["entry_id"])
    dest = a.dest.rstrip("/")
    rows_per = (len(rows) + a.shards_out - 1) // a.shards_out
    for si in range(a.shards_out):
        lo, hi = si * rows_per, min((si + 1) * rows_per, len(rows))
        if lo >= hi:
            break
        table = pa.Table.from_pylist(rows[lo:hi])
        with fs.open(f"{dest}/orig_r0_train-{si:05d}-of-{a.shards_out:05d}.parquet", "wb") as fh:
            pq.write_table(table, fh)
    print(f"wrote {len(rows)} docs to {dest}", flush=True)

    per_step = a.batch * a.seq
    spe = -(-total_tokens // per_step)  # ceil
    print(f"\n=== matched training budget ===", flush=True)
    print(f"total train tokens (Arm A; Arm B identical): {total_tokens:,}", flush=True)
    print(f"tokens/step (batch {a.batch} x seq {a.seq}): {per_step:,}", flush=True)
    print(f"EXP120_STEPS_PER_EPOCH = {spe}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
