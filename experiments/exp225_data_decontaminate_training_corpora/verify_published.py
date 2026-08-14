# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Check the *published* corpus, not the local build that produced it.

:mod:`filter_corpus` already asserts its own row count before it will write a
dataset README, but that proves something about a directory on a disk. What a
training run actually reads is the bucket, and between the two sit a 130 GB
upload and a sync tool. So this reads the published prefix back and checks four
things, all against the drop list rather than against the filter's own output:

* **Every shard is present.** A sync that quietly skipped one would otherwise
  surface much later as a slightly short training run.
* **The row count** equals ``n_documents - n_dropped`` exactly.
* **No dropped ``entry_id`` survives.** This is the one that matters — it is
  the direct statement that the published corpus is decontaminated.
* **``entry_id`` is still unique**, so nothing was duplicated in transit.

Only the ``entry_id`` column is read, so the check covers every row for a small
fraction of the corpus's bytes. It is still 3,338 round trips for ESM-Atlas;
run it where the bandwidth is.

    uv run python verify_published.py --arm afdb
    uv run python verify_published.py --arm esm_atlas --workers 48
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from build_corpus_index import BUCKET
from decontam_lib import ARMS, CORPORA, REFERENCE_VERSION, Corpus
from filter_corpus import _filesystem

HERE = Path(__file__).resolve().parent


def read_entry_ids(job: tuple[Corpus, int]) -> tuple[int, list[str] | None]:
    """The ``entry_id`` column of one published shard; ``None`` if it is absent."""
    corpus, shard = job
    path = f"{BUCKET}/{corpus.decontam_prefix}/{corpus.shard_name.format(shard)}"
    fs = _filesystem()
    if not fs.exists(path):
        return shard, None
    with fs.open(path, "rb") as fh:
        blob = fh.read()
    table = pq.read_table(pa.BufferReader(blob), columns=["entry_id"])
    return shard, table.column("entry_id").to_pylist()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--droplist", type=Path,
                    default=Path("/data/exp225_decontam/droplist_final.parquet"))
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", type=Path, default=None,
                    help="default: data/published_verification_<arm>.json")
    args = ap.parse_args()

    corpus = CORPORA[args.arm]
    out = args.out or HERE / f"data/published_verification_{args.arm}.json"

    droplist = pd.read_parquet(args.droplist, columns=["arm", "entry_id"])
    dropped = set(droplist.loc[droplist["arm"] == args.arm, "entry_id"])
    expected = corpus.n_documents - len(dropped)
    print(f"[{args.arm}] expecting {expected:,} rows over {corpus.n_shards} shards "
          f"at {corpus.decontam_prefix}", flush=True)

    surviving: set[str] = set()
    n_rows = 0
    contaminated: list[str] = []
    missing_shards: list[int] = []
    jobs = [(corpus, shard) for shard in range(corpus.n_shards)]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for done, (shard, ids) in enumerate(pool.map(read_entry_ids, jobs), 1):
            if ids is None:
                missing_shards.append(shard)
                continue
            n_rows += len(ids)
            surviving.update(ids)
            contaminated.extend(i for i in ids if i in dropped)
            if done % 500 == 0 or done == corpus.n_shards:
                print(f"[{args.arm}] {done}/{corpus.n_shards} shards, {n_rows:,} rows",
                      flush=True)

    report = {
        "arm": args.arm,
        "prefix": corpus.decontam_prefix,
        "reference_version": REFERENCE_VERSION,
        "shards_expected": corpus.n_shards,
        "shards_missing": missing_shards,
        "rows_expected": expected,
        "rows_found": n_rows,
        "rows_match": n_rows == expected,
        "contaminated_rows_surviving": len(contaminated),
        "contaminated_examples": contaminated[:10],
        "entry_id_unique": len(surviving) == n_rows,
        "documents_removed": len(dropped),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)

    failures = []
    if missing_shards:
        failures.append(f"{len(missing_shards)} shards missing from the bucket")
    if n_rows != expected:
        failures.append(f"row count {n_rows:,} != expected {expected:,}")
    if contaminated:
        failures.append(f"{len(contaminated):,} contaminated rows survived the filter")
    if len(surviving) != n_rows:
        failures.append(f"entry_id not unique: {len(surviving):,} distinct of {n_rows:,}")
    if failures:
        raise SystemExit(
            f"[{args.arm}] PUBLISHED CORPUS FAILED VERIFICATION: " + "; ".join(failures)
        )
    print(f"[{args.arm}] published corpus verified: {n_rows:,} rows, no contamination",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
