# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a light per-row index of a published corpus: ids only, no documents.

Two things need this and neither needs the documents themselves:

* **The fold-level purge (Tier C).** The AFDB corpus carries
  ``struct_cluster_id`` on every row, and the Foldseek axis produces a list of
  *clusters* to purge. Turning that into a document count — and later into a
  row filter — needs the cluster of every row.
* **The per-axis breakout.** "How many drops are structure-only" is a per-row
  join between the sequence drop list and the structural one, so both have to
  be expressible over the same rows.

The corpora are 13 GB (AFDB) and 133 GB (ESM-Atlas) of parquet, but the id
columns are a tiny fraction of that. Reading them over ``HfFileSystem`` with a
column projection pulls only the relevant column chunks — the whole AFDB index
costs a couple of hundred MB of transfer instead of 13 GB — so this is minutes,
not an overnight stream. (The bucket is public: ``token=False`` and no auth.)

    uv run python build_corpus_index.py --arm afdb --workers 24
    uv run python build_corpus_index.py --arm esm_atlas --limit-shards 20   # spot check
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

from decontam_lib import ARMS, CORPORA, Corpus

BUCKET = "buckets/open-athena/MarinFold"

#: Id columns worth indexing, in preference order. Only those a corpus actually
#: has are read — ESM-Atlas has no structural cluster, which is precisely why
#: its fold-level purge needs a Foldseek database built from scratch.
ID_COLUMNS = ("entry_id", "struct_cluster_id", "split")


def shard_columns(fs: HfFileSystem, corpus: Corpus) -> list[str]:
    """The subset of :data:`ID_COLUMNS` shard 0 actually carries."""
    with fs.open(f"{BUCKET}/{corpus.remote(0)}", "rb") as fh:
        present = set(pq.ParquetFile(fh).schema_arrow.names)
    columns = [c for c in ID_COLUMNS if c in present]
    if "entry_id" not in columns:
        raise SystemExit(f"{corpus.arm}: shard 0 has no entry_id column")
    return columns


def read_shard(args: tuple[Corpus, int, list[str]]) -> pa.Table:
    """One shard's id columns, plus its own shard/row coordinates.

    The ``(shard, row)`` pair is what makes an index row line up with an
    exp213 FASTA header, so the sequence drop list can be joined on either the
    ``entry_id`` (what a filter matches) or the coordinates (what proves the
    filter hit the row it meant to).
    """
    corpus, shard, columns = args
    fs = HfFileSystem(token=False)
    with fs.open(f"{BUCKET}/{corpus.remote(shard)}", "rb") as fh:
        table = pq.ParquetFile(fh).read(columns=columns)
    n = table.num_rows
    return table.append_column(
        "shard", pa.array([shard] * n, pa.int32())
    ).append_column("row", pa.array(range(n), pa.int32()))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <work>/index_<arm>.parquet")
    ap.add_argument("--work", type=Path, default=Path("/data/exp225_decontam"))
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit-shards", type=int, default=None,
                    help="spot check: index only this many shards (the row-count "
                         "assertion is skipped)")
    args = ap.parse_args()

    corpus = CORPORA[args.arm]
    out = args.out or args.work / f"index_{args.arm}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)

    fs = HfFileSystem(token=False)
    columns = shard_columns(fs, corpus)
    n_shards = min(args.limit_shards or corpus.n_shards, corpus.n_shards)
    print(f"[{corpus.arm}] {n_shards} shards, columns {columns}", flush=True)

    t0 = time.time()
    tables: list[pa.Table] = []
    jobs = [(corpus, shard, columns) for shard in range(n_shards)]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for done, table in enumerate(pool.map(read_shard, jobs), 1):
            tables.append(table)
            if done % 200 == 0 or done == n_shards:
                rate = done / max(time.time() - t0, 1e-9)
                print(f"[{corpus.arm}] {done}/{n_shards} shards, "
                      f"{time.time() - t0:.0f}s (eta {(n_shards - done) / rate / 60:.1f} min)",
                      flush=True)

    index = pa.concat_tables(tables)
    if args.limit_shards is None and index.num_rows != corpus.n_documents:
        raise SystemExit(
            f"{corpus.arm}: indexed {index.num_rows:,} rows but the registry says "
            f"{corpus.n_documents:,} — the corpus or the registry is wrong, and a "
            "survival percentage against the wrong denominator is worse than none"
        )
    pq.write_table(index, out, compression="zstd")
    print(f"[{corpus.arm}] {index.num_rows:,} rows -> {out} "
          f"({out.stat().st_size / 1e6:.0f} MB, {time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
