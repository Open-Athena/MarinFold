# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the ESM-Atlas source-part -> corpus-shard map, from parquet footers.

`redesign_esm_cw.py` joins the 65 M-entry ESM-Atlas source (2.08 TB of inline
cif, 3,338 parts) against the decontaminated corpus (3,338 shards of metadata)
that says which of those entries survived #225. Both sides are sharded over
the same entry ids, so the join *should* be file-for-file.

It very nearly is, and that near-miss cost this experiment a rerun. Both
shardings cut the id order into contiguous, non-overlapping ranges, and the
two agree on index for most files -- but the corpus index is not monotonic in
id order, because #225 rewrote the corpus after filtering. Pairing shard *i*
with part *i* is therefore right for most files and *silently partial* for
some: the join still matches rows, just not all of them. Two spot checks said
the assumption held; they were both inside the region where it does.

So the mapping is measured here rather than assumed there. The measurement is
cheap because parquet row-group statistics carry each column's min and max: the
id range of a file comes out of its **footer**, so indexing all 6,676 files
costs 6,676 small range requests and downloads no column data at all.

    uv run --no-project --with 'huggingface_hub>=1.5' --with pyarrow \\
        --with fsspec python build_shard_map.py --out esm_shard_map.json.gz

The output is a gzipped `{source part: [corpus shards]}` -- a few tens of KB,
small enough that the dispatcher ships it inline with the worker sources
instead of staging it, so a task cannot race a half-written copy.
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq

CORPUS = ("buckets/open-athena/MarinFold/data/document_structures/"
          "contacts_v1_esm_atlas_decontam/train")
SOURCE = "buckets/open-athena/esm-atlas-esmfold2-distill/structures/parts"
NUM_SHARDS = 3338


def _log(msg: str) -> None:
    print(f"[exp266-map] {msg}", file=sys.stderr, flush=True)


def id_range(fs, path: str, attempts: int = 5) -> tuple[int, str, str]:
    """(rows, min entry_id, max entry_id) from the parquet footer alone."""
    for attempt in range(attempts):
        try:
            with fs.open(path, "rb") as handle:
                meta = pq.ParquetFile(handle).metadata
                col = meta.schema.names.index("entry_id")
                stats = [meta.row_group(g).column(col).statistics
                         for g in range(meta.num_row_groups)]
                stats = [s for s in stats if s is not None and s.has_min_max]
            if not stats:
                # Without statistics the footer tells us nothing and the whole
                # approach collapses to reading 2 TB of column data. Fail rather
                # than silently fall back to something that slow.
                raise RuntimeError(f"{path}: no entry_id statistics in the footer")
            return (meta.num_rows,
                    min(s.min for s in stats), max(s.max for s in stats))
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def scan(paths: list[str], label: str, workers: int) -> list[tuple[int, str, str]]:
    import os
    import threading

    from huggingface_hub import HfFileSystem

    # Authenticated, and only modestly parallel. The bucket is public, but
    # anonymous reads share a low rate-limit bucket: at 48 threads HF answered
    # with 429s and 148-second backoffs, which is slower than doing it serially.
    # The redesign fan-out is reading the same bucket from 80 tasks at the same
    # time, so this budget is shared -- keep the concurrency here small.
    token = os.environ.get("HF_TOKEN") or True

    local = threading.local()

    def one(path: str):
        fs = getattr(local, "fs", None)
        if fs is None:
            fs = local.fs = HfFileSystem(token=token)
        return id_range(fs, path)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        out = list(pool.map(one, paths))
    _log(f"{label}: {len(out)} footers in {time.perf_counter() - t0:.0f}s")
    return out


def build_map(source: list[tuple[int, str, str]],
              corpus: list[tuple[int, str, str]]) -> dict[int, list[int]]:
    """Source part -> every corpus shard whose id range intersects it.

    Deliberately a brute-force interval intersection, 3,338 x 3,338 string
    comparisons in a few seconds. A binary search would need the corpus shards
    to be a sorted, non-overlapping partition of the id space -- and an earlier
    version asserted exactly that and blew up, because they are not. Assuming
    *anything* about how these two files are laid out relative to each other is
    what this whole module exists to stop doing, and the cheap version of the
    check is not cheap enough to be worth another assumption.

    Overlap is inclusive on both ends, so a shard that shares only its boundary
    id with the part is still returned; an unnecessary shard costs one small
    metadata read in the worker, a missing one costs silently dropped rows.
    """
    mapping: dict[int, list[int]] = {}
    for part, (_rows, lo, hi) in enumerate(source):
        mapping[part] = [i for i, (_r, clo, chi) in enumerate(corpus)
                         if clo <= hi and chi >= lo]
    return mapping


def describe_layout(label: str, ranges: list[tuple[int, str, str]]) -> None:
    """Report whether a sharding is a sorted, non-overlapping partition.

    Reported rather than enforced. The mapping does not depend on it, but it is
    the fact that was wrongly assumed, so it is worth printing every time.
    """
    order = sorted(range(len(ranges)), key=lambda i: ranges[i][1])
    sorted_by_index = order == list(range(len(ranges)))
    overlaps = sum(1 for k in range(len(order) - 1)
                   if ranges[order[k]][2] >= ranges[order[k + 1]][1])
    _log(f"{label}: index order == id order: {sorted_by_index}; "
         f"adjacent range overlaps: {overlaps}/{len(ranges) - 1}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="esm_shard_map.json.gz")
    ap.add_argument("--ranges-out", default=None,
                    help="Also write the raw per-file id ranges, for auditing "
                         "which files the old index assumption got wrong.")
    ap.add_argument("--ranges-in", default=None,
                    help="Reuse a previous scan instead of re-reading footers.")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    if args.ranges_in:
        cached = json.load(open(args.ranges_in))
        source = [tuple(x) for x in cached["source"]]
        corpus = [tuple(x) for x in cached["corpus"]]
        _log(f"reused ranges from {args.ranges_in}")
    else:
        source = scan([f"{SOURCE}/part_{j:05d}.parquet" for j in range(NUM_SHARDS)],
                      "source", args.workers)
        corpus = scan([f"{CORPUS}/shard-{i:05d}-of-{NUM_SHARDS:05d}.parquet"
                       for i in range(NUM_SHARDS)], "corpus", args.workers)

    # Persisted before the mapping, not after: the scan is the expensive half
    # and the mapping is the half likely to raise while its assumptions are
    # still being pinned down. Losing 5 minutes of footer reads to a failed
    # assertion is a self-inflicted wound.
    if args.ranges_out:
        json.dump({"source": source, "corpus": corpus}, open(args.ranges_out, "w"))
        _log(f"wrote {args.ranges_out}")

    describe_layout("source", source)
    describe_layout("corpus", corpus)

    mapping = build_map(source, corpus)
    empty = [j for j, v in mapping.items() if not v]
    if empty:
        raise ValueError(f"{len(empty)} source parts map to no corpus shard "
                         f"(e.g. {empty[:5]}); the two sides do not cover the "
                         f"same id space")

    widths = [len(v) for v in mapping.values()]
    identity = sum(1 for j, v in mapping.items() if v == [j])
    _log(f"corpus shards per source part: {min(widths)}-{max(widths)} "
         f"(mean {sum(widths)/len(widths):.2f}); "
         f"{identity}/{NUM_SHARDS} parts map to [j] alone")

    with gzip.open(args.out, "wt") as handle:
        json.dump({str(k): v for k, v in mapping.items()}, handle)
    _log(f"wrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
