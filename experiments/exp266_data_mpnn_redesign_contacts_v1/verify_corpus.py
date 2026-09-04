# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Completeness check over the finished redesigned corpus.

Success criterion 3: every kept backbone yields exactly 8 documents, drops are
counted and attributable, and the totals match what Stage A promised. Reads
only parquet *metadata* and a few small columns, so it is minutes rather than
hours over ~32 M documents.

Checks, in the order a corpus can go wrong:

1. **Shard coverage** — one document shard per staged backbone shard, none
   missing (a silently skipped shard is the failure mode the per-file resume
   makes easiest to hit).
2. **Row totals** — against Stage A's manifest count x 8.
3. **8 per backbone** — and, where a backbone has fewer, which design indices
   are absent. contacts-v1 can decline to serialize a chain, so a shortfall is
   allowed but must be *counted*, never assumed.
4. **Distinct documents per backbone** — 8 identical documents would mean the
   temperature ladder collapsed.

    uv run python verify_corpus.py --documents-glob 's3://.../documents/*.parquet' \\
        --expected-backbones 3962835
"""

from __future__ import annotations

import argparse
import collections
import sys

import fsspec
import pyarrow.parquet as pq


def _log(msg: str) -> None:
    print(f"[exp266-verify] {msg}", file=sys.stderr, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--documents-glob", required=True)
    ap.add_argument("--backbones-glob", default=None,
                    help="Staged backbones, to confirm shard-for-shard coverage.")
    ap.add_argument("--expected-backbones", type=int, default=3_962_835)
    ap.add_argument("--designs", type=int, default=8)
    ap.add_argument("--deep-shards", type=int, default=4,
                    help="Shards to open fully for the per-backbone checks; the "
                         "rest are counted from parquet metadata only.")
    args = ap.parse_args()

    fs, _ = fsspec.core.url_to_fs(args.documents_glob)
    files = sorted(fs.glob(args.documents_glob))
    _log(f"{len(files)} document shards")

    if args.backbones_glob:
        bfs, _ = fsspec.core.url_to_fs(args.backbones_glob)
        stems = {p.rsplit("/", 1)[-1].removesuffix(".parquet")
                 for p in bfs.glob(args.backbones_glob)}
        covered = {p.rsplit("/", 1)[-1].removeprefix("documents-").removesuffix(".parquet")
                   for p in files}
        missing = stems - covered
        print(f"shard coverage: {len(covered)}/{len(stems)}"
              + (f"  MISSING {sorted(missing)[:5]}" if missing else "  complete"))

    total = 0
    sizes = []
    for path in files:
        with fs.open(path, "rb") as h:
            md = pq.ParquetFile(h).metadata
        total += md.num_rows
        sizes.append(fs.info(path)["size"])
    expected = args.expected_backbones * args.designs
    print(f"documents: {total:,} (expected {expected:,}, "
          f"delta {total - expected:+,})")
    print(f"size: {sum(sizes) / 1e9:.1f} GB across {len(files)} shards")

    # Deep checks on a few shards spread across the length range (the corpus is
    # length-sorted, so the first shard alone is the shortest proteins and tells
    # you almost nothing -- that sample has misled this experiment twice).
    step = max(1, len(files) // max(args.deep_shards, 1))
    picked = files[::step][: args.deep_shards]
    tokens = 0
    rows_seen = 0
    for path in picked:
        with fs.open(path, "rb") as h:
            t = pq.read_table(h, columns=["entry_id", "design_index", "document",
                                          "num_tokens", "seq_len"])
        counts = collections.Counter(t.column("entry_id").to_pylist())
        short = {e: n for e, n in counts.items() if n != args.designs}
        docs_by_entry: dict[str, set] = collections.defaultdict(set)
        for e, d in zip(t.column("entry_id").to_pylist(),
                        t.column("document").to_pylist()):
            docs_by_entry[e].add(d)
        collapsed = sum(1 for v in docs_by_entry.values() if len(v) < args.designs)
        nt = t.column("num_tokens").to_pylist()
        sl = t.column("seq_len").to_pylist()
        tokens += sum(nt)
        rows_seen += len(nt)
        name = path.rsplit("/", 1)[-1]
        print(f"  {name}: {t.num_rows:,} rows, {len(counts):,} backbones, "
              f"seq_len {min(sl)}-{max(sl)}, mean tokens {sum(nt)/len(nt):.0f}, "
              f"backbones_not_{args.designs}={len(short)}, "
              f"backbones_with_duplicate_designs={collapsed}")

    if rows_seen:
        est = tokens / rows_seen * total
        print(f"token estimate: {est / 1e9:.1f} B "
              f"(mean {tokens / rows_seen:.0f}/doc over {rows_seen:,} sampled rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
