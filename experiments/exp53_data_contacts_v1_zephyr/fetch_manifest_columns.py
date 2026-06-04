# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Fetch afdb-24M's *small* columns into one local parquet (rate-limit safe).

Stage A needs only the small columns of all 12,005 afdb-24M shards (never
``cif_content``). Reading them straight through duckdb's ``hf://`` /
``HfFileSystem`` makes one ``paths-info`` **API** call per shard, which blows
HuggingFace's 3,000-request / 5-minute API quota at this scale.

This helper avoids the metadata API entirely: it lists the shard paths once
(one tree call), then reads each shard's small columns over the public CDN
``/resolve/`` URL with HTTP range requests (parquet footer + the small column
chunks only — tens of KB per shard, not the ~100 MB file), concurrently. The
result is a single consolidated parquet that ``selection.py`` then runs on
locally and repeatedly, with no further network traffic::

    uv run python fetch_manifest_columns.py --out ~/exp53_scratch/afdb24m_small.parquet
    uv run python selection.py --input ~/exp53_scratch/afdb24m_small.parquet --out ...
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_url

from selection import MANIFEST_COLUMNS

DEFAULT_REPO = "timodonnell/afdb-24M"


def list_shard_paths(repo: str) -> list[str]:
    """All ``.parquet`` shard paths in the dataset repo (one tree API call)."""
    files = HfApi().list_repo_files(repo, repo_type="dataset")
    return sorted(f for f in files if f.endswith(".parquet"))


def read_shard_columns(
    path: str, repo: str, columns: list[str], *, retries: int = 6
) -> pa.Table:
    """Range-read ``columns`` from one shard via its public CDN resolve URL.

    Hits ``huggingface.co/.../resolve/...`` (file serving), not the
    rate-limited ``/api/`` endpoints. Retries with exponential backoff on
    transient HTTP errors.
    """
    url = hf_hub_url(repo_id=repo, filename=path, repo_type="dataset")
    https = fsspec.filesystem("https")
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with https.open(url, "rb") as f:
                return pq.read_table(f, columns=columns)
        except Exception as exc:  # noqa: BLE001 — retry any transient read error
            last = exc
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"failed to read {path} after {retries} tries: {last}")


def fetch(
    repo: str,
    out_path: str,
    *,
    columns: list[str],
    concurrency: int = 24,
    limit: int | None = None,
) -> int:
    """Consolidate the small columns of every shard into one parquet at ``out_path``."""
    shards = list_shard_paths(repo)
    if limit is not None:
        shards = shards[:limit]
    print(f"[fetch] {len(shards)} shards, columns={columns}", file=sys.stderr)

    writer: pq.ParquetWriter | None = None
    done = 0
    rows = 0
    t0 = time.time()
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(read_shard_columns, s, repo, columns): s for s in shards
            }
            for fut in as_completed(futures):
                table = fut.result()
                if writer is None:
                    writer = pq.ParquetWriter(out_path, table.schema)
                writer.write_table(table)
                done += 1
                rows += table.num_rows
                if done % 500 == 0 or done == len(shards):
                    rate = done / (time.time() - t0)
                    eta = (len(shards) - done) / rate if rate else 0
                    print(f"[fetch] {done}/{len(shards)} shards  {rows:,} rows  "
                          f"{rate:.1f} shard/s  eta {eta/60:.1f} min", file=sys.stderr)
    finally:
        if writer is not None:
            writer.close()
    print(f"[fetch] wrote {rows:,} rows to {out_path} in {(time.time()-t0)/60:.1f} min",
          file=sys.stderr)
    return rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python fetch_manifest_columns.py")
    p.add_argument("--repo", default=DEFAULT_REPO, help=f"HF dataset (default {DEFAULT_REPO}).")
    p.add_argument("--out", required=True, help="Output consolidated parquet path.")
    p.add_argument("--concurrency", type=int, default=24,
                   help="Concurrent shard reads (default 24).")
    p.add_argument("--limit", type=int, default=None,
                   help="Read only the first N shards (smoke test).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fetch(args.repo, args.out, columns=list(MANIFEST_COLUMNS),
          concurrency=args.concurrency, limit=args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
