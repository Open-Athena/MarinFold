# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Fetch afdb-24M's *small* columns into local parquet (rate-limit safe + resumable).

Stage A needs only the small columns of all 12,005 afdb-24M shards (never
``cif_content``). Two failure modes have to be avoided at this scale:

* **API quota.** Reading via duckdb's ``hf://`` makes one ``paths-info``
  **API** call per shard, blowing HuggingFace's 3,000-req / 5-min API quota.
  We instead use :class:`HfFileSystem` with a **pre-warmed directory cache**
  (one ``ls`` per shard dir), so per-file ``info()`` is served from cache and
  the only network traffic per shard is the authenticated data read of the
  parquet footer + small-column chunks over the ``/resolve/`` CDN.
* **Download throttling.** Those data reads are **authenticated** (the stored
  HF token → team-tier limits) at modest concurrency, instead of anonymous
  reads that get throttled.

Output is **one parquet per shard** in a directory, written only if not
already present — so the job is resumable: re-run it and it fills only the
gaps. ``selection.py`` then points ``--input`` at that directory.

    uv run python fetch_manifest_columns.py --out ~/exp53_scratch/afdb24m_small
    uv run python selection.py --input ~/exp53_scratch/afdb24m_small --out ...
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

from selection import MANIFEST_COLUMNS

DEFAULT_REPO = "timodonnell/afdb-24M"


def list_shards_warming_cache(fs: HfFileSystem, repo: str) -> list[str]:
    """List every ``.parquet`` shard path, warming ``fs``'s dircache as we go.

    Lists each shard subdirectory once (a handful of API calls total) so the
    file sizes are cached; subsequent per-file ``info()`` during reads then
    needs no further API call.
    """
    base = f"datasets/{repo}"
    shards: list[str] = []
    for entry in fs.ls(base, detail=True):
        if entry["type"] != "directory":
            continue
        for f in fs.ls(entry["name"], detail=True):
            if f["name"].endswith(".parquet"):
                shards.append(f["name"])
    return sorted(shards)


def read_shard(
    fs: HfFileSystem, path: str, columns: list[str], *, retries: int = 8
) -> pa.Table:
    """Range-read ``columns`` from one shard (authenticated), with backoff."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with fs.open(path, "rb") as f:
                return pq.read_table(f, columns=columns)
        except Exception as exc:  # noqa: BLE001 — retry any transient read error
            last = exc
            time.sleep(min(5 * (attempt + 1), 60))  # 5,10,…,60s — rides out a 5-min window
    raise RuntimeError(f"failed to read {path} after {retries} tries: {type(last).__name__}: {last}")


def _out_name(shard_path: str) -> str:
    """Local filename for a shard (basenames are globally unique: shard_<N>.parquet)."""
    return shard_path.rsplit("/", 1)[-1]


def fetch(
    repo: str,
    out_dir: Path,
    *,
    columns: list[str],
    concurrency: int = 16,
    limit: int | None = None,
) -> tuple[int, list[str]]:
    """Write one small-columns parquet per shard into ``out_dir`` (resumable).

    Returns ``(num_shards_written_total, failed_paths)``. Skips shards whose
    output file already exists, so re-running fills only the gaps.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()  # uses the stored HF token automatically
    shards = list_shards_warming_cache(fs, repo)
    if limit is not None:
        shards = shards[:limit]
    todo = [s for s in shards if not (out_dir / _out_name(s)).exists()]
    have = len(shards) - len(todo)
    print(f"[fetch] {len(shards)} shards ({have} already present, {len(todo)} to fetch), "
          f"columns={columns}", file=sys.stderr)

    def work(path: str) -> str:
        table = read_shard(fs, path, columns)
        pq.write_table(table, out_dir / _out_name(path))
        return path

    failed: list[str] = []
    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(work, s): s for s in todo}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001 — one bad shard must not abort the run
                failed.append(futures[fut])
                print(f"[fetch] FAILED {futures[fut]}: {exc}", file=sys.stderr)
            done += 1
            if done % 500 == 0 or done == len(todo):
                rate = done / (time.time() - t0)
                eta = (len(todo) - done) / rate if rate else 0
                print(f"[fetch] {done}/{len(todo)} fetched  {rate:.1f} shard/s  "
                      f"eta {eta/60:.1f} min  failed={len(failed)}", file=sys.stderr)

    present = sum(1 for s in shards if (out_dir / _out_name(s)).exists())
    print(f"[fetch] {present}/{len(shards)} shards present in {out_dir} "
          f"({(time.time()-t0)/60:.1f} min; {len(failed)} failed this pass)", file=sys.stderr)
    return present, failed


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python fetch_manifest_columns.py")
    p.add_argument("--repo", default=DEFAULT_REPO, help=f"HF dataset (default {DEFAULT_REPO}).")
    p.add_argument("--out", required=True, type=Path,
                   help="Output directory (one small-columns parquet per shard).")
    p.add_argument("--concurrency", type=int, default=16, help="Concurrent shard reads (default 16).")
    p.add_argument("--limit", type=int, default=None, help="Fetch only the first N shards (smoke).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    present, failed = fetch(args.repo, args.out, columns=list(MANIFEST_COLUMNS),
                            concurrency=args.concurrency, limit=args.limit)
    if failed:
        print(f"[fetch] {len(failed)} shards still missing — re-run to fill gaps.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
