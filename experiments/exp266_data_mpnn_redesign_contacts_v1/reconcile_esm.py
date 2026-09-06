# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Did every kept ESM-Atlas backbone get redesigned exactly once?

Set comparison, not arithmetic. The ESM arm's completeness cannot be settled
by row counts alone for two reasons:

* Shards written before the shard-map fix used an index-aligned join. Where
  that join happened to be right the file is complete, and where it was wrong
  the task died -- but "wrong by a boundary window rather than entirely" would
  have produced a *short* file that no count-based check distinguishes from
  the legitimate `filtered` / `degenerate` drops.
* Those drops are real and are not recoverable from the logs of tasks that
  finished hours ago.

So this compares the actual id sets: everything in the published corpus
against everything that appears in the outputs. What comes back is the exact
list of kept backbones with no document, and -- via the shard map -- the
source parts to re-run with `--force-parts`.

Ids are compared as the first 16 hex characters read as a uint64, which keeps
65.5 M of them in a 520 MB numpy array instead of ~6 GB of Python strings.
At that size the expected number of collisions is ~1e-4, so a false "covered"
is far less likely than any other error in this pipeline.

Runs CoreWeave-side: the outputs are ~250 GB in CoreWeave object storage.

    uv run python dispatch_reconcile_cw.py
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import pathlib
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import fsspec
import numpy as np
import pyarrow.parquet as pq

CORPUS = ("hf://buckets/open-athena/MarinFold/data/document_structures/"
          "contacts_v1_esm_atlas_decontam/train")
NUM_SHARDS = 3338


def _log(msg: str) -> None:
    print(f"[exp266-reconcile] {msg}", file=sys.stderr, flush=True)


def _as_u64(ids: list[str]) -> np.ndarray:
    """First 16 hex chars of each id as uint64."""
    return np.array([int(x[:16], 16) for x in ids], dtype=np.uint64)


def _read_ids(uri: str, attempts: int = 5) -> list[str]:
    for attempt in range(attempts):
        try:
            with fsspec.open(uri, "rb") as handle:
                return pq.read_table(handle, columns=["entry_id"]) \
                         .column("entry_id").to_pylist()
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def _gather(paths: list[str], label: str, workers: int) -> list[np.ndarray]:
    done = [0]
    t0 = time.perf_counter()

    def one(path: str) -> np.ndarray:
        v = _as_u64(_read_ids(path))
        done[0] += 1
        if done[0] % 250 == 0:
            _log(f"  {label} {done[0]}/{len(paths)} ({time.perf_counter()-t0:.0f}s)")
        return v

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, paths))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--documents-glob", required=True)
    ap.add_argument("--shard-map", default="esm_shard_map.json.gz")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--force-out", default=None,
                    help="Write the comma-separated --force-parts list here.")
    args = ap.parse_args()

    fs, _ = fsspec.core.url_to_fs(args.documents_glob)
    out_files = sorted(fs.glob(args.documents_glob))
    _log(f"{len(out_files)} output shards")

    seen_idx = {int(m.group(1)) for p in out_files
                if (m := re.search(r"documents-(\d+)-of-(\d+)\.parquet$", p))}
    missing_shards = sorted(set(range(NUM_SHARDS)) - seen_idx)
    print(f"shard index coverage: {len(seen_idx)}/{NUM_SHARDS}"
          + (f"  MISSING {len(missing_shards)}: {missing_shards[:10]}"
             if missing_shards else "  complete"))

    corpus_paths = [f"{CORPUS}/shard-{i:05d}-of-{NUM_SHARDS:05d}.parquet"
                    for i in range(NUM_SHARDS)]
    per_shard = _gather(corpus_paths, "corpus", args.workers)
    corpus = np.concatenate(per_shard)
    _log(f"corpus: {corpus.size:,} kept backbones")

    produced = np.unique(np.concatenate(
        _gather([fs.unstrip_protocol(p) for p in out_files], "outputs",
                args.workers)))
    _log(f"outputs: {produced.size:,} distinct backbones")

    covered = np.isin(corpus, produced)
    n_missing = int((~covered).sum())
    print(f"\nkept backbones with no document: {n_missing:,} "
          f"({100 * n_missing / corpus.size:.3f}%)")
    print(f"documents' backbones not in the corpus: "
          f"{int(np.isin(produced, corpus, invert=True).sum()):,} "
          f"(must be 0 -- anything else means a bad join wrote foreign rows)")

    # Attribute the misses to corpus shards, then to the source parts that
    # read them. A drop that is spread evenly is the expected filtered /
    # degenerate loss; one concentrated in a few shards is a coverage hole.
    offset, by_shard = 0, collections.Counter()
    for i, ids in enumerate(per_shard):
        by_shard[i] = int((~covered[offset:offset + ids.size]).sum())
        offset += ids.size
    hot = [(i, n) for i, n in by_shard.most_common() if n]
    print(f"corpus shards with any miss: {len(hot)}/{NUM_SHARDS}; "
          f"worst: {hot[:10]}")

    mapping = {int(k): v for k, v in json.load(gzip.open(
        pathlib.Path(__file__).with_name(args.shard_map), "rt")).items()}
    # A shard is "holed" rather than merely lossy if it lost far more than the
    # corpus-wide rate; those are the ones worth re-running.
    rate = n_missing / corpus.size if corpus.size else 0
    holed = {i for i, n in by_shard.items() if n > max(20, 5 * rate * 20_000)}
    force = sorted({j for j, shards in mapping.items()
                    if holed & set(shards)} | set(missing_shards))
    print(f"\nsource parts to re-run (--force-parts): {len(force)}")
    if args.force_out:
        pathlib.Path(args.force_out).write_text(",".join(str(x) for x in force))
        print(f"wrote {args.force_out}")
    elif force:
        print(",".join(str(x) for x in force[:200]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
