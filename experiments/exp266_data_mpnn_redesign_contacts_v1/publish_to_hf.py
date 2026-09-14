# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish the redesigned corpus from CoreWeave object storage to the HF bucket.

Runs **on a CoreWeave pod**, not the workstation: the documents live in
CoreWeave object storage and the workstation uplink is ~2.5 MB/s, so pulling
~100 GB down and pushing it back up would take days. A pod has the CoreWeave
credentials injected and public internet to huggingface.co, so it can stream
shard-by-shard.

Uses `batch_bucket_files`, **not** `upload_file`. HF *buckets* are a distinct
repo kind: `upload_file(..., repo_type="bucket")` raises
`ValueError: Invalid repo type, must be one of [None, 'model', 'dataset',
'space']`. The bucket API is its own surface (`batch_bucket_files`,
`sync_bucket`, `list_bucket_tree`, `download_bucket_files`), which is the same
reason `snapshot_download` cannot see a bucket.

Each shard is streamed from object storage and handed over as raw bytes, so
nothing is staged to the pod's disk — 64 GB would not fit in the task's
allocation anyway.

Needs an **open-athena-scoped** token (`hf auth whoami` must list the org; the
bare workstation token is timodonnell-only). Pass it as `HF_TOKEN`.

Writes to `data/document_structures/contacts_v1_mpnn_redesign/train/`, matching
the layout of `contacts_v1` and `contacts_v1_decontam`.

    python publish_to_hf.py --documents-glob 's3://.../documents/*.parquet' \\
        --repo-path data/document_structures/contacts_v1_mpnn_redesign/train
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import fsspec

BUCKET_REPO = "open-athena/MarinFold"


def _log(msg: str) -> None:
    print(f"[exp266-publish] {msg}", file=sys.stderr, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--documents-glob", required=True)
    ap.add_argument("--repo-path", required=True,
                    help="Destination prefix inside the bucket repo.")
    ap.add_argument("--workers", type=int, default=4,
                    help="Parallel uploads. HF rate-limits the bucket write "
                         "endpoint: exp139 saw sustained 429s at 16 workers and "
                         "found 4 both stable and *faster* overall.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import batch_bucket_files

    token = os.environ.get("HF_TOKEN")
    if not token and not args.dry_run:
        raise SystemExit("HF_TOKEN is required (org-scoped for open-athena)")

    fs, _ = fsspec.core.url_to_fs(args.documents_glob)
    files = sorted(fs.glob(args.documents_glob))
    if not files:
        raise FileNotFoundError(f"no documents match {args.documents_glob}")
    total = sum(fs.info(f)["size"] for f in files)
    _log(f"{len(files)} shards, {total / 1e9:.1f} GB -> "
         f"{BUCKET_REPO}/{args.repo_path}")
    if args.dry_run:
        return 0

    started = time.perf_counter()
    from concurrent.futures import ThreadPoolExecutor

    def upload(path: str) -> None:
        name = path.rsplit("/", 1)[-1]
        with fs.open(path, "rb") as handle:
            blob = handle.read()
        batch_bucket_files(
            BUCKET_REPO,
            add=[(blob, f"{args.repo_path.rstrip('/')}/{name}")],
            token=token,
        )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, _ in enumerate(pool.map(upload, files), 1):
            if i % 10 == 0 or i == len(files):
                rate = i / (time.perf_counter() - started)
                _log(f"{i}/{len(files)} shards ({rate * 3600:.0f}/h)")

    _log(f"published in {(time.perf_counter() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
