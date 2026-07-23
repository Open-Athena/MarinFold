# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage the contacts-v1 corpus into the CoreWeave S3 bucket (issue #109).

CoreWeave task pods on ``cw-rno2a`` carry a SINGLE S3 endpoint/credential set,
so every input the training job reads must live under the CoreWeave bucket
``s3://marin-us-east-02a``. This script copies the contacts-v1 train/val (and
optionally test) parquet shards to::

    s3://marin-us-east-02a/MarinFold/data/document_structures/contacts_v1/<split>/

matching ``CONTACTS_V1_S3_CORPUS_BASE`` in ``contacts_v1_train_common.py``.

Source
------
Default is the **GCS working copy** exp53 wrote — byte-identical to the HF
publish and directly accessible with the workstation's gcloud creds::

    gs://marin-us-east5/protein-structure/MarinFold/exp53_contacts_v1_5x/documents/

(The canonical HF publish is ``hf://buckets/open-athena/MarinFold/data/
document_structures/contacts_v1/``. HF *buckets* aren't listable via the normal
dataset/model repo API — ``list_repo_files`` 401s — so pass ``--source hf`` only
if you have bucket-scoped HF auth; otherwise the GCS mirror is the reliable
source and produces identical bytes.)

Credentials / endpoint
----------------------
Reads the CoreWeave object-storage key from the environment. Source the
workstation env file first::

    set -a; source ~/.config/marin/cw-rno2a.env; set +a   # CW_KEY_ID/SECRET, endpoint
    python stage_data_to_coreweave.py --splits train val         # full stage
    python stage_data_to_coreweave.py --splits train --limit 1   # one-shard smoke test

Idempotent: an S3 object whose size already matches the source is skipped, so
re-running resumes an interrupted transfer.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import sys

import boto3
from botocore.config import Config

GCS_SOURCE_BASE = "gs://marin-us-east5/protein-structure/MarinFold/exp53_contacts_v1_5x/documents"
HF_SOURCE_BASE = "hf://buckets/open-athena/MarinFold/data/document_structures/contacts_v1"

S3_BUCKET = "marin-us-east-02a"
S3_PREFIX = "MarinFold/data/document_structures/contacts_v1"
# From OUTSIDE CoreWeave use https://cwobject.com (in-cluster jobs use LOTA at
# http://cwlota.com). Virtual-hosted addressing is required (path-style is
# rejected by CoreWeave AI Object Storage).
DEFAULT_ENDPOINT = "https://cwobject.com"


def _s3_client():
    key = os.environ.get("CW_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID")
    secret = os.environ.get("CW_KEY_SECRET") or os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not key or not secret:
        sys.exit(
            "CoreWeave object-storage creds not found. Run:\n"
            "  set -a; source ~/.config/marin/cw-rno2a.env; set +a"
        )
    endpoint = os.environ.get("AWS_ENDPOINT_URL", DEFAULT_ENDPOINT)
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key,
        aws_secret_access_key=secret,
        config=Config(s3={"addressing_style": "virtual"}, retries={"max_attempts": 10, "mode": "adaptive"}),
    )


def _open_fs(source: str):
    """Return an fsspec filesystem + the source base URL for ``source``."""
    import fsspec

    if source == "gcs":
        return fsspec.filesystem("gcs"), GCS_SOURCE_BASE
    if source == "hf":
        return fsspec.filesystem("hf"), HF_SOURCE_BASE
    raise ValueError(f"unknown source {source!r}")


def _list_shards(fs, base: str, split: str) -> list[str]:
    paths = fs.glob(f"{base}/{split}/*.parquet")
    # fsspec strips the protocol on gcs globs; normalize to bare keys.
    return sorted(p if "://" not in p else p.split("://", 1)[1] for p in paths)


def _stage_one(fs, s3, src_path: str, split: str, dry_run: bool) -> tuple[str, str]:
    fname = src_path.rsplit("/", 1)[-1]
    key = f"{S3_PREFIX}/{split}/{fname}"
    src_size = fs.size(src_path)
    try:
        head = s3.head_object(Bucket=S3_BUCKET, Key=key)
        if head["ContentLength"] == src_size:
            return key, "skip(exists)"
    except Exception:
        pass
    if dry_run:
        return key, f"would-copy({src_size}B)"
    with fs.open(src_path, "rb") as fh:
        s3.upload_fileobj(fh, S3_BUCKET, key)
    return key, f"copied({src_size}B)"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="stage_data_to_coreweave.py")
    ap.add_argument("--source", choices=("gcs", "hf"), default="gcs", help="corpus source (default gcs mirror)")
    ap.add_argument("--splits", nargs="+", default=["train", "val"], help="splits to stage (default: train val)")
    ap.add_argument("--limit", type=int, default=0, help="cap shards per split (0 = all); use 1 for a smoke test")
    ap.add_argument("--workers", type=int, default=16, help="parallel transfers")
    ap.add_argument("--dry-run", action="store_true", help="list what would copy, transfer nothing")
    args = ap.parse_args(argv)

    fs, base = _open_fs(args.source)
    s3 = _s3_client()

    total_copied = total_skipped = 0
    for split in args.splits:
        shards = _list_shards(fs, base, split)
        if args.limit:
            shards = shards[: args.limit]
        print(f"[{split}] {len(shards)} shard(s) from {args.source}:{base}/{split}/ "
              f"-> s3://{S3_BUCKET}/{S3_PREFIX}/{split}/")
        with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_stage_one, fs, s3, p, split, args.dry_run): p for p in shards}
            for i, fut in enumerate(cf.as_completed(futs), 1):
                key, status = fut.result()
                if status.startswith("skip"):
                    total_skipped += 1
                else:
                    total_copied += 1
                if i % 50 == 0 or i == len(shards) or args.limit:
                    print(f"  [{split}] {i}/{len(shards)} {status:>18}  {key}")
    print(f"done: {total_copied} copied, {total_skipped} skipped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
