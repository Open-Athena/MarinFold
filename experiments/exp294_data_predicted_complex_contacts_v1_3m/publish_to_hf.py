# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish the consolidated corpus to the public `open-athena/MarinFold` bucket.

Runs from the workstation, which is where consolidation put the shards. That
costs one uplink pass over the corpus; staging back through GCS to upload from a
pod would cost the same uplink pass plus a second hop, so there is nothing to
win by moving it.

HF *buckets* are their own repo kind: `upload_file(..., repo_type="bucket")` is
rejected outright, and `snapshot_download` cannot see a bucket either. The
surface is `batch_bucket_files` / `list_bucket_tree` / `download_bucket_files`.

The tokenizer travels with the data, not in a separate repo -- a corpus whose
`num_tokens` cannot be reproduced is not much use. `--verify` re-lists the
bucket afterwards and checks every intended file arrived at its expected size.

    HF_TOKEN=... uv run python publish_to_hf.py --release /data/exp294_release
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BUCKET_REPO = "open-athena/MarinFold"
DEFAULT_PREFIX = "data/document_structures/contacts_v1_complex"
#: HF rate-limits the bucket write endpoint. exp139 saw sustained 429s at 16
#: workers and found 4 both stable and faster overall.
DEFAULT_WORKERS = 4
#: Retries per file. A 429 or a dropped connection mid-shard is routine at this
#: size; a genuinely rejected write still fails after these.
MAX_ATTEMPTS = 6


def _log(message: str) -> None:
    print(f"[exp294-publish] {message}", file=sys.stderr, flush=True)


def plan(release: Path, prefix: str) -> list[tuple[Path, str]]:
    """Every local file paired with the bucket path it is published at.

    Fails if a piece is missing rather than publishing a partial corpus: a
    reader who finds documents but no tokenizer cannot tell that is a mistake.
    """
    prefix = prefix.rstrip("/")
    shards = sorted((release / "corpus" / "train").glob("shard-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no shards under {release / 'corpus' / 'train'}")
    items = [(p, f"{prefix}/train/{p.name}") for p in shards]

    tokenizer = sorted((release / "tokenizer").glob("*"))
    if not tokenizer:
        raise FileNotFoundError(
            f"no tokenizer under {release / 'tokenizer'}; the corpus tokenizer "
            "is published next to the data, not in a separate repo")
    items += [(p, f"{prefix}/tokenizer/{p.name}") for p in tokenizer if p.is_file()]

    for name in ("manifest_natural.parquet", "manifest_balanced.parquet",
                 "corpus_stats.json"):
        path = release / "corpus" / name
        if not path.exists():
            raise FileNotFoundError(f"missing {path}")
        items.append((path, f"{prefix}/{name}"))
    readme = release / "README.md"
    if readme.exists():
        items.append((readme, f"{prefix}/README.md"))
    return items


def _upload(token: str, local: Path, remote: str) -> int:
    from huggingface_hub import batch_bucket_files

    blob = local.read_bytes()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            batch_bucket_files(BUCKET_REPO, add=[(blob, remote)], token=token)
            return len(blob)
        except Exception as error:  # noqa: BLE001 - retried, then re-raised
            if attempt == MAX_ATTEMPTS:
                raise
            delay = min(60.0, 2.0 ** attempt)
            _log(f"{remote}: attempt {attempt} failed ({type(error).__name__}: "
                 f"{error}); retrying in {delay:.0f}s")
            time.sleep(delay)
    raise AssertionError("unreachable")


def verify(items: list[tuple[Path, str]], token: str, prefix: str) -> None:
    from huggingface_hub import list_bucket_tree

    published = {}
    for entry in list_bucket_tree(BUCKET_REPO, prefix.rstrip("/") + "/",
                                  recursive=True, token=token):
        path = getattr(entry, "path", None)
        if path is not None:
            published[path] = getattr(entry, "size", None)
    problems = []
    for local, remote in items:
        if remote not in published:
            problems.append(f"{remote}: absent")
        elif published[remote] not in (None, local.stat().st_size):
            problems.append(f"{remote}: {published[remote]} bytes on the bucket, "
                            f"{local.stat().st_size} locally")
    if problems:
        raise RuntimeError("publication did not verify:\n  " + "\n  ".join(problems))
    _log(f"verified {len(items)} files under {prefix}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)

    items = plan(args.release, args.prefix)
    total = sum(local.stat().st_size for local, _ in items)
    _log(f"{len(items)} files, {total / 1e9:.1f} GB -> {BUCKET_REPO}/{args.prefix}")
    if args.dry_run:
        for local, remote in items[:3] + items[-6:]:
            _log(f"  {local.stat().st_size:>12,}  {remote}")
        return 0

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required (org-scoped for open-athena)")
    if args.verify_only:
        verify(items, token, args.prefix)
        return 0

    started = time.perf_counter()
    sent = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_upload, token, local, remote) for local, remote in items]
        for done, future in enumerate(futures, 1):
            sent += future.result()
            if done % 10 == 0 or done == len(futures):
                elapsed = time.perf_counter() - started
                rate = sent / elapsed / 1e6
                eta = (total - sent) / max(sent / elapsed, 1.0) / 60
                _log(f"{done}/{len(futures)} files, {sent / 1e9:.1f}/{total / 1e9:.1f} GB "
                     f"({rate:.1f} MB/s, ~{eta:.0f} min left)")
    verify(items, token, args.prefix)
    _log(f"published in {(time.perf_counter() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
