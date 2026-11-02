# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Stream the materialized exp91 structure parts from S3 into a HuggingFace bucket.

The 66.76M-structure dataset lives as ~3,338 Parquet parts under
``s3://marinfold-exp91-usw2/exp91/structures/parts/`` (~1.9 TiB of mmCIF text).
The ``hf`` CLI cannot read ``s3://`` directly, so this uploader bridges the two:
for each part it downloads from S3 to a local staging dir, ``hf buckets cp``s it
into the target bucket, then deletes the local copy — so we never hold more than
``--workers`` parts on disk at once (never the full 1.9 TiB).

It mirrors the ``silterra/afdb-24M-foldseek-train-reps`` bucket layout: a flat
payload prefix (``structures/parts/part_NNNNN.parquet``) plus a README + manifest
uploaded separately.

RESUMABLE: a part already present in the target bucket at the same size is
skipped, so a killed run resumes for free. Use ``--overwrite`` to force re-upload.

SMOKE TEST: ``--limit N`` uploads only the first N parts (sorted by name), so
``--limit 2`` moves ~1 GB in a minute to prove the path end to end before you
commit to the full ~1.9 TiB (and its one-time ~$170 AWS->HF egress).

COST / SIGN-OFF: the full run is a cross-cloud, >10 GB transfer — a real egress
event that needs human sign-off per the repo rules. HF storage itself is cheap
(~$23/mo pay-as-you-go, free within an Enterprise plan's base) and HF read
egress is free; the AWS-side egress to push it is the one-time cost. The default
target is the ``silterra/...`` dev bucket for testing; point ``--target-bucket``
at ``open-athena/...`` for the real publish once write access is granted.

Requires: ``pyarrow`` (S3 reads via ``pyarrow.fs``, already an exp91 dep -- we do
NOT use boto3, which proved unreliable to install on the ephemeral upload workers),
AWS creds in the environment (``source aws_keys.env`` or an EC2 instance role), and
the ``hf`` CLI on PATH, logged in with write access to ``--target-bucket``.

Examples
--------
Smoke test (2 parts) into the dev bucket::

    source aws_keys.env
    uv run python upload_to_hf.py --limit 2

Dry run — show exactly what would move, transfer nothing::

    uv run python upload_to_hf.py --dry-run

Full publish into the org bucket (after sign-off + write access)::

    uv run python upload_to_hf.py \
        --target-bucket open-athena/MarinFold \
        --dest-prefix data/exp91-esm-atlas-esmfold2/structures/parts
"""

import argparse
import concurrent.futures as cf
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

# DO NOT add `import boto3`. S3 access here goes through pyarrow.fs (see _s3fs /
# _s3_download below), matching the materialize/funnel path. Reason: on the ephemeral
# EC2 upload workers (Ubuntu 22.04, launched from cloud-init), `pip install boto3`
# reliably wedged and the box self-terminated ~60s into bootstrap
# (Client.InstanceInitiatedShutdown) -- pinning boto3+botocore did not help, while
# pyarrow and huggingface_hub[cli] install cleanly. pyarrow is already required for the
# manifest conversion, so it costs us nothing extra. (This bug cost ~2 days to isolate.)
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
from pyarrow import fs as pafs

DEFAULT_SOURCE = "s3://marinfold-exp91-usw2/exp91/structures/parts/"
DEFAULT_TARGET_BUCKET = "silterra/esm-atlas-esmfold2-distill"
DEFAULT_DEST_PREFIX = "structures/parts"
DEFAULT_MANIFEST_URI = "s3://marinfold-exp91-usw2/exp91/out/selected_manifest.csv"

# Column types for the manifest Parquet. The CSV writes three pLDDT/pTM columns at
# full float64 precision (~18-20 ASCII chars each) — the dominant source of its
# 8.64 GB bulk. float32 is ample for 0-1 confidence scores and cuts each to 4 bytes.
# The two 32-hex-char hash columns are high-entropy (near-incompressible), so the
# Parquet lands ~3 GB (a ~2.5-3x shrink), not the order-of-magnitude the floats
# alone would suggest. Hashes stay string so they still join to the parts' string
# ``entry_id``. Unknown/absent columns are simply inferred/skipped.
_MANIFEST_TYPES = {
    "cluster_id": pa.string(),
    "protein_hash": pa.string(),
    "seq_len": pa.int32(),
    "mean_plddt": pa.float32(),
    "ptm": pa.float32(),
    "plddt_std": pa.float32(),
    "cluster_size": pa.int32(),
}

# ---------------------------------------------------------------------------
# Logging (mirrors materialize.py: timestamped, flushed).
# ---------------------------------------------------------------------------
_T0 = time.monotonic()
_LOG_LOCK = threading.Lock()


def _log(msg: str) -> None:
    with _LOG_LOCK:
        print(f"[{time.strftime('%H:%M:%S', time.gmtime())}]"
              f"[{time.monotonic() - _T0:7.1f}s] {msg}", flush=True)


def _human(n: int) -> str:
    x = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if x < 1024 or unit == "TiB":
            return f"{x:.1f} {unit}"
        x /= 1024
    return f"{x:.1f} TiB"


@dataclass(frozen=True)
class Part:
    """One S3 object to move: its key, basename, and size in bytes."""
    key: str
    name: str
    size: int


def _split_s3_uri(uri: str) -> tuple[str, str]:
    """Split ``s3://bucket/prefix`` into ``(bucket, prefix)`` (prefix keeps no leading /)."""
    p = urlparse(uri)
    if p.scheme != "s3":
        raise ValueError(f"--source must be an s3:// URI, got {uri!r}")
    return p.netloc, p.path.lstrip("/")


@lru_cache(maxsize=None)
def _s3fs(bucket: str) -> pafs.S3FileSystem:
    """A region-correct, thread-safe ``S3FileSystem`` for ``bucket`` (cached).

    We read S3 through pyarrow (not boto3) to match the materialize/funnel path:
    boto3 is deliberately NOT a dependency because installing it on the ephemeral
    upload workers proved unreliable, whereas pyarrow is already required for the
    manifest conversion. Credentials come from the instance role automatically.
    """
    region = pafs.resolve_s3_region(bucket)
    return pafs.S3FileSystem(region=region)


def _s3_download(bucket: str, key: str, local: Path) -> None:
    """Stream ``s3://bucket/key`` to ``local`` in chunks (no whole-file buffering)."""
    fs = _s3fs(bucket)
    with fs.open_input_stream(f"{bucket}/{key}") as src, open(local, "wb") as dst:
        while chunk := src.read(8 * 1024 * 1024):
            dst.write(chunk)


def list_source_parts(source_uri: str) -> list[Part]:
    """List ``*.parquet`` objects under the S3 source prefix, sorted by name.

    Sorted so ``--limit`` is deterministic (first N part_NNNNN by name) and a
    resumed run walks parts in the same order every time.
    """
    bucket, prefix = _split_s3_uri(source_uri)
    fs = _s3fs(bucket)
    selector = pafs.FileSelector(f"{bucket}/{prefix.rstrip('/')}", recursive=True)
    parts: list[Part] = []
    for info in fs.get_file_info(selector):
        if info.type != pafs.FileType.File or not info.path.endswith(".parquet"):
            continue
        key = info.path[len(bucket) + 1:]  # strip leading "bucket/"
        parts.append(Part(key=key, name=info.path.rsplit("/", 1)[-1], size=info.size))
    parts.sort(key=lambda p: p.name)
    return parts


def existing_bucket_sizes(target_bucket: str, dest_prefix: str) -> dict[str, int]:
    """Return ``{filename: size_bytes}`` already present under the bucket dest prefix.

    Uses ``hf buckets list --recursive --format json``. An empty/absent prefix
    yields an empty dict (nothing uploaded yet), which is the fresh-bucket case.
    """
    target = f"{target_bucket}/{dest_prefix}".rstrip("/")
    proc = subprocess.run(
        ["hf", "buckets", "list", target, "--recursive", "--format", "json"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        # Prefix not existing yet is normal on first run; log stderr for anything else.
        stderr = proc.stderr.strip()
        if stderr:
            _log(f"[skip-scan] '{target}' not listable yet ({stderr.splitlines()[-1]}); "
                 "treating as empty")
        return {}
    out = proc.stdout.strip()
    if not out:
        return {}
    sizes: dict[str, int] = {}
    for entry in json.loads(out):
        # Entries carry a path (relative to the bucket) and a size in bytes.
        path = entry.get("path") or entry.get("name") or ""
        name = path.rsplit("/", 1)[-1]
        if name.endswith(".parquet"):
            sizes[name] = int(entry.get("size", 0))
    return sizes


def _hf_cp(local: Path, dest: str) -> None:
    """``hf buckets cp`` a local file to a bucket ``hf://`` dest; raise on failure."""
    proc = subprocess.run(
        ["hf", "buckets", "cp", str(local), dest],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"hf buckets cp failed for {dest} (rc={proc.returncode}): "
            f"{proc.stderr.strip()[-500:]}")


def _root_join(dataset_root: str, name: str) -> str:
    """Join a bucket dataset-root prefix with a filename (root '' -> bucket root)."""
    dataset_root = dataset_root.strip("/")
    return f"{dataset_root}/{name}" if dataset_root else name


def convert_manifest_to_parquet(csv_path: Path, out_path: Path) -> tuple[int, int]:
    """Stream ``csv_path`` -> zstd Parquet at ``out_path`` with typed columns.

    Streaming (batch-at-a-time) so a multi-GB manifest never lands fully in RAM.
    Returns ``(rows, out_size_bytes)``.
    """
    read_opts = pacsv.ReadOptions(block_size=64 << 20)
    convert_opts = pacsv.ConvertOptions(column_types=_MANIFEST_TYPES)
    reader = pacsv.open_csv(csv_path, read_options=read_opts, convert_options=convert_opts)
    writer = None
    rows = 0
    try:
        for batch in reader:
            if writer is None:
                writer = pq.ParquetWriter(out_path, batch.schema, compression="zstd")
            writer.write_batch(batch)
            rows += batch.num_rows
    finally:
        if writer is not None:
            writer.close()
    return rows, out_path.stat().st_size


def stage_and_convert_manifest(manifest_uri: str, stage_dir: Path) -> Path:
    """Download the manifest CSV from S3, convert to Parquet in ``stage_dir``.

    The (large) CSV is removed once converted; the returned Parquet path is what
    gets uploaded. Returns the local Parquet path.
    """
    bucket, key = _split_s3_uri(manifest_uri)
    local_csv = stage_dir / Path(key).name
    _log(f"[manifest] downloading {manifest_uri}")
    t0 = time.monotonic()
    _s3_download(bucket, key, local_csv)
    csv_size = local_csv.stat().st_size
    _log(f"[manifest] downloaded {_human(csv_size)} in {time.monotonic() - t0:.0f}s; "
         "converting to zstd Parquet")
    out = stage_dir / "selected_manifest.parquet"
    t1 = time.monotonic()
    rows, out_size = convert_manifest_to_parquet(local_csv, out)
    local_csv.unlink(missing_ok=True)
    _log(f"[manifest] {rows:,} rows -> {_human(out_size)} Parquet in "
         f"{time.monotonic() - t1:.0f}s ({csv_size / max(out_size, 1):.1f}x smaller than CSV)")
    return out


def upload_one(part: Part, source_bucket: str, target_bucket: str, dest_prefix: str,
               stage_dir: Path) -> int:
    """Download one part from S3 to ``stage_dir`` and ``hf buckets cp`` it up.

    Returns the bytes moved. The local temp file is always removed, even on
    failure, so a long run never leaks disk. Any non-zero ``hf`` exit raises.
    """
    local = stage_dir / part.name
    dest = f"hf://buckets/{target_bucket}/{dest_prefix}/{part.name}"
    try:
        t0 = time.monotonic()
        _s3_download(source_bucket, part.key, local)
        dl = time.monotonic() - t0
        t1 = time.monotonic()
        _hf_cp(local, dest)
        up = time.monotonic() - t1
        _log(f"[ok] {part.name} {_human(part.size)} "
             f"(s3 {dl:.1f}s @ {_human(int(part.size / max(dl, 1e-3)))}/s, "
             f"hf {up:.1f}s @ {_human(int(part.size / max(up, 1e-3)))}/s)")
        return part.size
    finally:
        local.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default=DEFAULT_SOURCE,
                    help=f"S3 prefix of the *.parquet parts (default {DEFAULT_SOURCE})")
    ap.add_argument("--target-bucket", default=DEFAULT_TARGET_BUCKET,
                    help=f"HF bucket id <owner>/<name> (default {DEFAULT_TARGET_BUCKET})")
    ap.add_argument("--dest-prefix", default=DEFAULT_DEST_PREFIX,
                    help=f"path within the bucket (default {DEFAULT_DEST_PREFIX})")
    ap.add_argument("--limit", type=int, default=None,
                    help="upload only the first N parts (by name) — smoke test")
    ap.add_argument("--workers", type=int, default=4,
                    help="concurrent parts in flight (default 4; each holds ~0.6 GB on disk)")
    ap.add_argument("--stage-dir", type=Path, default=None,
                    help="local staging dir (default a fresh system temp dir)")
    ap.add_argument("--create-bucket", action="store_true",
                    help="create the target bucket first (exist-ok) before uploading")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-upload parts (and README/manifest) even if already present")
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would move and exit; transfer nothing")
    ap.add_argument("--dataset-root", default="",
                    help="bucket path for README/manifest (default bucket root). Set to "
                         "the dataset dir when publishing into a shared bucket, e.g. "
                         "data/exp91-esm-atlas-esmfold2")
    ap.add_argument("--with-readme", action="store_true",
                    help="also upload the dataset card to <dataset-root>/README.md")
    ap.add_argument("--readme", type=Path, default=Path(__file__).parent / "BUCKET_README.md",
                    help="local dataset-card file to upload (default BUCKET_README.md)")
    ap.add_argument("--with-manifest", action="store_true",
                    help="also convert selected_manifest.csv -> Parquet and upload to "
                         "<dataset-root>/selected_manifest.parquet")
    ap.add_argument("--manifest-uri", default=DEFAULT_MANIFEST_URI,
                    help=f"S3 URI of the manifest CSV to convert (default {DEFAULT_MANIFEST_URI})")
    args = ap.parse_args(argv)

    source_bucket, _ = _split_s3_uri(args.source)

    _log(f"=== upload: {args.source} -> hf://buckets/{args.target_bucket}/{args.dest_prefix}")
    parts = list_source_parts(args.source)
    _log(f"[scan] {len(parts):,} source parts, {_human(sum(p.size for p in parts))} total")
    if args.limit is not None:
        parts = parts[:args.limit]
        _log(f"[limit] capped to first {len(parts):,} parts "
             f"({_human(sum(p.size for p in parts))})")

    # Resume: skip parts already in the bucket at the same size.
    if args.overwrite:
        todo = parts
        if parts:
            _log("[skip-scan] --overwrite set; re-uploading all selected parts")
    else:
        existing = existing_bucket_sizes(args.target_bucket, args.dest_prefix)
        todo = [p for p in parts if existing.get(p.name) != p.size]
        _log(f"[skip-scan] {len(existing):,} parts already in bucket; "
             f"skipping {len(parts) - len(todo):,}, uploading {len(todo):,}")

    # Metadata: whether the README / manifest still need uploading.
    do_readme = args.with_readme
    root_existing = (existing_bucket_sizes(args.target_bucket, args.dataset_root)
                     if args.with_manifest and not args.overwrite else {})
    do_manifest = args.with_manifest and (
        args.overwrite or "selected_manifest.parquet" not in root_existing)
    if args.with_manifest and not do_manifest:
        _log("[manifest] selected_manifest.parquet already in bucket; skipping "
             "(use --overwrite to replace)")

    todo_bytes = sum(p.size for p in todo)
    if args.dry_run:
        _log(f"[dry-run] would upload {len(todo):,} parts ({_human(todo_bytes)})")
        for p in todo[:10]:
            _log(f"[dry-run]   {p.name}  {_human(p.size)}")
        if len(todo) > 10:
            _log(f"[dry-run]   ... and {len(todo) - 10:,} more")
        if do_readme:
            _log(f"[dry-run] would upload README -> {_root_join(args.dataset_root, 'README.md')}")
        if do_manifest:
            _log(f"[dry-run] would convert {args.manifest_uri} -> "
                 f"{_root_join(args.dataset_root, 'selected_manifest.parquet')}")
        _log("[dry-run] no transfer performed")
        return

    if not todo and not do_readme and not do_manifest:
        _log("[done] everything already present — nothing to do")
        return

    if args.create_bucket:
        _log(f"[create] hf buckets create {args.target_bucket} --exist-ok")
        proc = subprocess.run(
            ["hf", "buckets", "create", args.target_bucket, "--exist-ok"],
            capture_output=True, text=True)
        if proc.returncode != 0:
            _log(f"[create] FAILED: {proc.stderr.strip()}")
            sys.exit(1)

    stage_dir = args.stage_dir or Path(tempfile.mkdtemp(prefix="exp91-hf-upload-"))
    stage_dir.mkdir(parents=True, exist_ok=True)
    owns_stage = args.stage_dir is None

    moved = 0
    done = 0
    failures: list[str] = []
    t_start = time.monotonic()
    try:
        if todo:
            _log(f"[go] uploading {len(todo):,} parts ({_human(todo_bytes)}) "
                 f"via {args.workers} workers; staging in {stage_dir}")
            with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {
                    ex.submit(upload_one, p, source_bucket, args.target_bucket,
                              args.dest_prefix, stage_dir): p
                    for p in todo
                }
                for fut in cf.as_completed(futs):
                    p = futs[fut]
                    try:
                        moved += fut.result()
                        done += 1
                    except Exception as exc:  # noqa: BLE001 - report + continue, tally at end
                        failures.append(p.name)
                        _log(f"[FAIL] {p.name}: {exc}")
                    if done and done % 25 == 0:
                        rate = moved / max(time.monotonic() - t_start, 1e-3)
                        eta = (todo_bytes - moved) / max(rate, 1e-3)
                        _log(f"[progress] {done:,}/{len(todo):,} parts, "
                             f"{_human(moved)} moved @ {_human(int(rate))}/s, "
                             f"ETA {eta / 3600:.1f}h")
            _log(f"[parts] {done:,}/{len(todo):,} uploaded, {_human(moved)} in "
                 f"{(time.monotonic() - t_start) / 60:.1f} min ({len(failures):,} failures)")

        # Metadata after the parts, so a partial parts run never leaves a bucket
        # that looks documented/complete but isn't.
        if do_manifest:
            try:
                manifest_pq = stage_and_convert_manifest(args.manifest_uri, stage_dir)
                dest = f"hf://buckets/{args.target_bucket}/" \
                       f"{_root_join(args.dataset_root, 'selected_manifest.parquet')}"
                _hf_cp(manifest_pq, dest)
                manifest_pq.unlink(missing_ok=True)
                _log(f"[manifest] uploaded -> {dest}")
            except Exception as exc:  # noqa: BLE001 - report + continue to tally
                failures.append("selected_manifest.parquet")
                _log(f"[FAIL] manifest: {exc}")

        if do_readme:
            try:
                if not args.readme.exists():
                    raise FileNotFoundError(f"README not found: {args.readme}")
                dest = f"hf://buckets/{args.target_bucket}/" \
                       f"{_root_join(args.dataset_root, 'README.md')}"
                _hf_cp(args.readme, dest)
                _log(f"[readme] uploaded {args.readme.name} -> {dest}")
            except Exception as exc:  # noqa: BLE001 - report + continue to tally
                failures.append("README.md")
                _log(f"[FAIL] readme: {exc}")
    finally:
        if owns_stage:
            shutil.rmtree(stage_dir, ignore_errors=True)

    _log(f"=== DONE in {(time.monotonic() - t_start) / 60:.1f} min "
         f"({len(failures):,} failures)")
    if failures:
        _log(f"[failures] {', '.join(failures[:20])}"
             + (" ..." if len(failures) > 20 else ""))
        sys.exit(1)


if __name__ == "__main__":
    main()
