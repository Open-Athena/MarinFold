# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage 0 — mirror the ESM-Atlas distillation bucket HF → in-region GCS.

The source ``open-athena/esm-atlas-esmfold2-distill`` is an HF *bucket*
(CloudFront/S3-backed), ~2.08 TB, 3,338 ``structures/parts/part_*.parquet`` +
``selected_manifest.parquet``. Generation runs on Iris CPU workers that land in
**us-central1**, so we stage the parts into a us-central1 GCS bucket ONCE and
run all compute in-region — the only cross-cloud transfer in the whole
experiment (issue #139).

Run this **on an iris CPU pod**, not the workstation: the workstation uplink is
~2.5 MB/s (≈ days for 2 TB), while a pod does cloud-read + cloud-write at
~hundreds of MB/s. The pod's service account already has ``marin-us-central1``
write (same SA that writes marin checkpoints), so no GCS creds are needed; pass
an ``HF_TOKEN`` so the bucket ``paths-info`` endpoint doesn't 429 under
concurrency.

Because the pod disk is small (~32 GB) relative to 2 TB, each file is streamed
**one at a time per worker**: download → ``gcsfs`` put → delete local. Resumable
— a file already present in GCS with a matching size is skipped, so a preempted
run just re-lists and continues.

Launch (from this experiment dir so the pod bundles it), per the cloud-side
mirror recipe::

    /home/bizon/git/marin/.venv/bin/iris --cluster marin job run --no-wait \\
        --enable-extra-resources --cpu=8 --memory=16GB --disk=32GB \\
        --zone=us-central1-a \\
        -e MIRROR_WORKERS 6 -e HF_TOKEN <tok> \\
        -- uv run --with 'huggingface_hub>=1.5,<2' --with hf_xet --with gcsfs \\
           python mirror_source.py

Env vars (all optional except where noted):
    BUCKET_ID       source HF bucket (default open-athena/esm-atlas-esmfold2-distill)
    GCS_DEST        gs:// destination prefix (default the us-central1 source/ path)
    MIRROR_WORKERS  concurrent file movers (default 6; the recipe's 429-safe value)
    HF_TOKEN        HF token (authenticated = higher paths-info limit; recommended)
    MIRROR_TMP      pod-local scratch dir (default /tmp/esm_atlas_mirror)
    INCLUDE_README  "1" to also mirror README.md (default off)
    LIMIT           only mirror the first N parts (smoke; default: all)
    DRY_RUN         "1" to list the plan and exit without copying
"""

import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BUCKET_ID = os.environ.get("BUCKET_ID", "open-athena/esm-atlas-esmfold2-distill")
GCS_DEST = os.environ.get(
    "GCS_DEST",
    "gs://marin-us-central1/protein-structure/MarinFold/"
    "exp139_esm_atlas_contacts_v1/source",
).rstrip("/")
MIRROR_WORKERS = int(os.environ.get("MIRROR_WORKERS", "6"))
HF_TOKEN = os.environ.get("HF_TOKEN") or None
MIRROR_TMP = Path(os.environ.get("MIRROR_TMP", "/tmp/esm_atlas_mirror"))
INCLUDE_README = os.environ.get("INCLUDE_README", "") == "1"
LIMIT = int(os.environ["LIMIT"]) if os.environ.get("LIMIT") else None
DRY_RUN = os.environ.get("DRY_RUN", "") == "1"

_MAX_RETRIES = 6
_BASE_BACKOFF = 2.0  # seconds; exponential, for HF 429s + transient GCS errors


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _want(path: str) -> bool:
    """Which bucket paths we mirror: the parts + the manifest (+ README opt-in)."""
    if path.startswith("structures/parts/") and path.endswith(".parquet"):
        return True
    if path == "selected_manifest.parquet":
        return True
    if path == "README.md" and INCLUDE_README:
        return True
    return False


def _plan():
    """List the bucket and return [(src_path, size_or_None)] we intend to mirror."""
    from huggingface_hub import list_bucket_tree

    entries = []
    for entry in list_bucket_tree(BUCKET_ID, recursive=True, token=HF_TOKEN or None):
        path = getattr(entry, "path", None)
        # BucketFolder has no size; only take file entries.
        if path is None or not hasattr(entry, "size"):
            continue
        if _want(path):
            entries.append((path, getattr(entry, "size", None)))
    # parts sort naturally by name; manifest first is fine either way.
    entries.sort(key=lambda t: t[0])
    if LIMIT is not None:
        parts = [e for e in entries if e[0].startswith("structures/parts/")][:LIMIT]
        others = [e for e in entries if not e[0].startswith("structures/parts/")]
        entries = others + parts
    return entries


def _mirror_one(gcs, src_path: str, size: int | None) -> str:
    """Move one file HF→GCS: skip if already there (size match), else dl→put→rm."""
    from huggingface_hub import download_bucket_files

    dst = f"{GCS_DEST}/{src_path}"
    gcs_path = dst[len("gs://"):]

    # Resume: skip if present with a matching size (or present + size unknown).
    try:
        if gcs.exists(gcs_path):
            if size is None or gcs.size(gcs_path) == size:
                return f"skip  {src_path}"
    except Exception:  # noqa: BLE001 — existence check is best-effort
        pass

    local = MIRROR_TMP / src_path
    local.parent.mkdir(parents=True, exist_ok=True)
    last_exc = None
    for attempt in range(_MAX_RETRIES):
        try:
            download_bucket_files(
                BUCKET_ID, files=[(src_path, str(local))],
                raise_on_missing_files=True, token=HF_TOKEN or None,
            )
            gcs.put_file(str(local), gcs_path)
            got = local.stat().st_size
            return f"put   {src_path} ({got:,} B)"
        except Exception as exc:  # noqa: BLE001 — retry 429 / transient GCS
            last_exc = exc
            sleep = _BASE_BACKOFF * (2 ** attempt)
            _log(f"retry {src_path} attempt {attempt + 1}/{_MAX_RETRIES} "
                 f"after {sleep:.0f}s: {exc!r}")
            time.sleep(sleep)
        finally:
            try:
                if local.exists():
                    local.unlink()
            except OSError:
                pass
    raise RuntimeError(f"giving up on {src_path}: {last_exc!r}")


def main() -> int:
    plan = _plan()
    n_parts = sum(1 for p, _ in plan if p.startswith("structures/parts/"))
    total_bytes = sum(s or 0 for _, s in plan)
    _log(f"[mirror] {BUCKET_ID} -> {GCS_DEST}")
    _log(f"[mirror] {len(plan)} files to mirror ({n_parts} parts), "
         f"~{total_bytes / 1e12:.2f} TB, {MIRROR_WORKERS} workers"
         + (" [DRY_RUN]" if DRY_RUN else ""))
    for p, s in plan[:3]:
        _log(f"[mirror]   e.g. {p} ({(s or 0):,} B) -> {GCS_DEST}/{p}")
    if DRY_RUN:
        return 0

    import gcsfs

    gcs = gcsfs.GCSFileSystem()
    MIRROR_TMP.mkdir(parents=True, exist_ok=True)

    done = failed = 0
    with ThreadPoolExecutor(max_workers=MIRROR_WORKERS,
                            thread_name_prefix="mirror") as pool:
        futs = {pool.submit(_mirror_one, gcs, p, s): p for p, s in plan}
        for fut in as_completed(futs):
            src = futs[fut]
            try:
                _log(f"[mirror] {fut.result()}  [{done + failed + 1}/{len(plan)}]")
                done += 1
            except Exception:  # noqa: BLE001 — one file's failure is not fatal
                failed += 1
                _log(f"[mirror] FAIL {src}\n{traceback.format_exc()}")

    _log(f"[mirror] complete: {done} ok, {failed} failed of {len(plan)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
