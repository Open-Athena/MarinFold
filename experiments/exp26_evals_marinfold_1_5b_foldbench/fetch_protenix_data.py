# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Pull the Protenix-on-FoldBench artifacts we need from the HF bucket.

The full exp12 upload at ``open-athena/MarinFold`` is ~30 GB once
``best/`` and ``msa/`` are included. For exp20 we only need the small
subset:

- ``manifest.csv`` — 100 proteins
- ``scores.csv`` — 200 Protenix rows (single_seq + msa)
- ``scores_all_samples.csv`` — 8000 rows (per-sample, optional)
- ``scores_summary.csv`` — per-mode aggregates
- ``gt/*.cif`` — 100 GT biological-assembly mmCIFs (~26 MB)

We skip Protenix's ``best/`` predictions (curated top-1 sample CIFs +
per-seed distograms) — exp20's only Protenix dependency is the score
CSVs themselves.

Uses ``HfApi.download_bucket_files`` (requires huggingface_hub >=1.5).
The standard ``snapshot_download`` doesn't see bucket contents — buckets
are a separate storage layer from datasets/models.

This script's HF hub pin (>=1.5) is incompatible with the main exp20
venv (transformers 4.x caps huggingface_hub at <1.0). To work around
that, **run this with an ephemeral env** that pulls a newer hub:

    uv run --with "huggingface_hub>=1.5" python fetch_protenix_data.py

Output goes under ``./protenix_data/`` (gitignored). Re-runnable: the
bucket download is content-addressed, so unchanged files are no-ops.
"""

import argparse
import fnmatch
from pathlib import Path

from huggingface_hub import HfApi


BUCKET_ID = "open-athena/MarinFold"
REMOTE_PREFIX = "data/protenix-foldbench-monomers"

_ALLOW_PATTERNS: tuple[str, ...] = (
    f"{REMOTE_PREFIX}/manifest.csv",
    f"{REMOTE_PREFIX}/scores.csv",
    f"{REMOTE_PREFIX}/scores_all_samples.csv",
    f"{REMOTE_PREFIX}/scores_summary.csv",
    f"{REMOTE_PREFIX}/gt/*.cif",
)


def _files_to_download(api: HfApi, bucket_id: str) -> list[str]:
    """Return the subset of bucket paths that match one of _ALLOW_PATTERNS."""
    all_paths: list[str] = []
    for entry in api.list_bucket_tree(bucket_id, recursive=True):
        # entry is a BucketFile or BucketFolder; only download files.
        if getattr(entry, "type", None) == "directory":
            continue
        path = entry.path
        if any(fnmatch.fnmatch(path, pat) for pat in _ALLOW_PATTERNS):
            all_paths.append(path)
    return all_paths


def fetch(out_dir: Path) -> Path:
    """Download the Protenix data subset into ``out_dir``.

    Returns the path to the ``data/protenix-foldbench-monomers/``
    folder inside ``out_dir`` (i.e. the local mirror of REMOTE_PREFIX).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    paths = _files_to_download(api, BUCKET_ID)
    if not paths:
        raise RuntimeError(
            f"no matching files in bucket {BUCKET_ID} for patterns "
            f"{_ALLOW_PATTERNS!r}"
        )
    print(f"Downloading {len(paths)} files from bucket {BUCKET_ID} → {out_dir}/")
    # download_bucket_files takes a list of (remote, local) pairs.
    pairs = [(p, str(out_dir / p)) for p in paths]
    api.download_bucket_files(BUCKET_ID, files=pairs)
    return out_dir / REMOTE_PREFIX


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "protenix_data",
        help="Local mirror root (default: ./protenix_data/ next to this script).",
    )
    args = parser.parse_args()
    local_prefix = fetch(args.out)
    n_gt = sum(1 for _ in (local_prefix / "gt").glob("*.cif"))
    print(f"Fetched into {local_prefix} ({n_gt} GT mmCIFs).")


if __name__ == "__main__":
    main()
