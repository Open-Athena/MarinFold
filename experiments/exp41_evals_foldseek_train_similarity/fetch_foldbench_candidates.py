# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Pull the FoldBench-100 ground-truth mmCIFs as a candidate set.

These are the same 100 biological-assembly mmCIFs exp20/exp26 evaluate
against (one per FoldBench monomer, ~26 MB total). Here they are the
first *candidate* set for the foldseek train-similarity query tool: we
ask how close each is to anything in MarinFold's training set.

Mirrors ``exp20/fetch_protenix_data.py`` — bucket downloads use
``HfApi.download_bucket_files`` (needs ``huggingface_hub >= 1.5``);
``snapshot_download`` does not see bucket contents. Unlike exp20 this
experiment has no transformers pin, so the newer hub installs cleanly
in the experiment venv and no ephemeral ``uv run --with`` is required.

Output goes under ``./candidates/foldbench/`` (gitignored). Re-runnable:
the bucket download is content-addressed, so unchanged files are no-ops.
"""

import argparse
import fnmatch
from pathlib import Path

from huggingface_hub import HfApi

BUCKET_ID = "open-athena/MarinFold"
REMOTE_PREFIX = "data/protenix-foldbench-monomers"

_ALLOW_PATTERNS: tuple[str, ...] = (
    f"{REMOTE_PREFIX}/manifest.csv",
    f"{REMOTE_PREFIX}/gt/*.cif",
)


def _files_to_download(api: HfApi, bucket_id: str) -> list[str]:
    """Return the bucket paths matching one of ``_ALLOW_PATTERNS``."""
    paths: list[str] = []
    for entry in api.list_bucket_tree(bucket_id, recursive=True):
        if entry.type == "directory":
            continue
        path = entry.path
        if any(fnmatch.fnmatch(path, pat) for pat in _ALLOW_PATTERNS):
            paths.append(path)
    return paths


def fetch(out_dir: Path) -> Path:
    """Download the FoldBench GT cifs into ``out_dir``.

    Returns the directory holding the ``*.cif`` candidate structures.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    paths = _files_to_download(api, BUCKET_ID)
    if not paths:
        raise RuntimeError(
            f"no matching files in bucket {BUCKET_ID} for patterns {_ALLOW_PATTERNS!r}"
        )
    print(f"Downloading {len(paths)} files from bucket {BUCKET_ID} → {out_dir}/")
    pairs = [(p, str(out_dir / p)) for p in paths]
    api.download_bucket_files(BUCKET_ID, files=pairs)
    return out_dir / REMOTE_PREFIX / "gt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "candidates" / "foldbench",
        help="Local mirror root (default: ./candidates/foldbench/).",
    )
    args = parser.parse_args()
    gt_dir = fetch(args.out)
    n_gt = sum(1 for _ in gt_dir.glob("*.cif"))
    print(f"Fetched {n_gt} candidate mmCIFs into {gt_dir}")


if __name__ == "__main__":
    main()
