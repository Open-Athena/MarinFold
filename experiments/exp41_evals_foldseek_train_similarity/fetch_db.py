# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Download the built Foldseek representative DB from a Modal Volume.

``build_db_modal.py`` leaves a compact Foldseek DB (``db/targetDB*``) and
``reps_manifest.csv`` on the ``afdb-foldseek-reps`` Volume. This pulls them
to a local directory so ``query_similarity.py`` can run against the DB and
join hits to splits via the manifest -- exactly as it does for the
prototype DB, just with a far larger target set.

The DB is small (tens of MB for the 1-per-shard build), so a plain
iterdir + read_file walk (mirroring exp20's ``download_outputs``) is fine;
no bulk-transfer machinery needed.

Example::

    uv run python fetch_db.py --volume afdb-foldseek-reps --out db_1per_shard
"""

import argparse
from pathlib import Path

import modal
from modal.volume import FileEntryType


def _download_tree(vol: modal.Volume, remote_dir: str, local_dir: Path) -> int:
    """Copy every file under ``remote_dir`` from the Volume into ``local_dir``.

    ``Volume.iterdir`` is recursive by default in modal 1.x, so we flatten
    the walk: take each FILE entry's path relative to ``remote_dir`` and
    mirror it locally.
    """
    local_dir.mkdir(parents=True, exist_ok=True)
    base = remote_dir.rstrip("/")
    n = 0
    for entry in vol.iterdir(remote_dir, recursive=True):
        if entry.type != FileEntryType.FILE:
            continue
        rel = entry.path[len(base):].lstrip("/")
        dst = local_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        with dst.open("wb") as fh:
            for chunk in vol.read_file(entry.path):
                fh.write(chunk)
        n += 1
    return n


def _download_file(vol: modal.Volume, remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with local_path.open("wb") as fh:
        for chunk in vol.read_file(remote_path):
            fh.write(chunk)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--volume", default="afdb-foldseek-reps", help="Modal Volume name")
    ap.add_argument("--out", type=Path, default=Path("db_1per_shard"), help="Local output dir")
    args = ap.parse_args()

    vol = modal.Volume.from_name(args.volume)

    db_local = args.out / "db"
    n_db = _download_tree(vol, "/db", db_local)
    print(f"downloaded {n_db} DB files → {db_local}/")

    manifest_local = args.out / "reps_manifest.csv"
    _download_file(vol, "/reps_manifest.csv", manifest_local)
    print(f"downloaded manifest → {manifest_local}")

    print(
        f"\nQuery against it with:\n"
        f"  uv run python query_similarity.py \\\n"
        f"      --candidate-dir candidates/foldbench/data/protenix-foldbench-monomers/gt \\\n"
        f"      --db {db_local}/targetDB \\\n"
        f"      --reps-manifest {manifest_local} \\\n"
        f"      --out data/foldbench_vs_1per_shard_similarity.csv \\\n"
        f"      --db-tag afdb-24M-1per-shard"
    )


if __name__ == "__main__":
    main()
