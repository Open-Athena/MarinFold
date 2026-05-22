# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Pull the Modal outputs Volume into ``./outputs/``.

Use this when ``modal_app.py`` was launched detached (or when an
earlier --download-back failed) and the per-protein
``distogram.npz`` + ``provenance.json`` files are sitting in the
Modal Volume but not yet on local disk.

Idempotent — running again is a no-op for files that already exist
locally with the same byte count.
"""

import argparse
from pathlib import Path

import modal


_HERE = Path(__file__).resolve().parent

DEFAULT_VOLUME_NAME = "marinfold-1b-foldbench-runs"


def download(*, volume_name: str, out_dir: Path) -> int:
    """Stream every file under the Volume root into ``out_dir/<stem>/<file>``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    vol = modal.Volume.from_name(volume_name)
    n_files = 0
    n_skipped = 0
    for stem_entry in vol.iterdir("/"):
        if stem_entry.type != modal.FileEntryType.DIRECTORY:
            continue
        stem_dir = out_dir / Path(stem_entry.path).name
        stem_dir.mkdir(parents=True, exist_ok=True)
        for file_entry in vol.iterdir(stem_entry.path):
            if file_entry.type != modal.FileEntryType.FILE:
                continue
            local_file = stem_dir / Path(file_entry.path).name
            # Idempotency: skip if local file exists and matches reported size.
            if local_file.exists() and local_file.stat().st_size == file_entry.size:
                n_skipped += 1
                continue
            with local_file.open("wb") as fh:
                for chunk in vol.read_file(file_entry.path):
                    fh.write(chunk)
            n_files += 1
            print(f"  wrote {local_file.relative_to(out_dir)} ({file_entry.size} bytes)")
    print(f"Downloaded {n_files} new files into {out_dir}/ ({n_skipped} already present, skipped).")
    return n_files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--volume", default=DEFAULT_VOLUME_NAME,
        help=f"Modal Volume name (default: {DEFAULT_VOLUME_NAME!r}).",
    )
    parser.add_argument(
        "--out", type=Path, default=_HERE / "outputs",
        help="Local destination directory (default: ./outputs/).",
    )
    args = parser.parse_args()
    download(volume_name=args.volume, out_dir=args.out)


if __name__ == "__main__":
    main()
