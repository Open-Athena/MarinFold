"""Verify and stage the already co-located, published exp277 checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path

import fsspec

MODEL = "s3://marin-us-east-02a/MarinFold/exp304/model/exp277-step-266344"


def main() -> None:
    """Check every published file's size and SHA-256 before inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    args.out.mkdir(exist_ok=True, parents=True)
    fs, root = fsspec.core.url_to_fs(MODEL)
    for name, expected in manifest["files"].items():
        remote = f"{root}/{name}"
        if fs.info(remote)["size"] != expected["size"]:
            raise ValueError(f"Checkpoint size changed: {name}")
        destination = args.out / name
        fs.get_file(remote, str(destination))
        with destination.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected["sha256"]:
                raise ValueError(f"Checkpoint checksum mismatch: {name}")
    print("Published exp277 weights/config/tokenizer verified", flush=True)


if __name__ == "__main__":
    main()
