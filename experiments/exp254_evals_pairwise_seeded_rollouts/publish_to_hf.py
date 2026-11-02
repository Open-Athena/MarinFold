# /// script
# requires-python = ">=3.12"
# dependencies = ["huggingface-hub>=1.5"]
# ///
# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish or anonymously fetch the two input bundles audited in PR #255.

Archives contain the contents of the source directory, without its parent.
The committed manifest pins the archive and every input file by SHA-256.
Preparing and publishing are separate so the exact file list is reviewable.
"""

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

from huggingface_hub import HfFileSystem

BUCKET = "hf://buckets/open-athena/MarinFold"
SOURCE_COMMIT = "895e3357860bb70d8be3f9b155cd96cbc229fe5c"


def sha256(path: Path) -> str:
    """Hash a file without holding its contents in memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(source: Path, experiment: int, out: Path, manifest_path: Path) -> None:
    """Build a deterministic archive and write its public provenance manifest."""
    files = sorted(path for path in source.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"No input files in {source}")
    allowed = {".parquet", ".npz", ".csv"} if experiment == 254 else {
        ".json", ".jsonl", ".csv", ".gz", ".npz",
    }
    for path in files:
        if path.is_symlink() or path.suffix not in allowed:
            raise ValueError(f"Unexpected input file: {path}")
    out.mkdir(parents=True, exist_ok=True)
    archive = out / f"exp{experiment}-review-inputs.tar.gz"
    with archive.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as tar:
                for path in files:
                    info = tar.gettarinfo(str(path), arcname=str(path.relative_to(source)))
                    info.uid = info.gid = info.mtime = 0
                    info.uname = info.gname = ""
                    info.mode = 0o644
                    with path.open("rb") as stream:
                        tar.addfile(info, stream)
    digest = sha256(archive)
    manifest = {
        "schema_version": 1,
        "experiment": experiment,
        "source_pr_commit": SOURCE_COMMIT,
        "scope": "eval-val only; existing predictor outputs, no new inference",
        "archive": {
            "uri": f"{BUCKET}/data/exp{experiment}/pr255-audit/{digest[:16]}/{archive.name}",
            "sha256": digest,
            "bytes": archive.stat().st_size,
        },
        "files": {
            str(path.relative_to(source)): {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in files
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {archive}: {len(files)} files, {archive.stat().st_size:,} bytes")


def publish(manifest_path: Path, archive: Path) -> None:
    """Upload only an archive matching the reviewed manifest, plus the manifest."""
    manifest = json.loads(manifest_path.read_text())
    if sha256(archive) != manifest["archive"]["sha256"]:
        raise ValueError("Archive checksum does not match manifest")
    uri = manifest["archive"]["uri"]
    subprocess.run(["hf", "buckets", "cp", str(archive), uri], check=True)
    subprocess.run([
        "hf", "buckets", "cp", str(manifest_path), uri.rsplit("/", 1)[0] + "/manifest.json",
    ], check=True)


def fetch(manifest_path: Path, out: Path) -> None:
    """Download anonymously, verify all bytes, and extract into a new directory."""
    if out.exists():
        raise ValueError(f"Output must be a new directory: {out}")
    manifest = json.loads(manifest_path.read_text())
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as temporary:
        temp = Path(temporary)
        archive = temp / "inputs.tar.gz"
        fs = HfFileSystem(token=False)
        with fs.open(manifest["archive"]["uri"], "rb") as remote, archive.open("wb") as local:
            shutil.copyfileobj(remote, local)
        if sha256(archive) != manifest["archive"]["sha256"]:
            raise ValueError("Downloaded archive checksum mismatch")
        extracted = temp / "inputs"
        with tarfile.open(archive, "r:gz") as tar:
            members = tar.getmembers()
            names = [member.name for member in members]
            if len(names) != len(set(names)) or set(names) != set(manifest["files"]):
                raise ValueError("Archive file list does not match manifest")
            if not all(member.isfile() for member in members):
                raise ValueError("Archive must contain regular files only")
            tar.extractall(extracted, filter="data")
        for name, expected in manifest["files"].items():
            path = extracted / name
            if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
                raise ValueError(f"Extracted input checksum mismatch: {name}")
        extracted.rename(out)
    print(f"Verified {len(manifest['files'])} files anonymously -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--source", type=Path, required=True)
    prep.add_argument("--experiment", type=int, choices=(254, 256), required=True)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--manifest", type=Path, required=True)
    upload = commands.add_parser("publish")
    upload.add_argument("--manifest", type=Path, required=True)
    upload.add_argument("--archive", type=Path, required=True)
    download = commands.add_parser("fetch")
    download.add_argument("--manifest", type=Path, required=True)
    download.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.source, args.experiment, args.out, args.manifest)
    elif args.command == "publish":
        publish(args.manifest, args.archive)
    else:
        fetch(args.manifest, args.out)


if __name__ == "__main__":
    main()
