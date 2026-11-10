"""Publish complete, checksum-addressed pilot inputs, structures and analysis."""

import gzip
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

from huggingface_hub import HfFileSystem

HERE = Path(__file__).resolve().parent
DEST = "hf://buckets/open-athena/MarinFold/data/exp381-rollout-helico-low-neff/v1"


def archive(root: Path, files: list[Path], output: Path) -> None:
    """Build a deterministic archive without private caches or local paths."""
    with output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as tar:
                for path in sorted(files):
                    info = tar.gettarinfo(
                        str(path), arcname=str(path.relative_to(root))
                    )
                    info.mtime = 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    with path.open("rb") as stream:
                        tar.addfile(info, stream)


def main() -> None:
    """Upload and anonymously verify every published file, including raw structures."""
    run = json.loads((HERE / "data/run_manifest.json").read_text())
    if run["n_predictions"] != 200 or run["minimum_ca_coverage"] != 1:
        raise ValueError("cannot publish an incomplete experiment")
    results = HERE / "scratch/results"
    if len(list(results.glob("*/result.json"))) != 200:
        raise ValueError("expected all 200 saved predictions")
    public = HERE / "scratch/public"
    public.mkdir(parents=True, exist_ok=True)
    archive(
        HERE / "inputs",
        [p for p in (HERE / "inputs").rglob("*") if p.is_file()],
        public / "inputs.tar.gz",
    )
    archive(
        results,
        [p for p in results.rglob("*") if p.is_file()],
        public / "structures.tar.gz",
    )
    source_files = [
        *HERE.glob("*.py"),
        *HERE.glob("*.toml"),
        *HERE.glob("*.lock"),
        *HERE.glob("*.md"),
    ]
    archive(HERE, source_files, public / "source.tar.gz")
    for directory in ("data", "plots"):
        (public / directory).mkdir(exist_ok=True)
        for path in (HERE / directory).rglob("*"):
            if path.is_file() and path.name != "public_artifacts.json":
                destination = public / path.relative_to(HERE)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)
    shutil.copyfile(HERE / "README.md", public / "README.md")
    manifest = {
        str(p.relative_to(public)): dict(
            bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()
        )
        for p in sorted(public.rglob("*"))
        if p.is_file() and p.name != "artifact_manifest.json"
    }
    (public / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    subprocess.run(
        ["hf", "buckets", "sync", str(public), DEST + "/", "--no-delete"], check=True
    )
    fs = HfFileSystem(token=False, skip_instance_cache=True)
    if (
        fs.read_bytes(f"{DEST.removeprefix('hf://')}/artifact_manifest.json")
        != (public / "artifact_manifest.json").read_bytes()
    ):
        raise ValueError("artifact manifest anonymous readback mismatch")
    for relative, entry in manifest.items():
        data = fs.read_bytes(f"{DEST.removeprefix('hf://')}/{relative}")
        if (
            len(data) != entry["bytes"]
            or hashlib.sha256(data).hexdigest() != entry["sha256"]
        ):
            raise ValueError(f"anonymous readback mismatch: {relative}")
    (HERE / "data/public_artifacts.json").write_text(
        json.dumps(
            dict(root=DEST, anonymous_readback_verified=True, files=manifest), indent=2
        )
        + "\n"
    )
    subprocess.run(
        [
            "hf",
            "buckets",
            "cp",
            str(HERE / "data/public_artifacts.json"),
            f"{DEST}/data/public_artifacts.json",
        ],
        check=True,
    )
    # This file was created after the anonymous reader cached the data listing.
    fs.invalidate_cache()
    if (
        fs.read_bytes(f"{DEST.removeprefix('hf://')}/data/public_artifacts.json")
        != (HERE / "data/public_artifacts.json").read_bytes()
    ):
        raise ValueError("publication manifest anonymous readback mismatch")
    print(f"Published and anonymously verified {len(manifest)} files: {DEST}")


if __name__ == "__main__":
    main()
