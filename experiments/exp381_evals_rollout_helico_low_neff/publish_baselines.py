"""Publish the matched extension separately from the immutable original v1 pilot."""

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

from huggingface_hub import HfFileSystem

from publish_to_hf import archive

HERE = Path(__file__).resolve().parent
DEST = "hf://buckets/open-athena/MarinFold/data/exp381-rollout-helico-low-neff/v2-baselines"


def main(restore: bool = False) -> None:
    """Publish and verify all public files, or restore baseline inputs and structures."""
    fs = HfFileSystem(token=False, skip_instance_cache=True)
    public = HERE / "scratch/public_baselines"
    if restore:
        manifest = json.loads(
            (HERE / "data/baseline_public_artifacts.json").read_text()
        )["files"]
        for name, destination in (
            ("baseline_inputs.tar.gz", HERE / "inputs/baselines"),
            ("baseline_structures.tar.gz", HERE / "scratch/baselines/results"),
        ):
            payload = fs.read_bytes(f"{DEST.removeprefix('hf://')}/{name}")
            if hashlib.sha256(payload).hexdigest() != manifest[name]["sha256"]:
                raise ValueError(f"Archive checksum mismatch: {name}")
            path = HERE / "scratch" / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(payload)
            with tarfile.open(path) as tar:
                tar.extractall(destination, filter="data")
        print(
            "Restored 300 baseline predictions and their inputs; restore_public.py restores the original 200 and native atom arrays."
        )
        return
    run = json.loads((HERE / "data/baseline_run_manifest.json").read_text())
    results = HERE / "scratch/baselines/results"
    if (
        run["n_baseline_predictions"] != 300
        or len(list(results.glob("*/result.json"))) != 300
    ):
        raise ValueError("Refuse incomplete publication")
    public.mkdir(exist_ok=True)
    for root, name in (
        (HERE / "inputs/baselines", "baseline_inputs.tar.gz"),
        (results, "baseline_structures.tar.gz"),
    ):
        archive(root, [p for p in root.rglob("*") if p.is_file()], public / name)
    archive(
        HERE,
        [
            *HERE.glob("*.py"),
            *HERE.glob("*.toml"),
            *HERE.glob("*.lock"),
            *HERE.glob("*.md"),
        ],
        public / "source.tar.gz",
    )
    for directory in ("data", "plots"):
        for path in (HERE / directory).rglob("*"):
            if path.is_file() and path.name != "baseline_public_artifacts.json":
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
    for name, info in manifest.items():
        payload = fs.read_bytes(f"{DEST.removeprefix('hf://')}/{name}")
        if (
            len(payload) != info["bytes"]
            or hashlib.sha256(payload).hexdigest() != info["sha256"]
        ):
            raise ValueError(f"Anonymous readback mismatch: {name}")
    local_manifest = HERE / "data/baseline_public_artifacts.json"
    local_manifest.write_text(
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
            str(local_manifest),
            f"{DEST}/data/baseline_public_artifacts.json",
        ],
        check=True,
    )
    fs.invalidate_cache()
    if (
        fs.read_bytes(
            f"{DEST.removeprefix('hf://')}/data/baseline_public_artifacts.json"
        )
        != local_manifest.read_bytes()
    ):
        raise ValueError("Publication manifest readback mismatch")
    print(f"Published and anonymously verified {len(manifest)} files at {DEST}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore", action="store_true")
    main(parser.parse_args().restore)
