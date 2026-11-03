"""Publish and restore the frozen contact-diversity experiment."""

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

from huggingface_hub import HfFileSystem

from publish_to_hf import archive

HERE = Path(__file__).resolve().parent
DEST = "hf://buckets/open-athena/MarinFold/data/exp381-rollout-helico-low-neff/v3-diversity"


def main(
    restore: bool = False, selection_only: bool = False, maxmin: bool = False
) -> None:
    """Publish and verify all public files, or restore diversity inputs and structures."""
    fs = HfFileSystem(token=False, skip_instance_cache=True)
    public = HERE / "scratch/public_diversity"
    if selection_only:
        tag = "maxmin" if maxmin else "diversity"
        task_file = "maxmin_tasks.json" if maxmin else "tasks.json"
        identities = {}
        for relative in [
            "DIVERSITY_MAXMIN_PLAN.md" if maxmin else "DIVERSITY_PLAN.md",
            f"data/{tag}_plan.json",
            f"data/{tag}_selection_manifest.json",
            f"data/{tag}_contact_inputs.csv",
            f"inputs/diversity/{task_file}",
            "inputs/diversity/targets.json",
        ]:
            path = HERE / relative
            subprocess.run(
                ["hf", "buckets", "cp", str(path), f"{DEST}/{relative}"], check=True
            )
            fs.invalidate_cache()
            payload = fs.read_bytes(f"{DEST.removeprefix('hf://')}/{relative}")
            if payload != path.read_bytes():
                raise ValueError(f"Selection publication mismatch: {relative}")
            identities[relative] = hashlib.sha256(payload).hexdigest()
        (HERE / f"data/{tag}_selection_publication.json").write_text(
            json.dumps(
                dict(
                    root=DEST, anonymous_readback_verified=True, files_sha256=identities
                ),
                indent=2,
            )
            + "\n"
        )
        print("Published and anonymously verified frozen selection inputs")
        return
    if restore:
        manifest = json.loads(
            (HERE / "data/diversity_public_artifacts.json").read_text()
        )["files"]
        for name, destination in (
            ("diversity_inputs.tar.gz", HERE / "inputs/diversity"),
            ("diversity_structures.tar.gz", HERE / "scratch/diversity/results"),
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
            "Restored 600 diversity predictions and their inputs; restore_public.py restores the original 200 and native atom arrays."
        )
        return
    run = json.loads((HERE / "data/diversity_run_manifest.json").read_text())
    extra = json.loads((HERE / "data/maxmin_run_manifest.json").read_text())
    results = HERE / "scratch/diversity/results"
    if (
        run["n_predictions"] != 400
        or extra["n_predictions"] != 200
        or len(list(results.glob("*/result.json"))) != 600
    ):
        raise ValueError("Refuse incomplete publication")
    with (HERE / "data/diversity_summary.csv").open() as stream:
        summary = list(csv.DictReader(stream))
    if len(summary) != 96 or {r["arm"] for r in summary} != {
        "first10",
        "random10",
        "medoids10",
        "maxmin10",
    }:
        raise ValueError("Rebuild the final four-arm analysis before publication")
    public.mkdir(exist_ok=True)
    for root, name in (
        (HERE / "inputs/diversity", "diversity_inputs.tar.gz"),
        (results, "diversity_structures.tar.gz"),
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
            if path.is_file() and path.name != "diversity_public_artifacts.json":
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
    local_manifest = HERE / "data/diversity_public_artifacts.json"
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
            f"{DEST}/data/diversity_public_artifacts.json",
        ],
        check=True,
    )
    fs.invalidate_cache()
    if (
        fs.read_bytes(
            f"{DEST.removeprefix('hf://')}/data/diversity_public_artifacts.json"
        )
        != local_manifest.read_bytes()
    ):
        raise ValueError("Publication manifest readback mismatch")
    print(f"Published and anonymously verified {len(manifest)} files at {DEST}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore", action="store_true")
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--maxmin", action="store_true")
    args = parser.parse_args()
    main(args.restore, args.selection_only, args.maxmin)
