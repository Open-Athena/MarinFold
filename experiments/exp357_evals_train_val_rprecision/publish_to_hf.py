"""Publish compact results and raw samples from the co-located S3 working copy.

The separate export environment supplies a bucket-capable HF CLI without
changing the pinned evaluator's dependencies. Run with --submit locally after
the complete result marker exists; the CPU job performs the bulk upload.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import fsspec

HERE = Path(__file__).resolve().parent
ROOT = "marin-us-east-02a/MarinFold/exp357_evals_train_val_rprecision"


def publish(run_id: str) -> None:
    """Consolidate raw per-protein files into an archive and publish the record."""
    fs = fsspec.filesystem("s3")
    prefix = f"{ROOT}/{run_id}"
    if not fs.exists(prefix + "/results/complete.json"):
        raise ValueError("Refusing to publish an incomplete experiment")
    with tempfile.TemporaryDirectory(prefix="exp357-public-") as directory:
        local = Path(directory)
        for section in ("inputs", "results", "smoke"):
            for source in fs.find(prefix + "/" + section):
                target = local / source.removeprefix(prefix + "/")
                target.parent.mkdir(parents=True, exist_ok=True)
                fs.get_file(source, str(target))
        raw = local / "raw"
        sources = fs.glob(prefix + "/rollout/*/raw/*.json.gz")
        complete = json.loads((local / "results/complete.json").read_text())
        if len(sources) != complete["units"] * complete["models"]:
            raise ValueError(f"Incomplete raw outputs: {len(sources)}")
        for source in sources:
            relative = source.split("/rollout/", 1)[1]
            target = raw / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            fs.get_file(source, str(target))
        if not raw.exists():
            raise ValueError("No raw samples were found")
        shutil.make_archive(str(local / "raw_rollouts"), "tar", root_dir=raw)
        shutil.rmtree(raw)
        files = {}
        for path in sorted(local.rglob("*")):
            if not path.is_file():
                continue
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            files[str(path.relative_to(local))] = {
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
        (local / "artifact_manifest.json").write_text(json.dumps(files, indent=2))
        if sum(item["bytes"] for item in files.values()) > 10_000_000_000:
            raise ValueError(
                "Publication exceeds the 10 GB transfer budget; explicit approval is required."
            )
        destination = f"hf://buckets/open-athena/MarinFold/data/exp357-train-val-rprecision/{run_id}"
        subprocess.run(["hf", "buckets", "sync", str(local), destination], check=True)
        print(json.dumps({"published": destination}), flush=True)


def submit(run_id: str) -> None:
    """Run export on CoreWeave using the existing account's scoped credentials."""
    token = (
        os.environ.get("HF_TOKEN")
        or (Path.home() / ".cache/huggingface/token").read_text().strip()
    )
    with tempfile.TemporaryDirectory(prefix="exp357-export-bundle-") as directory:
        bundle = Path(directory)
        shutil.copyfile(__file__, bundle / "publish_to_hf.py")
        (bundle / "pyproject.toml").write_text(
            '[project]\nname="exp357-export"\nversion="0.1.0"\nrequires-python=">=3.12,<3.13"\ndependencies=["s3fs==2026.1.0","huggingface-hub==1.27.0"]\n'
        )
        command = [
            "/home/bizon/git/marin-freshiris/.venv/bin/iris",
            "--cluster=marin",
            "job",
            "run",
            "--target-cluster",
            "cw-us-east-02a",
            "--priority",
            "batch",
            "--enable-extra-resources",
            "--user",
            "bizon",
            "--job-name",
            f"exp357-export-{run_id}",
            "--cpu",
            "4",
            "--memory",
            "8GB",
            "--disk",
            "16GB",
            "--no-wait",
            "-e",
            "HF_TOKEN",
            token,
            "--",
            "python",
            "publish_to_hf.py",
            "--run-id",
            run_id,
        ]
        # subprocess arguments are intentionally never printed: they carry the token.
        subprocess.run(command, cwd=bundle, check=True)


def publish_report(run_id: str) -> None:
    """Publish the reviewed tables, plots, and report from the workstation."""
    required = [
        HERE / "data/results/per_protein_corrected.csv",
        HERE / "data/cap_checks.json",
        HERE / "plots/summary.pdf",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    with tempfile.TemporaryDirectory(prefix="exp357-report-") as directory:
        local = Path(directory)
        for path in (HERE / "data").iterdir():
            if path.is_file() and path.name != "targets.json":
                target = local / "data" / path.name
                target.parent.mkdir(exist_ok=True)
                shutil.copyfile(path, target)
        for name in ("per_protein_corrected.csv", "single_samples_corrected.csv"):
            shutil.copyfile(HERE / "data/results" / name, local / "data" / name)
        shutil.copytree(HERE / "plots", local / "plots")
        for name in ("README.md", "summary_narrative.md"):
            shutil.copyfile(HERE / name, local / name)
        manifest = {}
        for path in sorted(local.rglob("*")):
            if path.is_file():
                with path.open("rb") as handle:
                    digest = hashlib.file_digest(handle, "sha256").hexdigest()
                manifest[str(path.relative_to(local))] = {
                    "bytes": path.stat().st_size,
                    "sha256": digest,
                }
        (local / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2))
        base = "hf://buckets/open-athena/MarinFold/data/exp357-train-val-rprecision"
        hf_cli = str(Path.home() / ".local/bin/hf")
        subprocess.run(
            [
                hf_cli,
                "buckets",
                "sync",
                str(HERE / "scratch/cap_public"),
                base + "/v1-capcheck",
            ],
            check=True,
        )
        subprocess.run(
            [hf_cli, "buckets", "sync", str(local), base + f"/{run_id}/report"],
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="v1")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.report:
        publish_report(args.run_id)
    elif args.submit:
        submit(args.run_id)
    else:
        publish(args.run_id)


if __name__ == "__main__":
    main()
