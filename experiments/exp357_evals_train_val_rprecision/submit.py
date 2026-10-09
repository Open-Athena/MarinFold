"""Bundle the experiment and submit a regional CPU coordinator to Iris."""

import argparse
import dataclasses
import hashlib
import json
import runpy
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
IRIS = "/home/bizon/git/marin-freshiris/.venv/bin/iris"
MARINFOLD_REVISION = "d1bea417a64cc042ad931422200c3edeb873f2e0"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="v1")
    parser.add_argument("--attempt", default="a01")
    parser.add_argument("--cluster", default="cw-us-east-02a")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--repair",
        help="Retry just job_label:shard with the existing regional worker bundle",
    )
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--cap-check", action="store_true")
    args = parser.parse_args()
    specs = runpy.run_path(
        str(
            REPO
            / "experiments/exp277_models_single_mpnn_pilot/evals/2026-09-13_rollout_v2/checkpoint_specs.py"
        )
    )
    checkpoints = [
        dataclasses.asdict(specs[name])
        for name in ("EXP277_CHECKPOINT", "EXP277_EPOCH2_CHECKPOINT")
    ]
    (HERE / "data/checkpoints.json").write_text(json.dumps(checkpoints, indent=2))
    with tempfile.TemporaryDirectory(prefix="exp357-bundle-") as directory:
        bundle = Path(directory)
        for name in (
            "driver.py",
            "repair.py",
            "worker.py",
            "protocol.py",
            "scoring.py",
            "pyproject.toml",
            "uv.lock",
        ):
            shutil.copyfile(HERE / name, bundle / name)
        for name in (
            "targets.json",
            "cohort_manifest.csv",
            "input_provenance.json",
            "checkpoints.json",
        ):
            shutil.copyfile(HERE / "data" / name, bundle / name)
        if args.cap_check:
            shutil.copyfile(
                HERE / "data/capped_rollouts.json", bundle / "capped_rollouts.json"
            )
        shutil.copyfile(
            REPO
            / "experiments/exp89_evals_contacts_v1_model_on_eval_set/compute_metrics.py",
            bundle / "exp89_metrics.py",
        )
        wheel = HERE / "scratch/wheels/marinfold-0.1.0-py3-none-any.whl"
        if not wheel.exists():
            raise FileNotFoundError(
                "Build the pinned MarinFold wheel before submission; see README"
            )
        shutil.copyfile(wheel, bundle / wheel.name)
        manifest = {
            "marinfold_revision": MARINFOLD_REVISION,
            "files": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in bundle.iterdir()
            },
        }
        (bundle / "code_manifest.json").write_text(json.dumps(manifest, indent=2))
        manifest_name = "code_manifest.json"
        if args.repair:
            manifest_name = "repair_code_manifest.json"
        elif args.collect_only:
            manifest_name = "collection_code_manifest.json"
        elif args.cap_check:
            manifest_name = "cap_check_code_manifest.json"
        (HERE / "data" / manifest_name).write_text(json.dumps(manifest, indent=2))
        entrypoint = ["python", "driver.py", "--run-id", args.run_id]
        if args.repair:
            checkpoint, shard = args.repair.split(":")
            entrypoint = [
                "python",
                "repair.py",
                "--run-id",
                args.run_id,
                "--checkpoint",
                checkpoint,
                "--shard",
                shard,
            ]
        elif args.collect_only:
            entrypoint = [
                "python",
                "repair.py",
                "--run-id",
                args.run_id,
                "--collect-only",
            ]
        elif args.cap_check:
            entrypoint = ["python", "repair.py", "--run-id", args.run_id, "--cap-check"]
        command = [
            IRIS,
            "--cluster=marin",
            "job",
            "run",
            "--target-cluster",
            args.cluster,
            "--priority",
            "batch",
            "--enable-extra-resources",
            "--user",
            "bizon",
            "--job-name",
            f"exp357-{args.run_id}-{args.attempt}",
            "--cpu",
            "4",
            "--memory",
            "16GB",
            "--disk",
            "32GB",
            "--max-retries",
            "1",
            "--timeout",
            "172800",
            "--no-wait",
            "-e",
            "MARIN_PREFIX",
            "s3://marin-us-east-02a/MarinFold",
            "--",
            *entrypoint,
        ]
        print(
            json.dumps(
                {
                    "cluster": args.cluster,
                    "job": f"/bizon/exp357-{args.run_id}-{args.attempt}",
                    "bundle_bytes": sum(p.stat().st_size for p in bundle.iterdir()),
                    "models": [c["run_name"] for c in checkpoints],
                }
            ),
            flush=True,
        )
        if not args.dry_run:
            subprocess.run(command, cwd=bundle, check=True)


if __name__ == "__main__":
    main()
