# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit one named phase of this experiment from the workstation.

Phases run in order: `stage` mirrors the published complex corpus into
CoreWeave storage, `prepare-smoke` and `prepare` tokenize it, `audit` counts the
epoch with the trainer's packer, and `train-smoke` / `train` do the training.
Each is a single CPU driver job; the training phases dispatch their own
batch-priority GPU gang from inside the cluster (see `runtime.py`).

    python launch.py stage
    python launch.py prepare-smoke
    python launch.py prepare
    python launch.py audit
    python launch.py train-smoke --nodes 1
    python launch.py train --nodes 16
"""

import argparse
import configparser
import json
import netrc
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PREFIX = "s3://marin-us-east-02a/MarinFold/exp343_models_complex_corpus_training"
DEFAULT_IRIS = "/home/bizon/git/marin-freshiris/.venv/bin/iris"
VERSION = "2026.09.28.1"
#: exp277's four token caches and the #294 corpus both live in CoreWeave
#: US-EAST-02A, so training and tokenization stay in that region.
TARGET_CLUSTER = "cw-us-east-02a"
PHASES = ("stage", "prepare-smoke", "prepare", "audit", "train-smoke", "train")
#: Source files the workspace bundle needs beyond this experiment's own.
IMPORTED_SOURCES = (
    "experiments/exp232_sweep_cv1_decontam/training_contract.py",
    "experiments/exp277_models_single_mpnn_pilot/config.py",
    "experiments/exp277_models_single_mpnn_pilot/epoch_data.py",
)


def entrypoint(phase: str, env: dict[str, str]) -> list[str]:
    """The pod-side command for one phase, and any env the phase implies."""
    module = "experiments.exp343_models_complex_corpus_training."
    if phase == "stage":
        return ["python", "-m", module + "stage"]
    if phase.startswith("prepare"):
        command = ["python", "-m", module + "prepare"]
        if phase == "prepare-smoke":
            command.append("--smoke")
        return command
    if phase == "audit":
        return ["python", "-m", module + "audit_epoch"]
    env["SMOKE"] = "1" if phase == "train-smoke" else "0"
    return ["python", "-m", module + "train", "--version", VERSION, "--run"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=PHASES)
    parser.add_argument("--nodes", type=int, default=16)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--iris-bin", default=os.environ.get("IRIS_BIN", DEFAULT_IRIS))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="assemble the bundle and print the command without submitting",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    project = Path(__file__).resolve().parent
    credentials = configparser.ConfigParser()
    credentials.read(Path.home() / ".aws/credentials")
    cw = credentials["cw"]
    wandb = netrc.netrc().authenticators("api.wandb.ai")
    if wandb is None:
        raise ValueError("Missing W&B credentials")
    # The fleet default points at R2. This experiment's data is on CoreWeave;
    # retain its credentials and in-region LOTA endpoint explicitly.
    env = {
        "MARIN_PREFIX": PREFIX,
        "WANDB_ENTITY": "open-athena",
        "WANDB_PROJECT": "MarinFold",
        "WANDB_API_KEY": wandb[2],
        "NODES": str(args.nodes),
        "FSSPEC_S3": json.dumps(
            {
                "key": cw["aws_access_key_id"],
                "secret": cw["aws_secret_access_key"],
                "endpoint_url": "http://cwlota.com",
                "config_kwargs": {"s3": {"addressing_style": "virtual"}},
            }
        ),
        "PYTHONPATH": ".",
        "GIT_COMMIT": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
    }
    token_path = Path.home() / ".cache/huggingface/token"
    if token_path.exists():
        env["HF_TOKEN"] = token_path.read_text().strip()
    entry = entrypoint(args.phase, env)
    # `stage` downloads 19.6 GB through a temporary directory; the rest only read
    # parquet footers and cache ledgers.
    disk = "64GB" if args.phase == "stage" else "32GB"
    # Credentials reach only the child process -- never a source file, a command
    # transcript, or a state artifact.
    command = [
        args.iris_bin,
        "--cluster",
        "marin",
        "job",
        "run",
        "--target-cluster",
        TARGET_CLUSTER,
        "--priority",
        "batch",
        "--job-name",
        f"exp343-{args.phase}-a{args.attempt:02d}",
        "--no-wait",
        "--enable-extra-resources",
        "--cpu",
        "4",
        "--memory",
        "16GB",
        "--disk",
        disk,
    ]
    for key, value in env.items():
        command.extend(["-e", key, value])
    command.extend(["--", *entry])
    bundle = Path(tempfile.mkdtemp(prefix="exp343-bundle-"))
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copyfile(project / name, bundle / name)
    sources = list(project.glob("*.py")) + [root / name for name in IMPORTED_SOURCES]
    for source in sources:
        destination = bundle / source.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    gpus = f", {args.nodes * 8} H100" if args.phase.startswith("train") else ""
    print(
        f"Submitting {args.phase}: {TARGET_CLUSTER}, batch{gpus}\n"
        f"bundle {bundle} ({len(sources)} sources)",
        flush=True,
    )
    if args.dry_run:
        secrets = {env[key] for key in ("WANDB_API_KEY", "FSSPEC_S3") if key in env}
        secrets |= {env["HF_TOKEN"]} if "HF_TOKEN" in env else set()
        print(" ".join("<redacted>" if part in secrets else part for part in command))
        return
    result = subprocess.run(command, cwd=bundle, check=False)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
