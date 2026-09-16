"""Submit the pilot's preparation or training job from the workstation."""

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

PREFIX = "s3://marin-us-east-02a/MarinFold/exp277_models_single_mpnn_pilot"
DEFAULT_IRIS = "/home/bizon/git/marin-freshiris/.venv/bin/iris"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=(
            "prepare-smoke",
            "prepare",
            "train-smoke",
            "train",
            "continue-smoke",
            "continue",
        ),
    )
    parser.add_argument("--nodes", type=int, default=16)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--iris-bin", default=os.environ.get("IRIS_BIN", DEFAULT_IRIS))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    project = Path(__file__).resolve().parent
    credentials = configparser.ConfigParser()
    credentials.read(Path.home() / ".aws/credentials")
    cw = credentials["cw"]
    wandb = netrc.netrc().authenticators("api.wandb.ai")
    if wandb is None:
        raise ValueError("Missing W&B credentials")
    # The fleet default now points at R2. This experiment's existing data is on
    # CoreWeave; explicitly retain its credentials and in-region LOTA endpoint.
    env = {
        "MARIN_PREFIX": PREFIX,
        "WANDB_ENTITY": "open-athena",
        "WANDB_PROJECT": "MarinFold",
        "WANDB_API_KEY": wandb[2],
        "NODES": str(args.nodes),
        "ATTEMPT": str(args.attempt),
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
    module = "experiments.exp277_models_single_mpnn_pilot."
    if args.phase.startswith("prepare"):
        entry = ["python", "-m", module + "prepare"]
        if args.phase == "prepare-smoke":
            entry.append("--smoke")
    elif args.phase.startswith("train"):
        env["SMOKE"] = "1" if args.phase == "train-smoke" else "0"
        entry = ["python", "-m", module + "train", "--version", "2026.09.09.1", "--run"]
    else:
        env["SMOKE"] = "1" if args.phase == "continue-smoke" else "0"
        entry = [
            "python",
            "-m",
            module + "continue_train",
            "--version",
            "2026.09.09.1",
            "--run",
        ]
    # A production epoch is a multi-day gang on a contested cluster, and one
    # worker failing its dependency sync fails all sixteen: `/bizon/exp277-
    # continue-a01` died because a single worker's 480 MB nvidia-cudnn-cu13
    # download hit a stream error. Retry those phases so a transient does not
    # cost the whole gang; a retry re-runs the step, which resumes from the
    # run's own latest checkpoint once one exists. Smokes and preparation stay
    # at zero retries because they are gates and should fail loudly.
    max_retries = 3 if args.phase in ("train", "continue") else 0
    # The CLI and workers use the same committed lock. Credentials are passed
    # only to the child process,
    # never written into source files, command transcripts, or state artifacts.
    command = [
        args.iris_bin,
        "--cluster",
        "marin",
        "job",
        "run",
        "--target-cluster",
        "cw-us-east-02a",
        "--priority",
        "batch",
        "--job-name",
        f"exp277-{args.phase}-a{args.attempt:02d}",
        "--no-wait",
        "--max-retries",
        str(max_retries),
        "--enable-extra-resources",
        "--cpu",
        "4",
        "--memory",
        "16GB",
        "--disk",
        "32GB",
    ]
    for key, value in env.items():
        command.extend(["-e", key, value])
    command.extend(["--", *entry])
    print(
        f"Submitting {args.phase}: cw-us-east-02a, batch, {args.nodes * 8} H100 "
        f"for training, max_retries={max_retries}",
        flush=True,
    )
    bundle = Path(tempfile.mkdtemp(prefix="exp277-bundle-"))
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copyfile(project / name, bundle / name)
    sources = list(project.glob("*.py")) + [
        root / "experiments/exp232_sweep_cv1_decontam/training_contract.py",
    ]
    for source in sources:
        destination = bundle / source.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    result = subprocess.run(command, cwd=bundle, check=False)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
