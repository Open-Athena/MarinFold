"""Submit exp288 preparation checks or one size-sweep training trial."""

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

from experiments.exp288_models_chinchilla_size_sweep.config import PREFIX, TRIALS

DEFAULT_IRIS = "/home/bizon/git/marin-freshiris/.venv/bin/iris"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "train-smoke", "train"))
    parser.add_argument("--trial", choices=sorted(TRIALS), required=False)
    parser.add_argument("--nodes", type=int, default=None)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--target-cluster", default="cw-us-east-02a")
    parser.add_argument("--iris-bin", default=os.environ.get("IRIS_BIN", DEFAULT_IRIS))
    args = parser.parse_args()
    if args.phase.startswith("train") and args.trial is None:
        raise ValueError("--trial is required for train phases")

    root = Path(__file__).resolve().parents[2]
    project = Path(__file__).resolve().parent
    credentials = configparser.ConfigParser()
    credentials.read(Path.home() / ".aws/credentials")
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if wandb_api_key is None:
        try:
            wandb = netrc.netrc().authenticators("api.wandb.ai")
        except FileNotFoundError:
            wandb = None
        if wandb is not None:
            wandb_api_key = wandb[2]
    if wandb_api_key is None:
        raise ValueError("Missing W&B credentials; source ~/.config/marinfold/wandb.env")
    nodes = args.nodes or (TRIALS[args.trial].nodes if args.trial else 1)
    env = {
        "MARIN_PREFIX": PREFIX,
        "WANDB_ENTITY": "open-athena",
        "WANDB_PROJECT": "MarinFold",
        "WANDB_API_KEY": wandb_api_key,
        "NODES": str(nodes),
        "PYTHONPATH": ".",
        "GIT_COMMIT": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
    }
    if "FSSPEC_S3" in os.environ:
        env["FSSPEC_S3"] = os.environ["FSSPEC_S3"]
    elif "cw" in credentials:
        cw = credentials["cw"]
        env["FSSPEC_S3"] = json.dumps(
            {
                "key": cw["aws_access_key_id"],
                "secret": cw["aws_secret_access_key"],
                "endpoint_url": "http://cwlota.com",
                "config_kwargs": {"s3": {"addressing_style": "virtual"}},
            }
        )
    # Otherwise rely on the CoreWeave Iris task environment to inject FSSPEC_S3
    # into the root driver; runtime.py forwards that injected value to the child
    # training gang.
    token_path = Path.home() / ".cache/huggingface/token"
    if token_path.exists():
        env["HF_TOKEN"] = token_path.read_text().strip()
    module = "experiments.exp288_models_chinchilla_size_sweep."
    if args.phase == "prepare":
        entry = ["python", "-m", module + "prepare"]
        job_name = f"exp288-prepare-a{args.attempt:02d}"
    else:
        env["SMOKE"] = "1" if args.phase == "train-smoke" else "0"
        env["TRIAL"] = args.trial or ""
        entry = ["python", "-m", module + "train", "--version", "2026.09.14.1", "--run"]
        job_name = f"exp288-{args.phase}-{args.trial}-a{args.attempt:02d}"

    command = [
        args.iris_bin,
        "--cluster",
        "marin",
        "job",
        "run",
        "--target-cluster",
        args.target_cluster,
        "--priority",
        "batch",
        "--job-name",
        job_name,
        "--no-wait",
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

    if args.phase.startswith("train"):
        print(
            f"Submitting {args.phase} for {args.trial}: {args.target_cluster}, batch, {nodes * 8} H100",
            flush=True,
        )
    else:
        print(f"Submitting {args.phase}: {args.target_cluster}, batch", flush=True)

    bundle = Path(tempfile.mkdtemp(prefix="exp288-bundle-"))
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
