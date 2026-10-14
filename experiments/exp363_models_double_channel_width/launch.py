"""Submit the width experiment on CoreWeave or a regional TPU smoke test."""

import argparse
import configparser
import json
import netrc
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

PREFIX = "s3://marin-us-east-02a/MarinFold/exp363_models_double_channel_width"
VERSION = "2026.10.13.1"
DEFAULT_IRIS = "/home/bizon/git/marin-freshiris/.venv/bin/iris"
IMPORTED_SOURCES = (
    "experiments/exp232_sweep_cv1_decontam/training_contract.py",
    "experiments/exp277_models_single_mpnn_pilot/config.py",
    "experiments/exp277_models_single_mpnn_pilot/epoch_data.py",
    "experiments/exp277_models_single_mpnn_pilot/prepare.py",
    "experiments/exp277_models_single_mpnn_pilot/runtime.py",
)


def main() -> None:
    """Bundle the locked recipe and submit without exposing credentials."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train-smoke", "train", "tpu-smoke"))
    parser.add_argument("--nodes", type=int, choices=(1, 2, 4, 8, 16), default=4)
    parser.add_argument("--tpu", choices=("v5p-64", "v6e-32"), default="v5p-64")
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--iris-bin", default=os.environ.get("IRIS_BIN", DEFAULT_IRIS))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    project = Path(__file__).resolve().parent
    root = project.parents[1]
    tpu = args.phase == "tpu-smoke"
    prefix = (
        "gs://marin-us-east5/protein-structure/MarinFold/exp363_models_double_channel_width"
        if tpu else PREFIX
    )
    wandb = netrc.netrc().authenticators("api.wandb.ai")
    if wandb is None:
        raise ValueError("Missing W&B credentials")
    env = {
        "MARIN_PREFIX": prefix,
        "WANDB_ENTITY": "open-athena",
        "WANDB_PROJECT": "MarinFold",
        "WANDB_API_KEY": wandb[2],
        "NODES": str(args.nodes),
        "SMOKE": "1" if args.phase == "train-smoke" else "0",
        "SMOKE_TAG": f"-n{args.nodes}-a{args.attempt:02d}",
        "PYTHONPATH": ".",
        "GIT_COMMIT": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
    }
    if tpu:
        env.update(TPU=args.tpu, ATTEMPT=str(args.attempt))
    else:
        credentials = configparser.ConfigParser()
        credentials.read(Path.home() / ".aws/credentials")
        cw = credentials["cw"]
        env["FSSPEC_S3"] = json.dumps({
            "key": cw["aws_access_key_id"],
            "secret": cw["aws_secret_access_key"],
            "endpoint_url": "http://cwlota.com",
            "config_kwargs": {"s3": {"addressing_style": "virtual"}},
        })
    token_path = Path.home() / ".cache/huggingface/token"
    if token_path.exists():
        env["HF_TOKEN"] = token_path.read_text().strip()
    zone = "us-east5-a" if args.tpu.startswith("v5p") else "us-east5-b"
    placement = ["--zone", zone] if tpu else ["--target-cluster", "cw-us-east-02a"]
    job_name = f"exp363-{args.phase}"
    if tpu:
        job_name += f"-{args.tpu}"
    command = [
        args.iris_bin, "--cluster", "marin", "job", "run",
        *placement, "--priority", "batch",
        "--job-name", f"{job_name}-a{args.attempt:02d}",
        "--no-wait", "--enable-extra-resources",
        "--cpu", "2" if tpu else "4",
        "--memory", "8GB" if tpu else "16GB", "--disk", "32GB",
        "--timeout", "7200" if tpu else "2592000",
    ]
    for key, value in env.items():
        command.extend(["-e", key, value])
    module = "tpu_smoke" if tpu else "train"
    command.extend([
        "--", "python", "-m", f"experiments.exp363_models_double_channel_width.{module}",
        "--version", "2026.10.14.1" if tpu else VERSION, "--run",
    ])
    with tempfile.TemporaryDirectory(prefix="exp363-bundle-") as directory:
        bundle = Path(directory)
        for name in ("pyproject.toml", "uv.lock"):
            shutil.copyfile(project / name, bundle / name)
        sources = list(project.glob("*.py")) + [root / name for name in IMPORTED_SOURCES]
        for source in sources:
            destination = bundle / source.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        shutil.copytree(
            project / "tokenizer",
            bundle / project.relative_to(root) / "tokenizer",
            ignore=shutil.ignore_patterns(".cache"),
        )
        hardware = f"{args.tpu}, {zone}" if tpu else f"{args.nodes * 8} H100, cw-us-east-02a"
        print(
            f"{args.phase}: {hardware}, batch; "
            f"{len(sources)} source files, commit {env['GIT_COMMIT']}",
            flush=True,
        )
        if args.dry_run:
            return
        result = subprocess.run(command, cwd=bundle, check=False)
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
