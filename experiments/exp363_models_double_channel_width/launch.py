"""Submit the double-width smoke or production driver on regional CoreWeave."""

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
    parser.add_argument("phase", choices=("train-smoke", "train"))
    parser.add_argument("--nodes", type=int, choices=(1, 2, 4, 8, 16), default=16)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--iris-bin", default=os.environ.get("IRIS_BIN", DEFAULT_IRIS))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    project = Path(__file__).resolve().parent
    root = project.parents[1]
    credentials = configparser.ConfigParser()
    credentials.read(Path.home() / ".aws/credentials")
    cw = credentials["cw"]
    wandb = netrc.netrc().authenticators("api.wandb.ai")
    if wandb is None:
        raise ValueError("Missing W&B credentials")
    env = {
        "MARIN_PREFIX": PREFIX,
        "WANDB_ENTITY": "open-athena",
        "WANDB_PROJECT": "MarinFold",
        "WANDB_API_KEY": wandb[2],
        "NODES": str(args.nodes),
        "SMOKE": "1" if args.phase == "train-smoke" else "0",
        "SMOKE_TAG": f"-n{args.nodes}-a{args.attempt:02d}",
        "FSSPEC_S3": json.dumps({
            "key": cw["aws_access_key_id"],
            "secret": cw["aws_secret_access_key"],
            "endpoint_url": "http://cwlota.com",
            "config_kwargs": {"s3": {"addressing_style": "virtual"}},
        }),
        "PYTHONPATH": ".",
        "GIT_COMMIT": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
    }
    token_path = Path.home() / ".cache/huggingface/token"
    if token_path.exists():
        env["HF_TOKEN"] = token_path.read_text().strip()
    command = [
        args.iris_bin, "--cluster", "marin", "job", "run",
        "--target-cluster", "cw-us-east-02a", "--priority", "batch",
        "--job-name", f"exp363-{args.phase}-a{args.attempt:02d}",
        "--no-wait", "--enable-extra-resources",
        "--cpu", "4", "--memory", "16GB", "--disk", "32GB",
        "--timeout", "1209600",
    ]
    for key, value in env.items():
        command.extend(["-e", key, value])
    command.extend([
        "--", "python", "-m", "experiments.exp363_models_double_channel_width.train",
        "--version", VERSION, "--run",
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
        print(
            f"{args.phase}: cw-us-east-02a, batch, {args.nodes * 8} H100; "
            f"{len(sources)} source files, commit {env['GIT_COMMIT']}",
            flush=True,
        )
        if args.dry_run:
            return
        result = subprocess.run(command, cwd=bundle, check=False)
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
