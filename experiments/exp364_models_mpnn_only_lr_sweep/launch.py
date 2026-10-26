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

PREFIX = "s3://marin-us-east-02a/MarinFold/exp364_models_mpnn_only_lr_sweep"
DEFAULT_IRIS = "/home/bizon/git/marin-freshiris/.venv/bin/iris"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=(
            "audit",
            "train-smoke",
            "train",
            "cooldown-smoke",
            "cooldown",
        ),
    )
    parser.add_argument("--nodes", type=int, default=8)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--iris-bin", default=os.environ.get("IRIS_BIN", DEFAULT_IRIS))
    parser.add_argument(
        "--trial", choices=("lr5e-4", "lr1e-3", "lr2e-3"), default="lr1e-3"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.phase.startswith("cooldown") and args.trial != "lr5e-4":
        parser.error("The one-epoch cooldown forks only --trial lr5e-4")
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
        "CW_KEY_ID": cw["aws_access_key_id"],
        "CW_KEY_SECRET": cw["aws_secret_access_key"],
        "NODES": str(args.nodes),
        "TRIAL": args.trial,
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
    module = "experiments.exp364_models_mpnn_only_lr_sweep."
    if args.phase == "audit":
        entry = ["python", "-m", module + "audit_epoch"]
    else:
        env["SMOKE"] = "1" if args.phase.endswith("-smoke") else "0"
        driver = "cooldown_train" if args.phase.startswith("cooldown") else "train"
        entry = ["python", "-m", module + driver, "--version", "2026.10.13.1", "--run"]
    # Retry multi-day production drivers after transient worker failures;
    # the stable trial output path restores its own latest checkpoint.
    # Smokes and audits remain fail-fast gates with isolated identities.
    max_retries = 3 if args.phase in {"train", "cooldown"} else 0
    # The CLI and workers use the same committed lock. Credentials are passed
    # only to the child process,
    # never written into source files, command transcripts, or state artifacts.
    command = [
        args.iris_bin,
        "--cluster",
        "marin",
        "job",
        "run",
        "--user",
        "bizon",
        "--target-cluster",
        "cw-us-east-02a",
        "--priority",
        "batch",
        "--job-name",
        f"exp364-{args.trial}-{args.phase}-a{args.attempt:02d}",
        "--no-wait",
        "--max-retries",
        str(max_retries),
        "--enable-extra-resources",
        "--cpu",
        "4",
        "--memory",
        "64GB" if args.phase == "audit" else "16GB",
        "--disk",
        "32GB",
    ]
    for key, value in env.items():
        command.extend(["-e", key, value])
    command.extend(["--", *entry])
    resources = (
        "CPU audit" if args.phase == "audit" else f"{args.nodes * 8} H100 training gang"
    )
    print(
        f"Submitting {args.phase}: cw-us-east-02a, batch, {resources}, max_retries={max_retries}",
        flush=True,
    )
    bundle = Path(tempfile.mkdtemp(prefix="exp364-bundle-"))
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copyfile(project / name, bundle / name)
    sources = [
        source for source in project.glob("*.py") if not source.name.startswith("test_")
    ] + [
        root / "experiments/exp232_sweep_cv1_decontam/training_contract.py",
        *[
            root / "experiments/exp277_models_single_mpnn_pilot" / name
            for name in ("config.py", "prepare.py", "epoch_data.py")
        ],
    ]
    for source in sources:
        destination = bundle / source.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    if args.dry_run:
        print(f"Bundle: {bundle}; entrypoint: {entry}; forwarded keys: {sorted(env)}")
        return
    result = subprocess.run(command, cwd=bundle, check=False)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
