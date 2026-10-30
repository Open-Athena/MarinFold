"""Launch one exp362 trial with the baseline's locked worker environment."""

import argparse
import configparser
import json
import netrc
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

DEFAULT_IRIS = "/home/bizon/git/marin-freshiris/.venv/bin/iris"
PREFIX = "s3://marin-us-east-02a/MarinFold/exp362_models_smaller_model_scaling"


def main() -> None:
    """Bundle the experiment and unchanged baseline dependencies, then submit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("smoke", "train"))
    parser.add_argument("--trial", choices=("depth4", "depth8", "depth16"), required=True)
    parser.add_argument("--cluster", choices=("cw-us-east-02a", "cw-rno2a", "cw-us-east-08a"), required=True)
    parser.add_argument("--nodes", type=int, choices=(1, 2, 4, 8, 16), required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--user", default="bizon")
    parser.add_argument("--iris-bin", default=os.environ.get("IRIS_BIN", DEFAULT_IRIS))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    project = Path(__file__).resolve().parent
    env = {
        "MARIN_PREFIX": PREFIX,
        "WANDB_ENTITY": "open-athena", "WANDB_PROJECT": "MarinFold",
        "TRIAL": args.trial, "TARGET_CLUSTER": args.cluster,
        "NODES": str(args.nodes), "ATTEMPT": str(args.attempt),
        "SMOKE": "1" if args.phase == "smoke" else "0", "PYTHONPATH": ".",
        "GIT_COMMIT": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
    }
    command = [
        args.iris_bin, "--cluster", "marin", "job", "run", "--user", args.user,
        "--target-cluster", args.cluster, "--priority", "batch",
        "--job-name", f"exp362-{args.trial}-{args.phase}-a{args.attempt:02d}",
        "--no-wait", "--max-retries", "3" if args.phase == "train" else "0",
        "--enable-extra-resources", "--cpu", "4", "--memory", "16GB", "--disk", "32GB",
    ]
    entry = ["--", "python", "-m", "experiments.exp362_models_smaller_model_scaling.train",
             "--version", "2026.10.13.1", "--run"]
    display = command.copy()
    for key, value in env.items():
        display.extend(["-e", key, value])
    print(shlex.join(display + entry), flush=True)
    print("Secrets forwarded separately: WANDB_API_KEY, FSSPEC_S3, optional HF_TOKEN", flush=True)
    if args.dry_run:
        return
    credentials = configparser.ConfigParser()
    credentials.read(Path.home() / ".aws/credentials")
    cw = credentials["cw"]
    auth = netrc.netrc().authenticators("api.wandb.ai")
    if auth is None:
        raise ValueError("Missing W&B credentials")
    env["WANDB_API_KEY"] = auth[2]
    env["FSSPEC_S3"] = json.dumps({
        "key": cw["aws_access_key_id"], "secret": cw["aws_secret_access_key"],
        "endpoint_url": "http://cwlota.com",
        "config_kwargs": {"s3": {"addressing_style": "virtual"}},
    })
    token_path = Path.home() / ".cache/huggingface/token"
    if token_path.exists():
        env["HF_TOKEN"] = token_path.read_text().strip()
    for key, value in env.items():
        command.extend(["-e", key, value])
    command.extend(entry)
    with tempfile.TemporaryDirectory(prefix="exp362-bundle-") as temp:
        bundle = Path(temp)
        for name in ("pyproject.toml", "uv.lock"):
            shutil.copyfile(project / name, bundle / name)
        sources = list(project.glob("*.py")) + list(
            (root / "experiments/exp277_models_single_mpnn_pilot").glob("*.py")
        ) + [root / "experiments/exp232_sweep_cv1_decontam/training_contract.py"]
        for source in sources:
            destination = bundle / source.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        # Avoid CalledProcessError: its command representation contains secrets.
        result = subprocess.run(command, cwd=bundle, check=False)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
