"""Submit the independently named v5p fork or its approved input staging job."""

import argparse
import configparser
import json
import netrc
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

PREFIX = (
    "gs://marin-us-east5/protein-structure/MarinFold/exp364_models_mpnn_only_lr_sweep"
)
DEFAULT_IRIS = "/home/bizon/git/marin-freshiris/.venv/bin/iris"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("stage", "eval-stage", "smoke", "train"))
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--tpu", choices=("v5p-8", "v5p-64"), default="v5p-64")
    parser.add_argument("--iris-bin", default=os.environ.get("IRIS_BIN", DEFAULT_IRIS))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume-step", type=int)
    args = parser.parse_args()
    if args.resume_step is not None and (
        args.phase != "train" or args.resume_step != 186880
    ):
        parser.error(
            "--resume-step 186880 is only supported for final-batch training recovery"
        )
    root = Path(__file__).resolve().parents[2]
    project = Path(__file__).resolve().parent
    env = {
        "MARIN_PREFIX": PREFIX,
        "WANDB_ENTITY": "open-athena",
        "WANDB_PROJECT": "MarinFold",
        "PYTHONPATH": ".",
        "ATTEMPT": str(args.attempt),
        "TPU": args.tpu,
        "SMOKE": "1" if args.phase == "smoke" else "0",
        "GIT_COMMIT": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
    }
    module = "experiments.exp364_models_mpnn_only_lr_sweep."
    if args.resume_step is not None:
        env["RESUME_STEP"] = str(args.resume_step)
    staging = args.phase in {"stage", "eval-stage"}
    if staging:
        credentials = configparser.ConfigParser()
        credentials.read(Path.home() / ".aws/credentials")
        cw = credentials["cw"]
        env["FSSPEC_S3"] = json.dumps(
            {
                "key": cw["aws_access_key_id"],
                "secret": cw["aws_secret_access_key"],
                "endpoint_url": "https://cwobject.com",
                "config_kwargs": {"s3": {"addressing_style": "virtual"}},
            }
        )
        entry = [
            "python",
            "-m",
            module
            + ("stage_tpu_eval" if args.phase == "eval-stage" else "stage_tpu_inputs"),
            "--copy",
        ]
    else:
        wandb = netrc.netrc().authenticators("api.wandb.ai")
        if wandb is None:
            raise ValueError("Missing W&B credentials")
        env["WANDB_API_KEY"] = wandb[2]
        token_path = Path.home() / ".cache/huggingface/token"
        if token_path.exists():
            env["HF_TOKEN"] = token_path.read_text().strip()
        entry = [
            "python",
            "-m",
            module + "tpu_cooldown",
            "--version",
            "2026.10.13.1",
            "--run",
        ]
    command = [
        args.iris_bin,
        "--cluster",
        "marin",
        "job",
        "run",
        "--user",
        "bizon",
        "--zone",
        "us-east5-a",
        "--region",
        "us-east5",
        "--priority",
        "batch",
        "--job-name",
        f"exp364-tpu-cooldown-{args.phase}-a{args.attempt:02d}",
        "--no-wait",
        "--max-retries",
        "3" if args.phase == "train" else "0",
        "--enable-extra-resources",
        "--cpu",
        "4" if staging else "0.5",
        "--memory",
        "16GB" if staging else "2GB",
        "--disk",
        "32GB",
    ]
    for key, value in env.items():
        command.extend(["-e", key, value])
    command.extend(["--", *entry])
    bundle = Path(tempfile.mkdtemp(prefix="exp364-tpu-bundle-"))
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copyfile(project / "tpu" / name, bundle / name)
    sources = [p for p in project.glob("*.py") if not p.name.startswith("test_")]
    sources += [
        project / "data/tpu_transfer_plan.json",
        project / "data/contact_eval/plan.json",
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
    print(
        f"{args.phase}: us-east5-a, batch; training shape {args.tpu} "
        f"({int(args.tpu.split('-')[1]) // 2} chips)",
        flush=True,
    )
    if args.dry_run:
        print(f"Bundle: {bundle}; entrypoint: {entry}; forwarded keys: {sorted(env)}")
        return
    subprocess.run(command, cwd=bundle, check=True)


if __name__ == "__main__":
    main()
