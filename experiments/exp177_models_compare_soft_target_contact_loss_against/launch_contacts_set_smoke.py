# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit the contacts-set-v1 JAX loss smoke to a CoreWeave GB200."""

import argparse
import netrc
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_IRIS = "/tmp/marin-iris-origin-main-fresh/lib/iris"


def _wandb_api_key() -> str:
    env_key = os.environ.get("WANDB_API_KEY")
    if env_key:
        return env_key
    try:
        auth = netrc.netrc().authenticators("api.wandb.ai")
    except FileNotFoundError:
        auth = None
    if auth is None or not auth[2]:
        raise ValueError("Missing W&B credentials; source ~/.config/marinfold/wandb.env")
    return auth[2]


def _copy_package_subset(root: Path, bundle: Path) -> None:
    source = root / "marinfold/marinfold/document_structures/contacts_set_v1"
    dest = bundle / "marinfold/document_structures/contacts_set_v1"
    shutil.copytree(source, dest)
    for package_dir in [bundle / "marinfold", bundle / "marinfold/document_structures"]:
        package_dir.mkdir(parents=True, exist_ok=True)
        init = package_dir / "__init__.py"
        if not init.exists():
            init.write_text("")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-cluster", default="cw-us-east-08a")
    parser.add_argument("--job-name", default="exp177-contacts-set-v1-gb200-smoke-a01")
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--gpu", default="GB200x1")
    parser.add_argument("--mode", choices=("structured", "next-token", "delta-stream"), default="structured")
    parser.add_argument("--iris-project", default=os.environ.get("IRIS_PROJECT", DEFAULT_IRIS))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    project = Path(__file__).resolve().parent
    bundle = Path(tempfile.mkdtemp(prefix="exp177-contacts-set-smoke-"))
    if args.mode == "next-token":
        script_name = "train_contacts_set_next_token_smoke.py"
    elif args.mode == "delta-stream":
        script_name = "train_delta_stream_smoke.py"
    else:
        script_name = "train_contacts_set_smoke.py"
    for name in ["pyproject.toml", "uv.lock", script_name]:
        shutil.copyfile(project / name, bundle / name)
    _copy_package_subset(root, bundle)

    env_vars = {
        "PYTHONPATH": ".",
        "WANDB_API_KEY": _wandb_api_key(),
        "WANDB_ENTITY": "open-athena",
        "WANDB_PROJECT": "MarinFold",
        "WANDB_NAME": args.job_name,
        "EXP177_SMOKE_ROWS": str(args.rows),
        "EXP177_SMOKE_MAX_SEQ_LEN": str(args.max_seq_len),
        "EXP177_SMOKE_STEPS": str(args.steps),
        "EXP177_SMOKE_HIDDEN": str(args.hidden),
    }
    if "FSSPEC_S3" in os.environ:
        env_vars["FSSPEC_S3"] = os.environ["FSSPEC_S3"]

    command = [
        "uv",
        "run",
        "--project",
        args.iris_project,
        "iris",
        "--cluster=marin",
        "job",
        "run",
        "--target-cluster",
        args.target_cluster,
        "--priority",
        "batch",
        "--job-name",
        args.job_name,
        "--no-wait",
        "--enable-extra-resources",
        "--gpu",
        args.gpu,
        "--cpu",
        "8",
        "--memory",
        "64GB",
        "--disk",
        "64GB",
        "--extra",
        "gpu",
    ]
    for key, value in env_vars.items():
        command.extend(["-e", key, value])
    command.extend(["--", "uv", "run", "--extra", "gpu", "python", script_name])

    print(f"Submitting {args.job_name} to {args.target_cluster} ({args.gpu}) from bundle {bundle}", flush=True)
    result = subprocess.run(command, cwd=bundle, check=False)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
