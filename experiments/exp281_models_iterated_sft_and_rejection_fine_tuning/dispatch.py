"""Submit bounded exp281 stages through Iris's public CLI at batch priority.

The client checkout is supplied explicitly. Worker dependencies are pinned by
the experiment lockfile. A minimal workspace includes this experiment, its path
dependency, and history tools, avoiding Iris's 25 MB source-bundle limit.
"""

import argparse
import json
import netrc
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

from common import EXPERIMENT

STAGES = {"prepare": "prepare.py", "generate": "generate.py", "corpus": "build_corpus.py",
          "train": "train.py", "smoke": "smoke.py", "preflight": "preflight.py", "evaluate": "evaluate.py",
          "report": "report.py"}


def bundle(destination: Path) -> None:
    """Copy source and lockfiles, excluding artifacts and all virtualenvs."""
    root = Path(__file__).resolve().parents[2]
    ignore = shutil.ignore_patterns(".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "*.egg-info",
                                   "data", "plots", "tests", ".git")
    shutil.copytree(root / "marinfold", destination / "marinfold", ignore=ignore)
    shutil.copytree(Path(__file__).resolve().parent, destination / "experiments" / EXPERIMENT, ignore=ignore)
    (destination / "scripts").mkdir()
    for name in ("history.py", "_lib.py"):
        shutil.copy2(root / "scripts" / name, destination / "scripts" / name)
    (destination / "history" / "runs").mkdir(parents=True)


def worker_command(stage: str, arguments: list[str], gpus: int) -> list[str]:
    """Build an argv list with the exact experiment environment and entrypoint."""
    project = f"experiments/{EXPERIMENT}"
    command = ["uv", "run", "--project", project, "--locked", "--no-dev"]
    if stage == "generate":
        command += ["--extra", "generation"]
    command += ["python"]
    if stage == "train" and gpus > 1:
        command += ["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={gpus}"]
    return [*command, f"{project}/{STAGES[stage]}", *arguments]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iris-project", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--gpus", type=int, choices=[0, 1, 8], default=0)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    if args.stage == "train" and args.gpus not in (1, 8):
        raise ValueError("training requires one or eight H100s")
    if args.stage == "generate" and args.gpus != 1:
        raise ValueError("one vLLM worker uses one H100; fan out independent shard jobs")
    for arg in arguments:
        if arg.startswith(("gs://", "hf://", "https://")):
            raise ValueError("stage remote inputs into the CoreWeave bucket before dispatch")
        if arg.startswith("s3://") and not arg.startswith("s3://marin-us-east-02a/"):
            raise ValueError("worker data must use the co-located CoreWeave bucket")
    command = ["uv", "run", "--project", str(args.iris_project), "--package", "marin-iris", "iris",
               "--cluster", "marin", "job", "run", "--target-cluster", "cw-us-east-02a",
               "--priority", "batch", "--no-sync", "--no-wait", "--job-name", args.name,
               "--timeout", str(args.timeout), "--enable-extra-resources", "--cpu", "16" if args.gpus else "2",
               "--memory", "128GB" if args.gpus else "16GB", "--disk", "100GB", "--max-retries", "0"]
    if args.gpus:
        command += ["--gpu", f"H100x{args.gpus}"]
    worker = worker_command(args.stage, arguments, args.gpus)
    print(json.dumps({"cluster": "cw-us-east-02a", "gpus": args.gpus, "priority": "batch",
                      "command": shlex.join(command + ["--", *worker]), "submit": args.submit}, indent=2))
    if not args.submit:
        return
    if args.stage in ("train", "report") and "--no-wandb" not in arguments:
        key = os.environ.get("WANDB_API_KEY")
        if not key:
            auth = netrc.netrc().authenticators("api.wandb.ai")
            if not auth:
                raise ValueError("W&B credentials are required for training")
            key = auth[2]
        # Credential values are never included in the printed reviewable command.
        command += ["--env-vars", "WANDB_API_KEY", key]
    with tempfile.TemporaryDirectory(prefix="exp281-bundle-") as directory:
        bundle(Path(directory))
        subprocess.run([*command, "--", *worker], cwd=directory, check=True)


if __name__ == "__main__":
    main()
