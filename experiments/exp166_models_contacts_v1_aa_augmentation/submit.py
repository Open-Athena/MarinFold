# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit one independent exp166 evaluation job to marin-dev."""

import argparse
import os
import shlex
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

from checkpoint_specs import CHECKPOINTS, HF_BUCKET_ROOT


def build_command(
    *,
    iris: str,
    checkpoint: str,
    hf_token: str,
    smoke: bool,
    cluster: str,
    tpu: str,
    user: str = "eczech",
    job_suffix: str | None = None,
) -> list[str]:
    """Build an Iris command for exactly one checkpoint."""

    spec = CHECKPOINTS[checkpoint]
    run_kind = "smoke" if smoke else "scores"
    output_prefix = f"{HF_BUCKET_ROOT}/{run_kind}/{spec.output_name}"
    suffix = f"-{job_suffix}" if job_suffix else ""
    job_name = f"marinfold-exp166-{checkpoint}-{run_kind}{suffix}"
    command = [
        iris,
        f"--cluster={cluster}",
        "job",
        "run",
        "--user",
        user,
        "--job-name",
        job_name,
        "--no-wait",
        "--enable-extra-resources",
        "--priority",
        "interactive" if smoke else "batch",
        "--preemptible",
        "--tpu",
        tpu,
        "--cpu",
        "4",
        "--memory",
        "32GB",
        "--disk",
        "96GB",
        "--max-retries",
        "0" if smoke else "3",
        "--timeout",
        "21600",
        "--extra",
        "tpu",
        "-e",
        "HF_TOKEN",
        hf_token,
        "-e",
        "MARINFOLD_ACCELERATOR",
        tpu,
        "-e",
        "VLLM_TARGET_DEVICE",
        "tpu",
        "-e",
        "VLLM_WORKER_MULTIPROC_METHOD",
        "spawn",
        "--",
        "/app/.venv/bin/python",
        "eval_checkpoint.py",
        "--checkpoint",
        checkpoint,
        "--scratch",
        f"/app/scratch/{checkpoint}{suffix}",
        "--output-prefix",
        output_prefix,
    ]
    if smoke:
        command.extend(("--limit", "4", "--part-size", "4"))
    return command


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", choices=sorted(CHECKPOINTS), required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cluster", default="marin-dev")
    parser.add_argument("--tpu", default="v6e-4")
    parser.add_argument("--user", default="eczech")
    parser.add_argument("--job-suffix")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    iris = shutil.which("iris")
    if iris is None:
        raise FileNotFoundError("iris is not installed in the active environment")
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token and not args.dry_run:
        raise ValueError("HF_TOKEN must contain the open-athena write token")
    command = build_command(
        iris=iris,
        checkpoint=args.checkpoint,
        hf_token=hf_token or "<HF_TOKEN>",
        smoke=args.smoke,
        cluster=args.cluster,
        tpu=args.tpu,
        user=args.user,
        job_suffix=args.job_suffix,
    )
    if args.dry_run:
        safe = [
            "<HF_TOKEN>" if item == (hf_token or "<HF_TOKEN>") else item
            for item in command
        ]
        print(shlex.join(safe))
        return 0
    subprocess.run(command, cwd=Path(__file__).parent, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
