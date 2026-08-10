# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit one exp199 checkpoint contact evaluation through Iris."""

import argparse
import os
import shlex
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

from checkpoint_specs import CHECKPOINTS, HF_BUCKET_ROOT, validate_run_tag


def output_prefix(checkpoint: str, run_tag: str | None = None) -> str:
    """Return the durable prefix for one complete checkpoint evaluation."""

    spec = CHECKPOINTS[checkpoint]
    run_tag = validate_run_tag(run_tag)
    root = HF_BUCKET_ROOT
    if run_tag is not None:
        root = f"{root}/replicates/{run_tag}"
    return f"{root}/runs/{spec.run_name}/step-{spec.step}"


def build_command(
    *,
    iris: str,
    checkpoint: str,
    hf_token: str,
    cluster: str,
    tpu: str,
    region: str | None,
    user: str,
    job_suffix: str | None,
    run_tag: str | None,
) -> list[str]:
    """Build one explicitly placed Iris job command."""

    spec = CHECKPOINTS[checkpoint]
    run_tag = validate_run_tag(run_tag)
    suffix = f"-{job_suffix}" if job_suffix else ""
    if run_tag is not None:
        suffix = f"-{run_tag}{suffix}"
    job_name = f"marinfold-exp199-{checkpoint}-eval{suffix}"
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
        "interactive",
        "--preemptible",
    ]
    if region is not None:
        command.extend(["--region", region])
    command.extend(
        [
            "--tpu",
            tpu,
            "--cpu",
            "4",
            "--memory",
            "32GB",
            "--disk",
            "96GB",
            "--max-retries",
            "3",
            "--timeout",
            "21600",
            "--extra",
            "eval",
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
            "eval_contact_checkpoint.py",
            "--checkpoint",
            checkpoint,
            "--scratch",
            f"/app/scratch/{checkpoint}{suffix}",
            "--output-prefix",
            output_prefix(checkpoint, run_tag),
        ]
    )
    if region is not None and spec.region is not None and region != spec.region:
        print(
            f"[submit] warning: checkpoint is in {spec.region}, job requested {region}",
            flush=True,
        )
    return command


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", choices=sorted(CHECKPOINTS), required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cluster", default="marin")
    parser.add_argument("--tpu", default="v6e-4")
    parser.add_argument("--region")
    parser.add_argument("--user", default="eczech")
    parser.add_argument("--job-suffix")
    parser.add_argument("--run-tag")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    spec = CHECKPOINTS[args.checkpoint]
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
        cluster=args.cluster,
        tpu=args.tpu,
        region=args.region or spec.region,
        user=args.user,
        job_suffix=args.job_suffix,
        run_tag=args.run_tag,
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
