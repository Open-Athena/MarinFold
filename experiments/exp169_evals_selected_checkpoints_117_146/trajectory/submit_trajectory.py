# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit independent checkpoint-trajectory evaluations to marin-dev."""

import argparse
import os
import re
import shlex
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

from checkpoint_specs import CHECKPOINTS, HF_BUCKET_ROOT, RUNS

JOB_FAMILIES = {
    "exp146-3b-e8": "146-3b-e8",
    "exp117-1_5b-e16": "117-1p5b-e16",
    "exp117-1_5b-e8-bs64": "117-1p5b-e8-bs64",
}


def output_prefix(checkpoint: str) -> str:
    """Return the canonical durable output prefix for one checkpoint."""

    spec = CHECKPOINTS[checkpoint]
    return f"{HF_BUCKET_ROOT}/runs/{spec.run_name}/step-{spec.step}"


def chip_count(tpu: str) -> int:
    """Read the chip count from an Iris TPU shape such as v6e-16."""

    match = re.search(r"-(\d+)$", tpu)
    if match is None:
        raise ValueError(f"cannot infer chip count from TPU shape {tpu!r}")
    return int(match.group(1))


def checkpoint_selection(raw: str | None, run_key: str | None) -> list[str]:
    """Resolve an explicit list or one run family into catalog keys."""

    if raw:
        selected = [item for item in raw.split(",") if item]
        unknown = sorted(set(selected) - CHECKPOINTS.keys())
        if unknown:
            raise ValueError(f"unknown checkpoints: {unknown}")
        return selected
    return [
        key
        for key, spec in CHECKPOINTS.items()
        if run_key is None or spec.run_key == run_key
    ]


def build_command(
    *,
    iris: str,
    checkpoint: str,
    hf_token: str,
    cluster: str,
    tpu: str,
    user: str,
    job_suffix: str | None,
) -> list[str]:
    """Build one explicitly placed and independently resumable Iris job."""

    spec = CHECKPOINTS[checkpoint]
    suffix = f"-{job_suffix}" if job_suffix else ""
    short_family = JOB_FAMILIES[spec.run_key]
    job_name = f"marinfold-exp169-traj-{short_family}-s{spec.step}{suffix}"
    return [
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
        "--preemptible",
        "--region",
        spec.region,
        "--tpu",
        tpu,
        "--cpu",
        "8",
        "--memory",
        "64GB",
        "--disk",
        "96GB",
        "--max-retries",
        "3",
        "--timeout",
        "28800",
        "--extra",
        "eval",
        "-e",
        "HF_TOKEN",
        hf_token,
        "-e",
        "MARINFOLD_ACCELERATOR",
        tpu,
        "-e",
        "MARINFOLD_IRIS_JOB",
        f"/{user}/{job_name}",
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
        output_prefix(checkpoint),
        "--tensor-parallel-size",
        str(chip_count(tpu)),
    ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse a checkpoint family or explicit checkpoint submission."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", help="comma-separated catalog keys")
    parser.add_argument(
        "--run-key",
        choices=tuple(RUNS),
        help="submit every selected checkpoint from one run",
    )
    parser.add_argument("--tpu", default="v6e-8")
    parser.add_argument("--cluster", default="marin-dev")
    parser.add_argument("--user", default="eczech")
    parser.add_argument("--job-suffix")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Submit every selected checkpoint as a separate Iris job."""

    args = parse_args(argv)
    iris = shutil.which("iris")
    if iris is None:
        raise FileNotFoundError("iris is not installed in the active environment")
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token and not args.dry_run:
        raise ValueError("HF_TOKEN must contain the open-athena write token")
    selected = checkpoint_selection(args.checkpoints, args.run_key)
    for checkpoint in selected:
        command = build_command(
            iris=iris,
            checkpoint=checkpoint,
            hf_token=hf_token or "<HF_TOKEN>",
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
            continue
        subprocess.run(command, cwd=Path(__file__).parent, check=True)
    print(
        f"[submit] {'planned' if args.dry_run else 'submitted'} {len(selected)} job(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
