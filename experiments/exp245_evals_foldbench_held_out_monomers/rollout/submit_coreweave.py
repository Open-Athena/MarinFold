# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit the exp245 rollout driver to CoreWeave through Iris federation.

Adapted from PR #244's submitter. The driver is a small CPU job in the same
region as the checkpoints; it launches the GPU shards itself, so this only has
to hand it the run id and the embedded #89 metric script.
"""

import argparse
import base64
import hashlib
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from checkpoint_specs import CHECKPOINT_SUITES, MARIN_PREFIX

TARGET_CLUSTER = "cw-us-east-02a"
#: Iris rejects a client older than 14 days, so this must be a recent checkout
#: rather than a pinned wheel. Override with ``IRIS_BIN`` on another machine.
DEFAULT_IRIS = "/home/bizon/git/marin-freshiris/.venv/bin/iris"


def _metric_script() -> tuple[str, str]:
    repo_root = Path(__file__).resolve().parents[3]
    path = (
        repo_root
        / "experiments/exp89_evals_contacts_v1_model_on_eval_set"
        / "compute_metrics.py"
    )
    payload = path.read_bytes()
    return base64.b64encode(payload).decode(), hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=f"v2-{datetime.now(UTC):%Y%m%d-%H%M%S}")
    parser.add_argument("--model-mirror-run-id")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--contact-mult", type=int, default=6)
    parser.add_argument("--suite", choices=sorted(CHECKPOINT_SUITES), default="exp245")
    parser.add_argument(
        "--job-suffix",
        help="Optional suffix for a distinct Iris job that resumes the same run ID.",
    )
    parser.add_argument("--iris-bin", default=os.environ.get("IRIS_BIN", DEFAULT_IRIS))
    parser.add_argument("--user", default=os.environ.get("IRIS_USER", "eczech"),
                        help="Iris job owner; jobs land under /<user>/<job-name>.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    metric_b64, metric_sha256 = _metric_script()
    job_name = f"exp245-rollout-{args.run_id}"
    if args.suite != "exp245":
        job_name = f"exp245-{args.suite}-rollout-{args.run_id}"
    if args.job_suffix:
        job_name = f"{job_name}-{args.job_suffix}"
    command = [
        args.iris_bin,
        "--cluster=marin",
        "job",
        "run",
        "--target-cluster",
        TARGET_CLUSTER,
        "--priority",
        "batch",
        "--enable-extra-resources",
        "--user",
        args.user,
        "--job-name",
        job_name,
        "--cpu",
        "4",
        "--memory",
        "16GB",
        "--disk",
        "32GB",
        "--max-retries",
        "3",
        "--timeout",
        "21600",
        "--no-wait",
        "-e",
        "MARIN_PREFIX",
        MARIN_PREFIX,
        "-e",
        "EXP89_COMPUTE_METRICS_B64",
        metric_b64,
        "-e",
        "EXP89_COMPUTE_METRICS_SHA256",
        metric_sha256,
        "--",
        "python",
        "run_coreweave_eval.py",
        "--run-id",
        args.run_id,
        "--seed",
        str(args.seed),
        "--suite",
        args.suite,
        "--contact-mult",
        str(args.contact_mult),
    ]
    if args.model_mirror_run_id:
        command.extend(["--model-mirror-run-id", args.model_mirror_run_id])
    print(
        f"Submitting {job_name} to {TARGET_CLUSTER}; storage prefix {MARIN_PREFIX}; "
        f"suite {args.suite}; metric script sha256 {metric_sha256}"
    )
    if args.dry_run:
        print(
            "Dry run: command validated; embedded metric payload omitted from output."
        )
        return
    subprocess.run(command, cwd=Path(__file__).parent, check=True)


if __name__ == "__main__":
    main()
