# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit the exp260 MSA-depth evaluation to CoreWeave.

Unlike PR #257, which federated the job through the ``marin`` controller, this
submits straight to the ``cw-us-east-02a`` controller over the kubeconfig that
cluster's config names (``~/.kube/coreweave-iris-gpu``, context
``marin-gpu_US-EAST-02A``). Same cluster either way — it is where the checkpoint
already lives, so no weights move.

    uv sync
    uv run python submit_coreweave.py --dry-run     # validate the command
    uv run python submit_coreweave.py --run-id v1-01

The driver publishes its results to the public HF bucket at the end, so it needs
a write token for ``open-athena``: pass ``--hf-token``, set ``HF_TOKEN``, or let
it read the CLI's stored token. The token reaches the pod as a job environment
variable and is never printed here.
"""

import argparse
import base64
import hashlib
import os
import subprocess
import sys
from pathlib import Path

from checkpoint_specs import CHECKPOINT_SUITES, MARIN_PREFIX

CLUSTER = "cw-us-east-02a"
DEFAULT_USER = "timodonnell"
HF_TOKEN_FILE = Path.home() / ".cache/huggingface/token"


def _metric_script() -> tuple[str, str]:
    repo_root = Path(__file__).resolve().parents[3]
    path = (
        repo_root
        / "experiments/exp89_evals_contacts_v1_model_on_eval_set"
        / "compute_metrics.py"
    )
    payload = path.read_bytes()
    return base64.b64encode(payload).decode(), hashlib.sha256(payload).hexdigest()


def _iris_binary() -> str:
    """Return this directory's pinned ``iris``, falling back to PATH."""

    candidate = Path(__file__).resolve().parent / ".venv/bin/iris"
    if candidate.exists():
        return str(candidate)
    return os.environ.get("IRIS_BIN", "iris")


def _hf_token(explicit: str | None) -> str:
    """Resolve the HF write token the driver publishes results with."""

    token = explicit or os.environ.get("HF_TOKEN")
    if not token and HF_TOKEN_FILE.exists():
        token = HF_TOKEN_FILE.read_text().strip()
    if not token:
        raise SystemExit(
            "no Hugging Face token: pass --hf-token, set HF_TOKEN, or run `hf auth login`"
        )
    return token


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="v1-01")
    parser.add_argument("--model-mirror-run-id")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--contact-mult", type=int, default=6)
    parser.add_argument(
        "--suite", choices=sorted(CHECKPOINT_SUITES), default="training"
    )
    parser.add_argument(
        "--job-suffix",
        help="Optional suffix for a distinct Iris job that resumes the same run ID.",
    )
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--hf-token")
    parser.add_argument("--iris-bin", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    metric_b64, metric_sha256 = _metric_script()
    hf_token = _hf_token(args.hf_token)
    job_name = f"exp260-msa-depth-eval-{args.run_id}"
    if args.job_suffix:
        job_name = f"{job_name}-{args.job_suffix}"
    command = [
        args.iris_bin or _iris_binary(),
        f"--cluster={CLUSTER}",
        "job",
        "run",
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
        "-e",
        "HF_TOKEN",
        hf_token,
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
        f"Submitting {job_name} to {CLUSTER}; storage prefix {MARIN_PREFIX}; "
        f"suite {args.suite}; metric script sha256 {metric_sha256}",
        file=sys.stderr,
    )
    if args.dry_run:
        print(
            "Dry run: command validated; embedded metric payload and HF token "
            "omitted from output.",
            file=sys.stderr,
        )
        return
    # Never let the command line reach a log: it carries the HF token. A
    # CalledProcessError stringifies its whole argv, so failures are reported
    # by exit status alone and the iris CLI's own stderr stays the diagnostic.
    completed = subprocess.run(command, cwd=Path(__file__).parent, check=False)
    if completed.returncode != 0:
        raise SystemExit(
            f"iris submission failed with exit status {completed.returncode}; "
            "see the iris output above"
        )


if __name__ == "__main__":
    main()
