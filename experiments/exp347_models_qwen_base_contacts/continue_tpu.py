"""Run a bounded hardware gate, then the fixed continuation on the same slice.

This is a two-stage job, not a placement or recovery controller. Failure or a
two-hour timeout in validation prevents training. An Iris preemption restarts the
same job; an already committed successful gate is reused, and train_tpu.py owns
full-state checkpoint recovery. No other jobs or resources are submitted here.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import fsspec

from common import FORMATS

ROOT = "gs://marin-us-east5/protein-structure/MarinFold/exp347_qwen_base_contacts"


def main() -> None:
    """Require a successful report before replacing this process with training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=FORMATS, required=True)
    parser.add_argument("--attempt", type=int, required=True)
    args = parser.parse_args()
    snapshot = json.loads(Path(f"data/tpu_forks/{args.format}_parent.json").read_text())
    name = f"debug-{snapshot['fork_run']}-a{args.attempt:02d}"
    report_uri = f"{ROOT}/data/tpu-validation/{name}.json"
    fs = fsspec.filesystem("gcs")
    if not fs.exists(report_uri):
        subprocess.run(
            [
                "timeout",
                "--kill-after=60s",
                "7200s",
                sys.executable,
                "validate_tpu_runtime.py",
                "--format",
                args.format,
                "--attempt",
                str(args.attempt),
            ],
            check=True,
        )
    report = json.loads(fs.cat_file(report_uri))
    expected_source = f"{ROOT}/checkpoints/{snapshot['fork_run']}/source/step-{snapshot['parent']['step']}"
    if report["status"] != "passed" or report["source"] != expected_source:
        raise ValueError("Hardware gate did not pass for this frozen parent")
    print(
        json.dumps({"phase": "training", "validation_report": report_uri}), flush=True
    )
    os.execv(
        sys.executable,
        [
            sys.executable,
            "train_tpu.py",
            "--format",
            args.format,
            "--validation-report",
            report_uri,
        ],
    )


if __name__ == "__main__":
    main()
