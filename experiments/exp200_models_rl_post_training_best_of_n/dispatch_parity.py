# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit the Phase 1 parity gate to a marin v5p-8 — issue #200.

This directory is the iris workspace (see :mod:`_submit`), so the worker and its
modules travel in the bundle and the pod resolves exp200's own pinned manifest.
There is no marin checkout, no base64-inlined source, and no client/workspace
split — all of which earlier versions of this file needed.

Usage::

    uv run python dispatch_parity.py --limit 554 --n-generations 4 --max-sections 0
    uv run python dispatch_parity.py --dry-run
"""

import argparse

from _submit import check_clean, submit

EXP163 = "gs://marin-us-east5/MarinFold/exp163"
MODEL = f"{EXP163}/tpu/tpuF-bf16/step-404"
TARGETS = f"{EXP163}/eval554/targets.parquet"
PROMPTS = f"{EXP163}/eval554/prompts"
OUT = "gs://marin-us-east5/MarinFold/exp200/phase1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=554, help="proteins to sample (randomly)")
    ap.add_argument("--n-generations", type=int, default=4)
    ap.add_argument("--max-sections", type=int, default=0, help="0 = uncapped, matching #163")
    ap.add_argument("--tensor-parallel-size", type=int, default=4)
    ap.add_argument("--tpu", default="v5p-8")
    # us-east5 had 2 v5p-8 ready and both busy; us-central1-a had 96 ready and
    # zero demand. Measured, not guessed — which is the only good reason to pin a
    # zone rather than a region.
    ap.add_argument("--zone", default="us-central1-a")
    ap.add_argument("--model", default=MODEL,
                    help="checkpoint to score. A gs:// HF export works: phase1_parity "
                         "stages it locally, so the tokenizer loader never sees a URL.")
    ap.add_argument("--job-name", default="exp200-phase1-parity")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if not a.dry_run:
        check_clean()

    submit(
        job_name=a.job_name,
        extras=("tpu", "vllm"),
        tpu=a.tpu,
        zone=a.zone,
        command=[
            "python", "phase1_parity.py",
            "--model", a.model, "--targets", TARGETS, "--prompts", PROMPTS, "--out", OUT,
            "--limit", str(a.limit),
            "--n-generations", str(a.n_generations),
            "--max-sections", str(a.max_sections),
            "--tensor-parallel-size", str(a.tensor_parallel_size),
            "--tag", a.job_name,
        ],
        dry_run=a.dry_run,
    )
    print(f"  results: {OUT}/{a.job_name}/parity_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
