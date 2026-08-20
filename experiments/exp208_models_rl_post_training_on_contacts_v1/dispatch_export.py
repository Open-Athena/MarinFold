# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Export each trained arm to a bf16 HF checkpoint on a CPU pod — issue #208.

One job per arm. CPU because ``export_lm_to_hf`` runs with ``use_cpu=True``, so
this does not occupy a v5p; and in us-central1 because that is where both the
checkpoints and the compute live (see ``check_region_locality``).

    uv run python dispatch_export.py --arms 1em06,3em06,1em05
"""

import argparse

from _submit import check_clean, submit

PREFIX = "gs://marin-us-central1/protein-structure/MarinFold/exp208"
TOKENIZER = "timodonnell/plm-exp163-refine-cv1-1_5b-lr1e-4-e1-cos-tpuF-step404"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="1em06,3em06,1em05")
    ap.add_argument("--suffix", default="s8-v2", help="run-name suffix the sweep used")
    ap.add_argument("--step", default=None, help="step-N to export; default is the highest")
    ap.add_argument("--region", default="us-central1")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if not a.dry_run:
        check_clean()

    for arm in [x.strip() for x in a.arms.split(",") if x.strip()]:
        run = f"plm-exp208-rl-cv1-1_5b-lr{arm}-{a.suffix}"
        # The trainer namespaces its own checkpoints under {run}-train.
        checkpoint_dir = f"{PREFIX}/checkpoints/{run}/{run}-train"
        command = [
            "python", "export_checkpoint.py",
            "--checkpoint-dir", checkpoint_dir,
            "--out", f"{PREFIX}/exports/{run}",
            "--tokenizer", TOKENIZER,
        ]
        if a.step:
            command += ["--step", a.step]
        submit(
            job_name=f"exp208-export-{arm}",
            extras=("cpu",),
            cpu=16, memory="96GB",
            # f32 export plus the bf16 copy, both staged locally.
            disk="96GB",
            region=a.region,
            command=command,
            dry_run=a.dry_run,
        )
        print(f"  export: {PREFIX}/exports/{run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
