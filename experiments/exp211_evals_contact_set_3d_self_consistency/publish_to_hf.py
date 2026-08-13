# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish the issue #211 artifacts to the public open-athena/MarinFold bucket.

Two things are worth publishing, for different reasons.

**The rollouts** (554 proteins x 100, one row per emitted contact with rollout
index and emission order) are the reusable asset. Every existing contacts-v1 eval
throws this structure away at the vote-counting step, so this is the only copy —
and #200 / #208 want exactly this shape for RL work on the base format.

**The scored arms** are the evidence behind the headline, so a reader can redo the
statistics without a GPU.

Needs an **open-athena-scoped** token (``hf auth whoami`` must list the org).

    uv run python publish_to_hf.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
from pathlib import Path

PREFIX = "hf://buckets/open-athena/MarinFold/data/contacts-v1-consistency-exp211"


def hf_cli() -> str:
    """Find an ``hf`` that supports ``buckets``.

    Bucket support landed in huggingface_hub 1.5. This experiment's venv pins
    0.36 (that is what the analysis stack resolves to), so the venv's ``hf`` on
    PATH under ``uv run`` does NOT have the subcommand and fails with an argparse
    "invalid choice" rather than anything that looks like a version problem.
    Probe instead of trusting PATH order.
    """
    import shutil

    candidates = [c for c in (shutil.which("hf"), "/home/bizon/anaconda3/bin/hf",
                              "/usr/local/bin/hf") if c]
    for cand in candidates:
        probe = subprocess.run([cand, "buckets", "--help"],
                               capture_output=True, text=True)
        if probe.returncode == 0:
            return cand
    raise SystemExit(
        "no `hf` on this box supports `hf buckets` (needs huggingface_hub >= 1.5); "
        f"tried {candidates}"
    )


def upload(cli: str, local: Path, prefix: str, *, dry_run: bool) -> None:
    cmd = [cli, "buckets", "cp", str(local), f"{prefix}/{local.name}"]
    print("  " + " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", type=Path, default=Path("_scratch/rollouts"))
    ap.add_argument("--prefix", default=PREFIX)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    staged = Path("_scratch/publish")
    staged.mkdir(parents=True, exist_ok=True)

    # 554 + 554 small parquets would be 1108 bucket objects; one tarball instead.
    tarball = staged / "rollouts_exp199_n100.tar.gz"
    if not tarball.exists():
        print(f"[publish] packing {args.rollouts} -> {tarball}")
        with tarfile.open(tarball, "w:gz") as tf:
            tf.add(args.rollouts, arcname="rollouts")
    print(f"[publish] {tarball.name}: {tarball.stat().st_size / 2**20:.1f} MiB")

    files = [tarball, Path("data/arm_scores.csv.gz"), Path("data/results.txt"),
             Path("data/results.json"), Path("data/bounds.json"),
             Path("data/gt_gate.csv"), Path("data/power_check.csv"),
             Path("data/verify_exp82.csv"), Path("data/eval_targets.parquet")]
    missing = [f for f in files if not f.exists()]
    if missing:
        print(f"[publish] MISSING: {missing}", file=sys.stderr)
        return 1

    cli = hf_cli()
    print(f"[publish] using {cli}")
    print(f"[publish] -> {args.prefix}/")
    for f in files:
        upload(cli, f, args.prefix, dry_run=args.dry_run)
    print("[publish] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
