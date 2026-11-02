# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish the scored runs + predicted structures to the public HF bucket.

The predictions are the deliverable a reader most wants to poke at, and
``explore_predictions.ipynb`` has to reach them **without authentication**
(``experiments/AGENTS.md`` rule 2), so they go to the public
``open-athena/MarinFold`` bucket rather than staying on CoreWeave storage or
GCS.

Publishes under ``data/exp174-structural-eval/``:

* ``results/scores_all.csv`` — every (record, run) row, concatenated with a
  ``run`` column. This is what the notebook's tables and plots read.
* ``results/summary_all.csv`` — the stratified aggregate for every run.
* ``results/pred_<run>.tar.gz`` — one tarball of canonical PDBs per run.

The ground-truth bundle is published separately by ``publish_gt_bundle.py``.

Usage::

    uv run python publish_results.py --scores-dir _scratch/scores \\
        --pred-dir _scratch/pred --out data/
"""

import argparse
import subprocess
import tarfile
from pathlib import Path

import pandas as pd

BUCKET_PREFIX = "hf://buckets/open-athena/MarinFold/data/exp174-structural-eval"


def upload(local: Path, remote: str, *, dry_run: bool) -> None:
    command = ["hf", "buckets", "cp", str(local), remote]
    print("  $ " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scores-dir", type=Path, required=True)
    ap.add_argument("--pred-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="local dir for the combined CSVs")
    ap.add_argument("--prefix", default=BUCKET_PREFIX)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-structures", action="store_true",
                    help="publish the CSVs only (structures are the slow part)")
    ap.add_argument(
        "--runs",
        default=None,
        help="comma-separated run names whose structures to publish. Default is "
        "everything under --pred-dir, which includes the model-free quantization "
        "baselines - those are just requantized ground truth, so on a ~3 MB/s "
        "uplink they are 25 minutes of wasted upload.",
    )
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)

    # --- combined per-record scores + summaries ---
    score_frames, summary_frames = [], []
    for path in sorted(args.scores_dir.glob("*.csv")):
        if path.name.endswith(".summary.csv"):
            frame = pd.read_csv(path)
            frame.insert(0, "run", path.name[: -len(".summary.csv")])
            summary_frames.append(frame)
        else:
            frame = pd.read_csv(path)
            frame.insert(0, "run", path.stem)
            score_frames.append(frame)
    if not score_frames:
        raise SystemExit(f"no score CSVs under {args.scores_dir}")

    scores = pd.concat(score_frames, ignore_index=True)
    scores_path = args.out / "scores_all.csv"
    scores.to_csv(scores_path, index=False)
    print(f"[publish] {len(scores)} rows, {scores.run.nunique()} runs -> {scores_path}")

    summary_path = args.out / "summary_all.csv"
    pd.concat(summary_frames, ignore_index=True).to_csv(summary_path, index=False)

    upload(scores_path, f"{args.prefix}/results/scores_all.csv", dry_run=args.dry_run)
    upload(summary_path, f"{args.prefix}/results/summary_all.csv", dry_run=args.dry_run)

    if args.skip_structures:
        print("[publish] skipping structure tarballs")
        return 0

    # --- one tarball of predicted structures per run ---
    wanted = set(args.runs.split(",")) if args.runs else None
    for run_dir in sorted(p for p in args.pred_dir.iterdir() if p.is_dir()):
        if wanted is not None and run_dir.name not in wanted:
            continue
        pdbs = list(run_dir.rglob("*.pdb"))
        if not pdbs:
            continue
        archive = args.out / f"pred_{run_dir.name}.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for pdb in sorted(pdbs):
                tar.add(pdb, arcname=str(pdb.relative_to(run_dir)))
        print(
            f"[publish] {run_dir.name}: {len(pdbs)} structures, "
            f"{archive.stat().st_size / 1e6:.1f} MB",
            flush=True,
        )
        upload(archive, f"{args.prefix}/results/{archive.name}", dry_run=args.dry_run)

    print(f"[publish] done -> {args.prefix}/results/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
