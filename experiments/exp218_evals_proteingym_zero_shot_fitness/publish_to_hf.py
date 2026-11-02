# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish exp218's public artifacts to the ``open-athena/MarinFold`` HF bucket.

Two things go up:

- ``conditionals/`` — the ``(K, L, 20)`` amino-acid log-probability tensors, one
  ``.npz`` per assay (~623 MB at K=200). These are the point of the experiment
  as a reusable object: any variant-effect scoring rule, including ones nobody
  has written yet, can be re-derived from them without a GPU and without this
  code.
- ``results/`` — the small CSV/JSON outputs that back the README's numbers, so a
  reader can check the aggregation without re-running anything.

The bucket is world-readable with no auth (``HfFileSystem(token=False)``), which
is what lets a Colab notebook read these; writing needs an **open-athena-scoped**
token (`hf auth whoami` must list the org).

Usage::

    uv run python publish_to_hf.py                # upload
    uv run python publish_to_hf.py --dry-run      # list what would go
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi

HERE = Path(__file__).resolve().parent
REPO_ID = "open-athena/MarinFold"
PREFIX = "experiments/exp218_proteingym"


def files_to_publish() -> list[tuple[Path, str]]:
    """(local path, path in the bucket) for everything we publish."""
    out: list[tuple[Path, str]] = []
    for path in sorted((HERE / "data" / "conditionals").glob("*.npz")):
        out.append((path, f"{PREFIX}/conditionals/{path.name}"))
    for name in (
        "marinfold_spearman_dms_level.csv",
        "leaderboard_comparison.csv",
        "depth_breakdown.csv",
        "summary.json",
        "phase0_context_curve.csv",
        "phase0_per_protein.csv",
        "phase0_verdict.json",
        "timings.csv",
    ):
        path = HERE / "data" / name
        if path.exists():
            out.append((path, f"{PREFIX}/results/{name}"))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    entries = files_to_publish()
    total = sum(path.stat().st_size for path, _ in entries)
    print(f"{len(entries)} files, {total / 2**20:.0f} MiB -> {REPO_ID}/{PREFIX}")
    if args.dry_run:
        for path, target in entries[:5]:
            print(f"  {path.name:<50s} -> {target}")
        print(f"  ... and {max(len(entries) - 5, 0)} more")
        return

    api = HfApi()
    who = api.whoami()
    orgs = {o["name"] for o in who.get("orgs", [])}
    if "open-athena" not in orgs:
        raise SystemExit(
            f"Token belongs to {who['name']} with orgs {sorted(orgs)}; writing to "
            f"{REPO_ID} needs an open-athena-scoped token."
        )

    # One folder upload rather than per-file: the bucket API batches commits and
    # a 600 MB per-file loop would be both slower and non-atomic.
    api.upload_folder(
        repo_id=REPO_ID,
        repo_type="dataset",
        folder_path=str(HERE / "data" / "conditionals"),
        path_in_repo=f"{PREFIX}/conditionals",
        allow_patterns=["*.npz"],
    )
    for path, target in entries:
        if path.suffix == ".npz":
            continue
        api.upload_file(
            repo_id=REPO_ID,
            repo_type="dataset",
            path_or_fileobj=str(path),
            path_in_repo=target,
        )
    print(f"published to https://huggingface.co/buckets/{REPO_ID}/tree/{PREFIX}")


if __name__ == "__main__":
    main()
