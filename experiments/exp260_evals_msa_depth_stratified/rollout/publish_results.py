# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish the small result artifacts from CoreWeave to the public HF bucket.

The CoreWeave results prefix is not readable from a workstation, and this
evaluation deliberately holds no CoreWeave object-storage credentials outside
the cluster. So the driver ends by pushing the aggregates, per-protein rows,
timings, and provenance manifest to
``hf://buckets/open-athena/MarinFold/data/contacts-v1-msa-depth-exp260/<run-id>/``,
which every downstream reader (analysis scripts, notebooks, reviewers) can fetch
anonymously over HTTPS. The dense ``[L,L]`` vote matrices stay in CoreWeave —
they are the one artifact large enough that publishing them is not free.

Uploading goes through the ``hf`` CLI rather than ``HfFileSystem``: buckets are
writable through ``hf buckets cp`` but not through the filesystem adapter.

Runs as part of the driver, and standalone against a finished run:

    python publish_results.py --run-id v1-01
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import fsspec
from checkpoint_specs import PUBLISH_BUCKET, PUBLISH_PREFIX, run_root

#: Everything a reader needs to reproduce every number in the write-up. Paths
#: are relative to the run root on both sides, so the bucket layout mirrors the
#: CoreWeave one.
PUBLISHED_FILES = (
    "results/aggregate_metrics.csv",
    "results/subset_aggregate_metrics.csv",
    "results/marinfold_precision.csv",
    "results/contact_precision_all.csv",
    "results/timings.csv",
    "results/run_manifest.json",
    "results/published_reference_validation.json",
    "inputs/evaluation_subsets.csv",
    "inputs/eval_targets.parquet.validation.json",
)


def hf_binary() -> str:
    """Return the ``hf`` CLI that belongs to the running interpreter.

    The driver's venv is the one with ``huggingface_hub>=1.5`` in it; a bare
    ``hf`` on PATH could be an older install without the ``buckets`` subcommand.
    """

    candidate = Path(sys.executable).with_name("hf")
    if candidate.exists():
        return str(candidate)
    found = shutil.which("hf")
    if found is None:
        raise RuntimeError("no `hf` CLI available to publish results")
    return found


def publish(*, run_root_uri: str, run_id: str) -> dict:
    """Copy :data:`PUBLISHED_FILES` to the public bucket; return what landed."""

    filesystem, root = fsspec.core.url_to_fs(run_root_uri)
    destination = f"hf://buckets/{PUBLISH_BUCKET}/{PUBLISH_PREFIX}/{run_id}"
    binary = hf_binary()
    scratch = Path("/tmp/exp260_publish")
    if scratch.exists():
        shutil.rmtree(scratch)
    published: list[dict] = []
    for relative in PUBLISHED_FILES:
        source = f"{root.rstrip('/')}/{relative}"
        if not filesystem.exists(source):
            raise FileNotFoundError(f"expected result artifact is missing: {source}")
        local = scratch / relative
        local.parent.mkdir(parents=True, exist_ok=True)
        filesystem.get_file(source, str(local))
        subprocess.run(
            [binary, "buckets", "cp", str(local), f"{destination}/{relative}"],
            check=True,
        )
        published.append({"path": relative, "bytes": local.stat().st_size})
    record = {
        "bucket": PUBLISH_BUCKET,
        "prefix": f"{PUBLISH_PREFIX}/{run_id}",
        "https_root": (
            f"https://huggingface.co/buckets/{PUBLISH_BUCKET}/resolve/"
            f"{PUBLISH_PREFIX}/{run_id}"
        ),
        "files": published,
    }
    print(json.dumps({"event": "published", **record}, sort_keys=True), flush=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    publish(run_root_uri=run_root(args.run_id), run_id=args.run_id)


if __name__ == "__main__":
    main()
