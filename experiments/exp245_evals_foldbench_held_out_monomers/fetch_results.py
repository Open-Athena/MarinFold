# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 7a -- bring the CoreWeave evaluation's results back to this repo.

The run writes its results to ``s3://marin-us-east-02a/...``, which only jobs
inside that region can read, so ``rollout/export_results.py`` copies them to the
public bucket from inside the cluster. This downloads that copy anonymously and
commits the small tables under ``data/coreweave_results/``.

    uv run python fetch_results.py --run-id fbmono-20260818-01
"""
import argparse
import json
import urllib.request
from pathlib import Path

import upstream as U

BUCKET = "https://huggingface.co/buckets/open-athena/MarinFold/resolve"
BUCKET_PREFIX = "data/contacts-v1-foldbench-monomers-exp245"
OUT = U.DATA / "coreweave_results"

#: The tables the analysis and the README need. The dense per-unit vote matrices
#: stay in S3: they are large, and nothing downstream reads them.
WANTED = (
    "marinfold_precision.csv",
    "contact_precision_all.csv",
    "aggregate_metrics.csv",
    "subset_aggregate_metrics.csv",
    "run_manifest.json",
    "timings.csv",
)


def fetch(run_id: str, name: str, destination: Path) -> int:
    url = f"{BUCKET}/{BUCKET_PREFIX}/runs/{run_id}/{name}"
    with urllib.request.urlopen(url, timeout=300) as response:
        payload = response.read()
    destination.write_bytes(payload)
    return len(payload)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--files", nargs="*", default=list(WANTED))
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    fetched = {}
    for name in args.files:
        size = fetch(args.run_id, name, OUT / name)
        fetched[name] = {"bytes": size, "sha256": U.sha256(OUT / name)}
        print(f"[fetch] {name}: {size:,} bytes", flush=True)
    (OUT / "provenance.json").write_text(json.dumps({
        "run_id": args.run_id,
        "source": f"{BUCKET}/{BUCKET_PREFIX}/runs/{args.run_id}/",
        "files": fetched,
    }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
