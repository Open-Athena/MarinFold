# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Derive retained eval-set aggregates from the per-protein metric rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from finalize_coreweave import aggregate_subsets

HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    """Return a file's SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(
    precision_path: Path,
    subset_manifest_path: Path,
    output_path: Path,
) -> None:
    """Write all eval-val/eval-denovo aggregates and a provenance sidecar."""

    precision = pd.read_csv(precision_path)
    subset_manifest = pd.read_csv(subset_manifest_path)
    ordered_units = list(
        zip(subset_manifest.dataset, subset_manifest.stem, strict=True)
    )
    aggregate, counts = aggregate_subsets(
        precision,
        ordered_units=ordered_units,
        subset_manifest=subset_manifest,
    )
    eval_aggregate = aggregate[aggregate.subset.str.startswith("eval-")].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    eval_aggregate.to_csv(output_path, index=False)

    metadata = {
        "schema_version": 1,
        "description": (
            "Current eval-set aggregates retained independently of the legacy-554 "
            "comparison. Viral and nonviral partitions are included when present."
        ),
        "subset_units": {
            key: value for key, value in counts.items() if key.startswith("eval-")
        },
        "sources": {
            "per_protein_metrics": {
                "path": str(precision_path.relative_to(HERE)),
                "sha256": sha256(precision_path),
            },
            "subset_manifest": {
                "path": str(subset_manifest_path.relative_to(HERE)),
                "sha256": sha256(subset_manifest_path),
            },
        },
        "output_sha256": sha256(output_path),
    }
    output_path.with_suffix(output_path.suffix + ".meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(f"wrote {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse artifact paths."""

    default_results = HERE / "data" / "coreweave_results"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--precision",
        type=Path,
        default=default_results / "marinfold_precision.csv.gz",
    )
    parser.add_argument(
        "--subset-manifest",
        type=Path,
        default=default_results / "evaluation_subsets.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_results / "eval_split_metrics.csv",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    build(arguments.precision, arguments.subset_manifest, arguments.output)
