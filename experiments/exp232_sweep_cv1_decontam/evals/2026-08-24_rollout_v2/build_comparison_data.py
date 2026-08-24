# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Assemble the training-stage comparison from current and PR #244 results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REFERENCE_KEYS = (
    "exp146",
    "exp166",
    "exp232-m1-p02-decontam",
    "exp232-m2-p06-decontam",
    "cw-p06-aug",
    "cw-p06-cool",
)
REFERENCE_ROWS_KEYS = (*REFERENCE_KEYS, "protenix")
DISPLAY_NAME_OVERRIDES = {
    "exp146": "#146 3B",
    "cw-p06-aug": "CW m1-p06 aug",
    "cw-p06-cool": "CW m1-p06 cooldown",
    "exp232-m1-p02-decontam": "#232 m1-p02 sweep",
    "exp232-m2-p06-decontam": "#232 m2-p06 sweep",
}
VALIDATION_MODELS = {
    "marinfold-e8-reference-step35679": {
        "key": "exp75-reproduced",
        "display_name": "#75 E8 validation",
        "parameters": "1.5B",
        "evaluation": "validation",
        "wandb_run_id": "prot-exp75-cv1-1_5b-e8-lr1e-3-wd0p2-v1",
        "loss": 3.138312048873902,
        "fit_group": "exp75",
        "coreweave_checkpoint": (
            "s3://marin-us-east-02a/MarinFold/exp163/model/step-35679"
        ),
    }
}
NEW_MODELS = {
    "marinfold-exp232-decontam-train-m2-p06-step363000": {
        "key": "exp232-m2-p06-training",
        "display_name": "#232 m2-p06 training",
        "parameters": "1.5B",
        "evaluation": "computed_here",
        "wandb_run_id": (
            "prot-exp232-trc-cv1-decontam-train-s01-m2-p06-srcpeak-"
            "augcont-lr005-us-east1"
        ),
        "loss": 2.9680745601654053,
        "fit_group": "exp232-m2-p06-training",
        "coreweave_checkpoint": (
            "s3://marin-us-east-02a/marin/protein-structure/MarinFold/"
            "exp232_sweep_cv1_decontam/evals/rollout-v2/2026-08-24/v2-01/"
            "models/exp232-decontam-train-m2-p06-step363000/hf/step-363000"
        ),
    },
}


def sha256(path: Path) -> str:
    """Return a file's SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance_path(path: Path) -> str:
    """Return a portable repository-style path for provenance records."""

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(HERE.parents[3]))
    except ValueError:
        if "experiments" in resolved.parts:
            first = resolved.parts.index("experiments")
            return "/".join(resolved.parts[first:])
        return str(path)


def metric_value(aggregate: pd.DataFrame, *, model: str, range_name: str) -> float:
    """Return one legacy-554 R-precision aggregate."""

    selected = aggregate[
        (aggregate.model == model)
        & (aggregate.subset == "legacy_554")
        & (aggregate["range"] == range_name)
        & (aggregate.cut == "R")
    ]
    if len(selected) != 1:
        raise ValueError(
            f"expected one {model}/{range_name}/R row, found {len(selected)}"
        )
    return float(selected.iloc[0].precision)


def build(arguments: argparse.Namespace) -> None:
    """Build compact comparison tables and their provenance metadata."""

    prior = pd.read_csv(arguments.prior_comparison)
    reference = prior[prior.key.isin(REFERENCE_KEYS)].copy()
    if tuple(reference.key) != REFERENCE_KEYS:
        reference = reference.set_index("key").loc[list(REFERENCE_KEYS)].reset_index()
    loss_column = "loss" if "loss" in reference else "loss_current_scale"
    reference = reference[
        [
            "key",
            "display_name",
            "parameters",
            "wandb_run_id",
            loss_column,
            "r_all",
            "r_long",
            "fit_group",
            "coreweave_checkpoint",
            "metrics_source",
        ]
    ].rename(columns={loss_column: "loss"})
    reference.insert(3, "evaluation", "previous")
    reference["display_name"] = reference.apply(
        lambda row: DISPLAY_NAME_OVERRIDES.get(row["key"], row["display_name"]),
        axis=1,
    )

    aggregate = pd.read_csv(arguments.subset_aggregate)
    evaluated_models = {**VALIDATION_MODELS, **NEW_MODELS}
    evaluated_records = []
    for model, identity in evaluated_models.items():
        evaluated_records.append(
            {
                "key": identity["key"],
                "display_name": identity["display_name"],
                "parameters": identity["parameters"],
                "evaluation": identity["evaluation"],
                "wandb_run_id": identity["wandb_run_id"],
                "loss": identity["loss"],
                "r_all": metric_value(aggregate, model=model, range_name="all"),
                "r_long": metric_value(aggregate, model=model, range_name="long"),
                "fit_group": identity["fit_group"],
                "coreweave_checkpoint": identity["coreweave_checkpoint"],
                "metrics_source": str(arguments.subset_aggregate),
            }
        )
    comparison = pd.concat(
        [reference, pd.DataFrame(evaluated_records)], ignore_index=True
    )
    if len(comparison) != 8 or comparison.key.nunique() != 8:
        raise ValueError(
            "comparison must contain six references, one validation checkpoint, "
            "and one new checkpoint"
        )

    prior_rows = pd.read_csv(arguments.prior_rows)
    reference_rows = prior_rows[prior_rows.key.isin(REFERENCE_ROWS_KEYS)].copy()
    if reference_rows.groupby("key").size().to_dict() != {
        key: 554 for key in REFERENCE_ROWS_KEYS
    }:
        raise ValueError("reference per-protein rows are incomplete")

    subset_manifest = pd.read_csv(arguments.subset_manifest)
    legacy = subset_manifest.loc[
        subset_manifest.subset == "legacy_554", ["dataset", "stem"]
    ]
    legacy_units = pd.MultiIndex.from_frame(legacy[["dataset", "stem"]])
    if len(legacy_units) != 554 or not legacy_units.is_unique:
        raise ValueError("legacy target identities are not the expected 554 units")
    precision = pd.read_csv(arguments.precision)
    precision_units = pd.MultiIndex.from_frame(precision[["dataset", "stem"]])
    new_rows = []
    for model, identity in evaluated_models.items():
        selected = precision[
            (precision.model == model)
            & (precision["range"] == "all")
            & (precision.cut == "R")
            & precision_units.isin(legacy_units)
        ][["dataset", "stem", "precision"]].copy()
        if len(selected) != 554:
            raise ValueError(f"{model} has {len(selected)} legacy all-R rows")
        expected = comparison.set_index("key").loc[identity["key"], "r_all"]
        if not np.isclose(selected.precision.mean(), expected, atol=1e-14):
            raise ValueError(f"{model} per-protein mean does not match its aggregate")
        selected.insert(0, "key", identity["key"])
        new_rows.append(selected)
    rows = pd.concat([reference_rows, *new_rows], ignore_index=True)
    expected_keys = {
        *REFERENCE_ROWS_KEYS,
        *(value["key"] for value in evaluated_models.values()),
    }
    if set(rows.key) != expected_keys:
        raise ValueError("combined per-protein row keys are incomplete")

    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    comparison_path = arguments.output_directory / "comparison.csv"
    rows_path = arguments.output_directory / "all_r_rows.csv.gz"
    metadata_path = arguments.output_directory / "comparison_data.meta.json"
    comparison.to_csv(comparison_path, index=False)
    rows.to_csv(
        rows_path,
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    metadata = {
        "schema_version": 1,
        "reference_keys": list(REFERENCE_KEYS),
        "validation_keys": [value["key"] for value in VALIDATION_MODELS.values()],
        "new_keys": [value["key"] for value in NEW_MODELS.values()],
        "sources": {
            "prior_comparison": {
                "path": provenance_path(arguments.prior_comparison),
                "sha256": sha256(arguments.prior_comparison),
            },
            "prior_rows": {
                "path": provenance_path(arguments.prior_rows),
                "sha256": sha256(arguments.prior_rows),
            },
            "subset_aggregate": {
                "path": provenance_path(arguments.subset_aggregate),
                "sha256": sha256(arguments.subset_aggregate),
            },
            "precision": {
                "path": provenance_path(arguments.precision),
                "sha256": sha256(arguments.precision),
            },
            "subset_manifest": {
                "path": provenance_path(arguments.subset_manifest),
                "sha256": sha256(arguments.subset_manifest),
            },
        },
        "outputs": {
            "comparison": {
                "path": provenance_path(comparison_path),
                "sha256": sha256(comparison_path),
            },
            "all_r_rows": {
                "path": provenance_path(rows_path),
                "sha256": sha256(rows_path),
            },
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {comparison_path}")
    print(f"wrote {rows_path}")
    print(f"wrote {metadata_path}")


def parse_args() -> argparse.Namespace:
    """Parse data-source arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-comparison", type=Path, required=True)
    parser.add_argument("--prior-rows", type=Path, required=True)
    parser.add_argument("--subset-aggregate", type=Path, required=True)
    parser.add_argument("--precision", type=Path, required=True)
    parser.add_argument("--subset-manifest", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, default=HERE / "data")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
