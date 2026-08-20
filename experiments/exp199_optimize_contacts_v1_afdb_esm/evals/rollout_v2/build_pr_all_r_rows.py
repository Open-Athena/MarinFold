# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the compact per-protein all-range R table used by the PR figure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
COMPARISON = HERE / "data" / "pr_comparison.csv"
DEFAULT_OUTPUT = HERE / "data" / "pr_all_r_rows.csv.gz"
DEFAULT_PROVENANCE = HERE / "data" / "pr_all_r_rows.provenance.json"
PROTENIX_MEAN = 0.6031578401726864

LOCAL_SOURCES = {
    "exp75-historical": (
        REPO_ROOT
        / "experiments"
        / "exp82_evals_contacts_v1_contact_prediction"
        / "data"
        / "where_we_stand_rows.csv.gz"
    ),
    "exp146": (
        REPO_ROOT
        / "experiments"
        / "exp166_models_contacts_v1_aa_augmentation"
        / "data"
        / "historical_exp146_rprecision.csv.gz"
    ),
    "exp166": (
        REPO_ROOT
        / "experiments"
        / "exp166_models_contacts_v1_aa_augmentation"
        / "data"
        / "exp166_rows.csv.gz"
    ),
    "protenix": (
        REPO_ROOT
        / "experiments"
        / "exp89_evals_contacts_v1_model_on_eval_set"
        / "data"
        / "contact_precision_all.csv"
    ),
}
MODELS = {
    "exp75-historical": "marinfold-cv1-exp75-rollout",
    "exp75-reproduced": "marinfold-e8-reference-step35679",
    "exp146": "exp146_3b_e8_step17839",
    "exp166": "exp166_aaaug_step35679",
    "trc-p03-aug": "marinfold-trc-p03-aug-step72599",
    "trc-p03-base": "marinfold-trc-p03-base-step72599",
    "cw-p06-aug": "marinfold-cw-p06-aug-step145199",
    "cw-p06-cool": "marinfold-cw-p06-cool-step290400",
    "trc-cont": "marinfold-trc-cont-srcbase-aug100-step145199",
    "protenix": "protenix-v2",
}
S3_SOURCES = {
    "exp199": (
        "s3://marin-us-east-02a/marin/protein-structure/MarinFold/"
        "exp199_optimize_contacts_v1_afdb_esm/evals/rollout_v2/"
        "v2-20260812-06/results/contact_precision_all.csv"
    ),
    "continuation": (
        "s3://marin-us-east-02a/marin/protein-structure/MarinFold/"
        "exp199_optimize_contacts_v1_afdb_esm/evals/rollout_v2/"
        "contbase-v2-20260812-01/results/contact_precision_all.csv"
    ),
    "e8": (
        "s3://marin-us-east-02a/marin/protein-structure/MarinFold/"
        "exp199_optimize_contacts_v1_afdb_esm/evals/rollout_v2/"
        "e8ref-v2-20260812-01/results/contact_precision_all.csv"
    ),
    "cooldown": (
        "s3://marin-us-east-02a/marin/protein-structure/MarinFold/"
        "exp199_optimize_contacts_v1_afdb_esm/evals/rollout_v2/"
        "cooldown-v2-20260815-01/results/contact_precision_all.csv"
    ),
}


def sha256(path: Path) -> str:
    """Return a file's SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select(
    frame: pd.DataFrame,
    key: str,
    *,
    units: set[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Select one model's finite all-range R-precision rows."""

    rows = frame[
        (frame.model == MODELS[key]) & (frame.range == "all") & (frame.cut == "R")
    ]
    if "mode" in rows:
        rows = rows[rows["mode"] == "single_seq"]
    if "predictor" in rows:
        predictor = "structure" if key == "protenix" else "lm"
        rows = rows[rows.predictor == predictor]
    rows = rows.loc[np.isfinite(rows.precision), ["dataset", "stem", "precision"]]
    if units is not None:
        index = pd.MultiIndex.from_frame(rows[["dataset", "stem"]])
        rows = rows.loc[index.isin(units)]
    if len(rows) != 554:
        raise ValueError(f"{key} has {len(rows)} rows; expected 554")
    if rows.duplicated(["dataset", "stem"]).any():
        raise ValueError(f"{key} has duplicate evaluation units")
    return rows.assign(key=key)[["key", "dataset", "stem", "precision"]]


def run(
    *,
    exp199_results: Path,
    continuation_results: Path,
    e8_results: Path,
    cooldown_results: Path,
    output: Path,
    provenance: Path,
) -> None:
    """Build and validate the derived table."""

    source_paths = {
        **LOCAL_SOURCES,
        "exp199": exp199_results,
        "continuation": continuation_results,
        "e8": e8_results,
        "cooldown": cooldown_results,
    }
    frames = {name: pd.read_csv(path) for name, path in source_paths.items()}
    rows = {
        "exp75-historical": select(frames["exp75-historical"], "exp75-historical"),
        "exp75-reproduced": select(frames["e8"], "exp75-reproduced"),
        "exp146": select(frames["exp146"], "exp146"),
        "exp166": select(frames["exp166"], "exp166"),
        "trc-p03-aug": select(frames["exp199"], "trc-p03-aug"),
        "trc-p03-base": select(frames["exp199"], "trc-p03-base"),
        "cw-p06-aug": select(frames["exp199"], "cw-p06-aug"),
        "trc-cont": select(frames["continuation"], "trc-cont"),
        "protenix": select(frames["protenix"], "protenix"),
    }
    legacy_units = set(
        zip(
            rows["cw-p06-aug"].dataset,
            rows["cw-p06-aug"].stem,
            strict=True,
        )
    )
    rows["cw-p06-cool"] = select(
        frames["cooldown"],
        "cw-p06-cool",
        units=legacy_units,
    )

    comparison = pd.read_csv(COMPARISON).set_index("key")
    for key, values in rows.items():
        expected = PROTENIX_MEAN if key == "protenix" else comparison.loc[key, "r_all"]
        observed = float(values.precision.mean())
        if not np.isclose(observed, expected, atol=1e-14):
            raise ValueError(f"{key} mean {observed} != {expected}")

    derived = pd.concat(rows.values(), ignore_index=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    derived.to_csv(
        output,
        index=False,
        float_format="%.17g",
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )

    source_metadata = {}
    for name, path in source_paths.items():
        if name in S3_SOURCES:
            source_metadata[name] = {
                "s3_uri": S3_SOURCES[name],
                "sha256": sha256(path),
            }
        else:
            source_metadata[name] = {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": sha256(path),
            }
    metadata = {
        "schema_version": 1,
        "description": "Finite per-protein all-range R-precision rows for the PR primary figure.",
        "rows": len(derived),
        "rows_per_key": derived.groupby("key").size().to_dict(),
        "sources": source_metadata,
        "comparison_table": str(COMPARISON.relative_to(REPO_ROOT)),
        "comparison_table_sha256": sha256(COMPARISON),
        "output": str(output.relative_to(REPO_ROOT)),
        "output_sha256": sha256(output),
    }
    provenance.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {output}")
    print(f"wrote {provenance}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp199-results", type=Path, required=True)
    parser.add_argument("--continuation-results", type=Path, required=True)
    parser.add_argument("--e8-results", type=Path, required=True)
    parser.add_argument("--cooldown-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(
        exp199_results=arguments.exp199_results,
        continuation_results=arguments.continuation_results,
        e8_results=arguments.e8_results,
        cooldown_results=arguments.cooldown_results,
        output=arguments.output,
        provenance=arguments.provenance,
    )
