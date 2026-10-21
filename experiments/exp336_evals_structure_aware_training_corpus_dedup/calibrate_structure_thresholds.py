# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Calibrate TM thresholds from #292's measured sequence/structure pairs.

These galleries are deliberately curated rather than population-random, so the
result describes threshold behavior and concrete counterexamples, not corpus
prevalence. The AFDB anchor/anchor slice is the only slice where both rows were
already in the current training corpus; anchor/candidate slices describe the
prospective #292 supplement.
"""

import argparse
import hashlib
import io
import json
import subprocess
from pathlib import Path

import pandas as pd

EXP292_COMMIT = "f57387118363c6dfdf4a5c617903408562937b2e"
SOURCE_PATHS = {
    "afdb": (
        "experiments/exp292_models_structural_cluster_diversity/"
        "data/afdb-gallery-v2/pairs.csv"
    ),
    "esm_atlas": (
        "experiments/exp292_models_structural_cluster_diversity/"
        "data/esm-gallery-v1/pairs.csv"
    ),
}
SEQUENCE_THRESHOLDS = (0.90, 0.70, 0.50, 0.40, 0.30)
TM_THRESHOLDS = (0.50, 0.70, 0.80, 0.90)


def git_file(repo: Path, revision: str, path: str) -> bytes:
    """Read one pinned git blob without checking out #292's branch."""
    return subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout


def as_bool(series: pd.Series) -> pd.Series:
    """Normalize CSV bools across pandas inference versions."""
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().eq("true")


def summarize(source: str, frame: pd.DataFrame) -> list[dict]:
    """Build the two-dimensional sequence/TM calibration surface."""
    frame = frame.copy()
    frame["anchor_a"] = as_bool(frame.anchor_a)
    frame["anchor_b"] = as_bool(frame.anchor_b)
    frame["min_tm"] = frame[["tm_a", "tm_b"]].min(axis=1)
    frame["min_coverage"] = frame[["coverage_a", "coverage_b"]].min(axis=1)
    scopes = {
        "current_anchor_anchor": frame.loc[frame.anchor_a & frame.anchor_b],
        "prospective_anchor_candidate": frame.loc[frame.anchor_a ^ frame.anchor_b],
        "all_gallery_pairs": frame,
    }
    rows = []
    for scope, subset in scopes.items():
        comparable = subset.loc[subset.min_coverage >= 0.8]
        for sequence_threshold in SEQUENCE_THRESHOLDS:
            sequence_pairs = comparable.loc[
                comparable.aligned_sequence_identity >= sequence_threshold
            ]
            for tm_threshold in TM_THRESHOLDS:
                rescued = sequence_pairs.loc[sequence_pairs.min_tm < tm_threshold]
                rows.append(
                    {
                        "source": source,
                        "scope": scope,
                        "min_sequence_identity": sequence_threshold,
                        "min_bidirectional_coverage": 0.8,
                        "min_bidirectional_tm_for_redundancy": tm_threshold,
                        "sequence_candidate_pairs": len(sequence_pairs),
                        "pairs_redundant_by_joint_rule": len(sequence_pairs) - len(rescued),
                        "pairs_rescued_by_tm_rule": len(rescued),
                        "rescue_fraction": (
                            len(rescued) / len(sequence_pairs) if len(sequence_pairs) else None
                        ),
                        "median_contact_jaccard": (
                            sequence_pairs.ca_contact_jaccard.median()
                            if len(sequence_pairs)
                            else None
                        ),
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default=EXP292_COMMIT)
    parser.add_argument(
        "--output", type=Path, default=Path("data/structure_threshold_calibration.csv")
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=Path("data/structure_threshold_calibration.provenance.json"),
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    rows = []
    sources = {}
    for source, path in SOURCE_PATHS.items():
        content = git_file(repo, args.revision, path)
        frame = pd.read_csv(io.BytesIO(content))
        rows.extend(summarize(source, frame))
        sources[source] = {
            "path": path,
            "sha256": hashlib.sha256(content).hexdigest(),
            "pairs": len(frame),
        }
    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    args.provenance.write_text(
        json.dumps(
            {
                "status": "complete_calibration_not_prevalence",
                "exp292_commit": args.revision,
                "sources": sources,
                "tm_semantics": "min(tm_normalized_by_a, tm_normalized_by_b)",
                "coverage_semantics": "min(coverage_a, coverage_b) >= 0.8",
                "warning": (
                    "The source galleries are curated curation examples, not random corpus "
                    "samples. "
                    "Use these results to understand threshold behavior, not to extrapolate loss."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    requested = result.loc[
        (result.min_sequence_identity == 0.5)
        & (result.min_bidirectional_tm_for_redundancy == 0.7)
    ]
    print(requested.to_string(index=False))


if __name__ == "__main__":
    main()
