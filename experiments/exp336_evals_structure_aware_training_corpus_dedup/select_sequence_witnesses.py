# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Select direct sequence witnesses inside the 50%-identity Linclust neighborhoods.

The candidate neighborhoods are intentionally frozen across thresholds so the
>=50, 70, 90, 95 and 100 percent results are comparable. For each threshold,
the Linclust center is considered first, followed by numeric database key. A
row is removed only when it directly aligns to an already-retained row.
"""

import argparse
import json
import subprocess
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TextIO

import pandas as pd

DOCUMENTS = 69_516_181
ESM_DOCUMENTS = 65_553_178
THRESHOLDS = (0.5, 0.7, 0.9, 0.95, 1.0)


def pair_key(left: int, right: int) -> int:
    """Pack an unordered pair of uint32 MMseqs keys into one integer."""
    if left > right:
        left, right = right, left
    return (left << 32) | right


def source(key: int) -> str:
    """Map a combined-database key to its source arm."""
    return "esm_atlas" if key < ESM_DOCUMENTS else "afdb"


def load_identities(path: Path) -> dict[int, float]:
    """Load numeric MMseqs alignment output as pair -> sequence identity."""
    identities: dict[int, float] = {}
    with path.open() as input_file:
        for line_number, line in enumerate(input_file, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 4:
                raise ValueError(f"malformed alignment row {line_number:,}: {line!r}")
            left, right = int(fields[0]), int(fields[1])
            key = pair_key(left, right)
            if key in identities:
                raise ValueError(f"duplicate direct pair at alignment row {line_number:,}")
            identities[key] = float(fields[3])
    return identities


def select_star(
    representative: int,
    members: Iterable[int],
    identities: dict[int, float],
    threshold: float,
) -> tuple[list[int], list[tuple[int, int]]]:
    """Greedily select a star with a retained direct witness for every removal."""
    unique_members = set(members)
    if representative not in unique_members:
        raise ValueError(f"cluster {representative} does not contain its representative")
    order = [representative, *sorted(unique_members - {representative})]
    kept: list[int] = []
    removals: list[tuple[int, int]] = []
    for member in order:
        witnesses = [
            witness
            for witness in kept
            if identities.get(pair_key(member, witness), -1.0) >= threshold
        ]
        if witnesses:
            witness = max(
                witnesses,
                key=lambda candidate: (
                    identities[pair_key(member, candidate)],
                    -candidate,
                ),
            )
            removals.append((member, witness))
        else:
            kept.append(member)
    return kept, removals


def select_all(
    memberships_path: Path,
    identities: dict[int, float],
    thresholds: Sequence[float],
    removal_outputs: dict[float, TextIO] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    """Stream all candidate stars and aggregate direct-witness selections."""
    removal_counts = {threshold: 0 for threshold in thresholds}
    relation_counts: dict[tuple[float, str, str], int] = {}
    membership_rows = 0
    sequence_stars = 0

    def process(representative: int | None, members: list[int]) -> None:
        nonlocal sequence_stars
        if representative is None:
            return
        sequence_stars += 1
        if len(members) == 1:
            if members[0] != representative:
                raise ValueError(
                    f"singleton cluster {representative} contains member {members[0]}"
                )
            return
        for threshold in thresholds:
            _, removals = select_star(representative, members, identities, threshold)
            removal_counts[threshold] += len(removals)
            output = removal_outputs.get(threshold) if removal_outputs else None
            for member, witness in removals:
                relation = (threshold, source(witness), source(member))
                relation_counts[relation] = relation_counts.get(relation, 0) + 1
                if output is not None:
                    output.write(f"{member}\t{witness}\t{representative}\n")

    active_representative: int | None = None
    members: list[int] = []
    with memberships_path.open() as input_file:
        for line_number, line in enumerate(input_file, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 2:
                raise ValueError(f"malformed membership row {line_number:,}: {line!r}")
            representative, member = map(int, fields)
            membership_rows += 1
            if active_representative is not None and representative != active_representative:
                process(active_representative, members)
                members = []
            active_representative = representative
            members.append(member)
    process(active_representative, members)

    summary = pd.DataFrame(
        [
            {
                "min_sequence_identity": threshold,
                "documents_before": DOCUMENTS,
                "documents_removed": removal_counts[threshold],
                "documents_retained": DOCUMENTS - removal_counts[threshold],
                "removal_fraction": removal_counts[threshold] / DOCUMENTS,
            }
            for threshold in thresholds
        ]
    )
    relationships = pd.DataFrame(
        [
            {
                "min_sequence_identity": threshold,
                "witness_source": witness_source,
                "removed_source": removed_source,
                "documents_removed": count,
            }
            for (threshold, witness_source, removed_source), count in sorted(
                relation_counts.items()
            )
        ]
    )
    return summary, relationships, membership_rows, sequence_stars


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mmseqs",
        default="/home/bizon/.cache/marinfold/mmseqs/mmseqs/bin/mmseqs",
    )
    parser.add_argument("--work", type=Path, default=Path("/data/exp336_dedup"))
    parser.add_argument(
        "--summary", type=Path, default=Path("data/direct_sequence_thresholds.csv")
    )
    parser.add_argument(
        "--relationships",
        type=Path,
        default=Path("data/direct_sequence_threshold_relationships.csv"),
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=Path("data/direct_sequence_thresholds.provenance.json"),
    )
    args = parser.parse_args()

    tag = "id050_cov080_e1000p0"
    alignments_db = args.work / f"current_linclust_{tag}_direct_align_db"
    numeric_alignments = args.work / f"current_linclust_{tag}_direct_align_numeric.tsv"
    numeric_memberships = args.work / f"current_linclust_{tag}_numeric.tsv"
    removals_path = args.work / f"current_linclust_{tag}_selected_removals_numeric.tsv"
    if not numeric_alignments.exists():
        subprocess.run(
            [
                args.mmseqs,
                "prefixid",
                str(alignments_db),
                str(numeric_alignments),
                "--tsv",
                "1",
                "--threads",
                "64",
            ],
            check=True,
        )

    started = time.perf_counter()
    identities = load_identities(numeric_alignments)
    identity_load_seconds = time.perf_counter() - started
    with removals_path.open("w") as removals_file:
        started = time.perf_counter()
        summary, relationships, membership_rows, sequence_stars = select_all(
            numeric_memberships,
            identities,
            THRESHOLDS,
            removal_outputs={0.5: removals_file},
        )
        selection_seconds = time.perf_counter() - started
    if membership_rows != DOCUMENTS:
        raise ValueError(f"membership census {membership_rows:,} != {DOCUMENTS:,}")

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary, index=False)
    relationships.to_csv(args.relationships, index=False)
    args.provenance.write_text(
        json.dumps(
            {
                "status": "complete_sequence_only_candidate_limited",
                "documents": DOCUMENTS,
                "sequence_stars": sequence_stars,
                "direct_alignment_pairs": len(identities),
                "candidate_floor_identity": 0.5,
                "min_bidirectional_coverage": 0.8,
                "selection_order": "Linclust center, then ascending MMseqs database key",
                "numeric_memberships": str(numeric_memberships),
                "numeric_alignments": str(numeric_alignments),
                "numeric_removals_at_50": str(removals_path),
                "timings": {
                    "identity_load_seconds": identity_load_seconds,
                    "selection_seconds": selection_seconds,
                },
                "warning": (
                    "Results are direct-witness selections within the 50%-identity Linclust "
                    "candidate neighborhoods, not an exhaustive all-pairs graph. They are "
                    "sequence-only ceilings until structure/contact evidence is applied."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(summary.to_string(index=False))
    print(relationships.to_string(index=False))


if __name__ == "__main__":
    main()
