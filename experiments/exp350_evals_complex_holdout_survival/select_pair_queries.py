"""Select every sequence representation for pair-clean candidate complexes."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from select_survivor_queries import read_fasta

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Write the pair-clean query FASTA and stable selection provenance."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=HERE / "data")
    ap.add_argument("--out", type=Path, default=HERE / "data/pair_queries.fasta")
    args = ap.parse_args()
    complexes = list(csv.DictReader((args.data / "pair_per_complex.csv").open()))
    membership = list(csv.DictReader((args.data / "query_membership.csv").open()))
    by_complex: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in membership:
        by_complex[row["source"], row["target"]].add(row["query"])
    selected = [
        row
        for row in complexes
        if row["eligibility"] == "candidate"
        and row["complex_pair_status"] == "pair_clean"
    ]
    query_ids = set().union(
        *(by_complex[row["source"], row["target"]] for row in selected)
    )
    records = read_fasta(args.data / "queries.fasta")
    missing = query_ids - records.keys()
    if missing:
        raise ValueError(f"Missing {len(missing)} selected query records")
    with args.out.open("w") as fh:
        for query in sorted(query_ids):
            fh.write(f">{query}\n{records[query]}\n")
    selection_rows = [
        {
            key: row[key]
            for key in ["source", "target", "eligibility", "complex_pair_status"]
        }
        for row in sorted(complexes, key=lambda row: (row["source"], row["target"]))
    ]
    provenance = {
        "selection": "eligibility=candidate and no one-to-one 30%/50% chain-pair hit in one complex-training document",
        "complexes": len(selected),
        "complexes_by_source": Counter(row["source"] for row in selected),
        "queries": len(query_ids),
        "selection_columns": [
            "source",
            "target",
            "eligibility",
            "complex_pair_status",
        ],
        "selection_columns_sha256": hashlib.sha256(
            json.dumps(selection_rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "query_membership_sha256": hashlib.file_digest(
            (args.data / "query_membership.csv").open("rb"), "sha256"
        ).hexdigest(),
        "queries_fasta_sha256": hashlib.file_digest(
            (args.data / "queries.fasta").open("rb"), "sha256"
        ).hexdigest(),
        "output_sha256": hashlib.file_digest(args.out.open("rb"), "sha256").hexdigest(),
    }
    args.out.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
