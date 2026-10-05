"""Select every sequence representation for complexes clean in local complex arms."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read_fasta(path: Path) -> dict[str, str]:
    """Read an unwrapped FASTA into an identifier-to-sequence mapping."""
    records: dict[str, str] = {}
    name = ""
    parts: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if name:
                records[name] = "".join(parts)
            name, parts = line[1:].split()[0], []
        else:
            parts.append(line.strip())
    if name:
        records[name] = "".join(parts)
    return records


def main() -> None:
    """Write the reduced query FASTA and selection provenance."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=HERE / "data")
    ap.add_argument("--out", type=Path, default=HERE / "data/native_queries.fasta")
    args = ap.parse_args()
    complexes = list(csv.DictReader((args.data / "per_complex.csv").open()))
    membership = list(csv.DictReader((args.data / "query_membership.csv").open()))
    by_complex: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in membership:
        by_complex[row["source"], row["target"]].add(row["query"])
    selected = [
        row
        for row in complexes
        if row["eligibility"] == "candidate"
        and row["afcdb_status"] == "no_hit"
        and row["pinder_status"] == "no_hit"
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
    provenance = {
        "selection": "eligibility=candidate and no 30%/50% hit in afcdb or pinder complex-training arms",
        "complexes": len(selected),
        "complexes_by_source": Counter(row["source"] for row in selected),
        "queries": len(query_ids),
        "source_per_complex_sha256": hashlib.file_digest(
            (args.data / "per_complex.csv").open("rb"), "sha256"
        ).hexdigest(),
        "output_sha256": hashlib.file_digest(args.out.open("rb"), "sha256").hexdigest(),
    }
    args.out.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
