"""Screen curation candidates against exp225's full sequence-exclusion reference.

Uses the published identity/shorter-coverage rule (0.30 / 0.50), with the search
reporting ceiling recorded separately. This small-pool screen is not a final
production decontamination certificate: E-values depend on target database size.
"""

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from time import perf_counter

from structure_audit import load_protein, read_csv, write_csv


def read_fasta(path: Path) -> list[tuple[str, str]]:
    """Read a reference FASTA, preserving its identifiers and full sequences."""
    records = []
    name, parts = None, []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if name is not None:
                records.append((name, "".join(parts)))
            name, parts = line[1:], []
        elif line.strip():
            if name is None:
                raise ValueError("FASTA sequence before header")
            parts.append(line.strip())
    if name is not None:
        records.append((name, "".join(parts)))
    if not records or any(not sequence for _, sequence in records):
        raise ValueError("Empty reference sequence")
    return records


def main() -> None:
    """Run an explicit sequence screen and preserve hits, controls and provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, action="append", required=True)
    parser.add_argument(
        "--source",
        nargs=3,
        action="append",
        metavar=("NAME", "SAMPLE_DIR", "CACHE"),
        required=True,
    )
    parser.add_argument("--mmseqs", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    references = {}
    for path in args.reference:
        for name, sequence in read_fasta(path):
            references.setdefault(sequence, []).append(name)
    reference_rows = [
        {
            "query": f"eval{i:05d}",
            "sequence": sequence,
            "reference_ids": ";".join(names),
        }
        for i, (sequence, names) in enumerate(references.items())
    ]
    reference_file = args.work / "references.fasta"
    reference_file.write_text(
        "".join(f">{r['query']}\n{r['sequence']}\n" for r in reference_rows)
    )
    write_csv(
        args.output / "reference_map.csv",
        [{k: v for k, v in r.items() if k != "sequence"} for r in reference_rows],
    )
    candidates = []
    for name, directory, cache in args.source:
        for row in read_csv(Path(directory) / "sample.csv"):
            if row["is_anchor"].lower() == "true":
                continue
            protein = load_protein(Path(cache), row["entry_id"])
            candidates.append(
                {
                    "target": name + "|" + row["entry_id"],
                    "source": name,
                    "entry_id": row["entry_id"],
                    "sequence": protein.sequence,
                }
            )
    if len({r["target"] for r in candidates}) != len(candidates):
        raise ValueError("Duplicate curation target")
    target_file = args.work / "candidates.fasta"
    control = "positive_control"
    control_reference = next(
        r
        for r in reference_rows
        if len(r["sequence"]) >= 60
        and not set(r["sequence"]) - set("ACDEFGHIKLMNPQRSTVWY")
    )
    target_file.write_text(
        "".join(f">{r['target']}\n{r['sequence']}\n" for r in candidates)
        + f">{control}\n{control_reference['sequence']}\n"
    )
    hits_path = args.output / "sequence_hits.tsv"
    command = [
        str(args.mmseqs),
        "easy-search",
        str(reference_file),
        str(target_file),
        str(hits_path),
        str(args.work / "tmp"),
        "-s",
        "7.5",
        "-e",
        "1000",
        "--max-seqs",
        "100000",
        "--threads",
        "8",
        "--format-output",
        "query,target,fident,alnlen,qcov,tcov,evalue,bits",
    ]
    start = perf_counter()
    with (args.work / "mmseqs.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    elapsed = perf_counter() - start
    fields = ["query", "target", "fident", "alnlen", "qcov", "tcov", "evalue", "bits"]
    with hits_path.open() as handle:
        hits = list(csv.DictReader(handle, fieldnames=fields, delimiter="\t"))
    qualifying = [
        r
        for r in hits
        if float(r["fident"]) >= 0.3 and max(float(r["qcov"]), float(r["tcov"])) >= 0.5
    ]
    if not any(
        r["target"] == control
        and float(r["fident"]) >= 0.99
        and float(r["tcov"]) >= 0.99
        for r in qualifying
    ):
        raise ValueError("Exact-reference positive control was not excluded")
    best = {}
    for hit in qualifying:
        if hit["target"] not in best or float(hit["bits"]) > float(
            best[hit["target"]]["bits"]
        ):
            best[hit["target"]] = hit
    rows = []
    for candidate in candidates:
        hit = best.get(candidate["target"])
        rows.append(
            {
                "source": candidate["source"],
                "entry_id": candidate["entry_id"],
                "excluded": hit is not None,
                "nearest_reference": hit["query"] if hit else "",
                "identity": hit["fident"] if hit else "",
                "shorter_coverage": max(float(hit["qcov"]), float(hit["tcov"]))
                if hit
                else "",
                "evalue": hit["evalue"] if hit else "",
            }
        )
    write_csv(args.output / "candidate_sequence_screen.csv", rows)
    version = subprocess.check_output([str(args.mmseqs), "version"], text=True).strip()
    summary = {
        "candidates": len(rows),
        "unique_reference_sequences": len(references),
        "excluded": sum(r["excluded"] for r in rows),
        "positive_control_passed": True,
        "positive_control_reference": control_reference["query"],
        "elapsed_seconds": elapsed,
        "mmseqs_version": version,
        "command": command,
        "references": [
            {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in args.reference
        ],
        "candidate_fasta_sha256": hashlib.sha256(target_file.read_bytes()).hexdigest(),
        "rule": "identity >=0.30 and coverage of shorter sequence >=0.50; no additional E-value exclusion arm",
        "reporting_evalue_ceiling": 1000,
        "status": "preliminary curation screen; repeat against final candidate pool and frozen reference before training",
    }
    (args.output / "sequence_screen.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
