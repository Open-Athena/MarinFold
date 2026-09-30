"""Audit the five low-depth alignments without claiming ESMC training membership.

Run once with --snapshot to archive local A3Ms and UniSave accession histories.
Subsequent runs use only these small saved inputs. This is an alignment audit,
not a new homology search or a search of the ESMC pretraining corpus.
"""

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUTS = ROOT / "data/inputs/low_depth_training_audit"
WANTED = {"8ii8_A", "8oxk_A", "8qoh_A", "8ux2_A", "8wrx_A"}


def records(path: Path) -> list[tuple[str, str]]:
    """Read wrapped A3M records, preserving insertion residues and headers."""
    result = []
    header, sequence = None, ""
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        if line.startswith(">"):
            if header is not None:
                result.append((header, sequence))
            header, sequence = line[1:], ""
        else:
            if header is None:
                raise ValueError(f"Sequence before header: {path}")
            sequence += line
    if header is not None:
        result.append((header, sequence))
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a nonempty, consistently ordered audit table."""
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def snapshot() -> None:
    """Freeze the already-used alignments; fetch metadata only, not a sequence DB."""
    INPUTS.mkdir(parents=True, exist_ok=True)
    cache = Path.home() / ".cache/helico/data/benchmarks/FoldBench/foldbench-msas"
    source = Path("/home/bizon/git/helico/experiments/exp14_foldbench_held_out_monomers/data/targets.csv")
    with source.open() as stream:
        targets = [row for row in csv.DictReader(stream) if row["stem"] in WANTED]
    if {row["stem"] for row in targets} != WANTED or len(targets) != 5:
        raise ValueError("Incomplete low-depth target selection")
    for target in targets:
        digest = hashlib.sha256((target["input_seq"] + "\n").encode()).hexdigest()
        with gzip.open(cache / f"{digest}.a3m.gz", "rt") as stream:
            (INPUTS / f"{target['stem']}.a3m").write_text(stream.read())
    write_csv(INPUTS / "queries.csv", [{"stem": row["stem"], "sequence": row["input_seq"]} for row in targets])


def main() -> None:
    """Reproduce depths, inspect existing hits, and date their public accessions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", action="store_true")
    args = parser.parse_args()
    if args.snapshot:
        snapshot()
    with (INPUTS / "queries.csv").open() as stream:
        queries = {row["stem"]: row["sequence"] for row in csv.DictReader(stream)}
    with (ROOT / "data/targets.csv").open() as stream:
        targets = [row for row in csv.DictReader(stream) if row["designed"] == "0" and int(row["msa_depth"]) < 10]
    if set(queries) != WANTED or {row["stem"] for row in targets} != WANTED:
        raise ValueError("Low-depth cohort changed")
    summaries, hits = [], []
    for target in targets:
        stem, query = target["stem"], queries[target["stem"]]
        path = INPUTS / f"{stem}.a3m"
        alignment = records(path)
        if len(alignment) != int(target["msa_depth"]):
            raise ValueError(f"{stem}: archived depth mismatch")
        query_rows, protein_hits = 0, []
        for index, (header, sequence) in enumerate(alignment):
            aligned = "".join(aa for aa in sequence if not aa.islower())
            if len(aligned) != len(query):
                raise ValueError(f"{stem}: query-coordinate mismatch")
            fields = header.split()
            if fields[0] == "101":
                if aligned != query:
                    raise ValueError(f"{stem}: query record differs from input")
                query_rows += 1
                continue
            if len(fields) != 10:
                raise ValueError(f"{stem}: unexpected MMseqs2 hit metadata: {header}")
            covered = sum(aa != "-" for aa in aligned)
            hit = {
                "stem": stem, "accession": fields[0],
                "header_identity": float(fields[2]), "evalue": float(fields[3]),
                "query_coverage": covered / len(query),
                "identity_over_covered_query_positions": sum(a == b for a, b in zip(aligned, query, strict=True)) / covered,
                "source": str(path.relative_to(ROOT)), "source_record": index,
            }
            hits.append(hit)
            protein_hits.append(hit)
        best = min(protein_hits, key=lambda row: row["evalue"])
        accession = best["accession"].removeprefix("UniRef100_")
        history_path = INPUTS / f"{accession}_history.json"
        url = f"https://rest.uniprot.org/unisave/{accession}?format=json"
        if args.snapshot:
            with urllib.request.urlopen(url, timeout=30) as response:
                history_path.write_bytes(response.read())
        history = json.loads(history_path.read_text())["results"]
        first = min(history, key=lambda row: row["entryVersion"])
        summaries.append({
            "stem": stem, "archived_msa_rows": len(alignment), "query_rows": query_rows,
            "nonquery_hit_rows": len(protein_hits), "best_hit": best["accession"],
            "best_hit_header_identity": best["header_identity"], "best_hit_query_coverage": best["query_coverage"],
            "hit_first_public_date": dt.datetime.strptime(first["firstReleaseDate"], "%d-%b-%Y").date().isoformat(),
            "history_sequence_versions": "|".join(map(str, sorted({row['sequenceVersion'] for row in history}))),
            "msa_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "history_source": str(history_path.relative_to(ROOT)), "history_url": url,
            "esmc_training_membership": "not_verified", "esmc_training_homolog_count": "not_measured",
        })
    write_csv(ROOT / "data/low_depth_msa_hits.csv", hits)
    write_csv(ROOT / "data/low_depth_training_audit.csv", summaries)
    (ROOT / "data/low_depth_training_audit_manifest.json").write_text(json.dumps({
        "checked_date": "2026-09-29",
        "analysis": "Existing MSA hits and public accession histories; no new homology search",
        "inputs": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in sorted(INPUTS.iterdir()) if path.is_file()},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "esmc_sources": {
            "paper": "https://www.biorxiv.org/content/10.64898/2026.06.03.729735",
            "model_card": "https://huggingface.co/biohub/ESMC-6B#training-data",
            "uniref_release": "2023_02", "mgnify_release": "2023_02", "jgi_download": "2023-07",
        },
        "limitation": "No exact ESMC processed training set or membership manifest was available for this audit.",
    }, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
