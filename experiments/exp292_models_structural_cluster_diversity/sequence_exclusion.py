"""Frozen held-out sequence exclusion shared by both production sources.

Implements exp225's production rule exactly: a candidate is excluded when it
aligns to any held-out reference at >=30% identity over >=50% of the shorter
sequence. There is no E-value arm, so the screen is independent of target
database size and a per-shard search gives the same verdict as one global
search. Every run carries an exact-sequence positive control and fails loudly
if that control is not excluded.
"""

import csv
import shutil
import subprocess
from pathlib import Path
from time import perf_counter

from structure_audit import STANDARD_AMINO_ACIDS

MMSEQS_SHA256 = "1fb6d8dfe3c83379d2d59ccda19ffc151b69b80d0a62be4691abcd8b4c19e4f2"
MMSEQS_VERSION = "d401e78c2d18a822cdb1527d7464a043f6035a15"
MIN_IDENTITY = 0.3
MIN_SHORTER_COVERAGE = 0.5
REPORTING_EVALUE_CEILING = 1000
EXCLUSION_RULE = (
    "identity >=0.30 and coverage of shorter sequence >=0.50; no E-value arm"
)
MIN_CONTROL_LENGTH = 60
HIT_FIELDS = ("query", "target", "fident", "alnlen", "qcov", "tcov", "evalue", "bits")


def read_fasta(paths: list[Path]) -> list[tuple[str, str]]:
    """Read and sequence-deduplicate one or more reference FASTAs."""
    by_sequence: dict[str, list[str]] = {}
    for path in paths:
        name = None
        sequence_parts: list[str] = []
        for line in path.read_text().splitlines():
            if line.startswith(">"):
                if name is not None:
                    by_sequence.setdefault("".join(sequence_parts), []).append(name)
                name = line[1:]
                sequence_parts = []
            elif line.strip():
                if name is None:
                    raise ValueError(f"Sequence before FASTA header in {path}")
                sequence_parts.append(line.strip())
        if name is not None:
            by_sequence.setdefault("".join(sequence_parts), []).append(name)
    if not by_sequence or "" in by_sequence:
        raise ValueError("Held-out reference contains an empty sequence")
    return [
        (f"reference-{index:05d}", sequence)
        for index, sequence in enumerate(sorted(by_sequence))
    ]


def run_sequence_screen(
    candidates: list[dict],
    references: list[tuple[str, str]],
    mmseqs: Path,
    work: Path,
    threads: int,
) -> tuple[dict[str, dict], dict]:
    """Return the strongest qualifying held-out hit for each candidate.

    The returned mapping contains only excluded candidates, keyed by
    ``entry_id``. The second element records the exact command, the frozen
    binary's version and the positive-control outcome for provenance.
    """
    work.mkdir(parents=True, exist_ok=True)
    reference_fasta = work / "references.fasta"
    candidate_fasta = work / "candidates.fasta"
    reference_fasta.write_text(
        "".join(f">{name}\n{sequence}\n" for name, sequence in references)
    )
    positive_sequence = next(
        sequence
        for _, sequence in references
        if len(sequence) >= MIN_CONTROL_LENGTH
        and not set(sequence) - STANDARD_AMINO_ACIDS
    )
    positive_id = "positive-control"
    candidate_fasta.write_text(
        "".join(f">{row['entry_id']}\n{row['sequence']}\n" for row in candidates)
        + f">{positive_id}\n{positive_sequence}\n"
    )
    hits_path = work / "hits.tsv"
    tmp = work / "mmseqs-tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    command = [
        str(mmseqs),
        "easy-search",
        str(reference_fasta),
        str(candidate_fasta),
        str(hits_path),
        str(tmp),
        "-s",
        "7.5",
        "-e",
        str(REPORTING_EVALUE_CEILING),
        "--min-seq-id",
        str(MIN_IDENTITY),
        "--max-seqs",
        "1000000",
        "--threads",
        str(threads),
        "--format-output",
        ",".join(HIT_FIELDS),
    ]
    started = perf_counter()
    with (work / "mmseqs.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    strongest: dict[str, dict] = {}
    with hits_path.open() as handle:
        for hit in csv.DictReader(handle, fieldnames=list(HIT_FIELDS), delimiter="\t"):
            identity = float(hit["fident"])
            coverage = max(float(hit["qcov"]), float(hit["tcov"]))
            if identity < MIN_IDENTITY or coverage < MIN_SHORTER_COVERAGE:
                continue
            record = {
                "nearest_reference": hit["query"],
                "identity": identity,
                "shorter_coverage": coverage,
                "evalue": float(hit["evalue"]),
                "bits": float(hit["bits"]),
            }
            current = strongest.get(hit["target"])
            if current is None or (identity, coverage, record["bits"]) > (
                current["identity"],
                current["shorter_coverage"],
                current["bits"],
            ):
                strongest[hit["target"]] = record
    positive = strongest.pop(positive_id, None)
    if (
        positive is None
        or positive["identity"] < 0.99
        or positive["shorter_coverage"] < 0.99
    ):
        raise ValueError(
            "Exact held-out positive control did not pass sequence exclusion"
        )
    version = subprocess.check_output([str(mmseqs), "version"], text=True).strip()
    if version != MMSEQS_VERSION:
        raise ValueError("Frozen MMseqs2 binary reports an unexpected version")
    return strongest, {
        "command": command,
        "elapsed_seconds": perf_counter() - started,
        "candidates": len(candidates),
        "references": len(references),
        "excluded": len(strongest),
        "positive_control_passed": True,
        "rule": EXCLUSION_RULE,
        "reporting_evalue_ceiling": REPORTING_EVALUE_CEILING,
        "mmseqs_version": version,
    }
