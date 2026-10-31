"""Recover the exact chain sequences from the complex documents exp343 consumed."""

import argparse
import hashlib
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq

AA = dict(
    zip(
        [
            "ALA",
            "ARG",
            "ASN",
            "ASP",
            "CYS",
            "GLN",
            "GLU",
            "GLY",
            "HIS",
            "ILE",
            "LEU",
            "LYS",
            "MET",
            "PHE",
            "PRO",
            "SER",
            "THR",
            "TRP",
            "TYR",
            "VAL",
            "UNK",
        ],
        "ARNDCQEGHILKMFPSTWYVX",
        strict=True,
    )
)
RESIDUE = re.compile(r"<p(\d+)> <([A-Z]{3})>")
START = re.compile(r"<n-term> <p(\d+)>")
END = re.compile(r"<c-term> <p(\d+)>")


def chain_sequences(document: str, expected_lengths: list[int]) -> list[str]:
    """Decode disjoint chain runs on the 2000-position ring, checking all residues."""
    prefix = document.split("<begin_statements>", 1)[0]
    residues = {int(p): AA[a] for p, a in RESIDUE.findall(prefix)}
    starts = [int(p) for p in START.findall(prefix)]
    ends = {int(p) for p in END.findall(prefix)}
    if len(starts) != len(expected_lengths) or len(ends) != len(starts):
        raise ValueError("Missing or repeated chain termini")
    sequences = []
    used: set[int] = set()
    for start in starts:
        seq = []
        for offset in range(2000):
            p = (start + offset) % 2000
            if p in used or p not in residues:
                raise ValueError(f"Overlapping or incomplete chain at position {p}")
            used.add(p)
            seq.append(residues[p])
            if p in ends:
                break
        else:
            raise ValueError("Chain has no C terminus")
        sequences.append("".join(seq))
    if used != set(residues) or sorted(map(len, sequences)) != sorted(expected_lengths):
        raise ValueError("Decoded chains disagree with document metadata")
    return sequences


def extract_shard(job: tuple[str, str]) -> dict:
    """Write one FASTA per input shard, retaining one copy of identical chains per document."""
    source, out_dir = map(Path, job)
    output = out_dir / (source.stem + ".fasta")
    report = output.with_suffix(".json")
    if output.exists() and report.exists():
        return json.loads(report.read_text())
    n_docs = n_chains = 0
    columns = ["document_id", "source_arm", "document", "chain_lengths"]
    with output.open("w") as fh:
        for batch in pq.ParquetFile(source).iter_batches(
            batch_size=256, columns=columns
        ):
            for row in batch.to_pylist():
                sequences = chain_sequences(row["document"], row["chain_lengths"])
                for c, seq in enumerate(sorted(set(sequences))):
                    key = f"{row['source_arm']}|{source.stem}|{n_docs}|{c}|{row['document_id']}"
                    fh.write(f">{key}\n{seq}\n")
                    n_chains += 1
                n_docs += 1
    result = {
        "source": str(source),
        "source_bytes": source.stat().st_size,
        "fasta": str(output),
        "documents": n_docs,
        "unique_chains_per_document": n_chains,
        "sha256": hashlib.file_digest(output.open("rb"), "sha256").hexdigest(),
    }
    report.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    """Extract training shards 0–169; shard 170 is the validation holdout."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--corpus", type=Path, default=Path("/data/exp294_release/corpus/train")
    )
    ap.add_argument("--out", type=Path, default=Path("/data/exp350/complex_sequences"))
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sources = [args.corpus / f"shard-{i:05d}-of-00171.parquet" for i in range(170)]
    with ProcessPoolExecutor(args.workers) as pool:
        reports = []
        for result in pool.map(
            extract_shard, [(str(s), str(args.out)) for s in sources]
        ):
            reports.append(result)
            print(f"{len(reports)}/170 {result['documents']} documents", flush=True)
    (args.out / "manifest.json").write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
