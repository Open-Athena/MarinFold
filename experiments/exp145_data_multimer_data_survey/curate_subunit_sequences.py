# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Fetch amino-acid sequences for every subunit of the confident AFDB dimers.

Input is the output of ``curate_enumerate.py``. A homodimer contributes one
accession, a heterodimer two, giving 1,979,543 distinct subunits across the
2,010,800 confident complexes. The metadata CSVs carry accessions but no
sequences, and the structures are ~230 GB, so pulling sequences from UniProt is
several orders of magnitude cheaper than extracting them from mmCIF.

Written as **shards**, not one file, for two reasons: the job is long enough that
losing it to a transient failure matters, and sharding is what lets it run as a
fan-out. A shard whose FASTA already exists is skipped, so re-running resumes;
``--shard-index`` runs exactly one shard, which is the unit a remote worker takes.

Accessions are assigned to shards by ``index % n_shards`` over the sorted list
rather than in contiguous blocks. UniProt accession space is roughly ordered by
submission, so contiguous blocks would concentrate the deleted-from-UniProtKB
entries (which need the slower UniParc round trip) into a few shards and those
would set the wall clock.

Usage::

    python3 curate_subunit_sequences.py --out data/curation --n-shards 32
    python3 curate_subunit_sequences.py --out data/curation --n-shards 32 --shard-index 7
"""

import argparse
import csv
import gzip
import json
from pathlib import Path

from uniprot_sequences import fetch_sequences, write_fasta

INPUTS = ("confident_homodimers.csv.gz", "confident_heterodimers.csv.gz")


def read_subunit_accessions(curation_dir: Path) -> list[str]:
    """Collect the distinct subunit accessions from both confident-dimer lists.

    ``uniprot_2`` equals ``uniprot_1`` for homodimers by construction, so the set
    collapses them without a special case.
    """
    accessions: set[str] = set()
    for name in INPUTS:
        path = curation_dir / name
        if not path.exists():
            raise SystemExit(f"missing {path}; run curate_enumerate.py first")
        with gzip.open(path, "rt") as handle:
            for row in csv.DictReader(handle):
                for key in ("uniprot_1", "uniprot_2"):
                    value = row[key].strip()
                    if value:
                        accessions.add(value)
    return sorted(accessions)


def run_shard(accessions: list[str], index: int, out_dir: Path, threads: int) -> dict:
    fasta = out_dir / f"subunits-{index:05d}.fasta"
    stats_path = out_dir / f"subunits-{index:05d}.json"
    if fasta.exists() and stats_path.exists():
        print(f"[shard {index}] already done, skipping")
        return json.loads(stats_path.read_text())

    print(f"[shard {index}] {len(accessions):,} accessions")
    sequences, counts = fetch_sequences(accessions, threads=threads)
    written = write_fasta(fasta, sequences, order=accessions)
    unresolved = [a for a in accessions if a not in sequences]
    stats = {
        "shard": index,
        **counts,
        "written": written,
        "fasta": str(fasta),
        "fasta_bytes": fasta.stat().st_size,
        "unresolved_examples": unresolved[:20],
    }
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"[shard {index}] {json.dumps(counts)}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/curation"))
    parser.add_argument("--shard-dir", type=Path, default=None,
                        help="where shard FASTAs land (default <out>/subunit_shards)")
    parser.add_argument("--n-shards", type=int, default=32)
    parser.add_argument("--shard-index", type=int, default=None,
                        help="run only this shard; omit to run all in sequence")
    parser.add_argument("--threads", type=int, default=8, help="concurrent REST requests")
    args = parser.parse_args()

    shard_dir = args.shard_dir or (args.out / "subunit_shards")
    shard_dir.mkdir(parents=True, exist_ok=True)

    accessions = read_subunit_accessions(args.out)
    print(f"{len(accessions):,} distinct subunit accessions -> {args.n_shards} shards")

    indices = [args.shard_index] if args.shard_index is not None else range(args.n_shards)
    all_stats = []
    for index in indices:
        if not 0 <= index < args.n_shards:
            raise SystemExit(f"--shard-index {index} out of range for {args.n_shards} shards")
        subset = accessions[index::args.n_shards]
        all_stats.append(run_shard(subset, index, shard_dir, args.threads))

    if args.shard_index is None:
        summary = {
            "n_accessions": len(accessions),
            "n_shards": args.n_shards,
            "resolved": sum(s["resolved"] for s in all_stats),
            "from_uniprotkb": sum(s["from_uniprotkb"] for s in all_stats),
            "from_uniparc": sum(s["from_uniparc"] for s in all_stats),
            "unresolved": sum(s["unresolved"] for s in all_stats),
        }
        (args.out / "subunit_sequences_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
