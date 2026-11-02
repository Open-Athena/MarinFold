# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the monomer validation reference the multimer val split is defined against.

The contacts-v1 corpus (exp53) splits AFDB into train/val/test on afdb-24M's
``split`` column, which is cluster-consistent: entries sharing a structural
cluster land in the same split. The published val shards live in the HF bucket at
``data/document_structures/contacts_v1/val/`` and carry, per entry,
``uniprot_accession``, ``seq_cluster_id`` and ``struct_cluster_id``.

A dimer belongs in the multimer validation set if either subunit is close to a
protein the monomer models were validated on, so this script produces the target
side of that comparison: the distinct val accessions, their cluster IDs, and
their amino-acid sequences as a FASTA to search against.

Sequences come from UniProt rather than from the parquet. The ``document`` column
does encode residues, but as contacts-v1 tokens over a possibly-cropped window
(``start_index`` / ``truncated``), so reconstructing from it would give crops, not
the sequence the dimer subunit should be compared against. Retrieval, including
the UniParc fallback that most of these accessions need, lives in
``uniprot_sequences.py``.

Note what this does NOT do: it does not touch the train or test splits. A dimer
subunit matching a *train* monomer is fine and expected; only val matters here.

Usage::

    python3 curate_val_reference.py --out data/curation
"""

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from uniprot_sequences import fetch_sequences, write_fasta

BUCKET = "hf://buckets/open-athena/MarinFold/data/document_structures/contacts_v1/val/"
N_SHARDS = 22


def shard_uris() -> list[str]:
    return [f"{BUCKET}contacts_v1-{i:05d}-of-{N_SHARDS:05d}.parquet" for i in range(N_SHARDS)]


def read_val_entries(local_dir: Path) -> list[dict[str, str]]:
    """Read every val shard and return one row per distinct UniProt accession.

    The corpus emits crops (``round`` 0-4) rather than whole structures, but each
    accession appears exactly once across the 22 shards, so this collapses
    nothing in practice; the dict guards against that changing.
    """
    seen: dict[str, dict[str, str]] = {}
    n_rows = 0
    for path in sorted(local_dir.glob("contacts_v1-*.parquet")):
        table = pq.read_table(
            path,
            columns=["entry_id", "uniprot_accession", "seq_cluster_id", "struct_cluster_id",
                     "seq_len", "split", "round"],
        ).to_pylist()
        n_rows += len(table)
        for row in table:
            if row["split"] != "val":
                raise RuntimeError(f"{path.name}: split={row['split']!r} in a val shard")
            seen.setdefault(row["uniprot_accession"], row)
    print(f"  {n_rows:,} rows across {N_SHARDS} shards -> {len(seen):,} distinct accessions")
    return list(seen.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/curation"))
    parser.add_argument("--shards", type=Path, default=Path("sample/val_shards"))
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.shards.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] syncing {N_SHARDS} val shards into {args.shards}")
    for uri in shard_uris():
        target = args.shards / uri.rsplit("/", 1)[1]
        if target.exists() and target.stat().st_size > 0:
            continue
        subprocess.run(["hf", "buckets", "cp", uri, str(target)], check=True,
                       capture_output=True, text=True)
    print(f"  {len(list(args.shards.glob('*.parquet')))} shards present")

    print("[2/3] reading val entries")
    entries = read_val_entries(args.shards)
    accessions = sorted(e["uniprot_accession"] for e in entries)

    print(f"[3/3] fetching {len(accessions):,} sequences")
    sequences, counts = fetch_sequences(accessions, threads=args.threads)
    missing = [a for a in accessions if a not in sequences]

    fasta = args.out / "val_monomers.fasta"
    written = write_fasta(fasta, sequences, order=accessions)

    table = args.out / "val_monomers.csv"
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["uniprot_accession", "entry_id", "seq_cluster_id",
                        "struct_cluster_id", "seq_len", "uniprot_seq_len"],
        )
        writer.writeheader()
        for entry in sorted(entries, key=lambda e: e["uniprot_accession"]):
            writer.writerow({
                "uniprot_accession": entry["uniprot_accession"],
                "entry_id": entry["entry_id"],
                "seq_cluster_id": entry["seq_cluster_id"],
                "struct_cluster_id": entry["struct_cluster_id"],
                "seq_len": entry["seq_len"],
                "uniprot_seq_len": len(sequences.get(entry["uniprot_accession"], "")),
            })

    summary: dict[str, Any] = {
        "n_val_accessions": len(accessions),
        **counts,
        "unresolved_examples": missing[:20],
        "n_distinct_seq_clusters": len({e["seq_cluster_id"] for e in entries}),
        "n_distinct_struct_clusters": len({e["struct_cluster_id"] for e in entries}),
        "fasta": str(fasta),
        "fasta_records": written,
        "fasta_bytes": fasta.stat().st_size,
    }
    (args.out / "val_reference_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
