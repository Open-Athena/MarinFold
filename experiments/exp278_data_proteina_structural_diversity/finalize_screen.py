"""Materialize the provisional selected documents and verify their provenance."""

import csv
import hashlib
import io
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from marinfold.document_structures.contacts_v1.read import (
    fold_statements,
    iter_structure_statements,
    sequence_from_document,
)
from marinfold.document_structures.contacts_v1.vocab import NUM_POSITION_INDICES

from analyze_screen import write_csv
from launch import storage_filesystem

HERE = Path(__file__).resolve().parent
ROOT = "marin-us-east-02a/MarinFold/exp278-proteina"


def main() -> None:
    fs = storage_filesystem("cw-rno2a")
    report = HERE / "data/full-screen"
    with (report / "retention.csv").open() as handle:
        selected = {
            row["stem"]: row
            for row in csv.DictReader(handle)
            if row["selected"] == "True"
        }
    documents = []
    source_sizes = []
    for length in [60, 100, 200, 300, 400, 500]:
        root = f"{ROOT}/fold-l{length}-{'cis-v2' if length <= 300 else 'v1'}"
        for kind in ["candidates", "documents-provisional"]:
            paths = fs.glob(f"{root}/{kind}/*.parquet")
            source_sizes.append(
                {
                    "length": length,
                    "kind": kind,
                    "compressed_bytes": sum(fs.size(path) for path in paths),
                }
            )
            if kind != "documents-provisional":
                continue
            for path in paths:
                with fs.open(path, "rb") as handle:
                    for row in pq.read_table(handle).to_pylist():
                        stem = row["entry_id"]
                        if stem not in selected:
                            continue
                        if (
                            sequence_from_document(
                                row["document"],
                                len(row["sequence"]),
                                row["n_term_index"],
                            )
                            != row["sequence"]
                        ):
                            raise ValueError(f"Invalid sequence round-trip: {stem}")
                        contacts = fold_statements(
                            iter_structure_statements(row["document"])
                        )
                        if (
                            contacts.n_retract
                            or contacts.n_redundant_contact
                            or len(contacts.live) != row["contacts_emitted"]
                        ):
                            raise ValueError(
                                f"Contact metadata mismatch or duplicate statement: {stem}"
                            )
                        if any(
                            a == b
                            or any(
                                (position - row["n_term_index"]) % NUM_POSITION_INDICES
                                >= len(row["sequence"])
                                for position in (a, b)
                            )
                            for a, b in contacts.live
                        ):
                            raise ValueError(
                                f"Contact endpoint outside the intended monomer: {stem}"
                            )
                        if row["num_chains"] != 1 or row["seq_len"] != len(
                            row["sequence"]
                        ):
                            raise ValueError(f"Monomer metadata mismatch: {stem}")
                        documents.append(
                            {
                                **row,
                                "fine_cluster": selected[stem]["fine_cluster"],
                                "source_parquet": "s3://" + path,
                            }
                        )
    if len(documents) != len(selected) or {row["entry_id"] for row in documents} != set(
        selected
    ):
        raise ValueError(
            "Selected document count or provenance does not match retention"
        )
    documents.sort(key=lambda row: row["entry_id"])
    payload = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(documents), payload, compression="zstd")
    data = payload.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    uri = f"{ROOT}/initial-screen-selected-provisional/{digest}.parquet"
    fs.pipe_file(uri, data)
    write_csv(
        report / "selected-documents.csv",
        [
            {
                "stem": row["entry_id"],
                "length": len(row["sequence"]),
                "fine_cluster": row["fine_cluster"],
                "contacts_emitted": row["contacts_emitted"],
                "num_tokens": row["num_tokens"],
                "truncated": row["truncated"],
                "document_sha256": hashlib.sha256(row["document"].encode()).hexdigest(),
                "source_parquet": row["source_parquet"],
            }
            for row in documents
        ],
    )
    write_csv(report / "storage-by-length.csv", source_sizes)
    summary = {
        "documents": len(documents),
        "sha256": digest,
        "compressed_bytes": len(data),
        "uri": "s3://" + uri,
        "all_sequences_round_trip": True,
        "all_contact_counts_and_endpoints_validated": True,
        "total_tokens": sum(row["num_tokens"] for row in documents),
        "truncated_documents": sum(row["truncated"] for row in documents),
        "release_status": "provisional research screen; not in a training mixture",
        "selection": "global cross-length fine-cluster cap after quality and frozen evaluation screens",
    }
    (report / "selected-artifact.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
