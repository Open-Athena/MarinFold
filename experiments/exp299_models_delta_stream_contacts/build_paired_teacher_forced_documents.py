"""Build contacts-v1 and delta-V2 documents for the same eval proteins."""

import argparse
import json
from typing import Any

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
from marinfold.document_structures.contacts_v1.generate import build_document
from marinfold.document_structures.contacts_v1.parse import (
    RawContact,
    residues_from_sequence,
)
from marinfold.document_structures.contacts_v1.vocab import all_domain_tokens

from compute_contacts_delta_stream_documents import (
    CONTACTS_BEGIN_TOKEN_ID,
    DOC_END_TOKEN_ID,
    DOC_START_TOKEN_ID,
    STOP_TOKEN_ID,
    delta_to_token,
)
from delta_stream_rollout import AA_TO_TOKEN_ID, UNKNOWN_AA_TOKEN_ID

MIN_SEQUENCE_SEPARATION = 6
MIN_CONTACT_DEGREE = 0.001
MAX_ABS_DELTA = 1024
CONTACTS_V1_EOS_TOKEN_ID = 1


def read_jsonl(uri: str) -> list[dict[str, Any]]:
    """Read JSON objects from a local or fsspec URI."""
    with fsspec.open(uri, "rt") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def contacts_v1_vocabulary() -> dict[str, int]:
    """Return the exp177 tokenizer mapping for domain tokens."""
    return {token: token_id for token_id, token in enumerate(all_domain_tokens(), start=2)}


def contacts_v1_document(
    dataset: str, stem: str, sequence: str, contacts: list[list[float]]
) -> list[int]:
    """Serialize one eval target with the canonical contacts-v1 generator."""
    raw_contacts = [
        RawContact(seq_i=int(left), seq_j=int(right), degree=float(degree))
        for left, right, degree in contacts
    ]
    generated = build_document(
        f"{dataset}::{stem}",
        residues_from_sequence(sequence),
        raw_contacts,
    )
    if generated is None:
        raise ValueError(f"cannot serialize contacts-v1 document for {dataset}/{stem}")
    vocabulary = contacts_v1_vocabulary()
    try:
        token_ids = [vocabulary[token] for token in generated.document.split()]
    except KeyError as error:
        raise ValueError(f"unknown contacts-v1 token for {dataset}/{stem}: {error}") from error
    token_ids.append(CONTACTS_V1_EOS_TOKEN_ID)
    return token_ids


def delta_v2_document(sequence: str, contacts: list[list[float]]) -> list[int]:
    """Serialize one eval target with the deterministic sequence-prefix V2 format."""
    contacts_by_residue: list[list[int]] = [[] for _ in sequence]
    for left_value, right_value, degree_value in contacts:
        left = int(left_value)
        right = int(right_value)
        degree = float(degree_value)
        if abs(left - right) < MIN_SEQUENCE_SEPARATION or degree < MIN_CONTACT_DEGREE:
            continue
        contacts_by_residue[left].append(right - left)
        contacts_by_residue[right].append(left - right)

    contact_token_ids: list[int] = []
    for deltas in contacts_by_residue:
        contact_token_ids.extend(
            delta_to_token(delta, max_abs_delta=MAX_ABS_DELTA) for delta in sorted(deltas)
        )
        contact_token_ids.append(STOP_TOKEN_ID)
    sequence_token_ids = [AA_TO_TOKEN_ID.get(amino_acid, UNKNOWN_AA_TOKEN_ID) for amino_acid in sequence]
    return [
        DOC_START_TOKEN_ID,
        *sequence_token_ids,
        CONTACTS_BEGIN_TOKEN_ID,
        *contact_token_ids,
        DOC_END_TOKEN_ID,
    ]


def build_paired_rows(targets_uri: str, ground_truth_uri: str) -> list[dict[str, Any]]:
    """Build aligned rows for every target and assert exact key/length coverage."""
    with fsspec.open(targets_uri, "rb") as handle:
        targets = pq.read_table(handle).to_pylist()
    truth_by_key = {
        (str(row["dataset"]), str(row["stem"])): row for row in read_jsonl(ground_truth_uri)
    }
    rows: list[dict[str, Any]] = []
    for target in targets:
        dataset = str(target["dataset"])
        stem = str(target["stem"])
        sequence = str(target["input_seq"])
        key = (dataset, stem)
        truth = truth_by_key.pop(key)
        if len(sequence) != int(target["L"]) or len(sequence) != int(truth["L"]):
            raise ValueError(f"length mismatch for {dataset}/{stem}")
        contacts = truth["contacts"]
        contacts_v1_ids = contacts_v1_document(dataset, stem, sequence, contacts)
        delta_v2_ids = delta_v2_document(sequence, contacts)
        if len(contacts_v1_ids) > 8192 or len(delta_v2_ids) > 8192:
            raise ValueError(
                f"document exceeds 8192 tokens for {dataset}/{stem}: "
                f"contacts-v1={len(contacts_v1_ids)}, delta-v2={len(delta_v2_ids)}"
            )
        rows.append(
            {
                "dataset": dataset,
                "stem": stem,
                "entry_id": f"{dataset}::{stem}",
                "L": len(sequence),
                "input_seq": sequence,
                "contacts_v1_token_ids": contacts_v1_ids,
                "delta_v2_token_ids": delta_v2_ids,
            }
        )
    if truth_by_key:
        raise ValueError(f"ground truth has {len(truth_by_key)} unmatched proteins")
    rows.sort(key=lambda row: (row["dataset"], row["stem"]))
    return rows


def write_model_documents(rows: list[dict[str, Any]], token_column: str, destination: str) -> None:
    """Write one aligned model-specific parquet with a common token_ids column."""
    output = [
        {
            "dataset": row["dataset"],
            "stem": row["stem"],
            "entry_id": row["entry_id"],
            "L": row["L"],
            "input_seq": row["input_seq"],
            "token_ids": row[token_column],
        }
        for row in rows
    ]
    with fsspec.open(destination, "wb") as handle:
        pq.write_table(pa.Table.from_pylist(output), handle, compression="zstd")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--out", required=True, help="Output directory or URI")
    args = parser.parse_args()

    rows = build_paired_rows(args.targets, args.ground_truth)
    output_root = args.out.rstrip("/")
    write_model_documents(rows, "contacts_v1_token_ids", f"{output_root}/contacts_v1.parquet")
    write_model_documents(rows, "delta_v2_token_ids", f"{output_root}/delta_v2.parquet")
    lengths = {
        "contacts-v1": sum(len(row["contacts_v1_token_ids"]) for row in rows),
        "delta-v2": sum(len(row["delta_v2_token_ids"]) for row in rows),
    }
    print(f"wrote {len(rows)} aligned proteins to {output_root}: {lengths}")


if __name__ == "__main__":
    main()
