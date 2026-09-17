# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Compute sequence-prefix delta-stream contact documents from analyzed shards.

Each document places the complete amino-acid sequence before its contacts::

    DOC_START AA_TOKEN... CONTACTS_BEGIN DELTA_TOKEN... STOP_TOKEN ... DOC_END

The contact section has one ``DELTA_TOKEN... STOP_TOKEN`` segment per residue,
in sequence order. Signed deltas inside each segment are sorted left-to-right.
A residue with no contacts emits only ``STOP_TOKEN``. This lets a causal LM see
the entire sequence before predicting any contacts, as contacts-v1 does.
"""

import argparse
import os
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any

from fray.types import ResourceConfig
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

DEFAULT_INPUT = (
    "s3://marin-us-east-02a/protein-structure/MarinFold/"
    "exp139_esm_atlas_contacts_v1/analyzed/analyzed-*-of-03338.parquet"
)
DEFAULT_OUTPUT = (
    "s3://marin-us-east-02a/protein-structure/MarinFold/"
    "exp299_contacts_delta_stream_v2_sequence_prefix/documents/2026.09.16.1/"
    "shard-{shard:05d}-of-{total:05d}.parquet"
)
TOTAL_INPUT_SHARDS = 3338
SHARD_RE = re.compile(r"analyzed-(\d+)-of-\d+\.parquet$")

STOP_TOKEN_ID = 0
AA_BASE_TOKEN_ID = 1
DOC_START_TOKEN_ID = 21
CONTACTS_BEGIN_TOKEN_ID = 22
DOC_END_TOKEN_ID = 23
DELTA_BASE_TOKEN_ID = 32
DEFAULT_MAX_ABS_DELTA = 1024
AA_ORDER = (
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
)
AA_TO_TOKEN = {aa: AA_BASE_TOKEN_ID + idx for idx, aa in enumerate(AA_ORDER)}

ANALYZED_COLUMNS = (
    "entry_id",
    "seq_len",
    "global_plddt",
    "num_contacts",
    "residue_resname",
    "contact_seq_i",
    "contact_seq_j",
    "contact_degree",
)


@dataclass(frozen=True)
class ContactDelta:
    """One contact incident on a residue."""

    delta: int
    degree: float


def delta_to_token(delta: int, *, max_abs_delta: int) -> int:
    """Map a nonzero signed delta to a compact token id."""
    if delta == 0:
        raise ValueError("delta=0 is a self-contact")
    abs_delta = abs(delta)
    if abs_delta > max_abs_delta:
        raise ValueError(f"abs(delta)={abs_delta} exceeds max_abs_delta={max_abs_delta}")
    if delta < 0:
        offset = abs_delta - 1
    else:
        offset = max_abs_delta + delta - 1
    return DELTA_BASE_TOKEN_ID + offset


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "as_py"):
        value = value.as_py()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def document_row_from_analyzed(
    row: Mapping[str, Any],
    *,
    source_shard: int,
    min_seq_separation: int,
    min_contact_degree: float,
    max_abs_delta: int,
) -> dict[str, Any]:
    """Convert one analyzed row to a serialized delta-stream document."""
    entry_id = str(row["entry_id"])
    residue_resname = [str(x) for x in _as_list(row["residue_resname"])]
    seq_len = int(row.get("seq_len") or len(residue_resname))
    if seq_len != len(residue_resname):
        raise ValueError(f"{entry_id}: seq_len={seq_len} but residue_resname has {len(residue_resname)} entries")

    contacts_by_residue: list[list[ContactDelta]] = [[] for _ in range(seq_len)]
    raw_i = _as_list(row["contact_seq_i"])
    raw_j = _as_list(row["contact_seq_j"])
    raw_degree = _as_list(row["contact_degree"])
    if not (len(raw_i) == len(raw_j) == len(raw_degree)):
        raise ValueError(f"{entry_id}: contact arrays have inconsistent lengths")

    contacts_considered = 0
    contacts_used_undirected = 0
    for left, right, degree_value in zip(raw_i, raw_j, raw_degree):
        i = int(left)
        j = int(right)
        degree = float(degree_value)
        if not (0 <= i < seq_len and 0 <= j < seq_len):
            raise ValueError(f"{entry_id}: contact ({i}, {j}) outside seq_len={seq_len}")
        if i == j:
            raise ValueError(f"{entry_id}: self-contact at {i}")
        contacts_considered += 1
        if abs(i - j) < min_seq_separation or degree < min_contact_degree:
            continue
        delta_to_token(j - i, max_abs_delta=max_abs_delta)
        delta_to_token(i - j, max_abs_delta=max_abs_delta)
        contacts_by_residue[i].append(ContactDelta(delta=j - i, degree=degree))
        contacts_by_residue[j].append(ContactDelta(delta=i - j, degree=degree))
        contacts_used_undirected += 1

    sequence_token_ids: list[int] = []
    contact_token_ids: list[int] = []
    contact_deltas: list[list[int]] = []
    contact_degrees: list[list[float]] = []
    contacts_per_residue: list[int] = []
    max_contacts_per_residue = 0
    for aa, residue_contacts in zip(residue_resname, contacts_by_residue):
        sorted_contacts = sorted(residue_contacts, key=lambda contact: contact.delta)
        deltas = [contact.delta for contact in sorted_contacts]
        degrees = [float(contact.degree) for contact in sorted_contacts]
        contact_deltas.append(deltas)
        contact_degrees.append(degrees)
        contacts_per_residue.append(len(deltas))
        max_contacts_per_residue = max(max_contacts_per_residue, len(deltas))
        try:
            sequence_token_ids.append(AA_TO_TOKEN[aa])
        except KeyError as exc:
            raise ValueError(f"{entry_id}: unknown residue name {aa!r}") from exc
        contact_token_ids.extend(delta_to_token(delta, max_abs_delta=max_abs_delta) for delta in deltas)
        contact_token_ids.append(STOP_TOKEN_ID)

    token_ids = [
        DOC_START_TOKEN_ID,
        *sequence_token_ids,
        CONTACTS_BEGIN_TOKEN_ID,
        *contact_token_ids,
        DOC_END_TOKEN_ID,
    ]
    return {
        "entry_id": entry_id,
        "source_shard": int(source_shard),
        "seq_len": int(seq_len),
        "global_plddt": float(row["global_plddt"]) if row.get("global_plddt") is not None else None,
        "residue_resname": residue_resname,
        "min_seq_separation": int(min_seq_separation),
        "min_contact_degree": float(min_contact_degree),
        "max_abs_delta": int(max_abs_delta),
        "stop_token_id": int(STOP_TOKEN_ID),
        "doc_start_token_id": int(DOC_START_TOKEN_ID),
        "contacts_begin_token_id": int(CONTACTS_BEGIN_TOKEN_ID),
        "doc_end_token_id": int(DOC_END_TOKEN_ID),
        "aa_base_token_id": int(AA_BASE_TOKEN_ID),
        "delta_base_token_id": int(DELTA_BASE_TOKEN_ID),
        "vocab_size": int(DELTA_BASE_TOKEN_ID + 2 * max_abs_delta),
        "raw_contacts": int(len(raw_i)),
        "contacts_considered": int(contacts_considered),
        "contacts_used_undirected": int(contacts_used_undirected),
        "contacts_used_directed": int(2 * contacts_used_undirected),
        "contacts_per_residue": contacts_per_residue,
        "max_contacts_per_residue": int(max_contacts_per_residue),
        "sequence_token_count": int(len(sequence_token_ids)),
        "contact_token_count": int(len(contact_token_ids)),
        "token_count": int(len(token_ids)),
        "token_ids": token_ids,
        "contact_deltas": contact_deltas,
        "contact_degrees": contact_degrees,
    }


def documents_shard(
    items: Iterator[Mapping[str, Any]],
    shard_info,
    *,
    min_seq_separation: int,
    min_contact_degree: float,
    max_abs_delta: int,
    max_rows_per_shard: int | None,
) -> Iterator[dict[str, Any]]:
    emitted = 0
    for row in items:
        if max_rows_per_shard is not None and emitted >= max_rows_per_shard:
            break
        yield document_row_from_analyzed(
            row,
            source_shard=int(shard_info.shard_idx),
            min_seq_separation=min_seq_separation,
            min_contact_degree=min_contact_degree,
            max_abs_delta=max_abs_delta,
        )
        emitted += 1


def _input_files(input_pattern: str, *, num_shards: int | None, start_shard: int) -> Dataset[str]:
    if num_shards is None:
        return Dataset.from_files(input_pattern)
    if "analyzed-*-of-03338.parquet" not in input_pattern:
        raise ValueError("--num-shards requires the default analyzed-* input pattern")
    return Dataset.from_list(
        [
            input_pattern.replace("analyzed-*-of-03338.parquet", f"analyzed-{index:05d}-of-03338.parquet")
            for index in range(start_shard, start_shard + num_shards)
        ]
    )


def _optional_positive_int(value: str) -> int | None:
    if value.lower() in {"none", "null", "all"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive, or one of: none, null, all")
    return parsed


def run(args: argparse.Namespace) -> None:
    input_files = _input_files(args.input, num_shards=args.num_shards, start_shard=args.start_shard)
    rows = input_files.load_parquet(columns=list(ANALYZED_COLUMNS))
    out_rows = rows.map_shard(
        partial(
            documents_shard,
            min_seq_separation=args.min_seq_separation,
            min_contact_degree=args.min_contact_degree,
            max_abs_delta=args.max_abs_delta,
            max_rows_per_shard=args.max_rows_per_shard,
        )
    )
    ds = out_rows.write_parquet(args.output)
    ctx = ZephyrContext(
        max_workers=args.max_workers,
        resources=ResourceConfig(
            cpu=args.worker_cpu,
            ram=args.worker_memory,
            disk=args.worker_disk,
            regions=[args.region],
            preemptible=args.preemptible,
        ),
    )
    ctx.execute(ds)
    print(f"[exp299] wrote delta-stream contacts documents to {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--max-rows-per-shard", type=_optional_positive_int, default=128)
    parser.add_argument("--min-seq-separation", type=int, default=6)
    parser.add_argument("--min-contact-degree", type=float, default=0.001)
    parser.add_argument("--max-abs-delta", type=int, default=DEFAULT_MAX_ABS_DELTA)
    parser.add_argument("--max-workers", type=int, default=int(os.environ.get("EXP299_DOCUMENTS_MAX_WORKERS", "4")))
    parser.add_argument("--region", default="us-central1", help="Region for Zephyr workers and source-data locality.")
    parser.add_argument("--worker-cpu", type=float, default=1.0)
    parser.add_argument("--worker-memory", default="4GB")
    parser.add_argument("--worker-disk", default="16GB")
    parser.add_argument("--preemptible", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.num_shards is not None and args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if args.max_abs_delta <= 0:
        raise ValueError("--max-abs-delta must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
