# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Precompute exp177 compact soft-target examples with Zephyr.

This materializes the CPU-heavy part of ``FixedQuotaSoftTargetContactsDataset``:
read analyzed contacts-v1 rows, build block-causal documents, shard-local fixed
quota, and emit compact variable-length rows. CoreWeave GPU training can then
read these rows and only do cheap padding/JAX object construction.
"""

import argparse
import os
import re
import sys
from collections.abc import Iterator, Mapping
from functools import partial
from typing import Any

import numpy as np
from fray.types import ResourceConfig
from zephyr.dataset import Dataset, ShardInfo
from zephyr.execution import ZephyrContext

from marinfold.document_structures.contacts_v1 import ANALYZED_ROW_COLUMNS, CONTEXT_LENGTH
from marinfold.document_structures.contacts_v1.vocab import CONTACT, END
from marinfold_models.shard_documents import best_fit_pack_documents, fixed_quota_pack_slots
from premade_contacts_dataset import soft_target_contacts_v1_document_from_row

DEFAULT_INPUT = (
    "s3://marin-us-east-02a/protein-structure/MarinFold/"
    "exp139_esm_atlas_contacts_v1/analyzed/analyzed-*-of-03338.parquet"
)
SHARD_RE = re.compile(r"analyzed-(\d+)-of-\d+\.parquet$")


DEFAULT_OUTPUT = (
    "s3://marin-us-east-02a/protein-structure/MarinFold/"
    "exp177_soft_target_loss_h2h/preprocessed/soft_target_compact_v2/"
    "2026.08.15.1/shard-{shard:05d}-of-{total:05d}.parquet"
)


def _padded(values: np.ndarray, *, length: int, fill: int = 0) -> list[int]:
    padded = np.full(length, fill, dtype=np.int32)
    padded[: values.shape[0]] = values.astype(np.int32)
    return padded.tolist()


def _row_from_document(document, *, source_shard: int, slot_index: int, max_seq_len: int) -> dict[str, Any]:
    token_ids = np.asarray(document.token_ids, dtype=np.int32)
    query = np.asarray(document.query)
    query_positions = np.flatnonzero(query)
    if query_positions.size == 0:
        raise ValueError(f"source_shard={source_shard} slot={slot_index} has no query positions")
    prediction_start = int(query_positions[0])

    suffix = token_ids[prediction_start + 1 :]
    end_offsets = np.flatnonzero(suffix == int(END))
    if end_offsets.size == 0:
        raise ValueError(f"source_shard={source_shard} slot={slot_index} has no END token")
    suffix = suffix[: int(end_offsets[0])]
    if suffix.size % 3 != 0:
        raise ValueError(
            f"source_shard={source_shard} slot={slot_index} contact suffix length "
            f"{suffix.size} is not divisible by 3"
        )
    if np.any(suffix[0::3] != int(CONTACT)):
        raise ValueError(f"source_shard={source_shard} slot={slot_index} malformed contact triples")

    segment_ids = np.full(token_ids.shape[0], 0, dtype=np.int32)
    attention_blocks = np.zeros(token_ids.shape[0], dtype=np.int32)
    if prediction_start + 1 < token_ids.shape[0]:
        attention_blocks[prediction_start + 1 :] = np.arange(1, token_ids.shape[0] - prediction_start, dtype=np.int32)

    contact_first_ids = suffix[1::3].astype(np.int32)
    contact_second_ids = suffix[2::3].astype(np.int32)
    max_contacts = (max_seq_len - 2) // 3
    return {
        "source_shard": source_shard,
        "slot_index": slot_index,
        "token_ids": _padded(token_ids, length=max_seq_len),
        # Current soft-target docs do not set POSITION_IDS, so store zeros in
        # the final padded representation expected by training.
        "position_ids": [0] * max_seq_len,
        "segment_ids": _padded(segment_ids, length=max_seq_len, fill=-1),
        "attention_blocks": _padded(attention_blocks, length=max_seq_len),
        "prediction_start": prediction_start,
        "contact_first_ids": _padded(contact_first_ids, length=max_contacts),
        "contact_second_ids": _padded(contact_second_ids, length=max_contacts),
        "contact_count": int(suffix.size // 3),
        "target_position_count": int(suffix.size + 1),
    }


def preprocess_shard(
    items: Iterator[Mapping[str, Any]],
    shard_info: ShardInfo,
    *,
    examples_per_shard: int,
    max_seq_len: int,
    seed: int,
) -> Iterator[dict[str, Any]]:
    """Build fixed-quota compact examples for one analyzed shard."""
    rows = list(items)
    if not rows:
        raise ValueError(f"input shard {shard_info.shard_idx} contains no rows")

    row_rng = np.random.default_rng(np.random.SeedSequence([seed, 0, shard_info.shard_idx, 0]))
    documents = []
    for row_index in row_rng.permutation(len(rows)):
        document = soft_target_contacts_v1_document_from_row(rows[int(row_index)])
        if document is None:
            continue
        documents.append(document)
    if not documents:
        raise ValueError(f"input shard {shard_info.shard_idx} produced no documents")

    packs, _ = best_fit_pack_documents(
        documents,
        max_seq_len=max_seq_len,
        max_segments_per_example=1,
    )
    slot_rng = np.random.default_rng(np.random.SeedSequence([seed, 0, shard_info.shard_idx, 1]))
    slots = fixed_quota_pack_slots(packs, examples_per_shard=examples_per_shard, rng=slot_rng)

    real_slots = tuple(slot for slot in slots if slot is not None)
    if not real_slots:
        raise ValueError(f"input shard {shard_info.shard_idx} yielded no real slots")
    fill_rng = np.random.default_rng(np.random.SeedSequence([seed, 0, shard_info.shard_idx, 2]))

    for slot_index, slot in enumerate(slots):
        current = slot if slot is not None else real_slots[int(fill_rng.integers(len(real_slots)))]
        if len(current.documents) != 1:
            raise ValueError(
                f"soft-target preprocessing requires one document per slot; got "
                f"{len(current.documents)} in shard {shard_info.shard_idx} slot {slot_index}"
            )
        yield _row_from_document(
            current.documents[0],
            source_shard=shard_info.shard_idx,
            slot_index=slot_index,
            max_seq_len=max_seq_len,
        )


def _input_shard_index(path: str) -> int:
    match = SHARD_RE.search(path)
    if match is None:
        raise ValueError(f"Could not parse shard index from input path: {path}")
    return int(match.group(1))


def run(args: argparse.Namespace) -> None:
    input_files = Dataset.from_files(args.input)
    if args.num_shards is not None:
        input_files = input_files.filter(lambda path: _input_shard_index(path) < args.num_shards)
    rows = input_files.load_parquet(columns=list(ANALYZED_ROW_COLUMNS))
    out_rows = rows.map_shard(
        partial(
            preprocess_shard,
            examples_per_shard=args.examples_per_shard,
            max_seq_len=args.max_seq_len,
            seed=args.seed,
        )
    )
    ds = out_rows.write_parquet(args.output)
    ctx = ZephyrContext(
        max_workers=args.max_workers,
        resources=ResourceConfig(
            cpu=args.worker_cpu,
            ram=args.worker_memory,
            disk=args.worker_disk,
            preemptible=args.preemptible,
        ),
    )
    ctx.execute(ds)
    print(f"[exp177] wrote precomputed soft targets to {args.output}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--examples-per-shard", type=int, default=2650)
    parser.add_argument("--max-seq-len", type=int, default=CONTEXT_LENGTH)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=None, help="Debug cap after loading input shards.")
    parser.add_argument("--max-workers", type=int, default=int(os.environ.get("EXP177_PREPROCESS_MAX_WORKERS", "64")))
    parser.add_argument("--worker-cpu", type=float, default=1.0)
    parser.add_argument("--worker-memory", default="8GB")
    parser.add_argument("--worker-disk", default="16GB")
    parser.add_argument("--preemptible", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
