# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Compute exp177 sparse soft-target shape statistics with Zephyr.

This is a small, non-training preprocessing pass used to choose bucket shapes for
an exact sparse implementation of the contacts-v1 soft loss. It reads the
CoreWeave S3 exp139 analyzed shards, rebuilds the same exp177 soft-target
fixed-quota slots, and writes one compact statistics row per slot. It does not
copy token arrays or materialize a new training corpus.
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
from marinfold.document_structures.contacts_v1.vocab import CONTACT, END, NUM_POSITION_INDICES, POSITIONS
from marinfold_models.shard_documents import best_fit_pack_documents, fixed_quota_pack_slots
from premade_contacts_dataset import soft_target_contacts_v1_document_from_row

DEFAULT_INPUT = (
    "s3://marin-us-east-02a/protein-structure/MarinFold/"
    "exp139_esm_atlas_contacts_v1/analyzed/analyzed-*-of-03338.parquet"
)
SHARD_RE = re.compile(r"analyzed-(\d+)-of-\d+\.parquet$")

DEFAULT_OUTPUT = (
    "s3://marin-us-east-02a/protein-structure/MarinFold/"
    "exp177_soft_target_loss_h2h/stats/sparse_target_shapes_v1/"
    "2026.08.19.1/shard-{shard:05d}-of-{total:05d}.parquet"
)

POSITION_TOKEN_START = int(POSITIONS[0])
POSITION_TOKEN_STOP = POSITION_TOKEN_START + NUM_POSITION_INDICES


def _contact_suffix(token_ids: np.ndarray, *, prediction_start: int, source_shard: int, slot_index: int) -> np.ndarray:
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
    return suffix


def _quantile_int(values: np.ndarray, q: float) -> int:
    if values.size == 0:
        return 0
    return int(np.quantile(values, q, method="higher"))


def _row_from_document(
    document,
    *,
    source_shard: int,
    slot_index: int,
    filled_from_real_slot: bool,
    max_seq_len: int,
) -> dict[str, Any]:
    token_ids = np.asarray(document.token_ids, dtype=np.int32)
    query = np.asarray(document.query)
    query_positions = np.flatnonzero(query)
    if query_positions.size == 0:
        raise ValueError(f"source_shard={source_shard} slot={slot_index} has no query positions")
    prediction_start = int(query_positions[0])
    residue_count = prediction_start - 2
    if residue_count < 0:
        raise ValueError(f"source_shard={source_shard} slot={slot_index} invalid prediction_start={prediction_start}")

    suffix = _contact_suffix(
        token_ids,
        prediction_start=prediction_start,
        source_shard=source_shard,
        slot_index=slot_index,
    )
    first_ids = suffix[1::3].astype(np.int32)
    second_ids = suffix[2::3].astype(np.int32)
    contact_count = int(first_ids.shape[0])
    endpoints = np.concatenate([first_ids, second_ids]) if contact_count else np.asarray([], dtype=np.int32)
    if np.any((endpoints < POSITION_TOKEN_START) | (endpoints >= POSITION_TOKEN_STOP)):
        bad = endpoints[(endpoints < POSITION_TOKEN_START) | (endpoints >= POSITION_TOKEN_STOP)][:10]
        raise ValueError(f"source_shard={source_shard} slot={slot_index} non-position endpoints: {bad.tolist()}")

    local_endpoints = endpoints - POSITION_TOKEN_START
    endpoint_counts = np.bincount(local_endpoints, minlength=NUM_POSITION_INDICES) if endpoints.size else np.zeros(
        NUM_POSITION_INDICES, dtype=np.int64
    )
    nonzero_degrees = endpoint_counts[endpoint_counts > 0]
    unique_endpoint_count = int(nonzero_degrees.size)
    max_degree = int(nonzero_degrees.max(initial=0))
    degree_sum = int(nonzero_degrees.sum())

    # The exact sparse loss needs one row gather for the second endpoint target.
    # For an undirected contact graph, each emitted contact contributes one
    # neighbor entry to both endpoint rows, so total neighbor nnz is 2C and row
    # width is bounded by the maximum endpoint degree.
    num_tokens = int(token_ids.shape[0])
    return {
        "source_shard": int(source_shard),
        "slot_index": int(slot_index),
        "filled_from_real_slot": bool(filled_from_real_slot),
        "num_tokens": num_tokens,
        "padding_tokens_at_seq_len": int(max_seq_len - num_tokens),
        "prediction_start": int(prediction_start),
        "residue_count": int(residue_count),
        "contact_count": int(contact_count),
        "target_position_count": int(3 * contact_count + 1),
        "unique_endpoint_count": unique_endpoint_count,
        "endpoint_nnz": unique_endpoint_count,
        "neighbor_nnz": int(2 * contact_count),
        "max_degree": max_degree,
        "mean_degree_over_touched": float(degree_sum / unique_endpoint_count) if unique_endpoint_count else 0.0,
        "mean_degree_over_residues": float(degree_sum / residue_count) if residue_count else 0.0,
        "degree_p50": _quantile_int(nonzero_degrees, 0.50),
        "degree_p90": _quantile_int(nonzero_degrees, 0.90),
        "degree_p95": _quantile_int(nonzero_degrees, 0.95),
        "degree_p99": _quantile_int(nonzero_degrees, 0.99),
        "max_endpoint_local_index": int(local_endpoints.max(initial=-1)),
        "fits_position_vocab": bool(residue_count <= NUM_POSITION_INDICES),
    }


def stats_shard(
    items: Iterator[Mapping[str, Any]],
    shard_info: ShardInfo,
    *,
    examples_per_shard: int,
    max_seq_len: int,
    seed: int,
    include_quota_fill: bool,
) -> Iterator[dict[str, Any]]:
    """Emit sparse-shape stats for one analyzed shard's fixed-quota slots."""
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

    packs, truncated = best_fit_pack_documents(
        documents,
        max_seq_len=max_seq_len,
        max_segments_per_example=1,
    )
    if truncated:
        raise ValueError(f"soft-target stats unexpectedly truncated {truncated} documents in shard {shard_info.shard_idx}")

    slot_rng = np.random.default_rng(np.random.SeedSequence([seed, 0, shard_info.shard_idx, 1]))
    slots = fixed_quota_pack_slots(packs, examples_per_shard=examples_per_shard, rng=slot_rng)
    real_slots = tuple(slot for slot in slots if slot is not None)
    if not real_slots:
        raise ValueError(f"input shard {shard_info.shard_idx} yielded no real slots")
    fill_rng = np.random.default_rng(np.random.SeedSequence([seed, 0, shard_info.shard_idx, 2]))

    for slot_index, slot in enumerate(slots):
        filled_from_real_slot = slot is None
        if filled_from_real_slot:
            if not include_quota_fill:
                continue
            current = real_slots[int(fill_rng.integers(len(real_slots)))]
        else:
            current = slot
        if len(current.documents) != 1:
            raise ValueError(
                f"soft-target stats requires one document per slot; got "
                f"{len(current.documents)} in shard {shard_info.shard_idx} slot {slot_index}"
            )
        yield _row_from_document(
            current.documents[0],
            source_shard=shard_info.shard_idx,
            slot_index=slot_index,
            filled_from_real_slot=filled_from_real_slot,
            max_seq_len=max_seq_len,
        )


def _input_shard_index(path: str) -> int:
    match = SHARD_RE.search(path)
    if match is None:
        raise ValueError(f"Could not parse shard index from input path: {path}")
    return int(match.group(1))


def _input_files(args: argparse.Namespace) -> Dataset[str]:
    if args.num_shards is None:
        return Dataset.from_files(args.input)
    if "analyzed-*-of-03338.parquet" not in args.input:
        raise ValueError("--num-shards requires the default analyzed-* input pattern")
    return Dataset.from_list(
        [
            args.input.replace("analyzed-*-of-03338.parquet", f"analyzed-{index:05d}-of-03338.parquet")
            for index in range(args.num_shards)
        ]
    )


def run(args: argparse.Namespace) -> None:
    input_files = _input_files(args)
    rows = input_files.load_parquet(columns=list(ANALYZED_ROW_COLUMNS))
    out_rows = rows.map_shard(
        partial(
            stats_shard,
            examples_per_shard=args.examples_per_shard,
            max_seq_len=args.max_seq_len,
            seed=args.seed,
            include_quota_fill=args.include_quota_fill,
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
    print(f"[exp177] wrote sparse target stats to {args.output}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--examples-per-shard", type=int, default=2650)
    parser.add_argument("--max-seq-len", type=int, default=CONTEXT_LENGTH)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=None, help="Debug cap after loading input shards.")
    parser.add_argument("--max-workers", type=int, default=int(os.environ.get("EXP177_STATS_MAX_WORKERS", "64")))
    parser.add_argument("--worker-cpu", type=float, default=1.0)
    parser.add_argument("--worker-memory", default="8GB")
    parser.add_argument("--worker-disk", default="16GB")
    parser.add_argument("--preemptible", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--include-quota-fill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the duplicate filler slots used to pad underfull shards to the fixed quota.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
