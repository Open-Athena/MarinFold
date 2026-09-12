# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate exp177 precomputed soft-target contacts-v1 parquet format."""

import argparse
from collections.abc import Iterable

import fsspec
import pyarrow.parquet as pq

from marinfold.document_structures.contacts_v1.vocab import (
    BEGIN_SEQUENCE,
    BEGIN_STRUCTURE,
    CONTACT,
    DOC_TYPE,
    END,
    N_TERM,
    C_TERM,
    POSITIONS,
)

POSITION_TOKEN_START = int(POSITIONS[0])
POSITION_TOKEN_END = int(POSITIONS[-1]) + 1


def _shard_uri(prefix: str, shard: int, total: int) -> str:
    return f"{prefix.rstrip('/')}/shard-{shard:05d}-of-{total:05d}.parquet"


def _iter_rows(prefix: str, shards: Iterable[int], total: int, rows_per_shard: int):
    columns = [
        "token_ids",
        "position_ids",
        "segment_ids",
        "attention_blocks",
        "prediction_start",
        "contact_first_ids",
        "contact_second_ids",
        "contact_count",
        "target_position_count",
    ]
    for shard in shards:
        with fsspec.open(_shard_uri(prefix, shard, total), "rb") as source:
            table = pq.read_table(source, columns=columns)
        for row in table.slice(0, rows_per_shard).to_pylist():
            yield shard, row


def _is_position_token(token_id: int) -> bool:
    return POSITION_TOKEN_START <= token_id < POSITION_TOKEN_END


def validate(prefix: str, *, total_shards: int, shard_count: int, rows_per_shard: int) -> None:
    checked = 0
    contact_rows = 0
    for shard, row in _iter_rows(prefix, range(shard_count), total_shards, rows_per_shard):
        token_ids = [int(x) for x in row["token_ids"]]
        position_ids = [int(x) for x in row["position_ids"]]
        segment_ids = [int(x) for x in row["segment_ids"]]
        attention_blocks = [int(x) for x in row["attention_blocks"]]
        prediction_start = int(row["prediction_start"])
        contact_count = int(row["contact_count"])
        target_position_count = int(row["target_position_count"])
        first_ids = [int(x) for x in row["contact_first_ids"][:contact_count]]
        second_ids = [int(x) for x in row["contact_second_ids"][:contact_count]]

        assert token_ids[0] == int(DOC_TYPE), (shard, "bad doc type")
        assert token_ids[1] == int(BEGIN_SEQUENCE), (shard, "bad begin_sequence")
        assert token_ids[prediction_start] == int(BEGIN_STRUCTURE), (shard, "bad prediction_start")
        assert target_position_count == 3 * contact_count + 1, (shard, "bad target count")
        assert position_ids == list(range(len(position_ids))), (shard, "position_ids not absolute sequential")

        end_position = prediction_start + 3 * contact_count + 1
        assert token_ids[end_position] == int(END), (shard, "missing END after contacts")
        assert all(segment_id == 0 for segment_id in segment_ids[: end_position + 1]), (shard, "bad segment_ids")
        assert all(block == 0 for block in attention_blocks[: prediction_start + 1]), (shard, "bad prefix blocks")
        assert attention_blocks[prediction_start + 1 : end_position + 1] == list(range(1, 3 * contact_count + 2)), (
            shard,
            "bad suffix attention blocks",
        )

        declared_positions = {
            token_id for token_id in token_ids[:prediction_start] if _is_position_token(token_id)
        }
        assert declared_positions, (shard, "no declared positions in contacts-v1 prefix")
        assert int(N_TERM) in token_ids[:prediction_start], (shard, "missing n-term token")
        assert int(C_TERM) in token_ids[:prediction_start], (shard, "missing c-term token")

        for contact_index, (first, second) in enumerate(zip(first_ids, second_ids, strict=True)):
            contact_position = prediction_start + 1 + 3 * contact_index
            assert token_ids[contact_position] == int(CONTACT), (shard, contact_index, "missing CONTACT")
            assert token_ids[contact_position + 1] == first, (shard, contact_index, "first id mismatch")
            assert token_ids[contact_position + 2] == second, (shard, contact_index, "second id mismatch")
            assert first in declared_positions, (shard, contact_index, "first endpoint not declared in prefix")
            assert second in declared_positions, (shard, contact_index, "second endpoint not declared in prefix")
        contact_rows += int(contact_count > 0)
        checked += 1

    assert checked == shard_count * rows_per_shard, f"checked {checked}, expected {shard_count * rows_per_shard}"
    assert contact_rows > 0, "validated rows contained no contacts"
    print(f"validated {checked} rows from {shard_count} shards at {prefix}; contact_rows={contact_rows}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--total-shards", type=int, default=3338)
    parser.add_argument("--shard-count", type=int, default=8)
    parser.add_argument("--rows-per-shard", type=int, default=16)
    args = parser.parse_args()
    validate(
        args.prefix,
        total_shards=args.total_shards,
        shard_count=args.shard_count,
        rows_per_shard=args.rows_per_shard,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
