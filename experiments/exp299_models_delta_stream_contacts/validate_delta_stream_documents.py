# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate serialized V2 delta-stream document shards."""

import argparse
from collections.abc import Iterator

import fsspec
import pyarrow.parquet as pq

from compute_contacts_delta_stream_documents import (
    AA_BASE_TOKEN_ID,
    CONTACTS_BEGIN_TOKEN_ID,
    DELTA_BASE_TOKEN_ID,
    DOC_END_TOKEN_ID,
    DOC_START_TOKEN_ID,
    STOP_TOKEN_ID,
)


def _decode_delta(token_id: int, *, max_abs_delta: int) -> int:
    offset = token_id - DELTA_BASE_TOKEN_ID
    if not 0 <= offset < 2 * max_abs_delta:
        raise ValueError(f"token {token_id} is not a delta token")
    if offset < max_abs_delta:
        return -(offset + 1)
    return offset - max_abs_delta + 1


def _paths(pattern: str) -> list[str]:
    fs, path = fsspec.core.url_to_fs(pattern)
    protocol = fs.protocol[0] if isinstance(fs.protocol, tuple) else fs.protocol
    return [f"{protocol}://{item}" for item in sorted(fs.glob(path))]


def _rows(paths: list[str]) -> Iterator[dict]:
    columns = ["entry_id", "seq_len", "max_abs_delta", "token_count", "token_ids", "contact_deltas"]
    for path in paths:
        with fsspec.open(path, "rb") as handle:
            yield from pq.read_table(handle, columns=columns).to_pylist()


def validate_row(row: dict) -> None:
    """Assert that one V2 row contains a complete sequence prefix and contact suffix."""
    entry_id = str(row["entry_id"])
    seq_len = int(row["seq_len"])
    max_abs_delta = int(row["max_abs_delta"])
    token_ids = [int(token_id) for token_id in row["token_ids"]]
    expected_deltas = [[int(delta) for delta in deltas] for deltas in row["contact_deltas"]]

    if len(token_ids) != int(row["token_count"]):
        raise ValueError(f"{entry_id}: token_count does not match token_ids")
    if len(expected_deltas) != seq_len:
        raise ValueError(f"{entry_id}: contact_deltas length does not match seq_len")
    if token_ids[0] != DOC_START_TOKEN_ID or token_ids[-1] != DOC_END_TOKEN_ID:
        raise ValueError(f"{entry_id}: missing document framing")
    sequence = token_ids[1 : seq_len + 1]
    if len(sequence) != seq_len or any(not AA_BASE_TOKEN_ID <= token <= 20 for token in sequence):
        raise ValueError(f"{entry_id}: invalid sequence prefix")
    cursor = seq_len + 1
    if token_ids[cursor] != CONTACTS_BEGIN_TOKEN_ID:
        raise ValueError(f"{entry_id}: contacts do not begin after the complete sequence")
    cursor += 1

    for source_index, expected in enumerate(expected_deltas):
        observed: list[int] = []
        while token_ids[cursor] != STOP_TOKEN_ID:
            delta = _decode_delta(token_ids[cursor], max_abs_delta=max_abs_delta)
            target_index = source_index + delta
            if not 0 <= target_index < seq_len:
                raise ValueError(f"{entry_id}: delta {delta} from {source_index} leaves sequence")
            observed.append(delta)
            cursor += 1
        if observed != expected:
            raise ValueError(f"{entry_id}: contact deltas differ at residue {source_index}")
        if observed != sorted(observed):
            raise ValueError(f"{entry_id}: contact deltas are not sorted at residue {source_index}")
        cursor += 1
    if cursor != len(token_ids) - 1:
        raise ValueError(f"{entry_id}: trailing tokens after contact suffix")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()
    if args.max_rows is not None and args.max_rows <= 0:
        raise ValueError("--max-rows must be positive")

    paths = _paths(args.input)
    if not paths:
        raise ValueError(f"no documents matched {args.input}")
    checked = 0
    for row in _rows(paths):
        validate_row(row)
        checked += 1
        if args.max_rows is not None and checked >= args.max_rows:
            break
    if checked == 0:
        raise ValueError("matched documents contained no rows")
    print(f"validated {checked} V2 delta-stream documents from {len(paths)} shard(s)")


if __name__ == "__main__":
    main()
