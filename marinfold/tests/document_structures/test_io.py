# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for marinfold.document_structures.io.

- ``read_object_bytes``: fetch one object's bytes via fsspec.
  Contracts: returns the bytes on success; returns ``None`` on
  failure (never raises); emits a warning by default; can be silenced
  with ``warn=False``.

- ``thread_per_row_in_shard``: per-shard worker fan-out. Contracts:
  yields in input order; skips ``None`` results; empty input is a
  no-op (does not spawn a pool).

Tests use the local filesystem (via plain paths and ``file://`` URIs)
so they have no network dependency and run in milliseconds.
"""

import warnings

from marinfold.document_structures.io import (
    read_object_bytes,
    thread_per_row_in_shard,
)


# ---------- read_object_bytes ----------------------------------------------


def test_read_object_bytes_returns_full_contents(tmp_path):
    """Happy path: bytes round-trip through the fsspec dispatch."""
    path = tmp_path / "hello.bin"
    path.write_bytes(b"hello world")
    assert read_object_bytes(str(path)) == b"hello world"


def test_read_object_bytes_returns_none_and_warns_on_failure():
    """Failure contract: never raises; returns None; emits a warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = read_object_bytes("/no/such/path/that/exists")
    assert result is None
    assert any("fetch failed" in str(w.message) for w in caught)


def test_read_object_bytes_warn_false_silences_warning():
    """``warn=False`` suppresses the per-failure log (for bulk scans)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = read_object_bytes("/no/such/path", warn=False)
    assert result is None
    assert not any("fetch failed" in str(w.message) for w in caught)


# ---------- thread_per_row_in_shard ----------------------------------------


def test_thread_per_row_preserves_input_order_and_skips_none():
    """Output is in input order; rows whose worker returns None drop out.

    Bundles two contracts in one assertion because they're co-equal
    promises: a caller relying on either alone is relying on both.
    """
    result = list(thread_per_row_in_shard(
        [1, 2, 3, 4, 5],
        worker=lambda x: None if x % 2 == 0 else x * 10,
        fetch_concurrency=3,
    ))
    assert result == [10, 30, 50]


def test_thread_per_row_empty_input_is_no_op():
    """Empty input yields nothing without trying to spawn a 0-size pool."""
    assert list(thread_per_row_in_shard([], worker=lambda x: x)) == []
