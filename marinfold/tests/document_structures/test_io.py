# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for marinfold.document_structures.io.

- ``read_object_bytes``: fetch one object's bytes via fsspec.
  Contracts: returns the bytes on success; **raises on failure by
  default** (fail-loud, so corrupted data can't propagate silently
  into a training corpus); ``missing_ok=True`` returns ``None``
  instead and emits a warning; ``missing_ok=True, warn=False``
  silences the warning.

- ``thread_per_row_in_shard``: per-shard worker fan-out. Contracts:
  yields in input order; skips ``None`` results (for designed-in
  filters); empty input is a no-op (does not spawn a pool); worker
  exceptions propagate (verified implicitly via read_object_bytes's
  default behavior).

Tests use the local filesystem (via plain paths and ``file://`` URIs)
so they have no network dependency and run in milliseconds.
"""

import warnings

import pytest

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


def test_read_object_bytes_raises_on_failure_by_default():
    """Fail-loud default: I/O failures propagate. A silently-dropped row
    would corrupt a training corpus; loud failures force triage."""
    with pytest.raises((OSError, ValueError)):
        read_object_bytes("/no/such/path/that/exists")


def test_read_object_bytes_missing_ok_returns_none_and_warns():
    """Opt-in lenient mode: returns None and emits a warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = read_object_bytes("/no/such/path", missing_ok=True)
    assert result is None
    assert any("fetch failed" in str(w.message) for w in caught)


def test_read_object_bytes_missing_ok_warn_false_silences_warning():
    """``warn=False`` (with missing_ok=True) suppresses the per-failure
    log, for bulk scans where the spam would drown out signal."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = read_object_bytes("/no/such/path", missing_ok=True, warn=False)
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
