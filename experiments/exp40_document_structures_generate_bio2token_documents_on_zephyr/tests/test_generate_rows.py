# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""The Zephyr worker (``generate_shard``) end-to-end on CPU with inline cif.

Exercises the whole per-shard path without the distributed runtime: both the
inline-cif and the URI-fetch branches, the batched tokenization, provenance
passthrough, the ``max_context`` filter, and the fail-loud vs skip policy on
missing sources and bad structures — so the worker is covered by fast local
tests before it ever runs on a TPU.

Marked ``network``: downloads the pretrained checkpoint on first run.
"""

import os
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CIF = os.path.join(HERE, "tests", "data", "1QYS.cif")
sys.path.insert(0, HERE)

from generate_rows import generate_shard  # noqa: E402

with open(CIF) as _f:
    _CIF_TEXT = _f.read()


def _rows(n=3):
    return [
        {"entry_id": f"S{i}", "cif_content": _CIF_TEXT,
         "split": "val", "round": 4, "struct_cluster_id": 100 + i}
        for i in range(n)
    ]


@pytest.mark.network
def test_generate_shard_inline_batched():
    rows = _rows(3)
    out = list(generate_shard(rows, cif_text_column="cif_content",
                              device="cpu", max_batch=2))
    assert len(out) == 3
    for src, doc in zip(rows, out, strict=True):
        assert doc["structure"] == "bio2token-v2"
        assert doc["entry_id"] == src["entry_id"]           # preserved
        assert doc["split"] == "val" and doc["round"] == 4  # provenance passthrough
        assert doc["struct_cluster_id"] == src["struct_cluster_id"]
        assert doc["document"].startswith("<bio2token-v2> <begin_sequence> ")
        assert doc["document"].endswith(" <end>")
        assert doc["document"].count("<bt") == doc["num_atoms"]


@pytest.mark.network
def test_context_sampling_fits_budget():
    # A small context budget samples atoms down to fit — structures are kept
    # (Tim's ask), not dropped.
    out = list(generate_shard(_rows(2), cif_text_column="cif_content",
                              device="cpu", context_length=1000))
    assert len(out) == 2
    for doc in out:
        assert doc["num_tokens"] <= 1000
        assert doc["truncated"] is True
        assert doc["num_atoms"] < doc["num_atoms_total"]
        assert doc["document"].count("<bt") == doc["num_atoms"]


@pytest.mark.network
def test_on_error_policy():
    rows = _rows(1) + [{"entry_id": "BAD", "cif_content": "not a cif"}]
    # Fail-loud default: a bad structure kills the shard.
    with pytest.raises((ValueError, RuntimeError)):
        list(generate_shard(rows, cif_text_column="cif_content", device="cpu"))
    # Opt-in skip: the good row survives, the bad one is dropped.
    out = list(generate_shard(rows, cif_text_column="cif_content",
                              device="cpu", on_error="skip"))
    assert len(out) == 1 and out[0]["entry_id"] == "S0"


@pytest.mark.network
def test_uri_path_fetches_and_generates(tmp_path):
    # The production branch: cif_uri_column -> read_object_bytes -> thread pool.
    # Local paths exercise the same fsspec cat_file the GCS path uses.
    p = tmp_path / "s.cif"
    p.write_text(_CIF_TEXT)
    rows = [{"entry_id": "U0", "gcs_uri": str(p), "split": "val"}]
    out = list(generate_shard(rows, cif_uri_column="gcs_uri", device="cpu"))
    assert len(out) == 1
    assert out[0]["entry_id"] == "U0" and out[0]["split"] == "val"
    assert out[0]["document"].count("<bt") == out[0]["num_atoms"]


def test_uri_fetch_failure_raise_vs_skip(tmp_path):
    # A missing object must kill the shard under fail-loud, and be dropped under
    # skip — symmetrically with parse failures. (Fails at fetch, no model load.)
    rows = [{"entry_id": "MISS", "gcs_uri": str(tmp_path / "nope.cif")}]
    with pytest.raises((OSError, ValueError)):
        list(generate_shard(rows, cif_uri_column="gcs_uri", device="cpu"))
    assert list(generate_shard(rows, cif_uri_column="gcs_uri",
                               device="cpu", on_error="skip")) == []


def test_null_source_column_fails_loud():
    # A null/empty source column is bad data, not a designed-in filter: it must
    # raise under the default, never silently produce an empty corpus.
    rows = [{"entry_id": "NULL", "gcs_uri": None}]
    with pytest.raises(ValueError):
        list(generate_shard(rows, cif_uri_column="gcs_uri", device="cpu"))
    assert list(generate_shard(rows, cif_uri_column="gcs_uri",
                               device="cpu", on_error="skip")) == []
