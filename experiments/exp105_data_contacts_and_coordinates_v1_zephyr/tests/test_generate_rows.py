# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Worker tests for the Stage-B per-row generation.

Prove the worker (a) drops bad rows leniently without raising, and (b) is
**byte-identical** to calling the format's ``generate_document`` directly on
the same inline mmCIF — this experiment must not alter the document logic.
The pyconfind-backed cases use the bundled 1QYS mmCIF (marinfold test data)
and skip when pyconfind isn't installed.
"""

from pathlib import Path

import pytest

import generate_rows as gr

# 1QYS mmCIF shipped with the marinfold package tests (repo root is two
# levels up from this experiment dir: experiments/exp105.../).
_1QYS = (
    Path(gr.__file__).resolve().parents[2]
    / "marinfold" / "tests" / "data" / "1QYS.cif"
)


def test_missing_uri_row_is_dropped():
    assert gr.generate_doc_for_row({"entry_id": "x"}) is None


def test_missing_inline_cif_is_dropped():
    row = {"entry_id": "x", "cif_content": ""}
    assert gr.generate_doc_for_row(row, cif_text_column="cif_content") is None


def test_invalid_cif_is_dropped_not_raised():
    row = {"entry_id": "x", "cif_content": "not a real cif"}
    assert gr.generate_doc_for_row(row, cif_text_column="cif_content") is None


@pytest.mark.network
def test_worker_row_matches_direct_generation():
    pytest.importorskip("pyconfind")
    from marinfold.document_structures.contacts_and_coordinates_v1 import (
        generate_document,
    )

    cif = _1QYS.read_text()
    entry_id = "1QYS"
    row = {
        "entry_id": entry_id,
        "cif_content": cif,
        "round": 0,
        "struct_cluster_id": "clust-1",
        "split": "train",
    }
    out = gr.generate_doc_for_row(row, cif_text_column="cif_content")
    assert out is not None

    # Byte-identical document to a direct call with the same seed.
    structure = gr.structure_from_cif(cif, entry_id=entry_id)
    direct = generate_document(structure, entry_id=entry_id)
    assert direct is not None
    assert out["document"] == direct.document
    assert out["sha1"] == direct.sha1

    # Provenance passthrough + structure stamp are present.
    assert out["structure"] == "contacts-and-coordinates-v1"
    assert out["round"] == 0
    assert out["struct_cluster_id"] == "clust-1"
    assert out["split"] == "train"
    # entry_id comes from metadata_row (the seed), not duplicated.
    assert out["entry_id"] == entry_id


@pytest.mark.network
def test_generate_shard_inline_path():
    pytest.importorskip("pyconfind")
    cif = _1QYS.read_text()
    rows = [
        {"entry_id": "1QYS", "cif_content": cif, "round": 0, "split": "train"},
        {"entry_id": "bad", "cif_content": "garbage", "round": 1, "split": "train"},
    ]
    out = list(gr.generate_shard(rows, None, cif_text_column="cif_content"))
    # The bad row is dropped; the good one yields exactly one document.
    assert len(out) == 1
    assert out[0]["entry_id"] == "1QYS"
    assert out[0]["num_events"] > 0
