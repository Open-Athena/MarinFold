# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Worker tests for the exp139 ESM-Atlas generation pipeline.

Two layers:

- **Pure** (no pyconfind, no gemmi): the combined-row assembly and the drop
  policy, with ``analyze_structure`` / ``build_document`` monkeypatched. These
  prove the one structural novelty of this experiment — one pyconfind pass →
  one row carrying *both* the saved contacts and the document — without paying
  for pyconfind in CI.
- **Integration** (``importorskip('pyconfind')`` + a cached ESM-Atlas part):
  real end-to-end on inline ``cif_content``; skipped when neither is available.
"""

import glob
import math
import os

import pytest

import generate_rows as gr
from marinfold.document_structures.contacts_v1 import (
    AnalyzedStructure,
    RawContact,
    ResidueInfo,
    build_document,
)


def _toy_analyzed(entry_id="e1"):
    residues = tuple(
        ResidueInfo(seq_index=i, resname=rn, resnum=1 + i, chain="A")
        for i, rn in enumerate(["ALA", "GLY", "TRP", "SER", "LYS", "ILE"])
    )
    contacts = (
        RawContact(seq_i=0, seq_j=5, degree=0.9),
        RawContact(seq_i=1, seq_j=4, degree=0.3),
    )
    return AnalyzedStructure(
        entry_id=entry_id, residues=residues, contacts=contacts,
        global_plddt=88.0, source_path=__import__("pathlib").Path("<toy>"),
    )


def test_build_output_row_unions_contacts_document_and_provenance():
    analyzed = _toy_analyzed()
    result = build_document(analyzed.entry_id, analyzed.residues, analyzed.contacts,
                            global_plddt=analyzed.global_plddt)
    assert result is not None
    row = {
        "entry_id": "e1", "seq_cluster_id": "clust7", "cluster_size": 42,
        "ptm": 0.71, "plddt_std": 0.05, "source": "esm-atlas-v1", "split": "train",
    }
    out = gr.build_output_row(row, analyzed, result)

    # the saved-contacts record (reusable pyconfind output)
    assert out["residue_resname"] == ["ALA", "GLY", "TRP", "SER", "LYS", "ILE"]
    assert out["contact_seq_i"] == [0, 1]
    assert out["contact_seq_j"] == [5, 4]
    assert out["num_contacts"] == 2
    # the document + its metadata
    assert out["document"].startswith("<contacts-v1>")
    assert out["sha1"] == result.metadata_row()["sha1"]
    # shared keys agree (both derive from the same analyzed structure)
    assert out["seq_len"] == 6
    assert out["global_plddt"] == 88.0
    # provenance passthrough
    assert out["seq_cluster_id"] == "clust7"
    assert out["cluster_size"] == 42
    assert out["ptm"] == 0.71
    assert out["structure"] == "contacts-v1"


def test_generate_doc_for_row_missing_cif_drops():
    assert gr.generate_doc_for_row({"entry_id": "x"}) is None
    assert gr.generate_doc_for_row({"entry_id": "x", "cif_content": ""}) is None


def test_generate_doc_for_row_happy_path_monkeypatched(monkeypatch):
    """One analyze pass → combined row, without gemmi/pyconfind."""
    analyzed = _toy_analyzed("seed-id")

    monkeypatch.setattr(gr, "structure_from_cif", lambda data, *, entry_id: object())
    monkeypatch.setattr(gr, "analyze_structure", lambda structure, **kw: analyzed)
    captured = {}

    real_build = gr.build_document

    def spy_build(entry_id, residues, contacts, **kw):
        captured["args"] = (entry_id, residues, contacts)
        return real_build(entry_id, residues, contacts, **kw)

    monkeypatch.setattr(gr, "build_document", spy_build)

    out = gr.generate_doc_for_row(
        {"entry_id": "seed-id", "cif_content": "CIF", "seq_cluster_id": "c1"},
    )
    assert out is not None
    # build_document was fed the analyzed structure's residues + RAW contacts
    assert captured["args"][0] == "seed-id"
    assert captured["args"][2] == analyzed.contacts
    assert out["entry_id"] == "seed-id"
    assert out["contact_degree"] == [0.9, 0.3]


def test_generate_doc_for_row_unserializable_drops_whole_row(monkeypatch):
    """>2000 residues (build_document → None) drops BOTH doc and contacts (1:1)."""
    monkeypatch.setattr(gr, "structure_from_cif", lambda data, *, entry_id: object())
    monkeypatch.setattr(gr, "analyze_structure", lambda structure, **kw: _toy_analyzed())
    monkeypatch.setattr(gr, "build_document", lambda *a, **k: None)
    assert gr.generate_doc_for_row({"entry_id": "big", "cif_content": "CIF"}) is None


def test_generate_doc_for_row_multichain_valueerror_drops(monkeypatch):
    monkeypatch.setattr(gr, "structure_from_cif", lambda data, *, entry_id: object())

    def boom(structure, **kw):
        raise ValueError("expected a single protein chain, found 2")

    monkeypatch.setattr(gr, "analyze_structure", boom)
    with pytest.warns(UserWarning):
        assert gr.generate_doc_for_row({"entry_id": "mc", "cif_content": "CIF"}) is None


def test_generate_shard_filters_dropped_rows(monkeypatch):
    monkeypatch.setattr(gr, "_load_rotamer_library", lambda: None)
    monkeypatch.setattr(gr, "structure_from_cif", lambda data, *, entry_id: object())
    monkeypatch.setattr(
        gr, "analyze_structure",
        lambda structure, **kw: _toy_analyzed(kw.get("entry_id") or "e1"),
    )
    monkeypatch.setattr(gr, "build_document", gr.build_document)

    items = [
        {"entry_id": "a", "cif_content": "CIF"},
        {"entry_id": "b"},                     # missing cif → dropped
        {"entry_id": "c", "cif_content": "CIF"},
    ]
    out = list(gr.generate_shard(items))
    assert [r["entry_id"] for r in out] == ["a", "c"]


# --- integration: real gemmi + pyconfind on a cached ESM-Atlas part ---

_ATLAS_GLOBS = [
    os.path.expanduser("~/exp139_scratch/parts/*.parquet"),
    "/tmp/esm_atlas_smoke/*.parquet",
]


def _cached_atlas_rows(n: int) -> list[dict]:
    for pat in _ATLAS_GLOBS:
        shards = sorted(glob.glob(pat))
        if shards:
            import pyarrow.parquet as pq
            cols = ["entry_id", "cif_content", "seq_cluster_id", "cluster_size",
                    "ptm", "plddt_std", "source", "split"]
            tbl = pq.read_table(shards[0])
            keep = [c for c in cols if c in tbl.column_names]
            return tbl.select(keep).slice(0, n).to_pylist()
    pytest.skip("no cached ESM-Atlas part (see _ATLAS_GLOBS)")


def test_integration_real_pyconfind_on_atlas_part():
    pytest.importorskip("pyconfind")
    rows = _cached_atlas_rows(5)
    out = list(gr.generate_shard(rows))
    assert out, "expected at least one serializable structure in the first 5 rows"
    for r in out:
        assert r["document"].startswith("<contacts-v1>") and r["document"].endswith("<end>")
        assert r["num_contacts"] == len(r["contact_seq_i"]) == len(r["contact_degree"])
        assert 2 <= r["seq_len"] <= 2000
        # saved contacts are the RAW pyconfind set (>= what the doc emits)
        assert r["num_contacts"] >= r["contacts_emitted"]
        assert not math.isnan(r["global_plddt"])
