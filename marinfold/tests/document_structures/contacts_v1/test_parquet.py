# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for parquet (afdb-24M ``cif_content``) input to contacts-v1.

``_parquet_paths`` is pure; ``_gemmi_structure_from_cif_text`` needs gemmi
(a base dep); the end-to-end parquet generation needs pyconfind + the
rotamer library (``network`` marker).
"""

from pathlib import Path

import pytest

from marinfold.document_structures.contacts_v1.parse import _parquet_paths

_1QYS = Path(__file__).parents[2] / "data" / "1QYS.cif"


# ---------------------------------------------------------------------------
# _parquet_paths (pure)
# ---------------------------------------------------------------------------


def test_parquet_paths_single_file(tmp_path: Path):
    p = tmp_path / "shard.parquet"
    p.write_bytes(b"")
    assert _parquet_paths(p) == [p]


def test_parquet_paths_non_parquet_file(tmp_path: Path):
    p = tmp_path / "x.cif"
    p.write_text("")
    assert _parquet_paths(p) == []


def test_parquet_paths_directory_globs_sorted_recursive(tmp_path: Path):
    (tmp_path / "b.parquet").write_bytes(b"")
    (tmp_path / "a.parquet").write_bytes(b"")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.parquet").write_bytes(b"")
    (tmp_path / "notes.cif").write_text("")  # ignored
    assert [p.name for p in _parquet_paths(tmp_path)] == ["a.parquet", "b.parquet", "c.parquet"]


def test_parquet_paths_dir_without_parquet_is_empty(tmp_path: Path):
    (tmp_path / "a.cif").write_text("")
    assert _parquet_paths(tmp_path) == []


def test_parquet_paths_missing_is_empty(tmp_path: Path):
    assert _parquet_paths(tmp_path / "nope") == []


# ---------------------------------------------------------------------------
# gemmi string parsing (gemmi is a base dep)
# ---------------------------------------------------------------------------


def test_gemmi_structure_from_cif_text():
    pytest.importorskip("gemmi")
    from marinfold.document_structures.contacts_v1.parse import (
        _gemmi_structure_from_cif_text,
    )

    structure = _gemmi_structure_from_cif_text(_1QYS.read_text(), name="fallback")
    assert len(structure) >= 1
    assert sum(1 for chain in structure[0] for _ in chain) > 0


# ---------------------------------------------------------------------------
# End-to-end parquet generation (pyconfind + rotamer library)
# ---------------------------------------------------------------------------


def _write_parquet(path: Path, table_dict: dict) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.table(table_dict), str(path))
    return path


@pytest.mark.network
def test_generate_documents_from_parquet_uses_id_column(tmp_path: Path):
    pytest.importorskip("pyconfind")
    from marinfold.document_structures.contacts_v1 import generate_documents

    cif = _1QYS.read_text()
    shard = _write_parquet(
        tmp_path / "shard.parquet",
        {"entry_id": ["AF-A", "AF-B"], "cif_content": [cif, cif]},
    )
    docs = list(generate_documents(input_path=shard, num_docs=5))
    assert [d.entry_id for d in docs] == ["AF-A", "AF-B"]
    assert all(d.seq_len == 92 for d in docs)
    # Same structure, different entry id → different (deterministic) document.
    assert docs[0].document != docs[1].document


@pytest.mark.network
def test_generate_documents_from_parquet_num_docs_cap(tmp_path: Path):
    pytest.importorskip("pyconfind")
    from marinfold.document_structures.contacts_v1 import generate_documents

    cif = _1QYS.read_text()
    shard = _write_parquet(
        tmp_path / "shard.parquet",
        {"entry_id": [f"AF-{i}" for i in range(4)], "cif_content": [cif] * 4},
    )
    docs = list(generate_documents(input_path=shard, num_docs=2))
    assert len(docs) == 2  # early-stop honored across streamed batches


@pytest.mark.network
def test_generate_documents_from_parquet_synthetic_id(tmp_path: Path):
    pytest.importorskip("pyconfind")
    from marinfold.document_structures.contacts_v1 import generate_documents

    shard = _write_parquet(
        tmp_path / "noid.parquet", {"cif_content": [_1QYS.read_text()]}
    )
    docs = list(generate_documents(input_path=shard))
    assert len(docs) == 1
    assert docs[0].entry_id.startswith("noid:row")


def test_generate_documents_from_parquet_missing_cif_column(tmp_path: Path):
    """Wrong --cif-column errors loudly (no pyconfind needed to hit this)."""
    from marinfold.document_structures.contacts_v1.parse import (
        iter_parquet_analyzed_structures,
    )

    shard = _write_parquet(tmp_path / "s.parquet", {"foo": ["x"]})
    with pytest.raises(ValueError, match="no column 'cif_content'"):
        list(iter_parquet_analyzed_structures(shard))
