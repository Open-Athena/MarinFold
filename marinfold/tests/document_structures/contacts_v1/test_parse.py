# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure (no pyconfind) tests for contacts-v1 parsing helpers + the
shared ``write_docs`` metadata generalization."""

from pathlib import Path

import pytest

from marinfold import write_docs
from marinfold.document_structures.contacts_v1.parse import (
    _canonical_resname,
    iter_structure_paths,
)


@pytest.mark.parametrize("raw, expected", [
    ("ALA", "ALA"), ("GLY", "GLY"), ("TRP", "TRP"),
    ("MSE", "MET"),                       # selenomethionine
    ("SEC", "CYS"), ("CSO", "CYS"),       # seleno / oxidized cys
    ("SEP", "SER"), ("TPO", "THR"), ("PTR", "TYR"),   # phospho
    ("HSD", "HIS"), ("HSE", "HIS"), ("HSC", "HIS"),
    ("HSP", "HIS"), ("HIP", "HIS"),       # HIS variants
    ("xyz", "UNK"), ("FOO", "UNK"),       # unexpected → UNK
    (" ala ", "ALA"),                     # whitespace + case tolerant
])
def test_canonical_resname(raw, expected):
    assert _canonical_resname(raw) == expected


def test_iter_structure_paths_single_file(tmp_path: Path):
    f = tmp_path / "x.cif"
    f.write_text("dummy")
    assert list(iter_structure_paths(f)) == [f]


def test_iter_structure_paths_directory_filters_and_sorts(tmp_path: Path):
    (tmp_path / "a.cif").write_text("")
    (tmp_path / "b.pdb").write_text("")
    (tmp_path / "c.cif.gz").write_text("")
    (tmp_path / "notes.txt").write_text("")   # ignored
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "d.pdb.gz").write_text("")   # recursive
    found = [p.name for p in iter_structure_paths(tmp_path)]
    assert found == ["a.cif", "b.pdb", "c.cif.gz", "d.pdb.gz"]


def test_iter_structure_paths_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        list(iter_structure_paths(tmp_path / "nope"))


# ---------------------------------------------------------------------------
# Shared write_docs generalization: bare strings AND metadata dict rows.
# ---------------------------------------------------------------------------


def test_write_docs_accepts_metadata_dict_rows(tmp_path: Path):
    pq = pytest.importorskip("pyarrow.parquet")
    out = tmp_path / "docs.parquet"
    write_docs(
        out,
        [
            {"document": "<contacts-v1> <end>", "entry_id": "X", "seq_len": 2,
             "truncated": False},
            {"document": "<contacts-v1> <end>", "entry_id": "Y", "seq_len": 3,
             "truncated": True},
        ],
        structure_name="contacts-v1",
    )
    tbl = pq.read_table(str(out))
    cols = set(tbl.column_names)
    assert {"document", "entry_id", "seq_len", "truncated", "structure"} <= cols
    assert tbl.column("structure").to_pylist() == ["contacts-v1", "contacts-v1"]
    assert tbl.column("entry_id").to_pylist() == ["X", "Y"]


def test_write_docs_still_accepts_bare_strings(tmp_path: Path):
    out = tmp_path / "docs.jsonl"
    write_docs(out, ["<contacts-v1> <end>"], structure_name="contacts-v1")
    import json
    row = json.loads(out.read_text().splitlines()[0])
    assert row == {"document": "<contacts-v1> <end>", "structure": "contacts-v1"}


def test_write_docs_structure_name_wins(tmp_path: Path):
    out = tmp_path / "docs.jsonl"
    write_docs(
        out,
        [{"document": "d", "structure": "other"}],
        structure_name="contacts-v1",
    )
    import json
    assert json.loads(out.read_text())["structure"] == "contacts-v1"
