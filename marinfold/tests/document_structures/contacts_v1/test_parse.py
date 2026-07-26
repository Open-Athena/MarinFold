# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure (no pyconfind) tests for contacts-v1 parsing helpers + the
shared ``write_docs`` metadata generalization."""

import sys
import types
from pathlib import Path

import pytest

from marinfold import write_docs
from marinfold.document_structures.contacts_v1.parse import (
    _canonical_resname,
    analyze_structure,
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


class _FakeStructure:
    name = "fake_struct.cif"

    def __len__(self) -> int:
        return 0


@pytest.mark.parametrize("assembly", [None, 2, "bio1"])
def test_analyze_structure_passes_assembly_to_pyconfind(monkeypatch, assembly):
    captured: dict[str, object] = {}

    def fake_analyze(structure, **kwargs):
        captured["structure"] = structure
        captured["assembly"] = kwargs["assembly"]
        return types.SimpleNamespace(
            positions=[
                types.SimpleNamespace(
                    position=types.SimpleNamespace(chain="A", resname="ALA", resnum=10)
                ),
                types.SimpleNamespace(
                    position=types.SimpleNamespace(chain="A", resname="GLY", resnum=11)
                ),
            ],
            report=types.SimpleNamespace(
                contacts=[types.SimpleNamespace(pos_i=0, pos_j=1, degree=0.5)]
            ),
        )

    monkeypatch.setitem(sys.modules, "gemmi", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "pyconfind", types.SimpleNamespace(analyze=fake_analyze))

    analyzed = analyze_structure(_FakeStructure(), assembly=assembly)

    assert captured["structure"].__class__ is _FakeStructure
    assert captured["assembly"] == assembly
    assert analyzed.entry_id == "fake_struct"
    assert [r.resname for r in analyzed.residues] == ["ALA", "GLY"]
    assert analyzed.contacts[0].degree == 0.5


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


# --- AnalyzedStructure serialization round-trip (the reusable contacts record) ---

def _toy_analyzed():
    """A small AnalyzedStructure built by hand (no pyconfind)."""
    from marinfold.document_structures.contacts_v1.parse import (
        AnalyzedStructure,
        RawContact,
        ResidueInfo,
    )

    residues = tuple(
        ResidueInfo(seq_index=i, resname=rn, resnum=10 + i, chain="A")
        for i, rn in enumerate(["ALA", "GLY", "TRP", "SER", "LYS", "ILE", "PHE", "VAL"])
    )
    contacts = (
        RawContact(seq_i=0, seq_j=6, degree=0.812),
        RawContact(seq_i=1, seq_j=7, degree=0.4),
        RawContact(seq_i=2, seq_j=5, degree=0.0009),  # a weak-tail contact
    )
    return AnalyzedStructure(
        entry_id="toy-entry",
        residues=residues,
        contacts=contacts,
        global_plddt=87.5,
        source_path=Path("/tmp/toy.cif"),
    )


def test_analyzed_row_roundtrip_preserves_fields():
    from marinfold.document_structures.contacts_v1.parse import (
        ANALYZED_ROW_COLUMNS,
        analyzed_from_row,
        analyzed_to_row,
    )

    analyzed = _toy_analyzed()
    row = analyzed_to_row(analyzed)
    assert set(ANALYZED_ROW_COLUMNS) == set(row)
    assert row["seq_len"] == 8
    assert row["num_contacts"] == 3

    back = analyzed_from_row(row)
    assert back.entry_id == analyzed.entry_id
    assert back.global_plddt == analyzed.global_plddt
    assert back.residues == analyzed.residues        # seq_index re-derived 0..L-1
    assert back.contacts == analyzed.contacts
    # source_path is provenance-only (not serialized) → placeholder unless given
    assert str(back.source_path) == "<row>"


def test_analyzed_from_row_coerces_numpy_and_pyarrow_arrays():
    np = pytest.importorskip("numpy")
    from marinfold.document_structures.contacts_v1.parse import (
        analyzed_from_row,
        analyzed_to_row,
    )

    row = analyzed_to_row(_toy_analyzed())
    # Simulate a parquet reader handing back numpy arrays for the list columns.
    row = dict(row)
    row["contact_seq_i"] = np.array(row["contact_seq_i"], dtype=np.int64)
    row["contact_degree"] = np.array(row["contact_degree"], dtype=np.float32)
    back = analyzed_from_row(row)
    assert [c.seq_i for c in back.contacts] == [0, 1, 2]
    assert back.contacts[0].degree == pytest.approx(0.812, rel=1e-6)


def test_document_from_saved_contacts_is_byte_identical():
    """A document built from a round-tripped record == one built directly.

    This is the reuse guarantee: a future doc type can rebuild from the saved
    pyconfind contacts (analyzed_from_row) and get exactly what it would have
    from a fresh analyze_structure — no pyconfind needed.
    """
    from marinfold.document_structures.contacts_v1.generate import build_document
    from marinfold.document_structures.contacts_v1.parse import (
        analyzed_from_row,
        analyzed_to_row,
    )

    analyzed = _toy_analyzed()
    direct = build_document(
        analyzed.entry_id, analyzed.residues, analyzed.contacts,
        global_plddt=analyzed.global_plddt,
    )
    back = analyzed_from_row(analyzed_to_row(analyzed))
    from_saved = build_document(
        back.entry_id, back.residues, back.contacts,
        global_plddt=back.global_plddt,
    )
    assert direct is not None and from_saved is not None
    assert from_saved.document == direct.document
    assert from_saved.metadata_row() == direct.metadata_row()
