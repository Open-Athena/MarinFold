# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the exp65 fetch helpers (no network, no large data).

Covers the pure logic that's easy to get wrong: the manifest schema staying in
lockstep, the RCSB query shape, the CAMEO difficulty mapping, the committed
CASP FM classification loading, and gemmi residue counting on a blank-chain
PDB (the CASP-style file that needs ``setup_entities()``).
"""

import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP_DIR))

import fetch_casp_fm
import fetch_cameo_hard
import fetch_denovo_pdb
from _pdb_io import MANIFEST_FIELDS, ManifestRow, count_residues, write_manifest


def test_manifest_schema_lockstep():
    from dataclasses import fields

    assert tuple(f.name for f in fields(ManifestRow)) == MANIFEST_FIELDS


def test_write_manifest_roundtrip(tmp_path):
    import csv

    rows = [ManifestRow(source="denovo_pdb", stem="1abc_A", pdb_id="1abc", length=10)]
    out = write_manifest(rows, tmp_path / "m.csv")
    with out.open() as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == list(MANIFEST_FIELDS)
        got = next(reader)
    assert got["stem"] == "1abc_A" and got["length"] == "10"


def test_denovo_query_shape():
    q = fetch_denovo_pdb.build_query(
        min_len=40, max_len=400, max_resolution=3.0, monomer_only=True, after_date=None
    )
    assert q["return_type"] == "entry"
    assert q["request_options"]["return_all_hits"] is True
    attrs = [
        n["parameters"]["attribute"]
        for n in q["query"]["nodes"]
    ]
    assert "struct_keywords.pdbx_keywords" in attrs
    # The monomer + resolution gates are present when requested.
    assert "rcsb_entry_info.deposited_polymer_entity_instance_count" in attrs
    assert "rcsb_entry_info.resolution_combined" in attrs


def test_denovo_query_optional_nodes_drop():
    q = fetch_denovo_pdb.build_query(
        min_len=40, max_len=400, max_resolution=None, monomer_only=False, after_date=None
    )
    attrs = [n["parameters"]["attribute"] for n in q["query"]["nodes"]]
    assert "rcsb_entry_info.resolution_combined" not in attrs
    assert "rcsb_entry_info.deposited_polymer_entity_instance_count" not in attrs


def test_cameo_difficulty_mapping():
    assert fetch_cameo_hard.DIFF_LABELS["2"] == "hard"
    assert fetch_cameo_hard.LABEL_TO_DIFF["hard"] == "2"
    # round-trip every label
    for k, v in fetch_cameo_hard.DIFF_LABELS.items():
        assert fetch_cameo_hard.LABEL_TO_DIFF[v] == k


def test_casp_fm_classification_loads():
    # The committed classification has both editions and only FM-ish rows.
    rows = fetch_casp_fm.load_fm_domains({"FM", "FM/TBM"}, ["CASP14", "CASP15"])
    assert rows, "committed casp_fm_domains.csv is empty"
    assert {r["casp"] for r in rows} == {"CASP14", "CASP15"}
    assert {r["category"] for r in rows} <= {"FM", "FM/TBM", "TBM/FM"}
    # Filtering by category actually filters.
    fm_only = fetch_casp_fm.load_fm_domains({"FM"}, ["CASP14"])
    assert all(r["category"] == "FM" and r["casp"] == "CASP14" for r in fm_only)


# A minimal 3-residue poly-Gly with a BLANK chain id (CASP-style), which
# gemmi only counts as a polymer after setup_entities() / the CA fallback.
_BLANK_CHAIN_PDB = """\
REMARK test
ATOM      1  N   GLY     1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  GLY     1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   GLY     1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   GLY     1       1.251   2.390   0.000  1.00  0.00           O
ATOM      5  N   GLY     2       3.332   1.540   0.000  1.00  0.00           N
ATOM      6  CA  GLY     2       3.999   2.830   0.000  1.00  0.00           C
ATOM      7  C   GLY     2       5.510   2.690   0.000  1.00  0.00           C
ATOM      8  O   GLY     2       6.080   1.600   0.000  1.00  0.00           O
ATOM      9  N   GLY     3       6.190   3.820   0.000  1.00  0.00           N
ATOM     10  CA  GLY     3       7.640   3.860   0.000  1.00  0.00           C
ATOM     11  C   GLY     3       8.200   5.270   0.000  1.00  0.00           C
ATOM     12  O   GLY     3       7.470   6.260   0.000  1.00  0.00           O
END
"""


def test_count_residues_blank_chain(tmp_path):
    pdb = tmp_path / "blank.pdb"
    pdb.write_text(_BLANK_CHAIN_PDB)
    assert count_residues(pdb) == 3
