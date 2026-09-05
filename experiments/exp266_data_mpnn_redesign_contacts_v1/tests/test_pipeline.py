# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end over the shape that actually runs: staged row -> 8 documents."""

from __future__ import annotations

import sys
from pathlib import Path

import gemmi
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backbone import encode_backbone, prepare_structure, strip_to_backbone  # noqa: E402

PDB_MIRROR = Path("/data/tim/af3-db/mmcif_files")


def _staged_row(stem: str) -> dict:
    path = PDB_MIRROR / f"{stem}.cif"
    if not path.exists():
        pytest.skip(f"{path} not in the local PDB mirror")
    backbone = strip_to_backbone(prepare_structure(gemmi.read_structure(str(path))))
    return encode_backbone(backbone) | {
        "entry_id": stem,
        "split": "train",
        "round": 4,
        "struct_cluster_id": "cluster-x",
        "native_sha1": "deadbeef",
        "native_contacts_emitted": 123,
    }


def test_staged_row_yields_one_document_per_design() -> None:
    pytest.importorskip("torch")
    from generate_rows import documents_for_row
    from redesign import DESIGN_TEMPERATURES

    records = documents_for_row(_staged_row("1crn"), device="cpu")
    assert records is not None
    assert len(records) == len(DESIGN_TEMPERATURES)

    assert [r["design_index"] for r in records] == list(range(len(DESIGN_TEMPERATURES)))
    assert [r["mpnn_temperature"] for r in records] == list(DESIGN_TEMPERATURES)
    for r in records:
        assert r["entry_id"] == "1crn"              # per-design seed suffix stripped
        assert r["document"].startswith("<contacts-v1>")
        assert r["split"] == "train"                # provenance carried through
        assert r["native_sha1"] == "deadbeef"
        assert r["sha1"] != "deadbeef"              # and not confused with its own
        assert 0.0 <= r["identity_to_native"] <= 1.0


def test_designs_differ_across_slots() -> None:
    """Eight documents, eight different sequences — the whole point of the ladder."""
    pytest.importorskip("torch")
    from generate_rows import documents_for_row

    records = documents_for_row(_staged_row("1crn"), device="cpu")
    documents = {r["document"] for r in records}
    assert len(documents) == len(records)


def test_global_plddt_survives_staging() -> None:
    """`metadata_row` recomputes global_plddt from CA B-factors, which
    `decode_backbone` restores from the staged `ca_plddt`."""
    pytest.importorskip("torch")
    from generate_rows import documents_for_row

    row = _staged_row("101m")
    expected = sum(row["ca_plddt"]) / len(row["ca_plddt"])
    records = documents_for_row(row, device="cpu")
    assert records is not None
    assert records[0]["global_plddt"] == pytest.approx(expected)


def test_stage_row_filters_non_canonical_without_raising() -> None:
    """A designed-in filter returns None; anything else must raise."""
    from stage_rows import stage_row

    path = PDB_MIRROR / "1crn.cif"
    if not path.exists():
        pytest.skip("no local PDB mirror")
    cif = path.read_text()
    row = {"entry_id": "1crn", "cif": cif}
    assert stage_row(row, cif_text_column="cif") is not None

    # A structure whose residues are non-canonical is filtered, not an error.
    mutated = cif.replace("CYS", "MSE")
    assert stage_row({"entry_id": "1crn", "cif": mutated}, cif_text_column="cif") is None
