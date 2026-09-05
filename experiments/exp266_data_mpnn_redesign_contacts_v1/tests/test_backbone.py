# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""The load-bearing regression test for exp266.

exp266 computes contacts-v1 documents from a *stripped* backbone plus a
written-on sequence. If that path is not identical to the corpus path for
the native sequence, then every document we generate is computed under a
subtly different contact operator than `contacts_v1` — a silent
train-distribution mismatch that would be invisible until it showed up as
an unexplained accuracy regression.

So: strip an all-atom structure to backbone, write the native sequence back
on, and assert the resulting document is **byte-identical** to the document
built from the untouched structure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import gemmi
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backbone import (  # noqa: E402
    prepare_structure,
    relabel_sequence,
    residue_sequence,
    strip_to_backbone,
)
from marinfold.document_structures.contacts_v1 import generate_document  # noqa: E402
from marinfold.document_structures.contacts_v1.parse import analyze_structure  # noqa: E402

PDB_MIRROR = Path("/data/tim/af3-db/mmcif_files")

# Single-chain structures spanning small/medium/large and α/β/mixed.
CASES = ["1crn", "1ubq", "101m", "2lyz", "1mbn"]


def _load(stem: str) -> gemmi.Structure:
    path = PDB_MIRROR / f"{stem}.cif"
    if not path.exists():
        pytest.skip(f"{path} not in the local PDB mirror")
    return prepare_structure(gemmi.read_structure(str(path)))


@pytest.mark.parametrize("stem", CASES)
def test_stripped_and_relabelled_document_is_byte_identical(stem: str) -> None:
    """doc(all-atom) == doc(backbone + native sequence written back on)."""
    st = _load(stem)
    try:
        native = generate_document(st, entry_id=stem)
    except ValueError as exc:            # multi-chain in the mirror copy
        pytest.skip(f"{stem}: {exc}")
    assert native is not None

    stripped = strip_to_backbone(st)
    rebuilt = generate_document(
        relabel_sequence(stripped, residue_sequence(stripped)), entry_id=stem
    )
    assert rebuilt is not None
    assert rebuilt.document == native.document


@pytest.mark.parametrize("stem", CASES)
def test_contact_degrees_survive_side_chain_removal(stem: str) -> None:
    """Side chains are not an input to confind — degrees match exactly."""
    st = _load(stem)
    try:
        full = analyze_structure(st, entry_id=stem)
    except ValueError as exc:
        pytest.skip(f"{stem}: {exc}")
    stripped = analyze_structure(strip_to_backbone(st), entry_id=stem)

    assert {(c.seq_i, c.seq_j) for c in full.contacts} == {
        (c.seq_i, c.seq_j) for c in stripped.contacts
    }
    by_pair = {(c.seq_i, c.seq_j): c.degree for c in full.contacts}
    for c in stripped.contacts:
        assert c.degree == pytest.approx(by_pair[(c.seq_i, c.seq_j)], abs=0.0)


def test_relabel_rejects_length_mismatch() -> None:
    st = strip_to_backbone(_load("1crn"))
    with pytest.raises(ValueError, match="sequence length"):
        relabel_sequence(st, "ACDEF")


def test_relabel_rejects_non_canonical_letter() -> None:
    st = strip_to_backbone(_load("1crn"))
    seq = residue_sequence(st)
    with pytest.raises(ValueError, match="non-canonical"):
        relabel_sequence(st, "X" + seq[1:])


# --- staged backbone round-trip ----------------------------------------------


@pytest.mark.parametrize("stem", CASES)
def test_staged_backbone_round_trip_is_byte_identical(stem: str) -> None:
    """The staging encoding must be lossless *at the document level*.

    exp266 stages backbones to CoreWeave object storage because CoreWeave task
    pods have no GCP credentials. If the encode/decode round-trip perturbs
    geometry at all, every document in the corpus is computed from slightly
    different coordinates than the parent corpus was — the exact silent
    train-distribution mismatch the whole design is trying to avoid.
    """
    from backbone import decode_backbone, encode_backbone

    st = _load(stem)
    try:
        direct = generate_document(st, entry_id=stem)
    except ValueError as exc:
        pytest.skip(f"{stem}: {exc}")
    assert direct is not None

    stripped = strip_to_backbone(st)
    row = encode_backbone(stripped) | {"entry_id": stem}
    rebuilt = generate_document(decode_backbone(row), entry_id=stem)

    assert rebuilt is not None
    assert rebuilt.document == direct.document
    assert rebuilt.global_plddt == direct.global_plddt
    assert rebuilt.sha1 == direct.sha1


@pytest.mark.parametrize("stem", CASES)
def test_staged_coordinates_are_exact(stem: str) -> None:
    """int32 milli-angstrom storage reproduces the parsed doubles exactly."""
    from backbone import COORD_SCALE, encode_backbone

    stripped = strip_to_backbone(_load(stem))
    row = encode_backbone(stripped)
    original = [
        value
        for chain in stripped[0]
        for residue in chain
        for name in ("N", "CA", "C", "O")
        for value in _xyz(residue, name)
    ]
    assert [v / COORD_SCALE for v in row["coords_milli"]] == original


def _xyz(residue, name):
    atom = next(a for a in residue if a.name == name)
    return (atom.pos.x, atom.pos.y, atom.pos.z)


def test_encode_rejects_non_canonical_residues() -> None:
    from backbone import encode_backbone

    st = strip_to_backbone(_load("1crn"))
    st[0][0][0].name = "MSE"
    with pytest.raises(ValueError, match="non-canonical"):
        encode_backbone(st)


@pytest.mark.parametrize("stem", CASES)
def test_encode_does_not_need_a_stripped_structure(stem: str) -> None:
    """`encode_backbone` selects atoms by name, so stripping first is a no-op.

    Load-bearing: Stage A2 used to `strip_to_backbone` before encoding, and
    gemmi SIGSEGV'd inside that clone-then-delete path on a real AFDB entry
    mid-run. Dropping the strip removes the crash — but only if it genuinely
    changes nothing, which is what this asserts.
    """
    from backbone import encode_backbone

    st = _load(stem)
    try:
        direct = encode_backbone(st)
    except ValueError as exc:
        pytest.skip(f"{stem}: {exc}")
    stripped = encode_backbone(strip_to_backbone(st))
    assert direct == stripped
