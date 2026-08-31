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
