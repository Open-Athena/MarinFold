# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``read.sequence_from_document`` — the sequence-section inverse (issue #213).

The contract that matters is a **round trip**: whatever
``build_document`` writes for a residue list, the reader must give back as
the same one-letter sequence. That is what lets a downstream job (exp94's
KNN index, exp213's train-set overlap audit) recover the training corpus's
sequences from the published documents, which carry no ``sequence`` column.

Pure — no pyconfind, no tokenizer, no torch.
"""

import pytest

from marinfold.document_structures.contacts_v1 import vocab
from marinfold.document_structures.contacts_v1.generate import (
    GenerationConfig,
    build_document,
    generate_sequence_only_document,
)
from marinfold.document_structures.contacts_v1.parse import (
    RawContact,
    ResidueInfo,
    residues_from_sequence,
)
from marinfold.document_structures.contacts_v1.read import sequence_from_document

# Every standard one-letter code, so the round trip exercises the whole map.
_ALL_20 = "ARNDCQEGHILKMFPSTWYV"


def _residues_from(sequence: str) -> list[ResidueInfo]:
    return list(residues_from_sequence(sequence))


def _roundtrip(sequence: str, *, entry_id: str = "e", contacts=()) -> str:
    """build_document(sequence) -> sequence_from_document(...) -> sequence."""
    result = build_document(entry_id, _residues_from(sequence), list(contacts))
    return sequence_from_document(
        result.document, result.seq_len, result.n_term_index
    )


# ---------------------------------------------------------------------------
# Round trip through the real generator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("entry_id", ["e", "afdb-A0A123", "x" * 40])
def test_roundtrip_recovers_the_sequence_for_any_framing(entry_id):
    """Different entry_ids give different start offsets + shuffles; all invert."""
    sequence = _ALL_20 * 3
    assert _roundtrip(sequence, entry_id=entry_id) == sequence


def test_roundtrip_with_contacts_ignores_the_structure_section():
    """`<contact> <pX> <pY>` statements must not be read back as residues."""
    sequence = _ALL_20 * 2
    contacts = [
        RawContact(seq_i=0, seq_j=20, degree=1.0),
        RawContact(seq_i=3, seq_j=30, degree=0.5),
    ]
    assert _roundtrip(sequence, contacts=contacts) == sequence


def test_roundtrip_of_a_sequence_only_document():
    """The sequence-only variant has no `<begin_statements>` at all."""
    sequence = _ALL_20
    result = generate_sequence_only_document(
        sequence, entry_id="e", config=GenerationConfig(sequence_only=True)
    )
    assert sequence_from_document(
        result.document, result.seq_len, result.n_term_index
    ) == sequence


def test_roundtrip_across_the_position_wraparound():
    """A sequence long enough that (start + k) wraps past NUM_POSITION_INDICES."""
    n = vocab.NUM_POSITION_INDICES - 3
    sequence = (_ALL_20 * (n // 20 + 1))[:n]
    # Search entry_ids for one whose start offset actually forces a wrap, so
    # this test fails if the modulo inversion is dropped.
    for entry_id in (f"wrap{i}" for i in range(200)):
        result = build_document(entry_id, _residues_from(sequence), [])
        if result.n_term_index + result.seq_len > vocab.NUM_POSITION_INDICES:
            break
    else:  # pragma: no cover - 200 tries without a wrap would be a broken RNG
        pytest.fail("no wrapping framing found")
    assert sequence_from_document(
        result.document, result.seq_len, result.n_term_index
    ) == sequence


# ---------------------------------------------------------------------------
# Non-standard residues and edge cases
# ---------------------------------------------------------------------------


def test_unknown_residues_read_back_as_x():
    """`<UNK>` is the document's only non-standard residue; it reads as "X"."""
    # B/Z/J/U/O and X all become UNK on the way in, so all read back as X.
    assert _roundtrip("ABZJUOX") == "AXXXXXX"


def test_positions_never_written_default_to_x():
    """Residues a document never states (a truncated rollout) read back as X."""
    sequence = _ALL_20
    result = build_document("e", _residues_from(sequence), [])
    # Drop the statements for sequence positions 3 and 11. The `<n-term>` /
    # `<c-term>` markers are interleaved into the resampled residue stream, so
    # target the two `<pN> <AAA>` pairs by position token rather than by slicing.
    dropped = {3, 11}
    tokens = result.document.split()
    keep, i = [], 0
    while i < len(tokens):
        token, nxt = tokens[i], tokens[i + 1] if i + 1 < len(tokens) else ""
        if (
            token.startswith("<p")
            and nxt.startswith("<")
            and len(nxt) == 5
            and nxt[1:4].isupper()
            and (int(token[2:-1]) - result.n_term_index) % vocab.NUM_POSITION_INDICES
            in dropped
        ):
            i += 2
            continue
        keep.append(token)
        i += 1
    recovered = sequence_from_document(
        " ".join(keep), result.seq_len, result.n_term_index
    )
    assert len(recovered) == len(sequence)
    assert {i for i, c in enumerate(recovered) if c == "X"} == dropped
    assert all(recovered[i] == sequence[i] for i in range(len(sequence)) if i not in dropped)


def test_empty_document_gives_all_x():
    assert sequence_from_document("", 5, 0) == "XXXXX"


def test_zero_length_sequence():
    assert sequence_from_document("<p0> <ALA>", 0, 0) == ""


def test_negative_seq_len_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        sequence_from_document("", -1, 0)
