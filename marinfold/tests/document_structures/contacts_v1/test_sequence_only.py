# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure (no pyconfind) tests for the contacts-v1 *sequence-only* variant.

The sequence-only path turns a bare one-letter amino-acid sequence (e.g. a
UniRef50 record; see exp64) into a ``<contacts-v1.sequence_only>`` document:
the same sequence section contacts-v1 emits, but with no structure section
and no contacts. These tests pin the residue builder, the document framing,
and — crucially — that the sequence section is byte-identical to the one the
full contacts-v1 builder produces for the same ``entry_id`` + residues.
"""

import pytest

from marinfold import build_tokenizer
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


_AA_CYCLE = ["MET", "ALA", "GLY", "LYS", "PHE", "SER", "THR", "VAL", "LEU", "ILE"]

# The flag under test. min_seq_separation is irrelevant in sequence-only mode
# (no contacts), so the default config plus the flag is enough.
_SEQ_ONLY = GenerationConfig(sequence_only=True)


def _residues(n: int, *, start_resnum: int = 1) -> list[ResidueInfo]:
    return [
        ResidueInfo(seq_index=i, resname=_AA_CYCLE[i % len(_AA_CYCLE)],
                    resnum=start_resnum + i, chain="A")
        for i in range(n)
    ]


def _seq_section(document: str) -> list[str]:
    """Tokens of the sequence section (between <begin_sequence> and terminator).

    Works for both document types: the terminator is whichever of
    ``<begin_statements>`` (full contacts-v1) or ``<end>`` (sequence-only)
    comes first.
    """
    toks = document.split()
    start = toks.index("<begin_sequence>") + 1
    end = min(toks.index(m) for m in ("<begin_statements>", "<end>") if m in toks)
    return toks[start:end]


# ---------------------------------------------------------------------------
# residues_from_sequence
# ---------------------------------------------------------------------------


def test_residues_from_sequence_maps_one_letter_codes():
    res = residues_from_sequence("MAG")
    assert [r.resname for r in res] == ["MET", "ALA", "GLY"]
    assert [r.seq_index for r in res] == [0, 1, 2]
    assert [r.resnum for r in res] == [1, 2, 3]   # 1-based synthetic numbering
    assert {r.chain for r in res} == {"A"}


def test_residues_from_sequence_ignores_whitespace_and_case():
    assert residues_from_sequence(" m a\ng ") == residues_from_sequence("MAG")


def test_residues_from_sequence_unknown_codes_become_unk():
    # Standard-20 map through; the ambiguity/placeholder letters and any
    # stray symbol fall back to UNK (matching the structure path).
    res = residues_from_sequence("AXBUO")
    assert [r.resname for r in res] == ["ALA", "UNK", "UNK", "UNK", "UNK"]


# ---------------------------------------------------------------------------
# Document shape
# ---------------------------------------------------------------------------


def test_framing_is_sequence_only():
    res = build_document("e", _residues(6), [], config=_SEQ_ONLY)
    toks = res.document.split()
    assert toks[0] == "<contacts-v1.sequence_only>"
    assert toks[1] == "<begin_sequence>"
    assert "<begin_statements>" not in toks   # no structure section
    assert "<contact>" not in toks
    assert toks[-1] == "<end>"


def test_sequence_section_defines_each_residue_once_plus_two_termini():
    residues = _residues(7)
    res = build_document("seqcheck", residues, [], config=_SEQ_ONLY)
    seq = _seq_section(res.document)
    assert seq.count("<n-term>") == 1
    assert seq.count("<c-term>") == 1


def test_num_tokens_formula_and_matches_split():
    length = 7
    res = build_document("e", _residues(length), [], config=_SEQ_ONLY)
    # frame (<doc> <begin_sequence> ... <end>) = 3; sequence section = 2 per
    # residue + 2 per terminus * 2 termini.
    assert res.num_tokens == 3 + 2 * length + 2 * 2
    assert res.num_tokens == len(res.document.split())


def test_tokenizes_with_no_unk_and_new_token_is_real():
    res = build_document("vc", _residues(9), [], config=_SEQ_ONLY)
    tok = build_tokenizer(vocab.all_domain_tokens())
    ids = tok.encode(res.document, add_special_tokens=False)
    assert len(ids) == len(res.document.split())
    assert tok.convert_tokens_to_ids("<UNK>") not in ids
    assert tok.convert_tokens_to_ids("<contacts-v1.sequence_only>") != tok.unk_token_id


def test_unk_resname_emits_unk_token():
    residues = [ResidueInfo(0, "MET", 1, "A"), ResidueInfo(1, "UNK", 2, "A"),
                ResidueInfo(2, "ALA", 3, "A")]
    res = build_document("e", residues, [], config=_SEQ_ONLY)
    assert "<UNK>" in res.document.split()


def test_contacts_are_ignored_and_metadata_zeroed():
    # Even if contacts are passed, sequence-only drops them entirely and
    # reports the contact-statistics fields as 0 / None.
    res = build_document("e", _residues(10), [RawContact(0, 5, 0.9)], config=_SEQ_ONLY)
    assert res.contacts == ()
    assert res.contacts_emitted == 0
    assert res.contacts_pre_filter == 0
    assert res.contacts_passing_min_degree == 0
    assert res.contacts_excluded == 0
    assert res.truncated is False
    assert res.highest_contact_degree is None
    assert res.lowest_nonzero_contact_degree is None
    assert res.lowest_included_contact_degree is None
    assert "<contact>" not in res.document


# ---------------------------------------------------------------------------
# Determinism + bounds
# ---------------------------------------------------------------------------


def test_deterministic_and_entry_id_sensitive():
    a = build_document("X", _residues(8), [], config=_SEQ_ONLY)
    b = build_document("X", _residues(8), [], config=_SEQ_ONLY)
    assert a.document == b.document
    assert a.start_index == b.start_index
    c = build_document("Y", _residues(8), [], config=_SEQ_ONLY)
    assert a.document != c.document   # different start + shuffle


@pytest.mark.parametrize("n", [0, 1])
def test_returns_none_for_too_few_residues(n):
    assert build_document("e", _residues(n), [], config=_SEQ_ONLY) is None


def test_returns_none_for_too_many_residues():
    too_many = vocab.NUM_POSITION_INDICES + 1
    assert build_document("e", _residues(too_many), [], config=_SEQ_ONLY) is None


# ---------------------------------------------------------------------------
# The guarantee: sequence section == contacts-v1's sequence section
# ---------------------------------------------------------------------------


def test_sequence_section_byte_identical_to_full_contacts_v1():
    # The variant exists so a sequence-only corpus shares contacts-v1's
    # representation: same entry_id + residues -> identical sequence section
    # (same wrap-around start, same shuffle). Only the doc type and the
    # dropped structure section differ.
    residues = _residues(40)
    contacts = [RawContact(0, 10, 0.9), RawContact(3, 30, 0.4)]
    full = build_document("AF-shared", residues, contacts,
                          config=GenerationConfig(min_seq_separation=1))
    seq_only = build_document("AF-shared", residues, contacts,
                              config=GenerationConfig(min_seq_separation=1,
                                                      sequence_only=True))
    assert _seq_section(full.document) == _seq_section(seq_only.document)
    assert full.start_index == seq_only.start_index
    assert full.document.split()[0] == "<contacts-v1>"
    assert seq_only.document.split()[0] == "<contacts-v1.sequence_only>"


# ---------------------------------------------------------------------------
# generate_sequence_only_document (string -> document)
# ---------------------------------------------------------------------------


def test_generate_sequence_only_document_from_string():
    seq = "MAGFSTKVLIDEWYQNRHPC"   # 20 standard residues
    res = generate_sequence_only_document(seq, entry_id="u1")
    assert res is not None
    assert res.seq_len == len(seq)
    toks = res.document.split()
    assert toks[0] == "<contacts-v1.sequence_only>"
    assert toks[-1] == "<end>"
    assert "<begin_statements>" not in toks
    assert [r.resname for r in res.residues] == [
        r.resname for r in residues_from_sequence(seq)
    ]


def test_generate_sequence_only_forces_the_flag():
    # A plain (full) config still yields a sequence-only document.
    res = generate_sequence_only_document("MAGFST", entry_id="u2",
                                          config=GenerationConfig())
    assert res.document.split()[0] == "<contacts-v1.sequence_only>"


def test_generate_sequence_only_matches_build_document():
    seq = "MAGFSTKVLI"
    a = generate_sequence_only_document(seq, entry_id="z")
    b = build_document("z", residues_from_sequence(seq), (),
                       config=GenerationConfig(sequence_only=True))
    assert a.document == b.document


def test_generate_sequence_only_too_short_returns_none():
    assert generate_sequence_only_document("M", entry_id="u3") is None
