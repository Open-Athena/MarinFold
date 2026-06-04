# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure (no pyconfind) tests for the contacts-v1 document builder.

``build_document`` takes already-computed residues + contacts, so the
serialization, determinism, wrap-around indexing, ordering, and
truncation logic can all be exercised without running pyconfind.
"""

from pathlib import Path
import re

import pytest

from marinfold import build_tokenizer
from marinfold.document_structures.contacts_v1 import generate as generate_module
from marinfold.document_structures.contacts_v1 import vocab
from marinfold.document_structures.contacts_v1.generate import (
    GenerationConfig,
    build_document,
)
from marinfold.document_structures.contacts_v1.parse import (
    AnalyzedStructure,
    RawContact,
    ResidueInfo,
)


_AA_CYCLE = ["MET", "ALA", "GLY", "LYS", "PHE", "SER", "THR", "VAL", "LEU", "ILE"]


def _residues(n: int, *, start_resnum: int = 1) -> list[ResidueInfo]:
    return [
        ResidueInfo(seq_index=i, resname=_AA_CYCLE[i % len(_AA_CYCLE)],
                    resnum=start_resnum + i, chain="A")
        for i in range(n)
    ]


def _pos_indices(document: str) -> list[int]:
    # Positions are reused <pX> tokens from contacts-and-distances-v1.
    return [int(m) for m in re.findall(r"<p(\d+)>", document)]


def _sections(document: str) -> tuple[list[str], list[str]]:
    """Split into (sequence-section tokens, structure-section tokens)."""
    toks = document.split()
    bs = toks.index("<begin_statements>")
    seq = toks[toks.index("<begin_sequence>") + 1: bs]
    struct = toks[bs + 1: toks.index("<end>")]
    return seq, struct


# ---------------------------------------------------------------------------
# Document shape
# ---------------------------------------------------------------------------


def test_document_framing():
    res = build_document("e", _residues(5), [RawContact(0, 2, 0.9)])
    toks = res.document.split()
    assert toks[0] == "<contacts-v1>"
    assert toks[1] == "<begin_sequence>"     # reused from c-and-d-v1
    assert "<begin_statements>" in toks      # reused from c-and-d-v1
    assert toks[-1] == "<end>"


def test_sequence_section_defines_each_residue_once_plus_two_termini():
    residues = _residues(7)
    res = build_document("seqcheck", residues, [])
    seq, _ = _sections(res.document)
    assert seq.count("<n-term>") == 1
    assert seq.count("<c-term>") == 1
    # One <pX> <AA> statement per residue → exactly len residues AA tokens,
    # and the residue position indices are the L distinct wrapped indices.
    n = vocab.NUM_POSITION_INDICES
    expected = {(res.start_index + k) % n for k in range(len(residues))}
    # Drop the two termini <pos> tokens (they repeat n_term/c_term indices,
    # which are already in `expected`); the set of all <pos> indices equals
    # exactly the residue index set.
    assert set(_pos_indices(res.document)) == expected


def test_token_count_matches_split_and_num_tokens():
    res = build_document("e", _residues(6), [RawContact(0, 3, 0.5), RawContact(1, 4, 0.2)])
    assert res.num_tokens == len(res.document.split())


def test_tokenizes_with_no_unk():
    res = build_document("vocabcheck", _residues(9),
                         [RawContact(0, 4, 0.7), RawContact(2, 8, 0.3)])
    tok = build_tokenizer(vocab.all_domain_tokens())
    ids = tok.encode(res.document, add_special_tokens=False)
    assert len(ids) == len(res.document.split())
    assert tok.convert_tokens_to_ids("<UNK>") not in ids


def test_unk_resname_emits_unk_token():
    residues = [
        ResidueInfo(0, "MET", 1, "A"),
        ResidueInfo(1, "UNK", 2, "A"),
        ResidueInfo(2, "ALA", 3, "A"),
    ]
    res = build_document("e", residues, [])
    assert "<UNK>" in res.document.split()


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic_same_entry_id():
    residues = _residues(8)
    contacts = [RawContact(0, 4, 0.9), RawContact(1, 6, 0.5), RawContact(2, 7, 0.1)]
    a = build_document("AF-X", residues, contacts)
    b = build_document("AF-X", residues, contacts)
    assert a.document == b.document
    assert a.start_index == b.start_index


def test_different_entry_id_differs():
    residues = _residues(8)
    a = build_document("AF-X", residues, [])
    b = build_document("AF-Y", residues, [])
    # Overwhelmingly likely to differ (different random start + shuffle).
    assert a.document != b.document


# ---------------------------------------------------------------------------
# Wrap-around indexing
# ---------------------------------------------------------------------------


def test_indices_in_range_and_terminus_formula():
    residues = _residues(40)
    res = build_document("wrap", residues, [])
    n = vocab.NUM_POSITION_INDICES
    assert 0 <= res.start_index < n
    assert res.n_term_index == res.start_index
    assert res.c_term_index == (res.start_index + res.seq_len - 1) % n
    assert all(0 <= p < n for p in _pos_indices(res.document))


def test_wraparound_actually_wraps():
    # Small index space forces a wrap for most starts; scan a few seeds.
    cfg = GenerationConfig(num_position_indices=10)
    found_wrap = False
    for i in range(50):
        res = build_document(f"e{i}", _residues(8), [], config=cfg)
        assert 0 <= res.start_index < 10
        assert all(0 <= p < 10 for p in _pos_indices(res.document))
        # Residue indices must stay unique even across the wrap.
        idxs = [(res.start_index + k) % 10 for k in range(8)]
        assert len(set(idxs)) == 8
        if res.c_term_index < res.n_term_index:
            found_wrap = True
    assert found_wrap, "expected at least one wrapped chain in 50 seeds"


# ---------------------------------------------------------------------------
# Contact ordering + pair-order flip
# ---------------------------------------------------------------------------


def test_contacts_selected_by_strength_multiset_preserved():
    residues = _residues(10)
    contacts = [RawContact(0, 5, 0.2), RawContact(1, 7, 0.9), RawContact(2, 9, 0.5)]
    res = build_document("order", residues, contacts)
    # All fit → all included; the multiset of degrees is preserved.
    assert res.contacts_emitted == 3
    assert sorted(c.degree for c in res.contacts) == [0.2, 0.5, 0.9]


def test_contacts_listed_in_random_order_not_degree_sorted():
    # Many distinct degrees: a random shuffle being exactly descending is
    # ~1/39! — so a non-sorted order is a reliable signal of randomization.
    residues = _residues(40)
    contacts = [RawContact(i, i + 1, float(39 - i)) for i in range(39)]
    res = build_document("randorder", residues, contacts)
    degrees = [c.degree for c in res.contacts]
    assert res.contacts_emitted == 39
    assert sorted(degrees) == [float(d) for d in range(1, 40)]  # multiset intact
    assert degrees != sorted(degrees, reverse=True), "doc order should be randomized"


def test_emitted_pair_order_matches_flip_flag():
    residues = _residues(12)
    contacts = [RawContact(0, 5, 0.9), RawContact(1, 8, 0.6), RawContact(3, 11, 0.3)]
    res = build_document("flip", residues, contacts)
    _, struct = _sections(res.document)
    # struct is groups of <contact> <pA> <pB>.
    triples = [struct[i:i + 3] for i in range(0, len(struct), 3)]
    assert len(triples) == len(res.contacts)
    for (marker, a, b), c in zip(triples, res.contacts):
        assert marker == "<contact>"
        first, second = (c.pos_j, c.pos_i) if c.flipped else (c.pos_i, c.pos_j)
        assert a == f"<p{first}>"
        assert b == f"<p{second}>"


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


def test_truncation_keeps_strongest_and_respects_budget():
    residues = _residues(50)
    # 40 contacts with strictly decreasing degrees.
    contacts = [RawContact(i, i + 1, float(40 - i)) for i in range(40)]
    context_length = 200  # frame(4)+seq(2*50)+termini(4)=108 → 30 contacts fit
    res = build_document("trunc", residues, contacts, context_length=context_length)
    assert res.contacts_pre_filter == 40
    assert res.contacts_passing_min_degree == 40  # all degrees >= 0.001
    assert res.contacts_emitted == 30
    assert res.contacts_excluded == 10
    assert res.truncated is True
    assert res.num_tokens <= context_length
    # The 30 strongest (degrees 40..11) survive — as a multiset, since the
    # in-document order is randomized.
    assert sorted((c.degree for c in res.contacts), reverse=True) == [
        float(d) for d in range(40, 10, -1)
    ]
    # Degree statistics.
    assert res.highest_contact_degree == 40.0
    assert res.lowest_nonzero_contact_degree == 1.0       # weakest overall
    assert res.lowest_included_contact_degree == 11.0      # weakest that fit


def test_not_truncated_when_everything_fits():
    residues = _residues(6)
    contacts = [RawContact(0, 3, 0.9), RawContact(1, 5, 0.4)]
    res = build_document("fits", residues, contacts)
    assert res.truncated is False
    assert res.contacts_excluded == 0
    assert res.contacts_passing_min_degree == 2
    assert res.contacts_emitted == res.contacts_pre_filter == 2
    # No truncation → weakest-included equals weakest-overall.
    assert res.highest_contact_degree == 0.9
    assert res.lowest_nonzero_contact_degree == 0.4
    assert res.lowest_included_contact_degree == 0.4


def test_degree_stats_none_when_no_contacts():
    res = build_document("nocontacts", _residues(5), [])
    assert res.contacts_pre_filter == 0
    assert res.contacts_passing_min_degree == 0
    assert res.contacts_emitted == 0
    assert res.contacts_excluded == 0
    assert res.highest_contact_degree is None
    assert res.lowest_nonzero_contact_degree is None
    assert res.lowest_included_contact_degree is None


# ---------------------------------------------------------------------------
# Minimum-degree filter
# ---------------------------------------------------------------------------


def test_min_degree_filter_excludes_weak_contacts_even_with_space():
    residues = _residues(10)
    # Two strong, two below the 0.001 default — plenty of budget room.
    contacts = [
        RawContact(0, 5, 0.9),
        RawContact(1, 6, 0.5),
        RawContact(2, 7, 0.0005),   # below threshold
        RawContact(3, 8, 1e-8),     # below threshold
    ]
    res = build_document("weak", residues, contacts)
    assert res.contacts_pre_filter == 4
    assert res.contacts_passing_min_degree == 2
    assert res.contacts_emitted == 2          # weak ones never included
    assert res.contacts_excluded == 2
    assert res.truncated is False             # dropped by filter, not budget
    # Emitted degrees are only the strong ones; none below threshold.
    assert sorted(c.degree for c in res.contacts) == [0.5, 0.9]
    assert all(c.degree >= 0.001 for c in res.contacts)
    # Whole-protein stats still reflect the raw contacts.
    assert res.highest_contact_degree == 0.9
    assert res.lowest_nonzero_contact_degree == 1e-8
    assert res.lowest_included_contact_degree == 0.5


def test_custom_min_contact_degree_threshold():
    residues = _residues(10)
    contacts = [RawContact(0, 5, 0.9), RawContact(1, 6, 0.05), RawContact(2, 7, 0.005)]
    cfg = GenerationConfig(min_contact_degree=0.1)
    res = build_document("thresh", residues, contacts, config=cfg)
    assert res.contacts_passing_min_degree == 1   # only 0.9 >= 0.1
    assert res.contacts_emitted == 1
    assert res.contacts_excluded == 2
    assert res.lowest_included_contact_degree == 0.9


def test_all_contacts_below_threshold():
    residues = _residues(6)
    contacts = [RawContact(0, 3, 0.0002), RawContact(1, 5, 1e-7)]
    res = build_document("allweak", residues, contacts)
    assert res.contacts_pre_filter == 2
    assert res.contacts_passing_min_degree == 0
    assert res.contacts_emitted == 0
    assert res.contacts_excluded == 2
    assert res.truncated is False
    # Whole-protein degree stats exist; included-degree is null.
    assert res.highest_contact_degree == 0.0002
    assert res.lowest_nonzero_contact_degree == 1e-7
    assert res.lowest_included_contact_degree is None
    # Document still valid: sequence section + empty structure section.
    assert res.document.split()[-1] == "<end>"
    assert "<contact>" not in res.document.split()


# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [0, 1])
def test_returns_none_for_too_few_residues(n):
    assert build_document("e", _residues(n), []) is None


def test_returns_none_for_too_many_residues():
    cfg = GenerationConfig(num_position_indices=10)
    assert build_document("e", _residues(11), [], config=cfg) is None
    # Exactly at the cap is allowed.
    assert build_document("e", _residues(10), [], config=cfg) is not None


def test_returns_none_when_fixed_section_exceeds_context_length():
    assert build_document("e", _residues(6), [], context_length=19) is None


def test_generate_documents_warns_and_skips_when_fixed_section_exceeds_context_length(
    monkeypatch,
):
    analyzed = AnalyzedStructure(
        entry_id="too-long",
        residues=tuple(_residues(30)),
        contacts=(),
        global_plddt=80.0,
        source_path=Path("too-long.cif"),
    )
    monkeypatch.setattr(
        generate_module,
        "iter_analyzed_structures",
        lambda *args, **kwargs: iter([analyzed]),
    )
    with pytest.warns(
        UserWarning,
        match=r"skipping too-long: fixed sequence section needs 68 tokens > context_length 67",
    ):
        assert list(generate_module.generate_documents("ignored", context_length=67)) == []


# ---------------------------------------------------------------------------
# Metadata surface
# ---------------------------------------------------------------------------


def test_metadata_row_and_summary_dict():
    residues = _residues(5)
    contacts = [RawContact(0, 2, 0.9), RawContact(1, 4, 0.5)]
    res = build_document("AF-META", residues, contacts, global_plddt=87.5)
    row = res.metadata_row()
    assert row["document"] == res.document
    assert row["entry_id"] == "AF-META"
    assert row["seq_len"] == 5
    assert row["global_plddt"] == 87.5
    assert row["contacts_pre_filter"] == 2
    assert row["contacts_passing_min_degree"] == 2
    assert row["contacts_emitted"] == 2
    assert row["contacts_excluded"] == 0
    assert row["truncated"] is False
    assert row["highest_contact_degree"] == 0.9
    assert row["lowest_nonzero_contact_degree"] == 0.5
    assert row["lowest_included_contact_degree"] == 0.5
    assert len(row["sha1"]) == 40
    summary = res.summary_dict()
    assert "document" not in summary
    assert summary["sequence"] == ["MET", "ALA", "GLY", "LYS", "PHE"]
    assert len(summary["contacts"]) == 2
    # Contacts are in (random) document order; check as a set.
    assert {c["degree"] for c in summary["contacts"]} == {0.9, 0.5}
    assert {"resnum_i", "resname_i", "pos_i", "degree"} <= summary["contacts"][0].keys()
