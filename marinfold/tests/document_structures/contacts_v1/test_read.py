# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the contacts-v1 structure-section fold (``read.py``, issue #158).

Pure — no pyconfind, no tokenizer, no torch. The fold is the semantic
definition of retraction: the structure section is an ordered edit list and
the live contact set is whatever survives to ``<end>``.
"""

from marinfold.document_structures.contacts_v1.read import (
    fold_statements,
    iter_structure_statements,
    live_contacts,
)


def _doc(*statements: str) -> str:
    """Wrap structure statements in a minimal (frame is ignored by the fold)."""
    return (
        "<contacts-v1> <begin_sequence> <begin_statements> "
        + " ".join(statements)
        + " <end>"
    )


# --------------------------------------------------------------------------
# Parsing statements out of the stream
# --------------------------------------------------------------------------


def test_iter_statements_preserves_kind_and_order():
    text = _doc(
        "<contact> <p10> <p20>",
        "<retract> <p10> <p20>",
        "<contact> <p5> <p30>",
    )
    assert list(iter_structure_statements(text)) == [
        ("contact", 10, 20),
        ("retract", 10, 20),
        ("contact", 5, 30),
    ]


def test_iter_statements_ignores_sequence_section_tokens():
    # <pX> <AA> pairs and the termini in the sequence section must not be
    # mistaken for statements (only <contact>/<retract> triples match).
    text = (
        "<contacts-v1> <begin_sequence> <p22> <PHE> <n-term> <p20> "
        "<begin_statements> <contact> <p20> <p21> <end>"
    )
    assert list(iter_structure_statements(text)) == [("contact", 20, 21)]


# --------------------------------------------------------------------------
# The fold: contact adds, retract removes, order matters
# --------------------------------------------------------------------------


def test_no_retract_is_the_emitted_set():
    text = _doc(
        "<contact> <p10> <p20>",
        "<contact> <p5> <p30>",
    )
    assert live_contacts(text) == frozenset({(10, 20), (5, 30)})


def test_retract_removes_the_pair():
    text = _doc(
        "<contact> <p10> <p20>",
        "<contact> <p5> <p30>",
        "<retract> <p10> <p20>",
    )
    assert live_contacts(text) == frozenset({(5, 30)})


def test_orientation_is_canonicalised():
    # Contact written j-first (coin-flip), retract written i-first — still
    # matches, because both canonicalise to (min, max).
    text = _doc(
        "<contact> <p20> <p10>",
        "<retract> <p10> <p20>",
    )
    assert live_contacts(text) == frozenset()


def test_long_distance_retract():
    # A contact retracted many statements later still parses correctly.
    stmts = ["<contact> <p0> <p50>"]
    stmts += [f"<contact> <p{i}> <p{i + 40}>" for i in range(1, 20)]
    stmts.append("<retract> <p0> <p50>")
    live = live_contacts(_doc(*stmts))
    assert (0, 50) not in live
    assert (1, 41) in live and (19, 59) in live
    assert len(live) == 19


def test_reemit_after_retract():
    text = _doc(
        "<contact> <p10> <p20>",
        "<retract> <p10> <p20>",
        "<contact> <p10> <p20>",
    )
    assert live_contacts(text) == frozenset({(10, 20)})


# --------------------------------------------------------------------------
# Malformed edit lists: tolerated, but counted
# --------------------------------------------------------------------------


def test_clean_document_has_no_anomalies():
    res = fold_statements(
        iter_structure_statements(
            _doc("<contact> <p1> <p10>", "<retract> <p1> <p10>", "<contact> <p2> <p20>")
        )
    )
    assert res.live == frozenset({(2, 20)})
    assert res.n_contact == 2
    assert res.n_retract == 1
    assert res.n_retract_absent == 0
    assert res.n_reemit == 0
    assert res.n_redundant_contact == 0


def test_retract_of_absent_pair_is_noop_and_counted():
    res = fold_statements(iter_structure_statements(_doc("<retract> <p1> <p2>")))
    assert res.live == frozenset()
    assert res.n_retract == 1
    assert res.n_retract_absent == 1


def test_double_retract_counts_second_as_absent():
    res = fold_statements(
        iter_structure_statements(
            _doc("<contact> <p1> <p2>", "<retract> <p1> <p2>", "<retract> <p1> <p2>")
        )
    )
    assert res.live == frozenset()
    assert res.n_retract == 2
    assert res.n_retract_absent == 1


def test_reemit_is_counted():
    res = fold_statements(
        iter_structure_statements(
            _doc("<contact> <p1> <p2>", "<retract> <p1> <p2>", "<contact> <p1> <p2>")
        )
    )
    assert res.live == frozenset({(1, 2)})
    assert res.n_reemit == 1
    assert res.n_redundant_contact == 0


def test_redundant_contact_is_counted():
    res = fold_statements(
        iter_structure_statements(
            _doc("<contact> <p1> <p2>", "<contact> <p1> <p2>")
        )
    )
    assert res.live == frozenset({(1, 2)})
    assert res.n_redundant_contact == 1
    assert res.n_reemit == 0


def test_empty_structure_section():
    assert live_contacts(_doc()) == frozenset()
