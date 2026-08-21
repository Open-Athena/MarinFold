# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
"""The Gate B counters, pinned.

Gate B is a threshold on a count, so the count is the whole gate: if
``split_sections`` is off by one, "the multi-draft habit leaked into plain
generation" becomes unmeasurable. These pin the three behaviours that decide it.
"""
from __future__ import annotations

from eval_modes_worker import contacts_of, f1_of, jaccard, split_sections

BEGIN = "<begin_statements>"


def test_a_clean_single_section_counts_as_one():
    """The prompt ends ON <begin_statements>, so the first section is un-prefixed."""
    assert len(split_sections("<contact> <p10> <p20> <end>")) == 1


def test_a_leaked_second_section_counts_as_two():
    """This is the #163 failure Gate B exists to detect."""
    text = f"<contact> <p10> <p20> <end> {BEGIN} <contact> <p11> <p21> <end>"
    assert len(split_sections(text)) == 2


def test_a_trailing_marker_does_not_invent_a_section():
    """#200: the empty tail after a final marker changed the COUNT on 3/2216."""
    assert len(split_sections(f"<contact> <p10> <p20> {BEGIN}")) == 1
    assert len(split_sections(f"<contact> <p10> <p20> {BEGIN}   ")) == 1


def test_contacts_unwrap_the_nterm_offset_and_drop_near_diagonal():
    """Position tokens carry contacts-v1's random N-terminal offset."""
    # nterm=100, so <p110>/<p130> are residues 10 and 30 -- separation 20, kept.
    assert contacts_of("<contact> <p110> <p130>", nterm=100, L=50) == {(10, 30)}
    # separation 3 is below MIN_SEP and must be dropped
    assert contacts_of("<contact> <p110> <p113>", nterm=100, L=50) == set()
    # out of range for this protein
    assert contacts_of("<contact> <p110> <p199>", nterm=100, L=50) == set()
    # canonical ordering: the same pair either way round is one contact
    both = contacts_of("<contact> <p130> <p110> <contact> <p110> <p130>", nterm=100, L=50)
    assert both == {(10, 30)}


def test_f1_and_jaccard_edges():
    assert f1_of(set(), {(1, 10)}) == 0.0
    assert f1_of({(1, 10)}, set()) == 0.0
    assert f1_of({(1, 10)}, {(1, 10)}) == 1.0
    # two empty sections are identical, not undefined -- they must not count as
    # diverse, or a collapsed model would look healthy.
    assert jaccard(set(), set()) == 1.0
    assert jaccard({(1, 10)}, {(1, 10)}) == 1.0
    assert jaccard({(1, 10)}, {(2, 20)}) == 0.0
