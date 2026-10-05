"""Chain reconstruction, including the ring wrap and the disjointness guard."""

import pytest

from experiments.exp343_models_complex_corpus_training.complex_sections import (
    RING,
    Role,
    chain_runs,
    index_to_chain,
    label,
    position,
)


def document(
    sequence: list[tuple[int, str]],
    statements: list[str],
    *,
    termini_in_sequence: list[str] | None = None,
) -> str:
    """Assemble a document, optionally with termini inside the sequence section.

    The published corpus puts them there -- the section is shuffled and the
    terminus statements shuffle with it -- so both placements are exercised.
    """
    parts = [f"<p{index}> <{code}>" for index, code in sequence]
    if termini_in_sequence:
        parts = termini_in_sequence + parts
    return " ".join(
        ["<contacts-v1>", "<begin_sequence>", " ".join(parts), "<begin_statements>",
         *statements, "<end>"]
    )


def test_position_reads_only_position_tokens() -> None:
    assert position("<p1999>") == 1999
    assert position("<p0>") == 0
    assert position("<contact>") is None
    assert position("<PHE>") is None
    assert position("<pad>") is None


def test_two_chains_are_paired_by_the_first_c_term_reached() -> None:
    runs = chain_runs(
        [("<n-term>", 643), ("<n-term>", 1816), ("<c-term>", 793), ("<c-term>", 1966)]
    )
    assert runs == [(643, 793), (1816, 1966)]


def test_a_chain_wrapping_past_index_1999_is_paired_correctly() -> None:
    # 1900..1999, 0..49 is one chain; 200..300 is the other.
    runs = chain_runs(
        [("<n-term>", 1900), ("<c-term>", 49), ("<n-term>", 200), ("<c-term>", 300)]
    )
    assert runs == [(200, 300), (1900, 49)]
    owner = index_to_chain(runs)
    assert owner[1999] == 1 and owner[0] == 1 and owner[49] == 1
    assert owner[250] == 0
    assert 150 not in owner
    assert len(owner) == 101 + 150


def test_unbalanced_termini_are_rejected() -> None:
    with pytest.raises(ValueError, match="N-termini and"):
        chain_runs([("<n-term>", 10), ("<n-term>", 500), ("<c-term>", 100)])


def test_a_chain_running_into_the_next_chains_start_is_rejected() -> None:
    with pytest.raises(ValueError, match="before any C-terminus"):
        chain_runs(
            [("<n-term>", 10), ("<n-term>", 50), ("<c-term>", 9), ("<c-term>", 500)]
        )


def test_overlapping_runs_are_rejected() -> None:
    with pytest.raises(ValueError, match="is claimed by chains"):
        index_to_chain([(10, 30), (20, 40)])


def test_contacts_split_by_chain_membership() -> None:
    # Chain 0: 100..102. Chain 1: 500..502.
    text = document(
        [(100, "ALA"), (101, "GLY"), (102, "SER"), (500, "PHE"), (501, "LYS"),
         (502, "THR")],
        [
            "<n-term> <p100>", "<c-term> <p102>", "<n-term> <p500>", "<c-term> <p502>",
            "<contact> <p100> <p102>",
            "<contact> <p101> <p501>",
            "<contact> <p900> <p101>",
        ],
    )
    sections = label(text)
    assert sections.num_chains == 2
    assert (sections.contacts_intra, sections.contacts_inter) == (1, 1)
    # A contact naming an index no chain claims is counted apart, never folded
    # into either class.
    assert sections.contacts_unresolved == 1
    assert sections.roles[sections.tokens.index("<begin_sequence>") + 1] == Role.SEQUENCE
    assert sections.roles[-1] == Role.END
    assert sections.roles.count(Role.CONTACT_INTER) == 3
    assert sections.roles.count(Role.CONTACT_INTRA) == 3
    assert sections.roles.count(Role.TERMINUS) == 8


def test_every_token_gets_exactly_one_role() -> None:
    text = document(
        [(5, "ALA"), (6, "GLY"), (1998, "SER"), (1999, "PHE")],
        [
            "<n-term> <p5>", "<c-term> <p6>", "<n-term> <p1998>", "<c-term> <p1999>",
            "<contact> <p5> <p1999>",
        ],
    )
    sections = label(text)
    assert len(sections.roles) == len(sections.tokens)
    assert sections.contacts_inter == 1
    assert sections.num_chains == 2
    assert set(sections.chain_of_index) == {5, 6, 1998, 1999}


def test_a_monomer_document_has_one_chain_and_no_interface() -> None:
    text = document(
        [(0, "ALA"), (1, "GLY"), (2, "SER")],
        ["<n-term> <p0>", "<c-term> <p2>", "<contact> <p0> <p2>"],
    )
    sections = label(text)
    assert sections.num_chains == 1
    assert (sections.contacts_intra, sections.contacts_inter) == (1, 0)


def test_a_single_chain_spanning_the_whole_ring_is_accepted() -> None:
    termini = [("<n-term>", 0), ("<c-term>", RING - 1)]
    assert chain_runs(termini) == [(0, RING - 1)]
    assert len(index_to_chain(chain_runs(termini))) == RING


def test_a_non_contacts_v1_document_is_rejected() -> None:
    with pytest.raises(ValueError, match="not a contacts-v1 document"):
        label("<contacts-and-crops-v1> <begin_sequence> <p0> <ALA> <end>")


def test_an_unexpected_statement_token_is_rejected() -> None:
    text = document(
        [(0, "ALA"), (1, "GLY")],
        ["<n-term> <p0>", "<c-term> <p1>", "<retract> <p0> <p1>"],
    )
    with pytest.raises(ValueError, match="Unexpected statement token"):
        label(text)


def test_termini_inside_the_sequence_section_are_labelled_terminus() -> None:
    # This is the published corpus's own layout: the sequence section is
    # shuffled and the terminus statements travel inside it.
    text = document(
        [(100, "ALA"), (101, "GLY"), (102, "SER"), (500, "PHE"), (501, "LYS"),
         (502, "THR")],
        ["<contact> <p100> <p102>", "<contact> <p101> <p501>"],
        termini_in_sequence=[
            "<n-term> <p100>", "<c-term> <p102>", "<n-term> <p500>", "<c-term> <p502>",
        ],
    )
    sections = label(text)
    assert sections.num_chains == 2
    assert (sections.contacts_intra, sections.contacts_inter) == (1, 1)
    # Eight terminus tokens, and none of them charged to the sequence loss.
    assert sections.roles.count(Role.TERMINUS) == 8
    assert sections.roles.count(Role.SEQUENCE) == 12
