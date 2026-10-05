"""Tests for leakage decisions whose mistakes would silently certify dirty targets."""

import pytest

from analyze_survival import contaminates
from build_candidates import foldbench_members
from build_complex_sequences import chain_sequences


def test_ring_wrap_and_shuffled_statements() -> None:
    document = (
        "<contacts-v1> <begin_sequence> <p12> <TRP> <p0> <GLY> "
        "<c-term> <p12> <p1999> <ALA> <n-term> <p1999> "
        "<c-term> <p0> <n-term> <p12> <begin_statements> <end>"
    )
    assert sorted(chain_sequences(document, [1, 2])) == ["AG", "W"]


def test_incomplete_document_is_not_a_shorter_training_sequence() -> None:
    with pytest.raises(ValueError, match="incomplete"):
        chain_sequences(
            "<begin_sequence> <n-term> <p0> <p0> <ALA> <c-term> <p2> <begin_statements>",
            [3],
        )


def test_short_training_fragment_excludes_long_query() -> None:
    hit = {
        "nident": "30",
        "alnlen": "100",
        "qstart": "101",
        "qend": "200",
        "qlen": "1000",
        "tstart": "1",
        "tend": "100",
        "tlen": "100",
    }
    assert contaminates(hit)


def test_rounded_identity_does_not_create_a_false_exclusion() -> None:
    hit = {
        "nident": "29",
        "alnlen": "97",
        "qstart": "1",
        "qend": "97",
        "qlen": "100",
        "tstart": "1",
        "tend": "97",
        "tlen": "100",
    }
    assert not contaminates(hit)


def test_high_identity_short_motif_does_not_exclude_whole_protein() -> None:
    hit = {
        "nident": "20",
        "alnlen": "20",
        "qstart": "1",
        "qend": "20",
        "qlen": "200",
        "tstart": "1",
        "tend": "20",
        "tlen": "200",
    }
    assert not contaminates(hit)


def test_foldbench_symmetry_copy_labels_are_not_collapsed() -> None:
    rows = [
        {"interface_chain_id_1": "A", "interface_chain_id_2": "A-2"},
        {"interface_chain_id_1": "A-2", "interface_chain_id_2": "B"},
    ]
    assert foldbench_members(rows) == ["A", "A-2", "B"]
