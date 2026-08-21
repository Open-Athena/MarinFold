# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pieces of exp254 that a wrong answer would look plausible in.

The metric code is copied verbatim from exp89 and is tested there; what is new
here is the seeding — the prompt construction, the orientation coin flip, the
rank ordering of the seed list, and the claim that the two arms share their
document realizations. Each of those can be wrong in a way that still produces a
number.

    uv run pytest test_exp254.py
"""

import random

import numpy as np
import pytest

from common import BEGIN, MIN_SEP, parse_rollout, realization, seed_statement
from rank_pairwise import top_pairs

SEQUENCE = "MSEVKELLEEFLKRNKPVRIHHKNGEEIKVRITHIGEDTVEFELNGRSAAEIL"


@pytest.fixture(scope="module")
def residues():
    from marinfold.document_structures.contacts_v1 import residues_from_sequence

    return residues_from_sequence(SEQUENCE)


def test_arms_share_their_realizations(residues):
    """The seeded prompt is the i.i.d. prompt plus exactly one statement.

    This is what makes the two arms paired: realization *r* is the same document
    in both, so any difference in the scores is the seeding and not a different
    draw of the N-terminus or the statement order.
    """
    for r in range(5):
        prefix, seq_positions = realization("test", residues, f"r{r}")
        again, again_positions = realization("test", residues, f"r{r}")
        assert prefix == again and seq_positions == again_positions
        seeded = prefix + seed_statement(seq_positions[3], seq_positions[20],
                                         random.Random(r))
        assert seeded.startswith(prefix)
        assert seeded[len(prefix):].split() == [
            "<contact>", f"<p{seq_positions[3]}>", f"<p{seq_positions[20]}>",
        ] or seeded[len(prefix):].split() == [
            "<contact>", f"<p{seq_positions[20]}>", f"<p{seq_positions[3]}>",
        ]


def test_realization_prefix_ends_at_the_structure_boundary(residues):
    prefix, seq_positions = realization("test", residues, "r0")
    assert prefix.endswith(BEGIN)
    assert len(seq_positions) == len(SEQUENCE)
    assert len(set(seq_positions)) == len(seq_positions)


def test_seed_orientation_is_flipped_not_fixed():
    """contacts-v1 randomizes each pair's orientation, so the seed must too."""
    seen = {seed_statement(11, 77, random.Random(k)) for k in range(50)}
    assert seen == {" <contact> <p11> <p77>", " <contact> <p77> <p11>"}


def test_parse_rollout_orders_dedupes_and_filters():
    pos_to_seq = {100 + k: k for k in range(30)}
    text = (
        "<contact> <p110> <p120> "      # (10, 20) -- kept, first
        "<contact> <p101> <p103> "      # separation 2 -- below MIN_SEP
        "<contact> <p120> <p110> "      # the same pair, reversed -- deduped
        "<contact> <p105> <p125> "      # (5, 25) -- kept, second
        "<contact> <p105> <p900> "      # not this realization's ring -- dropped
        "<end>"
    )
    assert parse_rollout(text, pos_to_seq) == [(10, 20), (5, 25)]
    assert MIN_SEP == 6


def test_top_pairs_are_ranked_and_respect_min_separation():
    rng = np.random.default_rng(0)
    L = 40
    matrix = rng.random((L, L))
    matrix = matrix + matrix.T
    ii, jj, scores = top_pairs(matrix, 25, MIN_SEP)
    assert len(ii) == 25
    assert np.all(jj - ii >= MIN_SEP)
    assert np.all(np.diff(scores) <= 0)
    assert len({(int(a), int(b)) for a, b in zip(ii, jj)}) == 25


def test_top_pairs_caps_at_the_candidate_count():
    matrix = np.ones((12, 12))
    ii, _, _ = top_pairs(matrix, 1000, MIN_SEP)
    assert len(ii) == sum(max(0, 12 - MIN_SEP - i) for i in range(12))
