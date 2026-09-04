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
from rank_pairwise import SEED_RANGES, select_seeds, stratum_quotas, top_pairs

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


def test_seed_ranges_match_the_metric_ranges():
    """A seed labelled `long` must be what the long-range metric scores.

    ``build_metrics.RANGES`` is copied verbatim from exp89 and must not be
    edited, so this is the direction the check has to run: the seed bins are
    asserted against it, not the other way round.
    """
    from build_metrics import RANGES

    assert SEED_RANGES == {k: v for k, v in RANGES.items() if k != "all"}


def test_stratum_quotas_spend_every_seed_longest_first():
    assert stratum_quotas(100) == {"long": 34, "medium": 33, "short": 33}
    assert stratum_quotas(99) == {"long": 33, "medium": 33, "short": 33}
    assert sum(stratum_quotas(101).values()) == 101


def _descending_matrix(L: int) -> np.ndarray:
    """A symmetric matrix whose scores fall with separation.

    Under this matrix `top` alone would take only short-range pairs, which is
    what makes the stratified and long strategies visibly different from it.
    """
    i, j = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    matrix = 1.0 / (1.0 + np.abs(i - j))
    return (matrix + matrix.T) / 2


def test_strategies_select_the_bins_they_claim():
    matrix = _descending_matrix(120)

    ii, jj, _, labels = select_seeds(matrix, 100, MIN_SEP, "top")
    assert set(labels) == {"short"}, "the fixture should make top short-heavy"

    ii, jj, _, labels = select_seeds(matrix, 100, MIN_SEP, "stratified")
    assert len(ii) == 100
    counts = {name: int((labels == name).sum()) for name in SEED_RANGES}
    assert counts == stratum_quotas(100)
    separation = jj - ii
    assert np.all(separation[labels == "long"] >= 24)
    assert np.all((separation[labels == "medium"] >= 12)
                  & (separation[labels == "medium"] <= 23))
    assert np.all((separation[labels == "short"] >= MIN_SEP)
                  & (separation[labels == "short"] <= 11))
    # Round-robin: the first three seeds cover all three bins, so a partial run
    # is balanced rather than 34 long rollouts followed by 33 medium ones.
    assert set(labels[:3]) == set(SEED_RANGES)

    ii, jj, _, labels = select_seeds(matrix, 100, MIN_SEP, "long")
    assert len(ii) == 100 and set(labels) == {"long"}
    assert np.all(jj - ii >= 24)


def test_every_strategy_returns_distinct_pairs():
    matrix = _descending_matrix(120)
    for strategy in ("top", "stratified", "long"):
        ii, jj, _, _ = select_seeds(matrix, 100, MIN_SEP, strategy)
        assert len({(int(a), int(b)) for a, b in zip(ii, jj)}) == 100, strategy


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="unknown seed strategy"):
        select_seeds(_descending_matrix(60), 10, MIN_SEP, "sideways")
