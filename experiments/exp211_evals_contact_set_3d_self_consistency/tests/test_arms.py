# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the issue #211 contact-set arms.

The load-bearing property across all of them is **size matching** — #142 measured
rollouts emitting ~0.70x the ground-truth contact count, and a sparser set embeds
more easily, so an arm that quietly returns the wrong number of contacts would
manufacture the experiment's headline effect.
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arms import (  # noqa: E402
    decoy_protein,
    ground_truth,
    marginal_chimera,
    separation_matched_random,
    splice_chimera,
    subsample,
)


def rng(seed=0):
    return np.random.default_rng(seed)


def a_contact_set(n=40, length=100, seed=0):
    r = rng(seed)
    out = set()
    while len(out) < n:
        i, j = sorted(r.integers(0, length, size=2))
        if j - i >= 6:
            out.add((int(i), int(j)))
    return sorted(out)


# --------------------------------------------------------------------------


def test_ground_truth_canonicalizes_orientation():
    # contacts-v1 coin-flips each pair's orientation, so both spellings must fold
    # to one canonical pair.
    assert ground_truth([(9, 2), (2, 9), (3, 11)]) == [(2, 9), (3, 11)]


def test_subsample_is_size_matched_and_a_subset():
    src = a_contact_set(40)
    got = subsample(src, 25, rng())
    assert len(got) == 25
    assert set(got) <= set(src)


def test_subsample_caps_at_available():
    src = a_contact_set(10)
    assert len(subsample(src, 999, rng())) == 10


# --------------------------------------------------------------------------


def test_marginal_chimera_is_size_matched_and_deduplicated():
    votes = {p: i + 1 for i, p in enumerate(a_contact_set(60))}
    got = marginal_chimera(votes, 30, rng())
    assert len(got) == 30 == len(set(got))
    assert set(got) <= set(votes)


def test_marginal_chimera_respects_the_vote_weights():
    # Pairs with 100x the votes should dominate. This is the property that makes
    # arm 4 marginal-matched; if it degenerated to uniform sampling the null would
    # be far too easy to beat and the headline effect would be inflated.
    heavy = a_contact_set(10, seed=1)
    light = a_contact_set(10, seed=2)
    light = [p for p in light if p not in set(heavy)]
    votes = {**{p: 100 for p in heavy}, **{p: 1 for p in light}}
    picks = Counter()
    for s in range(200):
        picks.update(marginal_chimera(votes, len(heavy), rng(s)))
    heavy_share = sum(picks[p] for p in heavy) / sum(picks.values())
    assert heavy_share > 0.9


def test_marginal_chimera_returns_everything_when_size_exceeds_pool():
    votes = {p: 1 for p in a_contact_set(5)}
    assert len(marginal_chimera(votes, 50, rng())) == 5


def test_marginal_chimera_rejects_all_zero_votes():
    with pytest.raises(ValueError, match="zero"):
        marginal_chimera({p: 0 for p in a_contact_set(5)}, 3, rng())


# --------------------------------------------------------------------------


def test_splice_chimera_is_size_matched_despite_overlap():
    # Two rollouts sharing most of their contacts: the naive union dedupes below
    # `size`, and the top-up has to make it back up.
    shared = a_contact_set(30, seed=3)
    a = shared + a_contact_set(10, seed=4)
    b = shared + a_contact_set(10, seed=5)
    pool = sorted(set(a) | set(b))
    got = splice_chimera(a, b, 30, rng(), pool=pool)
    assert len(got) == 30 == len(set(got))


def test_splice_chimera_draws_from_both_rollouts():
    a = a_contact_set(30, seed=6)
    b = a_contact_set(30, seed=7)
    b = [p for p in b if p not in set(a)]
    got = set(splice_chimera(a, b, 20, rng(), pool=sorted(set(a) | set(b))))
    assert got & set(a) and got & set(b)


# --------------------------------------------------------------------------


def test_separation_matched_random_preserves_the_separation_profile():
    src = a_contact_set(40, length=120)
    got = separation_matched_random(src, 120, rng())
    assert Counter(j - i for i, j in got) == Counter(j - i for i, j in src)


def test_separation_matched_random_respects_min_sep_and_bounds():
    src = a_contact_set(30, length=80)
    got = separation_matched_random(src, 80, rng())
    assert all(j - i >= 6 and 0 <= i < j < 80 for i, j in got)
    assert len(got) == len(set(got))


# --------------------------------------------------------------------------


def test_decoy_protein_clips_to_length_and_size():
    # Donor is longer than the target, so out-of-range pairs are dropped before
    # the size match. Pick a donor dense enough that enough survive — in the real
    # run the donor is chosen from proteins of similar length for this reason.
    donor = a_contact_set(200, length=120)
    got = decoy_protein(donor, 100, 25, rng())
    assert len(got) == 25
    assert all(j < 100 for _, j in got)


def test_decoy_protein_caps_at_what_survives_clipping():
    # A donor whose contacts nearly all lie beyond the target length cannot fill
    # the quota; the arm returns what it has rather than inventing pairs.
    donor = [(i, i + 6) for i in range(150, 190)]
    got = decoy_protein(donor, 100, 25, rng())
    assert got == []
