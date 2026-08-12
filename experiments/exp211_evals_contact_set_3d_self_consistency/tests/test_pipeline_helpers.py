# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the issue #211 pipeline helpers.

These cover the pure functions that decide what gets measured — sequence
reconstruction, per-rollout accuracy, and the memory-bounded batching — where a
silent bug would not crash anything, it would just produce a wrong number.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prepare_targets import sequence_from_prefix  # noqa: E402
from score_arms import accuracy, chunk_by_pairs  # noqa: E402


# --------------------------------------------------------------------------
# Sequence reconstruction from a published prompt
# --------------------------------------------------------------------------


def test_sequence_from_prefix_uses_the_realization_map():
    # The realization starts at p1990 and wraps: p1990 -> position 0, p0 ->
    # position 10. Reading <pX> as a position directly would scramble the chain.
    positions = [(1990 + t) % 2000 for t in range(12)]
    prefix = " ".join(f"<p{p}> <ALA>" for p in positions)
    assert sequence_from_prefix(prefix, positions, 12) == "A" * 12


def test_sequence_from_prefix_is_order_free():
    # contacts-v1 shuffles the sequence section, so the statements arrive in a
    # random order and the map — not the order — decides each residue's index.
    positions = list(range(4))
    forward = "<p0> <MET> <p1> <LYS> <p2> <THR> <p3> <ALA>"
    shuffled = "<p2> <THR> <p0> <MET> <p3> <ALA> <p1> <LYS>"
    assert sequence_from_prefix(forward, positions, 4) == "MKTA"
    assert sequence_from_prefix(shuffled, positions, 4) == "MKTA"


def test_sequence_from_prefix_maps_unk_to_x():
    # The UNK round trip is what makes the whole reconstruction lossless:
    # <UNK> -> "X", and residues_from_sequence maps "X" back to UNK.
    positions = list(range(3))
    assert sequence_from_prefix("<p0> <MET> <p1> <UNK> <p2> <ALA>", positions, 3) == "MXA"


def test_sequence_from_prefix_rejects_an_incomplete_prefix():
    # A prefix that does not define every position must fail loudly — silently
    # emitting a short sequence would misalign every contact index downstream.
    with pytest.raises(ValueError, match="undefined"):
        sequence_from_prefix("<p0> <MET>", [0, 1, 2], 3)


def test_sequence_from_prefix_ignores_contact_statements():
    # Real prefixes end at <begin_statements>, but be robust to trailing text:
    # a <contact> statement has no amino acid and must not be read as one.
    positions = list(range(2))
    prefix = "<p0> <MET> <p1> <ALA> <begin_statements> <contact> <p0> <p1>"
    assert sequence_from_prefix(prefix, positions, 2) == "MA"


# --------------------------------------------------------------------------
# Per-rollout accuracy
# --------------------------------------------------------------------------


def test_accuracy_basic():
    gt = {(0, 10), (1, 20), (2, 30)}
    a = accuracy([(0, 10), (1, 20), (5, 40)], gt)
    assert a["precision"] == pytest.approx(2 / 3)
    assert a["recall"] == pytest.approx(2 / 3)
    assert a["f1"] == pytest.approx(2 / 3)
    assert a["n_pred"] == 3


def test_accuracy_is_orientation_and_order_free():
    gt = {(0, 10)}
    assert accuracy([(0, 10)], gt)["precision"] == 1.0
    # Callers hand in canonicalized pairs; a duplicate must not inflate n_pred.
    assert accuracy([(0, 10), (0, 10)], gt)["n_pred"] == 1


def test_accuracy_empty_prediction():
    a = accuracy([], {(0, 10)})
    assert np.isnan(a["precision"]) and a["recall"] == 0.0 and a["n_pred"] == 0


def test_accuracy_no_overlap():
    a = accuracy([(0, 10)], {(1, 20)})
    assert a["precision"] == 0.0 and a["f1"] == 0.0


# --------------------------------------------------------------------------
# Memory-bounded batching
# --------------------------------------------------------------------------


def test_chunk_by_pairs_shrinks_with_length():
    # The non-contact constraint list is O(L^2) per row, so a batch that fits at
    # L=150 must not be attempted at L=761.
    small = chunk_by_pairs(1000, 150, 4, 40_000_000)
    large = chunk_by_pairs(1000, 761, 4, 40_000_000)
    assert small > large >= 1


def test_chunk_by_pairs_scales_with_the_budget():
    assert (chunk_by_pairs(1000, 200, 4, 80_000_000)
            > chunk_by_pairs(1000, 200, 4, 40_000_000))


def test_chunk_by_pairs_accounts_for_restarts():
    # Restarts are extra rows, so more of them means fewer sets per call.
    assert (chunk_by_pairs(1000, 200, 1, 40_000_000)
            > chunk_by_pairs(1000, 200, 8, 40_000_000))


def test_chunk_by_pairs_never_returns_zero():
    # A single very long protein must still be scored, one set at a time.
    assert chunk_by_pairs(10, 2000, 8, 1) == 1
