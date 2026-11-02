# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the issue #211 consistency tiers."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from consistency import (  # noqa: E402
    Bounds,
    contact_matrix,
    embed_residual,
    packing_score,
    separation,
    smooth_upper_bounds,
    triangle_violations,
)


# --------------------------------------------------------------------------
# Representation
# --------------------------------------------------------------------------


def test_contact_matrix_is_symmetric_and_order_free():
    a = contact_matrix([(0, 7), (3, 12)], 20)
    b = contact_matrix([(7, 0), (12, 3)], 20)
    assert np.array_equal(a, b)
    assert np.array_equal(a, a.T)
    assert not a.diagonal().any()


def test_contact_matrix_rejects_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        contact_matrix([(0, 20)], 20)


def test_contact_matrix_drops_self_pairs():
    assert not contact_matrix([(5, 5)], 10).any()


def test_separation():
    s = separation(4)
    assert s[0, 3] == 3 and s[2, 2] == 0
    assert np.array_equal(s, s.T)


# --------------------------------------------------------------------------
# T1 — packing
# --------------------------------------------------------------------------


def test_packing_score_counts_partners_not_pairs():
    # Residue 0 contacts 3 partners; the pair count is 3, the max degree is 3.
    p = packing_score(contact_matrix([(0, 6), (0, 7), (0, 8)], 20))
    assert p["max_degree"] == 3
    assert p["n_contacts"] == 3


def test_packing_score_empty():
    p = packing_score(contact_matrix([], 10))
    assert p["max_degree"] == 0 and p["n_contacts"] == 0


# --------------------------------------------------------------------------
# T2 — triangle-inequality bound smoothing
# --------------------------------------------------------------------------


def test_smooth_upper_bounds_walks_the_backbone():
    # With no contacts the only path between i and j is along the chain.
    u = smooth_upper_bounds(contact_matrix([], 5), Bounds())
    assert u[0, 1] == pytest.approx(3.80)
    assert u[0, 4] == pytest.approx(4 * 3.80)


def test_smooth_upper_bounds_shortcuts_through_a_contact():
    # A contact (0, 10) is a 12 A shortcut, beating the 10 * 3.8 = 38 A chain path.
    u = smooth_upper_bounds(contact_matrix([(0, 10)], 11), Bounds())
    assert u[0, 10] == pytest.approx(12.0)
    # And it tightens the neighbour: 0 -> 10 -> 9 is 12 + 3.8, vs 9 * 3.8 along the chain.
    assert u[0, 9] == pytest.approx(12.0 + 3.80)


def test_triangle_violations_fires_when_the_bound_system_demands_it():
    # The tier reports 0 on real data (Phase 0), so prove it *can* fire: with an
    # absurd 100 A non-contact lower bound, every scored pair is a violation.
    mask = contact_matrix([(0, 10)], 20)
    v = triangle_violations(mask, Bounds(l_noncontact=100.0))
    assert v["n_triangle_violations"] == v["n_scored_pairs"] > 0


def test_triangle_violations_silent_at_realistic_bounds():
    # The documented null: a dense-ish contact set produces no violations.
    rng = np.random.default_rng(0)
    length = 60
    pairs = set()
    while len(pairs) < 40:
        i, j = sorted(rng.integers(0, length, size=2))
        if j - i >= 6:
            pairs.add((int(i), int(j)))
    v = triangle_violations(contact_matrix(pairs, length), Bounds())
    assert v["n_triangle_violations"] == 0


def test_triangle_violations_excludes_close_in_sequence_pairs():
    # Pairs below min_sep are neither contacts nor non-contacts, so they are never
    # scored — otherwise every (i, i+1) would "violate" a 6 A lower bound.
    v = triangle_violations(contact_matrix([], 20), Bounds())
    n_far = int(((separation(20) >= 6) & np.triu(np.ones((20, 20), bool), 1)).sum())
    assert v["n_scored_pairs"] == n_far
    assert v["n_triangle_violations"] == 0


# --------------------------------------------------------------------------
# T3 — 3D embeddability
# --------------------------------------------------------------------------


def test_embed_residual_satisfies_an_unconstrained_chain():
    # No contacts: an extended chain satisfies everything, residual must be 0.
    r = embed_residual(contact_matrix([], 30), n_restarts=1, iters=400, device="cpu")
    assert r["contact_excess"] == pytest.approx(0.0)
    assert r["bond_err"] < 0.2


def test_embed_residual_flags_a_geometrically_impossible_set():
    # Every pair at separation >= 6 declared a contact: 60 residues cannot all sit
    # within 12 A of each other given a 4 A steric floor. (At L=40 this is only
    # *strained* — excess 5.1 A but nothing individually unsatisfiable — which is
    # itself a useful calibration of how much the metric asks for.)
    length = 60
    pairs = [(i, j) for i in range(length) for j in range(i + 6, length)]
    r = embed_residual(
        contact_matrix(pairs, length), n_restarts=2, iters=600, device="cpu"
    )
    assert r["contact_excess"] > 1.0
    assert r["unsat_frac"] > 0.0


def test_embed_residual_does_not_leak_across_a_batch():
    # The index_add_ reduction is the easiest place for one set's residual to land
    # in another's row. Scores are *not* expected to be bitwise identical to a
    # solo run — the batch shares one RNG stream, so a set's initial coordinates
    # depend on its position (see embed_residual's docstring). What must hold is
    # that a contact-free set reports exactly 0 whatever it sits next to, and an
    # impossible one stays impossible.
    sets = {
        "easy": contact_matrix([], 60),
        "hard": contact_matrix([(i, j) for i in range(60) for j in range(i + 6, 60)], 60),
    }
    for order in (["easy", "hard"], ["hard", "easy"]):
        rows = embed_residual(
            np.stack([sets[k] for k in order]), n_restarts=1, iters=400, seed=3, device="cpu"
        )
        by_set = dict(zip(order, rows))
        assert by_set["easy"]["contact_excess"] == pytest.approx(0.0)
        assert by_set["hard"]["contact_excess"] > 1.0


def test_embed_residual_min_over_restarts_is_monotone():
    # More restarts can only lower a min. Guards the torch.where reduction.
    mask = contact_matrix([(i, i + 8) for i in range(0, 30, 3)], 40)
    few = embed_residual(mask, n_restarts=1, iters=400, seed=0, device="cpu")
    many = embed_residual(mask, n_restarts=4, iters=400, seed=0, device="cpu")
    assert many["contact_excess"] <= few["contact_excess"] + 1e-6


def test_embed_residual_per_contact_normalization():
    pairs = [(0, 6), (1, 8), (2, 10)]
    r = embed_residual(contact_matrix(pairs, 20), n_restarts=1, iters=300, device="cpu")
    assert r["contact_excess_per_contact"] == pytest.approx(r["contact_excess"] / 3)


def test_embed_residual_single_vs_batch_shape():
    single = embed_residual(contact_matrix([], 15), n_restarts=1, iters=100, device="cpu")
    batched = embed_residual(
        contact_matrix([], 15)[None], n_restarts=1, iters=100, device="cpu"
    )
    assert isinstance(single, dict) and isinstance(batched, list) and len(batched) == 1
