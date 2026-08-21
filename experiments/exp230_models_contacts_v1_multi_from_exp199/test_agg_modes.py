# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
"""The five aggregation rules, pinned against hand-built cases.

These decide what "consensus", "best", "last" and "second-to-last" MEAN, so they
are worth more than the code that implements them.
"""
from __future__ import annotations

import numpy as np
import pytest

from score_agg_modes import f1, score_matrix
from score_gate_a import metrics_for

L = 40
A, B, C = (0, 10), (1, 12), (2, 15)      # true contacts, all separation >= 6
FALSE = (5, 30)


def test_consensus_ranks_by_vote_count():
    """A pair in every section must outrank one in a single section."""
    secs = [{A, B, FALSE}, {A, B}, {A}]
    M, n_pred = score_matrix("consensus", secs, {A, B, C}, L)
    assert M[A] == 3 and M[B] == 2 and M[FALSE] == 1
    assert M[A[1], A[0]] == 3            # symmetrised
    assert n_pred == 3                   # union of all sections
    # R = 3 true contacts; top-3 by votes = A, B, FALSE -> 2/3
    assert abs(metrics_for(M.astype(np.float16), {A, B, C}, L)["all:R"] - 2 / 3) < 1e-9


def test_best_is_an_oracle_and_picks_the_highest_f1_section():
    secs = [{FALSE}, {A, B, C}, {A}]     # index 1 is perfect
    M, n_pred = score_matrix("best", secs, {A, B, C}, L)
    assert n_pred == 3
    assert M[A] == 1 and M[B] == 1 and M[C] == 1 and M[FALSE] == 0
    assert metrics_for(M.astype(np.float16), {A, B, C}, L)["all:R"] == 1.0


def test_last_and_second_last_pick_by_position_not_quality():
    secs = [{A, B, C}, {A}, {FALSE}]     # best is FIRST; last is worst
    M_last, _ = score_matrix("last", secs, {A, B, C}, L)
    assert M_last[FALSE] == 1 and M_last[A] == 0
    M_2nd, n2 = score_matrix("second_last", secs, {A, B, C}, L)
    assert M_2nd[A] == 1 and M_2nd[FALSE] == 0 and n2 == 1


def test_a_short_prediction_is_capped_at_n_over_R():
    """The honest consequence of scoring an unranked SET at R = n_true."""
    secs = [{A}]                          # 1 predicted, 3 true
    M, n_pred = score_matrix("last", secs, {A, B, C}, L)
    assert n_pred == 1
    # top-3 = the one predicted pair (correct) + 2 index-ordered zeros -> 1/3
    assert abs(metrics_for(M.astype(np.float16), {A, B, C}, L)["all:R"] - 1 / 3) < 1e-9


def test_empty_rollout_scores_zero_not_nan():
    M, n_pred = score_matrix("last", [], {A, B, C}, L)
    assert n_pred == 0 and M.sum() == 0
    assert metrics_for(M.astype(np.float16), {A, B, C}, L)["all:R"] == 0.0


def test_second_last_falls_back_only_inside_score_matrix():
    """score_matrix tolerates one section; the SCORER skips it instead.

    That distinction matters: silently falling back to the only section would
    make `second_last` identical to `last` on every single-section rollout, which
    is exactly the population where the two differ most.
    """
    M, _ = score_matrix("second_last", [{A}], {A, B, C}, L)
    assert M[A] == 1                     # tolerated here
    # and the caller's guard is what keeps it honest -- see score_agg_modes.main


def test_f1_helper_edges():
    assert f1(set(), {A}) == 0.0
    assert f1({A}, set()) == 0.0
    assert f1({A, B}, {A, B}) == 1.0
    assert abs(f1({A, FALSE}, {A, B}) - 0.5) < 1e-9
