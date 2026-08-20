# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for exp208's consensus scorer — issue #208.

The load-bearing one is :func:`test_matches_exp89_metric_rows`. exp208's reward
is only "literally the deployed metric" if this module computes the same number
as ``build_rollout_rows.metric_rows``, which carries exp89's ``compute_metrics``
verbatim and is what every published MarinFold R-precision came from. exp82's
README already records that its *own* earlier ``metrics()`` disagreed with exp89's
by up to 0.4/protein on small proteins through float16 tie-breaking, so "close
enough" is a documented way to be wrong here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2] / "exp82_evals_contacts_v1_contact_prediction"),
)

import consensus as C  # noqa: E402


def _exp89_rprecision(score, resolved, true_pairs, length, rng="all"):
    """R-precision through exp82/exp89's own code, as an independent oracle."""
    # build_rollout_rows imports sklearn for AUC at module scope. The RL workspace
    # venv deliberately does not carry the analysis stack (pandas/sklearn would
    # join a pinned, fragile marin resolution), so this oracle is only available
    # in an ordinary env. Skip loudly rather than let the most important test in
    # this file quietly not run.
    pytest.importorskip(
        "sklearn",
        reason="run the fidelity test in an analysis env (system python or a venv with "
               "scikit-learn); the exp208 RL workspace pins marin and omits it on purpose",
    )
    from build_rollout_rows import RANGES, metric_rows, resolved_pairs

    tmat = np.zeros((length, length), bool)
    for i, j in true_pairs:
        tmat[i, j] = True
    pi, pj, psep = resolved_pairs(np.asarray(resolved, dtype=np.int64))
    rows = metric_rows(score, tmat, pi, pj, psep, length, with_precision=True)
    lo, hi = RANGES[rng]
    for row in rows:
        if row["range"] == rng and row["cut"] == "R":
            return row["precision"], row["n_true"]
    raise AssertionError("no R row")


@pytest.mark.parametrize("seed", range(8))
def test_matches_exp89_metric_rows(seed):
    """Same votes, same GT, same universe -> same number as the published path."""
    rs = np.random.default_rng(seed)
    length = int(rs.integers(40, 90))
    n_resolved = int(rs.integers(length // 2, length + 1))
    resolved = np.sort(rs.choice(length, size=n_resolved, replace=False))

    pairs, position = C.candidate_index(length, resolved=resolved)
    if len(pairs) < 20:
        pytest.skip("degenerate universe")

    n_true = int(rs.integers(5, max(6, len(pairs) // 8)))
    true_rows = rs.choice(len(pairs), size=n_true, replace=False)
    gt = {(int(i), int(j)) for i, j in pairs[true_rows]}

    # A group of rollouts, each a random subset of candidates, biased toward truth
    # so the vote matrix looks like a real one (many ties at low counts).
    n_rollouts = 12
    pair_sets = []
    for _ in range(n_rollouts):
        picks = rs.choice(len(pairs), size=int(rs.integers(5, 40)), replace=False)
        chosen = {(int(i), int(j)) for i, j in pairs[picks]}
        chosen |= {p for p in gt if rs.random() < 0.3}
        pair_sets.append(chosen)

    votes = C.vote_counts(pair_sets, position, len(pairs))
    is_true = C.truth_mask(pairs, gt)
    mine = C.rprecision(votes.sum(axis=0), is_true, int(is_true.sum()))

    # exp89's path wants the dense [L, L] float16 matrix fetch_cw_scores writes.
    score = np.zeros((length, length), np.float32)
    total = votes.sum(axis=0)
    for (i, j), v in zip(map(tuple, pairs), total):
        score[i, j] = v
        score[j, i] = v
    theirs, their_n_true = _exp89_rprecision(
        score.astype(np.float16).astype(np.float64), resolved, gt, length
    )

    assert their_n_true == int(is_true.sum())
    assert mine == pytest.approx(theirs, abs=1e-12)


def test_candidate_order_is_triu_over_resolved():
    """Ties fall back on this order, so it has to be the metric's order."""
    resolved = [1, 4, 7, 9]
    pairs, _ = C.candidate_index(12, resolved=resolved, min_sep=1)
    assert [tuple(p) for p in pairs] == [
        (1, 4), (1, 7), (1, 9), (4, 7), (4, 9), (7, 9),
    ]


def test_empty_rollout_has_exactly_zero_marginal():
    """The property that makes this term oppose collapse-to-silence."""
    pairs, position = C.candidate_index(40)
    gt = {(0, 10), (1, 20), (5, 30)}
    is_true = C.truth_mask(pairs, gt)
    pair_sets = [set(gt), {(0, 10)}, set()]
    votes = C.vote_counts(pair_sets, position, len(pairs))

    _, marginals = C.loo_marginals(votes, is_true, len(gt))
    assert marginals[2] == 0.0
    # ... and after centring it is strictly negative, because its siblings helped.
    centred = marginals - marginals.mean()
    assert centred[2] < 0


def test_unique_true_contact_beats_a_duplicate_one():
    """A rollout is paid for what its siblings missed, not for agreeing."""
    pairs, position = C.candidate_index(60)
    gt = {(0, 10), (1, 20), (2, 30), (3, 40)}
    is_true = C.truth_mask(pairs, gt)

    # Three siblings all emit the same two true contacts. One rollout adds a
    # third, unique true contact; the other just repeats the consensus.
    common = {(0, 10), (1, 20)}
    unique = [common | {(2, 30)}, common, common, common]
    duplicate = [set(common), common, common, common]

    v_unique = C.vote_counts(unique, position, len(pairs))
    v_dup = C.vote_counts(duplicate, position, len(pairs))
    _, m_unique = C.loo_marginals(v_unique, is_true, len(gt))
    _, m_dup = C.loo_marginals(v_dup, is_true, len(gt))

    assert m_unique[0] > 0
    assert m_dup[0] == 0.0
    assert m_unique[0] > m_dup[0]


def test_wrong_contact_that_displaces_a_true_one_is_penalised():
    """The other half of the signal: adding noise into the top-R costs you.

    Two rollouts vote the truth; two vote a pair of false contacts that tie on
    vote count and win the stable-sort tie-break on index order, pushing both
    true pairs out of top-R. Dropping either false-voting rollout breaks the tie
    and restores a perfect consensus, so its marginal is strongly negative.
    """
    pairs, position = C.candidate_index(60)
    gt = {(0, 10), (1, 20)}
    is_true = C.truth_mask(pairs, gt)

    false_pairs = {(0, 6), (0, 7)}          # candidate rows 0 and 1 -- earliest
    assert not (false_pairs & gt)
    groups = [set(gt), set(gt), set(false_pairs), set(false_pairs)]
    votes = C.vote_counts(groups, position, len(pairs))
    consensus, marginals = C.loo_marginals(votes, is_true, len(gt))

    assert consensus == 0.0                  # both true pairs displaced
    assert marginals[2] < 0 and marginals[3] < 0
    assert marginals[2] == pytest.approx(-1.0)


def test_union_below_r_is_visible_in_diagnostics():
    """Vote collapse in its most direct form: not enough distinct pairs to fill top-R."""
    pairs, position = C.candidate_index(80)
    gt = {(i, i + 10) for i in range(20)}
    is_true = C.truth_mask(pairs, gt)

    narrow = [{(0, 10), (1, 11)} for _ in range(8)]
    diag = C.group_diagnostics(C.vote_counts(narrow, position, len(pairs)), is_true, len(gt))
    assert diag["union"] == 2.0
    assert diag["union_over_r"] < 1.0
    assert diag["mean_jaccard"] == pytest.approx(1.0)


def test_rollout_precision_recall_basic():
    pairs, position = C.candidate_index(40)
    gt = {(0, 10), (1, 20), (2, 30)}
    is_true = C.truth_mask(pairs, gt)
    votes = C.vote_counts([{(0, 10), (1, 20), (0, 7)}], position, len(pairs))

    out = C.rollout_precision_recall(votes[0], is_true, len(gt))
    assert out["n_pred"] == 3
    assert out["precision"] == pytest.approx(2 / 3)
    assert out["recall"] == pytest.approx(2 / 3)
    assert out["f1"] == pytest.approx(2 / 3)


def test_unscoreable_protein_is_nan_not_zero():
    pairs, position = C.candidate_index(40)
    is_true = C.truth_mask(pairs, set())
    votes = C.vote_counts([{(0, 10)}, {(1, 20)}], position, len(pairs))
    consensus, marginals = C.loo_marginals(votes, is_true, 0)
    assert np.isnan(consensus)
    assert np.isnan(marginals).all()
