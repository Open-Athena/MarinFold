# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Pin the rollout readout: what a completion is taken to have claimed (#160).

``read_rollout`` is the only place where a sampled document becomes numbers, and
every downstream figure inherits its choices. The cases below are the ones where
a plausible-looking implementation would be wrong in a way no aggregate metric
would reveal — a vote matrix that quietly counts retracted pairs looks exactly
like a model that never retracted.

Run in exp159's venv (it has ``marinfold``)::

    ../exp159_data_backtracking_corpus/.venv/bin/python -m pytest \\
        test_score_backtracking_worker.py -q
"""

from __future__ import annotations

import pytest

from score_backtracking_worker import MIN_SEP, read_rollout

# Sequence index k lives at position 100+k, so positions and indices can never be
# confused by accident: any off-by-one shows up as a 100-sized error.
SEQIDX = {100 + k: k for k in range(60)}


def stmt(kind: str, pos_a: int, pos_b: int) -> str:
    return f"<{kind}> <p{pos_a}> <p{pos_b}>"


def doc(*parts: str) -> str:
    return " ".join(["<begin_statements>", *parts, "<end>"])


def test_contacts_vote_and_map_to_sequence_space():
    out = read_rollout(doc(stmt("contact", 100, 110), stmt("contact", 105, 120)), SEQIDX)
    assert out.live == sorted([(0, 10), (5, 20)])
    assert out.statements == [(0, 0, 10), (0, 5, 20)]
    assert out.n_unmapped == 0


def test_retracted_pair_does_not_vote():
    """The whole reason this worker exists: exp82's regex would count (0, 10)."""
    out = read_rollout(
        doc(stmt("contact", 100, 110), stmt("contact", 105, 120), stmt("retract", 100, 110)),
        SEQIDX,
    )
    assert out.live == [(5, 20)]
    # ...but the retraction is still visible in the edit list, which is what the
    # diagnostics score.
    assert out.statements == [(0, 0, 10), (0, 5, 20), (1, 0, 10)]
    assert out.fold.n_retract == 1


def test_retract_matches_its_contact_written_in_either_orientation():
    out = read_rollout(doc(stmt("contact", 110, 100), stmt("retract", 100, 110)), SEQIDX)
    assert out.live == []
    assert out.fold.n_retract_absent == 0


def test_reemitted_pair_votes_again():
    out = read_rollout(
        doc(stmt("contact", 100, 110), stmt("retract", 100, 110), stmt("contact", 100, 110)),
        SEQIDX,
    )
    assert out.live == [(0, 10)]
    assert out.fold.n_reemit == 1


def test_votes_drop_near_diagonal_but_statements_keep_it():
    """Ordering is load-bearing: filtering the stream would shift retraction distances."""
    near = MIN_SEP - 1
    out = read_rollout(
        doc(stmt("contact", 100, 100 + near), stmt("contact", 100, 130)), SEQIDX
    )
    assert out.live == [(0, 30)]
    assert out.statements == [(0, 0, near), (0, 0, 30)]


def test_self_pair_never_votes():
    out = read_rollout(doc(stmt("contact", 100, 100)), SEQIDX)
    assert out.live == []


def test_unmapped_position_is_counted_not_silently_dropped():
    out = read_rollout(doc(stmt("contact", 100, 999), stmt("contact", 100, 110)), SEQIDX)
    assert out.live == [(0, 10)]
    assert out.n_unmapped == 1
    assert [s[0] for s in out.statements] == [0]


def test_retract_of_a_pair_never_emitted_is_a_counted_no_op():
    out = read_rollout(doc(stmt("contact", 100, 110), stmt("retract", 105, 120)), SEQIDX)
    assert out.live == [(0, 10)]
    assert out.fold.n_retract_absent == 1


def test_a_document_with_no_retractions_reads_exactly_like_exp82():
    """The control arm has to be measured by the same code path, unchanged."""
    pairs = [(100, 110), (105, 120), (101, 140)]
    out = read_rollout(doc(*[stmt("contact", a, b) for a, b in pairs]), SEQIDX)
    assert out.live == sorted((min(a, b) - 100, max(a, b) - 100) for a, b in pairs)
    assert out.fold.n_retract == 0


def test_truncated_completion_still_reads_its_complete_statements():
    """Rollouts that hit the token budget end mid-statement; the prefix still counts."""
    text = "<begin_statements> " + stmt("contact", 100, 110) + " <contact> <p105>"
    out = read_rollout(text, SEQIDX)
    assert out.live == [(0, 10)]


@pytest.mark.parametrize("kind", ["contact", "retract"])
def test_whitespace_variation_is_tolerated(kind):
    out = read_rollout(f"<{kind}>\n <p100>  <p110>", SEQIDX)
    assert len(out.statements) == 1
