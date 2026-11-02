# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the rollout readout (issue #211).

``parse_rollout`` is the one piece of the worker that runs without vLLM and the
one place a silent bug would poison every downstream number — in particular the
per-realization ``<pX>`` map, which differs between rollouts of the same protein
and whose omission would misalign rollouts against each other while leaving every
count looking plausible.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen_rollouts_worker import MIN_SEP, parse_rollout  # noqa: E402


def straight_map(length, start=0, num_pos=2000):
    """The realization map contacts-v1 builds: wrap-around index -> 0-based position."""
    return {(start + t) % num_pos: t for t in range(length)}


def test_reads_contacts_in_emission_order():
    m = straight_map(50)
    text = "<contact> <p0> <p10> <contact> <p3> <p20> <end>"
    contacts, n_emitted, oor, close = parse_rollout(text, m)
    assert [(i, j) for i, j, _ in contacts] == [(0, 10), (3, 20)]
    assert n_emitted == 2 and oor == 0 and close == 0


def test_canonicalizes_pair_orientation():
    # contacts-v1 coin-flips each pair's orientation, so both spellings are the
    # same contact and must land on the same (min, max) key.
    m = straight_map(50)
    a, *_ = parse_rollout("<contact> <p0> <p10>", m)
    b, *_ = parse_rollout("<contact> <p10> <p0>", m)
    assert [(i, j) for i, j, _ in a] == [(i, j) for i, j, _ in b] == [(0, 10)]


def test_applies_the_wraparound_map():
    # The realization starts at p1990 on a 20-residue chain, so p1990 is position
    # 0 and p5 is position 15 — a rollout parsed without this map would record
    # (5, 1990) and be nonsense.
    m = straight_map(20, start=1990)
    contacts, *_ = parse_rollout("<contact> <p1990> <p5>", m)
    assert [(i, j) for i, j, _ in contacts] == [(0, 15)]


def test_flags_duplicates_without_dropping_them():
    m = straight_map(50)
    contacts, n_emitted, _, _ = parse_rollout(
        "<contact> <p0> <p10> <contact> <p10> <p0> <contact> <p1> <p20>", m
    )
    assert [d for *_, d in contacts] == [False, True, False]
    assert n_emitted == 3
    # The distinct-contact count the rollouts table stores excludes duplicates.
    assert sum(1 for *_, d in contacts if not d) == 2


def test_rejects_out_of_range_positions():
    # A position the realization never defined: the model invented an index
    # outside this protein's map.
    m = straight_map(20)
    contacts, n_emitted, oor, _ = parse_rollout("<contact> <p0> <p900>", m)
    assert contacts == [] and n_emitted == 1 and oor == 1


def test_rejects_pairs_below_min_separation():
    # contacts-v1 forbids |i - j| < 6 by definition, so a model emitting one is
    # producing something the format cannot mean.
    m = straight_map(50)
    contacts, n_emitted, _, close = parse_rollout(
        f"<contact> <p0> <p{MIN_SEP - 1}>", m
    )
    assert contacts == [] and n_emitted == 1 and close == 1


def test_rejects_self_contact():
    m = straight_map(50)
    contacts, _, oor, _ = parse_rollout("<contact> <p7> <p7>", m)
    assert contacts == [] and oor == 1


def test_ignores_surrounding_tokens_and_partial_statements():
    # Real completions carry the prompt's sequence section, whitespace variation,
    # and — when the token budget runs out — a truncated trailing statement.
    m = straight_map(60)
    text = (
        "<p0> <ALA>\n<n-term> <p0>\n<begin_statements>\n"
        "<contact>  <p0>  <p30>\n<contact> <p2> <p40>\n<contact> <p5>"
    )
    contacts, n_emitted, _, _ = parse_rollout(text, m)
    assert [(i, j) for i, j, _ in contacts] == [(0, 30), (2, 40)]
    assert n_emitted == 2


def test_empty_completion():
    contacts, n_emitted, oor, close = parse_rollout("<end>", straight_map(30))
    assert contacts == [] and n_emitted == 0 and oor == 0 and close == 0
