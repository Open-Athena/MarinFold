# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the retraction diagnostics (#160) — pure, no model.

These metrics are the experiment's pass/fail, so the definitions are pinned
here on hand-built edit lists with known answers.

Run from this directory::

    uv run pytest test_retraction_diagnostics.py -q
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from retraction_diagnostics import aggregate, diagnose_document  # noqa: E402


def C(a, b):
    return ("contact", a, b)


def R(a, b):
    return ("retract", a, b)


def test_perfect_discrimination():
    # Emits 2 true + 2 false; retracts exactly the two false ones.
    gt = {(1, 10), (2, 20)}
    stmts = [C(1, 10), C(5, 50), C(2, 20), C(6, 60), R(5, 50), R(6, 60)]
    d = diagnose_document(stmts, gt)
    assert (d.retracted_fp, d.retracted_tp, d.kept_fp, d.kept_tp) == (2, 0, 0, 2)

    s = aggregate([d])
    assert s["fp_base_rate"] == 0.5
    assert s["retract_precision_fp"] == 1.0     # everything retracted was wrong
    assert s["retract_recall_fp"] == 1.0        # every mistake was caught
    assert s["retract_enrichment"] == 2.0       # == 1 / base_rate, the ceiling


def test_no_signal_gives_enrichment_one():
    # Retracts one true and one false — exactly the 50/50 base rate.
    gt = {(1, 10), (2, 20)}
    stmts = [C(1, 10), C(5, 50), C(2, 20), C(6, 60), R(1, 10), R(5, 50)]
    d = diagnose_document(stmts, gt)
    s = aggregate([d])
    assert s["fp_base_rate"] == 0.5
    assert s["retract_precision_fp"] == 0.5
    assert s["retract_enrichment"] == 1.0       # no discrimination


def test_no_retraction_at_all():
    gt = {(1, 10)}
    d = diagnose_document([C(1, 10), C(5, 50)], gt)
    s = aggregate([d])
    assert s["frac_docs_with_retraction"] == 0.0
    assert s["mean_retracts_per_doc"] == 0.0
    assert math.isnan(s["retract_precision_fp"])   # undefined, not 0
    assert math.isnan(s["retract_enrichment"])


def test_orientation_is_canonicalised():
    # Contact written j-first, retract i-first — same pair.
    gt = {(1, 10)}
    d = diagnose_document([C(50, 5), R(5, 50)], gt)
    assert d.retracted_fp == 1
    assert d.kept_fp == 0


def test_retract_distance_and_immediacy():
    gt = set()
    # (5,50) retracted 4 statements later; (6,60) retracted immediately.
    stmts = [C(5, 50), C(1, 11), C(2, 22), C(3, 33), R(5, 50), C(6, 60), R(6, 60)]
    d = diagnose_document(stmts, gt)
    assert sorted(d.distances) == [1, 4]
    s = aggregate([d])
    assert s["mean_retract_distance"] == 2.5
    assert s["frac_immediate_retractions"] == 0.5


def test_recovery_counts_true_contact_on_freed_residue():
    # Retract the wrong (5,50), then emit the TRUE (5,40) — a recovery.
    gt = {(5, 40)}
    d = diagnose_document([C(5, 50), C(1, 11), R(5, 50), C(5, 40)], gt)
    assert d.n_recovered == 1
    assert aggregate([d])["recovery_rate"] == 1.0


def test_recovery_requires_a_true_contact():
    # The follow-up contact is also wrong -> not a recovery.
    gt = {(9, 90)}
    d = diagnose_document([C(5, 50), R(5, 50), C(5, 41)], gt)
    assert d.n_recovered == 0
    assert aggregate([d])["recovery_rate"] == 0.0


def test_reemission_still_counts_as_retracted():
    # Emitted, retracted, re-emitted: the pair was retracted at some point, and
    # it is a TRUE contact, so it lands in retracted_tp (a trigger false alarm).
    gt = {(1, 10)}
    d = diagnose_document([C(1, 10), C(2, 20), R(1, 10), C(1, 10)], gt)
    assert d.retracted_tp == 1
    assert d.kept_tp == 0


def test_malformed_retract_is_counted_not_crashed():
    d = diagnose_document([R(7, 70), C(1, 10)], {(1, 10)})
    assert d.n_retract_absent == 1
    assert aggregate([d])["n_retract_absent"] == 1


def test_aggregate_pools_across_documents():
    gt = {(1, 10)}
    a = diagnose_document([C(1, 10), C(5, 50), R(5, 50)], gt)
    b = diagnose_document([C(1, 10), C(6, 60), R(6, 60)], gt)
    s = aggregate([a, b])
    assert s["n_documents"] == 2
    assert s["n_emitted_pairs"] == 4
    assert s["retract_precision_fp"] == 1.0
    assert s["retract_enrichment"] == 2.0
