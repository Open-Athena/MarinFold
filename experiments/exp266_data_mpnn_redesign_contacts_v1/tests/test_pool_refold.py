# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Matching the two refold arms on backbone id.

The bug this pins: `pool_refold` compared every design refold against every
native refold without joining on `entry_id`, so the two rates were measured
over different protein sets while the README called the result "paired". It
moved the headline scTM ratio by 3.5 points. The join is one line, and it was
untested, which is why nothing caught it.
"""

from __future__ import annotations

from pool_refold import (bootstrap_ratio, by_entry, match_arms,
                         matched_ratio, rate)


def refold(entry_id: str, rmsd: float, tm: float, seq_len: int = 150) -> dict:
    return {"entry_id": entry_id, "sc_rmsd": rmsd, "sc_tm": tm,
            "seq_len": seq_len, "mpnn_temperature": 0.1}


def test_matching_excludes_design_backbones_with_no_control():
    """The design arm covers backbones the native arm does not; drop them.

    Here the uncontrolled backbone passes everything, so pooling would inflate
    the design rate to 2/3 while the matched rate is the honest 1/2.
    """
    design = [refold("a", 1.0, 0.9), refold("b", 5.0, 0.2), refold("c", 1.0, 0.9)]
    native = [refold("a", 1.0, 0.9), refold("b", 1.0, 0.9)]
    dby, nby = by_entry(design), by_entry(native)
    shared = sorted(set(dby) & set(nby))
    matched = [r for k in shared for r in dby[k]]

    assert shared == ["a", "b"]
    assert rate(design, "sc_rmsd", 2.0, False) == 2 / 3     # pooled, wrong
    assert rate(matched, "sc_rmsd", 2.0, False) == 1 / 2     # matched, right


def test_siblings_stay_together_in_the_bootstrap():
    """Resampling backbones, not refolds — otherwise the CI is far too narrow.

    Each backbone here is internally unanimous and the two disagree, so with
    two backbones resampled with replacement the design rate can only be 0,
    1/2 or 1 — and the ratio only 0.0, 0.5 or 1.0. A refold-level bootstrap
    would mix a backbone's siblings and manufacture intermediate values the
    sampling design cannot produce, reporting a tighter interval than the data
    support. The full 0–1 span here is the honest answer for n=2.
    """
    design = ([refold("a", 1.0, 0.9) for _ in range(8)]
              + [refold("b", 5.0, 0.2) for _ in range(8)])
    native = [refold("a", 1.0, 0.9), refold("b", 1.0, 0.9)]
    dby, nby = by_entry(design), by_entry(native)

    lo, hi = bootstrap_ratio(dby, nby, "sc_rmsd", 2.0, False, draws=4000, seed=0)
    assert (lo, hi) == (0.0, 1.0)

    # Every achievable ratio is one a whole-backbone draw can produce.
    seen = {round(bootstrap_ratio(dby, nby, "sc_rmsd", 2.0, False,
                                  draws=1, seed=k)[0], 3) for k in range(60)}
    assert seen <= {0.0, 0.5, 1.0}, seen


def test_ratio_is_one_when_designs_match_native():
    design = [refold(k, 1.0, 0.9) for k in "abcd"]
    native = [refold(k, 1.0, 0.9) for k in "abcd"]
    lo, hi = bootstrap_ratio(by_entry(design), by_entry(native),
                             "sc_tm", 0.5, True, draws=500, seed=0)
    assert lo == hi == 1.0


def test_no_shared_backbones_is_not_silently_zero():
    """Disjoint arms must not produce a number that looks like a measurement."""
    lo, hi = bootstrap_ratio(by_entry([refold("a", 1.0, 0.9)]),
                             by_entry([refold("z", 1.0, 0.9)]),
                             "sc_tm", 0.5, True, draws=100, seed=0)
    assert lo != lo and hi != hi          # NaN


def test_native_only_backbone_is_excluded_from_the_denominator():
    """The estimate and its interval must be computed on the same population.

    Reported on PR #267: the design arm was filtered to shared ids but the
    native denominator still ran over every native row, while `bootstrap_ratio`
    matched internally. One native-only backbone that fails drags the native
    rate down without touching the design rate, so the point estimate reads 2.0
    against a CI of [1.0, 1.0] — an estimate outside its own interval.
    """
    design = [refold("a", 1.0, 0.9)]
    native = [refold("a", 1.0, 0.9), refold("z", 5.0, 0.2)]   # "z" has no design

    dm, nm, shared = match_arms(design, native)
    assert shared == ["a"]
    assert len(nm) == 1 and nm[0]["entry_id"] == "a"

    ratio = matched_ratio(design, native, "sc_rmsd", 2.0, False)
    lo, hi = bootstrap_ratio(by_entry(design), by_entry(native),
                             "sc_rmsd", 2.0, False, draws=200, seed=0)
    assert ratio == 1.0                       # not 2.0
    assert lo <= ratio <= hi                  # estimate inside its own interval


def test_match_arms_drops_both_directions():
    design = [refold("a", 1.0, 0.9), refold("b", 1.0, 0.9)]
    native = [refold("b", 1.0, 0.9), refold("z", 1.0, 0.9)]
    dm, nm, shared = match_arms(design, native)
    assert shared == ["b"]
    assert {r["entry_id"] for r in dm} == {"b"}
    assert {r["entry_id"] for r in nm} == {"b"}


def test_matched_ratio_agrees_with_the_bootstrap_when_arms_are_unequal():
    """Fuzz the disagreement the bug produced: estimate must sit in its CI."""
    import random

    rng = random.Random(7)
    for trial in range(25):
        shared_ids = [f"s{i}" for i in range(6)]
        design = [refold(k, rng.choice([1.0, 5.0]), 0.9) for k in shared_ids for _ in range(4)]
        native = [refold(k, rng.choice([1.0, 5.0]), 0.9) for k in shared_ids]
        native += [refold(f"z{j}", 5.0, 0.2) for j in range(rng.randrange(0, 4))]
        r = matched_ratio(design, native, "sc_rmsd", 2.0, False)
        lo, hi = bootstrap_ratio(by_entry(design), by_entry(native),
                                 "sc_rmsd", 2.0, False, draws=400, seed=trial)
        if r == r and lo == lo:               # both defined
            assert lo <= r <= hi, (trial, r, lo, hi)
