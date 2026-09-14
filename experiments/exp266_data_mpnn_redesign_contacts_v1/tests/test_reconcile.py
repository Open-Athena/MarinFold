# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Which source parts the ESM reconciliation asks to re-run.

The output of `parts_to_rerun` decides whether ~2,800 already-written shards
get regenerated, so its distinction between "lost a few backbones the way
every shard does" and "has a hole where a join went wrong" is worth pinning.
"""

from __future__ import annotations

from reconcile_esm import parts_to_rerun

# Every part reads its own shard plus the two neighbours it overlaps, which is
# the real shape of the map (mean 3.00 corpus shards per source part).
MAP = {j: [j - 1, j, j + 1] for j in range(1, 9)}


def test_uniform_low_loss_reruns_nothing():
    """0.2 % lost per shard is the measured filtered/degenerate rate."""
    by_shard = {i: 40 for i in range(10)}
    assert parts_to_rerun(by_shard, MAP, [], rate=0.002) == []


def test_a_hole_reruns_every_part_that_reads_it():
    """One shard losing everything pulls in the three parts that overlap it."""
    by_shard = {i: 40 for i in range(10)}
    by_shard[5] = 19_600
    assert parts_to_rerun(by_shard, MAP, [], rate=0.002) == [4, 5, 6]


def test_missing_outputs_are_always_rerun():
    """A part with no output file at all needs no evidence beyond its absence."""
    by_shard = {i: 0 for i in range(10)}
    assert parts_to_rerun(by_shard, MAP, [7, 2], rate=0.0) == [2, 7]


def test_the_floor_stops_a_clean_corpus_flagging_everything():
    """With rate ~0 the relative threshold collapses; the floor holds it up."""
    by_shard = {i: 5 for i in range(10)}
    assert parts_to_rerun(by_shard, MAP, [], rate=0.0) == []


def test_threshold_scales_with_the_measured_rate():
    """A shard is judged against how lossy this corpus actually is.

    300 misses is a hole when shards typically lose 40, and unremarkable when
    they typically lose 400 -- the same count, two different verdicts.
    """
    by_shard = {i: 40 for i in range(10)}
    by_shard[5] = 300
    assert parts_to_rerun(by_shard, MAP, [], rate=0.002) == [4, 5, 6]
    assert parts_to_rerun(by_shard, MAP, [], rate=0.02) == []
