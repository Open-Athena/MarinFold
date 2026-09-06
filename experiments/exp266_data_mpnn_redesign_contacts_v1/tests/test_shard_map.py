# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""The source-part -> corpus-shard join, on hand-built partitions.

These cover the bug that cost the ESM-Atlas arm a rerun. The two shardings of
the 65 M ESM-Atlas entry ids agree on index for most files and disagree near
the end, so pairing shard *i* with part *i* passes any spot check taken from
the agreeing region and silently drops rows outside it. `build_map` must
therefore key off the ids and never off the index -- including in the case
where the index order and the id order are the *same*, which is what makes the
wrong implementation look right.
"""

from __future__ import annotations

from build_shard_map import build_map


def part(lo: str, hi: str, rows: int = 100):
    return (rows, lo, hi)


def test_identical_partitions_map_to_identity():
    source = [part("00", "0f"), part("10", "1f"), part("20", "2f")]
    assert build_map(source, list(source)) == {0: [0], 1: [1], 2: [2]}


def test_corpus_is_a_subset_of_each_part():
    """Decontamination removes rows, so the corpus range sits inside the source's."""
    source = [part("00", "0f"), part("10", "1f")]
    corpus = [part("01", "0e"), part("11", "1e")]
    assert build_map(source, corpus) == {0: [0], 1: [1]}


def test_offset_cuts_span_two_shards():
    """The two shardings need not cut the id order at the same places."""
    source = [part("00", "0f"), part("10", "1f"), part("20", "2f")]
    corpus = [part("00", "17"), part("18", "27"), part("28", "2f")]
    assert build_map(source, corpus) == {0: [0], 1: [0, 1], 2: [1, 2]}


def test_permuted_corpus_index_follows_ids_not_index():
    """The real failure: corpus index order is not id order.

    Shards 1 and 2 are swapped relative to the source. An index-aligned join
    would pair part 1 with the shard holding part 2's ids and find nothing --
    which is exactly the `no documents from 0 backbones` the fan-out hit.
    """
    source = [part("00", "0f"), part("10", "1f"), part("20", "2f")]
    corpus = [part("00", "0f"), part("20", "2f"), part("10", "1f")]
    assert build_map(source, corpus) == {0: [0], 1: [2], 2: [1]}


def test_partial_overlap_is_not_silently_dropped():
    """A part overlapping two out-of-order shards must name both.

    This is the case that produced incomplete output rather than an error: the
    index-aligned join matched *some* of the part's rows, so it wrote a file
    and reported success.
    """
    source = [part("00", "1f"), part("20", "3f")]
    corpus = [part("18", "2f"), part("00", "17"), part("30", "3f")]
    assert build_map(source, corpus) == {0: [0, 1], 1: [0, 2]}


def test_source_part_with_no_kept_rows_maps_to_nothing():
    """An empty list is reported, not hidden -- `main` turns it into an error."""
    source = [part("00", "0f"), part("10", "1f")]
    corpus = [part("00", "0f")]
    assert build_map(source, corpus) == {0: [0], 1: []}


def test_overlapping_corpus_shards_are_all_returned():
    """The corpus shards really do overlap each other, so this must work.

    An earlier version rejected this case, on the theory that both sides were
    non-overlapping partitions. The source is; the corpus is not. Nothing about
    the mapping needs it to be -- an interval intersection is indifferent to
    overlap, and returning both shards is the answer that keeps every row.
    """
    source = [part("00", "1f")]
    corpus = [part("00", "17"), part("10", "1f")]
    assert build_map(source, corpus) == {0: [0, 1]}


def test_boundary_touch_counts_as_overlap():
    """Inclusive on both ends: an extra shard is cheap, a dropped row is not."""
    source = [part("10", "1f")]
    corpus = [part("00", "10"), part("1f", "2f")]
    assert build_map(source, corpus) == {0: [0, 1]}
