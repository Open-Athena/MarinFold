# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
"""Gate A's reducer, pinned against hand-computed values.

The metric is copied verbatim from exp89 and the whole point of copying it is
that it must not drift.  These tests are the cheap proof that the copy behaves
the way the definition says: R-precision cuts at the number of TRUE contacts,
ranks by vote count, and honours the min-separation-6 rule.
"""
from __future__ import annotations

import numpy as np

from score_gate_a import metrics_for, paired_report


def _score(L, votes):
    m = np.zeros((L, L), np.float32)
    for (i, j), v in votes.items():
        m[i, j] = v
        m[j, i] = v
    return m.astype(np.float16)


def test_r_precision_cuts_at_the_true_count():
    L = 20
    gt = {(0, 10), (1, 12), (2, 15)}                 # 3 true, all sep >= 6
    # (5, 19) is a high-scoring FALSE pair that displaces one true pair.
    score = _score(L, {(0, 10): 100, (1, 12): 50, (5, 19): 30, (2, 15): 10})
    m = metrics_for(score, gt, L)
    assert m["all:n_true"] == 3
    # top-3 by score = (0,10) true, (1,12) true, (5,19) false -> 2/3
    assert abs(m["all:R"] - 2.0 / 3.0) < 1e-9


def test_min_separation_excludes_near_diagonal_truth():
    """A contact at separation < 6 is not a candidate and not counted true."""
    L = 20
    gt = {(0, 3), (0, 10)}                            # (0,3) is sep 3 -> dropped
    score = _score(L, {(0, 10): 100, (0, 3): 999})
    m = metrics_for(score, gt, L)
    assert m["all:n_true"] == 1
    # the sep-3 pair cannot be ranked at all, so the one true pair is found
    assert abs(m["all:R"] - 1.0) < 1e-9


def test_perfect_and_empty_predictions():
    L = 30
    gt = {(0, 10), (1, 12), (2, 15)}
    perfect = _score(L, {p: 100 for p in gt})
    assert abs(metrics_for(perfect, gt, L)["all:R"] - 1.0) < 1e-9
    # No votes at all: every candidate ties at 0 and mergesort takes the first
    # three by index, which are near-diagonal pairs -- so R-precision is 0.
    assert metrics_for(_score(L, {}), gt, L)["all:R"] == 0.0


def test_paired_report_signs_and_ci():
    """delta is finetune - base, and a uniform +0.1 shift must land outside 0."""
    base = {f"u{i}": {"all:R": 0.5} for i in range(200)}
    ft = {f"u{i}": {"all:R": 0.6} for i in range(200)}
    r = paired_report(base, ft, n_boot=2000)
    assert r["n"] == 200
    assert abs(r["delta_mean"] - 0.1) < 1e-9
    assert r["delta_ci95"][0] > 0            # a real gain excludes zero
    assert r["frac_finetune_better"] == 1.0
    # and the reverse direction is symmetric
    r2 = paired_report(ft, base, n_boot=2000)
    assert abs(r2["delta_mean"] + 0.1) < 1e-9
    assert r2["delta_ci95"][1] < 0


def _write_votes(root, label, unit, L, votes, part=0):
    import pyarrow as pa
    import pyarrow.parquet as pq
    d = root / label
    d.mkdir(parents=True, exist_ok=True)
    ds, stem = unit
    rows = [dict(dataset=ds, stem=stem, L=L, i=i, j=j, votes=v)
            for (i, j), v in votes.items()]
    schema = pa.schema([("dataset", pa.string()), ("stem", pa.string()),
                        ("L", pa.int32()), ("i", pa.int16()), ("j", pa.int16()),
                        ("votes", pa.int16())])
    pq.write_table(pa.Table.from_pylist(rows, schema=schema),
                   d / f"shard-000-of-001-part-{part:04d}.parquet")


def test_load_votes_round_trips_local_paths(tmp_path):
    """The worker writes fsspec URIs; Gate A reduces from LOCAL disk on the node.

    Local paths go through fsspec's LocalFileSystem, where `glob` returns bare
    paths and `unstrip_protocol` puts `file://` back on.  This is the one part of
    the chain that never ran on the TPU path, so it gets an explicit test.
    """
    from score_gate_a import load_votes
    _write_votes(tmp_path, "base", ("cameo", "7abc_A"), 20,
                 {(0, 10): 100, (1, 12): 50})
    mats, lengths = load_votes(str(tmp_path), "base")
    assert list(mats) == [("cameo", "7abc_A")]
    m = mats[("cameo", "7abc_A")]
    assert m.shape == (20, 20)
    assert m[0, 10] == 100 and m[10, 0] == 100      # symmetrised
    assert lengths[("cameo", "7abc_A")] == 20


def test_duplicate_parts_are_fatal(tmp_path):
    """A retried shard covering a protein twice would SUM its votes."""
    import pytest
    from score_gate_a import load_votes
    _write_votes(tmp_path, "base", ("cameo", "7abc_A"), 20, {(0, 10): 100}, part=0)
    _write_votes(tmp_path, "base", ("cameo", "7abc_A"), 20, {(0, 10): 100}, part=1)
    with pytest.raises(SystemExit, match="more than one part"):
        load_votes(str(tmp_path), "base")
